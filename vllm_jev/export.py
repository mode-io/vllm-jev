"""Compile an Open-Jev LoRA/head checkpoint into a native vLLM pooling model."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from peft import PeftModel
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig, AutoTokenizer, Qwen3_5ForCausalLM

from .model import ARCHITECTURE


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export(base: Path, adapter: Path, head_path: Path, output: Path) -> dict:
    base = base.resolve()
    adapter = adapter.resolve()
    head_path = head_path.resolve()
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    config = AutoConfig.from_pretrained(base, local_files_only=True).get_text_config()
    if config.hidden_size != 2560 or config.vocab_size != 248320:
        raise ValueError("this pinned export recipe expects Qwen3.5-4B text dimensions")
    model = Qwen3_5ForCausalLM.from_pretrained(
        base, config=config, dtype=torch.bfloat16,
        device_map={"": "cuda:0"}, local_files_only=True,
    )
    # Open-Jev trained LoRA on Qwen3_5TextModel, not the multimodal wrapper.
    text_backbone = model.model
    reference_weight = text_backbone.layers[0].linear_attn.in_proj_qkv.weight.detach().float().clone()
    peft = PeftModel.from_pretrained(text_backbone, adapter, is_trainable=False)
    model.model = peft.merge_and_unload(safe_merge=True)
    merged_weight = model.model.layers[0].linear_attn.in_proj_qkv.weight.detach().float()
    merge_max_delta = float((merged_weight - reference_weight).abs().max())
    if merge_max_delta <= 0:
        raise ValueError("LoRA merge did not change the backbone")
    del reference_weight
    model.eval()

    head = torch.load(head_path, map_location="cpu", weights_only=True)
    weight = head["weight"].detach().float().contiguous()
    bias = head["bias"].detach().float().contiguous()
    if weight.shape != (1, config.hidden_size) or bias.shape != (1,):
        raise ValueError(f"unexpected scalar head shapes: {weight.shape}, {bias.shape}")
    if not torch.isfinite(weight).all() or not torch.isfinite(bias).all():
        raise ValueError("nonfinite score head")

    model.save_pretrained(output, safe_serialization=True, max_shard_size="4GB")
    AutoTokenizer.from_pretrained(base, local_files_only=True).save_pretrained(output)

    # vLLM's Qwen3_5ForCausalLM expects ``model.layers.*`` while the HF
    # Qwen3_5ForCausalLM wrapper saves ``model.language_model.layers.*``.
    # Rewrite keys, retaining the exact tensor values and dtypes.
    index_path = output / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError("expected sharded text checkpoint index")
    index = json.loads(index_path.read_text())
    old_prefix = "model.language_model."
    new_prefix = "model."
    weight_map = index["weight_map"]
    if not all(name.startswith(old_prefix) for name in weight_map):
        raise ValueError("unexpected HF text checkpoint weight names")
    for filename in sorted(set(weight_map.values())):
        shard_path = output / filename
        with safe_open(shard_path, framework="pt", device="cpu") as file:
            tensors = {new_prefix + name.removeprefix(old_prefix): file.get_tensor(name)
                       for name in file.keys()}
        temporary = shard_path.with_suffix(".tmp.safetensors")
        save_file(tensors, temporary)
        os.replace(temporary, shard_path)
    index["weight_map"] = {new_prefix + name.removeprefix(old_prefix): filename
                           for name, filename in weight_map.items()}

    score_path = output / "score.safetensors"
    save_file({"score.weight": weight, "score.bias": bias}, score_path)
    if "score.weight" in index["weight_map"] or "score.bias" in index["weight_map"]:
        raise ValueError("export already contains a score head")
    index["weight_map"].update({"score.weight": score_path.name,
                                "score.bias": score_path.name})
    index["metadata"]["total_size"] += weight.numel() * weight.element_size() + bias.numel() * bias.element_size()
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")

    config_path = output / "config.json"
    saved_config = json.loads(config_path.read_text())
    saved_config["architectures"] = [ARCHITECTURE]
    saved_config["num_labels"] = 1
    saved_config["id2label"] = {"0": "jev_score"}
    saved_config["label2id"] = {"jev_score": 0}
    config_path.write_text(json.dumps(saved_config, indent=2, sort_keys=True) + "\n")

    manifest = {
        "format": "vllm-jev-pooling-v1",
        "architecture": ARCHITECTURE,
        "foundation_revision": base.name,
        "foundation_config_sha256": sha256(base / "config.json"),
        "adapter_config_sha256": sha256(adapter / "adapter_config.json"),
        "adapter_weights_sha256": sha256(adapter / "adapter_model.safetensors"),
        "head_sha256": sha256(head_path),
        "score_weight_dtype": str(weight.dtype),
        "merge_max_abs_delta_layer0": merge_max_delta,
        "score_bias": float(bias[0]),
        "score_semantics": "final hidden scalar; candidate softmax after all candidates; no token generation",
        "files": {p.name: sha256(p) for p in output.iterdir() if p.is_file()},
    }
    (output / "jev_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.base, args.adapter, args.head, args.output), indent=2))


if __name__ == "__main__":
    main()
