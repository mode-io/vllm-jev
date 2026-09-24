"""Compile an Open-Jev LoRA/head checkpoint into a native vLLM pooling model."""

import argparse
import json
import math
import os
from pathlib import Path

import torch
from peft import PeftModel
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoTokenizer, Qwen3_5ForCausalLM, Qwen3ForCausalLM

from . import ARCHITECTURE, QWEN3_ARCHITECTURE
from .checkpoint import sha256


def export(base: Path, adapter: Path, head_path: Path, output: Path) -> dict:
    base = base.resolve()
    adapter = adapter.resolve()
    head_path = head_path.resolve()
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    config = AutoConfig.from_pretrained(base, local_files_only=True).get_text_config()
    if config.hidden_size <= 0:
        raise ValueError("invalid text backbone hidden size")
    if config.model_type == "qwen3_5_text":
        architecture, prompt_protocol = ARCHITECTURE, "open_jev_choice"
        model = Qwen3_5ForCausalLM.from_pretrained(
            base,
            config=config,
            dtype=torch.bfloat16,
            device_map={"": "cuda:0"},
            local_files_only=True,
        )
        # This release trained LoRA on the inner text backbone.
        peft = PeftModel.from_pretrained(model.model, adapter, is_trainable=False)
        model.model = peft.merge_and_unload(safe_merge=True)
        old_prefix = "model.language_model."
    elif config.model_type == "qwen3":
        architecture, prompt_protocol = QWEN3_ARCHITECTURE, "openjev_branch_v03"
        model = Qwen3ForCausalLM.from_pretrained(
            base,
            config=config,
            dtype=torch.bfloat16,
            device_map={"": "cuda:0"},
            local_files_only=True,
        )
        peft = PeftModel.from_pretrained(model, adapter, is_trainable=False)
        model = peft.merge_and_unload(safe_merge=True)
        old_prefix = "model."
    else:
        raise ValueError(f"unsupported scalar-head backbone: {config.model_type}")
    model.eval()

    head = (
        load_file(str(head_path))
        if head_path.suffix == ".safetensors"
        else torch.load(head_path, map_location="cpu", weights_only=True)
    )
    weight = head["weight"].detach().float().contiguous()
    bias = head.get("bias", torch.zeros(1)).detach().float().contiguous()
    if weight.shape != (1, config.hidden_size) or bias.shape != (1,):
        raise ValueError(f"unexpected scalar head shapes: {weight.shape}, {bias.shape}")
    if not torch.isfinite(weight).all() or not torch.isfinite(bias).all():
        raise ValueError("nonfinite score head")

    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    if not tokenizer.chat_template:
        raise ValueError(
            "base snapshot is missing its chat template; download the complete tokenizer"
        )
    model.save_pretrained(output, safe_serialization=True, max_shard_size="4GB")
    tokenizer.save_pretrained(output)

    # vLLM's Qwen3_5ForCausalLM expects ``model.layers.*`` while the HF
    # Qwen3_5ForCausalLM wrapper saves ``model.language_model.layers.*``.
    # Rewrite keys, retaining the exact tensor values and dtypes.
    index_path = output / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
    else:
        # Smaller Qwen3.5 checkpoints may fit in one safetensors file.
        single = output / "model.safetensors"
        if not single.exists():
            raise FileNotFoundError("expected a safetensors checkpoint")
        with safe_open(single, framework="pt", device="cpu") as file:
            names = list(file.keys())
            total_size = sum(
                file.get_tensor(name).numel() * file.get_tensor(name).element_size()
                for name in names
            )
        index = {
            "metadata": {"total_size": total_size},
            "weight_map": {name: single.name for name in names},
        }
    new_prefix = "model."
    weight_map = index["weight_map"]
    if not all(
        name.startswith(old_prefix) or name == "lm_head.weight" for name in weight_map
    ):
        raise ValueError("unexpected HF text checkpoint weight names")
    rewritten_total_size = 0
    for filename in sorted(set(weight_map.values())):
        shard_path = output / filename
        with safe_open(shard_path, framework="pt", device="cpu") as file:
            tensors = {
                new_prefix + name.removeprefix(old_prefix): file.get_tensor(name)
                for name in file.keys()
                if name.startswith(old_prefix)
            }
        if tensors:
            rewritten_total_size += sum(
                tensor.numel() * tensor.element_size() for tensor in tensors.values()
            )
            temporary = shard_path.with_suffix(".tmp.safetensors")
            save_file(tensors, temporary)
            os.replace(temporary, shard_path)
        else:
            shard_path.unlink()
    index["weight_map"] = {
        new_prefix + name.removeprefix(old_prefix): filename
        for name, filename in weight_map.items()
        if name.startswith(old_prefix)
    }

    score_path = output / "score.safetensors"
    save_file({"score.weight": weight, "score.bias": bias}, score_path)
    if "score.weight" in index["weight_map"] or "score.bias" in index["weight_map"]:
        raise ValueError("export already contains a score head")
    index["weight_map"].update(
        {"score.weight": score_path.name, "score.bias": score_path.name}
    )
    index["metadata"]["total_size"] = (
        rewritten_total_size
        + weight.numel() * weight.element_size()
        + bias.numel() * bias.element_size()
    )
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")

    config_path = output / "config.json"
    saved_config = json.loads(config_path.read_text())
    saved_config["architectures"] = [architecture]
    saved_config["num_labels"] = 1
    saved_config["id2label"] = {"0": "jev_score"}
    saved_config["label2id"] = {"jev_score": 0}
    config_path.write_text(json.dumps(saved_config, indent=2, sort_keys=True) + "\n")

    calibration_path = head_path.with_name("temperature.json")
    calibration_temperature = None
    calibration_temperatures = None
    if calibration_path.is_file():
        calibration_temperature = float(
            json.loads(calibration_path.read_text())["temperature"]
        )
        if not math.isfinite(calibration_temperature) or calibration_temperature <= 0:
            raise ValueError("invalid calibration temperature")
    elif (head_path.with_name("calibration-v03.json")).is_file():
        calibration_path = head_path.with_name("calibration-v03.json")
        raw = json.loads(calibration_path.read_text())
        calibration_temperatures = {
            key: float(raw[key]["temperature"]) for key in ("choice", "noul", "score")
        }
        if any(
            not math.isfinite(value) or value <= 0
            for value in calibration_temperatures.values()
        ):
            raise ValueError("invalid calibration temperatures")

    manifest = {
        "format": "vllm-jev-pooling-v1",
        "architecture": architecture,
        "prompt_protocol": prompt_protocol,
        "foundation_revision": base.name,
        "foundation_config_sha256": sha256(base / "config.json"),
        "adapter_config_sha256": sha256(adapter / "adapter_config.json"),
        "adapter_weights_sha256": sha256(adapter / "adapter_model.safetensors"),
        "head_sha256": sha256(head_path),
        "score_weight_dtype": str(weight.dtype),
        "score_bias": float(bias[0]),
        "score_semantics": "final hidden scalar; candidate softmax after all candidates; no token generation",
        "files": {p.name: sha256(p) for p in output.iterdir() if p.is_file()},
    }
    if calibration_temperature is not None:
        manifest["calibration_temperature"] = calibration_temperature
        manifest["source_temperature_sha256"] = sha256(calibration_path)
    if calibration_temperatures is not None:
        manifest["calibration_temperatures"] = calibration_temperatures
        manifest["source_temperature_sha256"] = sha256(calibration_path)
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
