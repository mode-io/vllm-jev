"""Export a JevAny Pointer LoRA checkpoint for native vLLM pooling."""

import argparse
import json
import math
import os
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from . import (
    JEVANY_POINTER_QWEN3_VL_MOE_ARCHITECTURE,
    JEVANY_POINTER_QWEN35_ARCHITECTURE,
    JEVANY_POINTER_QWEN35_MOE_ARCHITECTURE,
)
from .checkpoint import sha256, verify_files

PROTOCOL = "jevany_pointer_v1"
MODELS = {
    "SimpleJev/JevAny-Qwen3.5-4B-LoRA": "1c7aa9bab14ac347aeb917c0bcd757838a8a78ce",
    "SimpleJev/JevAny-Qwen3.8-27B-LoRA": "09c9e9102d5b8cc7d56558d25da1202a761b6c0d",
}
FORMAT = "vllm-jev-jevany-pointer-v1"
HEAD_FILE = "jevany_pointer_head.safetensors"
MANIFEST_FILE = "jevany_pointer_manifest.json"
COPIED_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "processor_config.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "vocab.json",
    "merges.txt",
)


def _index(path: Path) -> dict:
    index = json.loads((path / "model.safetensors.index.json").read_text())
    weights = index.get("weight_map")
    if (
        not isinstance(weights, dict)
        or not weights
        or any(
            not isinstance(name, str)
            or not isinstance(shard, str)
            or Path(shard).name != shard
            for name, shard in weights.items()
        )
    ):
        raise ValueError("base shard index must use local filenames")
    return index


def _adapter_pairs(
    source: Path, scale: float
) -> dict[str, tuple[torch.Tensor, torch.Tensor, float]]:
    tensors = load_file(str(source / "adapter_model.safetensors"))
    left_suffix = ".lora_A.weight"
    right_suffix = ".lora_B.weight"
    left_names = {name for name in tensors if name.endswith(left_suffix)}
    right_names = {name for name in tensors if name.endswith(right_suffix)}
    unexpected = set(tensors) - left_names - right_names
    if unexpected:
        raise ValueError(
            f"unsupported JevAny adapter tensors: {sorted(unexpected)[:3]}"
        )

    pairs = {}
    paired_right = set()
    for name in sorted(left_names):
        prefix = name[: -len(left_suffix)]
        right_name = prefix + right_suffix
        if right_name not in tensors:
            raise ValueError(f"LoRA B tensor missing for {prefix}")
        if not prefix.startswith("base_model.model."):
            raise ValueError(f"unsupported JevAny LoRA tensor: {name}")
        destination = (
            "model.language_model."
            + prefix[len("base_model.model.") :]
            + ".weight"
        )
        pairs[destination] = (tensors[name], tensors[right_name], scale)
        paired_right.add(right_name)
    if paired_right != right_names or not pairs:
        raise ValueError("JevAny adapter contains incomplete LoRA pairs")
    return pairs


def _link_or_copy(source: Path, output: Path) -> None:
    try:
        os.link(source, output)
    except OSError:
        shutil.copy2(source, output)


def _copy_metadata(base: Path, source: Path, output: Path) -> None:
    for name in COPIED_FILES:
        item = source / name
        if not item.is_file():
            item = base / name
        if item.is_file():
            shutil.copy2(item, output / name)


def export_jevany_pointer(
    base: Path,
    source: Path,
    output: Path,
    model_id: str = "jevany-pointer",
) -> dict:
    """Merge LoRA one safetensors shard at a time and retain the VL backbone."""
    base, source, output = base.resolve(), source.resolve(), output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)

    meta = torch.load(source / "head.pt", map_location="cpu", weights_only=True)
    adapter = json.loads((source / "adapter_config.json").read_text())
    head_dim = meta.get("head_dim", 256)
    residual_dim = meta.get("head_residual_dim", 0)
    head_type = meta.get("head_type", "residual" if residual_dim else "linear")
    if (
        meta.get("decision_mode") != "pointer"
        or meta.get("multimodal") is not True
        or meta.get("backbone_adapter") != "qwen_vl"
        or meta.get("branch_mode") != "rows"
        or meta.get("option_isolation", False)
        or meta.get("special_embeddings", False)
        or meta.get("query_readout", "decide") != "decide"
        or meta.get("option_readout", "close") != "close"
        or type(head_dim) is not int
        or head_dim <= 0
        or type(residual_dim) is not int
        or residual_dim < 0
        or head_type != ("residual" if residual_dim else "linear")
    ):
        raise ValueError("unsupported JevAny Pointer checkpoint configuration")
    if (
        adapter.get("peft_type") != "LORA"
        or adapter.get("bias", "none") != "none"
        or adapter.get("use_rslora", False)
        or adapter.get("use_dora", False)
        or adapter.get("fan_in_fan_out", False)
        or adapter.get("lora_bias", False)
        or adapter.get("rank_pattern")
        or adapter.get("alpha_pattern")
        or adapter.get("modules_to_save")
    ):
        raise ValueError("unsupported JevAny adapter type")
    rank, alpha = adapter.get("r"), adapter.get("lora_alpha")
    if type(rank) is not int or rank <= 0 or not isinstance(alpha, (int, float)):
        raise ValueError("invalid JevAny LoRA scale")
    if adapter.get("base_model_name_or_path") != meta.get("base"):
        raise ValueError("JevAny adapter and head refer to different bases")

    config = json.loads((base / "config.json").read_text())
    text = config.get("text_config", {})
    architecture = {
        (
            "qwen3_5",
            "qwen3_5_text",
            ("Qwen3_5ForConditionalGeneration",),
        ): JEVANY_POINTER_QWEN35_ARCHITECTURE,
        (
            "qwen3_5_moe",
            "qwen3_5_moe_text",
            ("Qwen3_5MoeForConditionalGeneration",),
        ): JEVANY_POINTER_QWEN35_MOE_ARCHITECTURE,
        (
            "qwen3_vl_moe",
            "qwen3_vl_moe_text",
            ("Qwen3VLMoeForConditionalGeneration",),
        ): JEVANY_POINTER_QWEN3_VL_MOE_ARCHITECTURE,
    }.get(
        (
            config.get("model_type"),
            text.get("model_type"),
            tuple(config.get("architectures", ())),
        )
    )
    if architecture is None:
        raise ValueError("JevAny base is not a supported Qwen3.5 VL model")
    hidden_size = text.get("hidden_size")
    if type(hidden_size) is not int or hidden_size <= 0:
        raise ValueError("invalid JevAny hidden size")

    base_index = _index(base)
    pairs = _adapter_pairs(source, float(alpha) / rank)
    missing = set(pairs) - set(base_index["weight_map"])
    if missing:
        raise ValueError(f"LoRA targets are missing from the base: {sorted(missing)[:3]}")

    by_shard = {}
    for name in pairs:
        by_shard.setdefault(base_index["weight_map"][name], []).append(name)
    for shard in sorted(set(base_index["weight_map"].values())):
        source_shard, output_shard = base / shard, output / shard
        targets = by_shard.get(shard, [])
        if not targets:
            _link_or_copy(source_shard, output_shard)
            continue
        with safe_open(source_shard, framework="pt", device="cpu") as file:
            tensors = {name: file.get_tensor(name) for name in file.keys()}
        for name in targets:
            left, right, scale = pairs[name]
            original = tensors[name]
            delta = right.float() @ left.float()
            if delta.shape != original.shape:
                raise ValueError(f"LoRA shape mismatch for {name}")
            tensors[name] = (original.float() + delta * scale).to(original.dtype)
        save_file(
            {name: value.contiguous() for name, value in tensors.items()},
            str(output_shard),
            metadata={"format": "pt"},
        )

    config["architectures"] = [architecture]
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (output / "model.safetensors.index.json").write_text(
        json.dumps(base_index, indent=2) + "\n"
    )
    _copy_metadata(base, source, output)

    head = meta.get("head")
    if not isinstance(head, dict):
        raise ValueError("JevAny Pointer head is missing")
    head = {name: value.float().contiguous() for name, value in head.items()}
    save_file(head, str(output / HEAD_FILE))

    tokenizer_config = json.loads((output / "tokenizer_config.json").read_text())
    marker_names = tokenizer_config.get("jevany_decision_tokens")
    if not isinstance(marker_names, list) or len(marker_names) != 5:
        marker_names = [
            "<|fim_prefix|>",
            "<|fim_middle|>",
            "<|box_start|>",
            "<|box_end|>",
            "<|fim_suffix|>",
        ]
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(output, local_files_only=True)
    markers = [tokenizer.convert_tokens_to_ids(name) for name in marker_names]
    if any(token is None for token in markers) or len(set(markers)) != 5:
        raise ValueError("JevAny delimiter tokens are missing")

    manifest = {
        "format": FORMAT,
        "architecture": architecture,
        "prompt_protocol": PROTOCOL,
        "source_repository": model_id,
        "source_checkpoint": str(source),
        "foundation_repository": meta["base"],
        "foundation_revision": meta.get("base_revision"),
        "hidden_size": hidden_size,
        "head_dim": head_dim,
        "head_residual_dim": residual_dim,
        "head_type": head_type,
        "query_readout": "decide",
        "option_readout": "close",
        "max_state": int(meta.get("args", {}).get("max_state", 8192)),
        "max_length": int(meta.get("args", {}).get("max_branch", 8192)),
        "max_images": 2,
        "image_token_id": config["image_token_id"],
        "markers": markers,
        "marker_names": marker_names,
        "temperature": float(meta.get("temperature", 1.0)),
        "lora_merge": "fp32_delta_then_source_dtype",
        "files": {
            item.name: sha256(item) for item in output.iterdir() if item.is_file()
        },
    }
    (output / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2) + "\n")
    verify_jevany_pointer(output, full=True)
    return manifest


def verify_jevany_pointer(path: Path, *, full: bool = True) -> dict:
    path = path.resolve()
    manifest = json.loads((path / MANIFEST_FILE).read_text())
    if (
        manifest.get("format") != FORMAT
        or manifest.get("architecture")
        not in {
            JEVANY_POINTER_QWEN35_ARCHITECTURE,
            JEVANY_POINTER_QWEN35_MOE_ARCHITECTURE,
            JEVANY_POINTER_QWEN3_VL_MOE_ARCHITECTURE,
        }
        or manifest.get("prompt_protocol") != PROTOCOL
        or manifest.get("max_images") != 2
    ):
        raise ValueError("unrecognized JevAny Pointer checkpoint")
    hidden = manifest.get("hidden_size")
    head_dim = manifest.get("head_dim")
    residual_dim = manifest.get("head_residual_dim")
    markers = manifest.get("markers")
    if (
        type(hidden) is not int
        or hidden <= 0
        or type(head_dim) is not int
        or head_dim <= 0
        or type(residual_dim) is not int
        or residual_dim < 0
        or manifest.get("head_type")
        != ("residual" if residual_dim else "linear")
        or manifest.get("query_readout") != "decide"
        or manifest.get("option_readout") != "close"
        or not isinstance(markers, list)
        or len(markers) != 5
        or len(set(markers)) != 5
        or any(type(token) is not int or token < 0 for token in markers)
    ):
        raise ValueError("invalid JevAny Pointer dimensions or markers")
    for name in ("max_state", "max_length"):
        if type(manifest.get(name)) is not int or manifest[name] <= 0:
            raise ValueError(f"invalid JevAny Pointer {name}")
    temperature = manifest.get("temperature")
    if (
        not isinstance(temperature, (int, float))
        or not math.isfinite(temperature)
        or temperature <= 0
    ):
        raise ValueError("invalid JevAny Pointer temperature")

    config = json.loads((path / "config.json").read_text())
    if config.get("architectures") != [manifest["architecture"]]:
        raise ValueError("JevAny Pointer architecture mismatch")
    index = _index(path)
    checked = verify_files(
        path,
        manifest,
        index,
        (
            "config.json",
            "model.safetensors.index.json",
            HEAD_FILE,
            "tokenizer.json",
            "tokenizer_config.json",
            "processor_config.json",
        ),
        full=full,
    )
    with safe_open(path / HEAD_FILE, framework="pt", device="cpu") as file:
        expected = {
            "q.weight": (head_dim, hidden),
            "q.bias": (head_dim,),
            "k.weight": (head_dim, hidden),
            "k.bias": (head_dim,),
        }
        if residual_dim:
            expected.update(
                {
                    "residual.0.weight": (residual_dim, head_dim),
                    "residual.0.bias": (residual_dim,),
                    "residual.2.weight": (1, residual_dim),
                    "residual.2.bias": (1,),
                }
            )
        if set(file.keys()) != set(expected):
            raise ValueError("invalid JevAny Pointer head tensors")
        for name, shape in expected.items():
            tensor = file.get_tensor(name)
            if tensor.shape != shape or tensor.dtype != torch.float32:
                raise ValueError(f"invalid JevAny Pointer tensor: {name}")
            if not torch.isfinite(tensor).all():
                raise ValueError("nonfinite JevAny Pointer head")
    return {
        "format": manifest["format"],
        "architecture": manifest["architecture"],
        "checked_files": checked,
        "full": full,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", default="jevany-pointer")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    manifest = export_jevany_pointer(
        args.base, args.checkpoint, args.output, args.model_id
    )
    if args.quick:
        verify_jevany_pointer(args.output, full=False)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
