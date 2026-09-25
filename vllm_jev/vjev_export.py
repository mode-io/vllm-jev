"""Prepare the released merged vjev vision checkpoints for native vLLM pooling."""

import json
import os
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from . import VJEV_QWEN35_ARCHITECTURE
from .checkpoint import sha256, verify_files

REVISIONS = {
    "yah01/vjev-vision": "2fa8b58e40e5bc351a7d6dd39b953469a8f3ded2",
    "yah01/vjev-vision-pilot": "616a851a6b828c466d81d287659918b71cfb5803",
}


def export_vjev(source: Path, output: Path, model_id: str) -> dict:
    if model_id not in REVISIONS:
        raise ValueError("unsupported vjev checkpoint")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((source / "config.json").read_text())
    meta = json.loads((source / "vjev.json").read_text())
    if (
        config.get("model_type") != "qwen3_5"
        or config.get("architectures") != ["Qwen3_5ForConditionalGeneration"]
        or meta.get("arch") != "vl"
        or meta.get("readout") != "trailing"
        or meta.get("pause") != 0
    ):
        raise ValueError("unsupported vjev vision format")
    hidden = config["text_config"]["hidden_size"]
    head = torch.load(source / "head.pt", map_location="cpu", weights_only=True)
    if (
        set(head) != {"weight", "bias"}
        or head["weight"].shape != (1, hidden)
        or head["bias"].shape != (1,)
        or any(not torch.isfinite(tensor).all() for tensor in head.values())
    ):
        raise ValueError("invalid vjev decision head")
    save_file(
        {key: tensor.float().contiguous() for key, tensor in head.items()},
        output / "vjev_head.safetensors",
    )

    index = json.loads((source / "model.safetensors.index.json").read_text())
    weights = index["weight_map"]
    if not any(name.startswith("model.visual.") for name in weights) or not any(
        name.startswith("model.language_model.") for name in weights
    ):
        raise ValueError("vjev vision or language weights missing")
    for name in set(weights.values()):
        if Path(name).name != name:
            raise ValueError("vjev shard path must stay within the checkpoint")
        try:
            os.link(source / name, output / name)
        except OSError:
            shutil.copy2(source / name, output / name)
    (output / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2) + "\n"
    )
    config["architectures"] = [VJEV_QWEN35_ARCHITECTURE]
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "processor_config.json",
        "chat_template.jinja",
        "generation_config.json",
    ):
        file = source / name
        if file.is_file():
            shutil.copy2(file, output / name)
    manifest = {
        "format": "vllm-jev-vjev-v1",
        "architecture": VJEV_QWEN35_ARCHITECTURE,
        "prompt_protocol": "vjev_vision_v1",
        "source_repository": model_id,
        "source_revision": REVISIONS[model_id],
        "max_length": min(4096, int(meta.get("max_len", 2560)) + 1536),
        "readout": "trailing",
        "files": {
            file.name: sha256(file) for file in output.iterdir() if file.is_file()
        },
    }
    (output / "vjev_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def verify_vjev(path: Path, *, full: bool = True) -> dict:
    path = path.resolve()
    manifest = json.loads((path / "vjev_manifest.json").read_text())
    model_id = manifest.get("source_repository")
    if (
        manifest.get("format") != "vllm-jev-vjev-v1"
        or manifest.get("architecture") != VJEV_QWEN35_ARCHITECTURE
        or manifest.get("prompt_protocol") != "vjev_vision_v1"
        or model_id not in REVISIONS
        or manifest.get("source_revision") != REVISIONS[model_id]
        or manifest.get("readout") != "trailing"
        or type(manifest.get("max_length")) is not int
        or not 1 <= manifest["max_length"] <= 4096
    ):
        raise ValueError("unrecognized vjev checkpoint")
    config = json.loads((path / "config.json").read_text())
    if config.get("architectures") != [VJEV_QWEN35_ARCHITECTURE]:
        raise ValueError("vjev model architecture mismatch")
    hidden = config["text_config"]["hidden_size"]
    with safe_open(
        path / "vjev_head.safetensors", framework="pt", device="cpu"
    ) as file:
        if set(file.keys()) != {"weight", "bias"}:
            raise ValueError("invalid vjev head tensors")
        if file.get_tensor("weight").shape != (1, hidden) or file.get_tensor(
            "bias"
        ).shape != (1,):
            raise ValueError("invalid vjev head shape")
        if any(not torch.isfinite(file.get_tensor(name)).all() for name in file.keys()):
            raise ValueError("nonfinite vjev head")
    index = json.loads((path / "model.safetensors.index.json").read_text())
    if not any(name.startswith("model.visual.") for name in index["weight_map"]):
        raise ValueError("vjev visual weights missing")
    if not any(
        name.startswith("model.language_model.") for name in index["weight_map"]
    ):
        raise ValueError("vjev language weights missing")
    checked_files = verify_files(
        path,
        manifest,
        index,
        (
            "config.json",
            "model.safetensors.index.json",
            "vjev_head.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "processor_config.json",
        ),
        full=full,
    )
    return {
        "format": manifest["format"],
        "architecture": VJEV_QWEN35_ARCHITECTURE,
        "checked_files": checked_files,
        "full": full,
    }
