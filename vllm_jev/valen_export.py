"""Export the pinned Valen preview to native vLLM token pooling."""

import json
import shutil
import sys
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn
from transformers import Qwen3_5ForConditionalGeneration

from . import VALEN_QWEN35_ARCHITECTURE
from .checkpoint import sha256, verify_files

MODEL_ID = "Valen-Team/Valen-Preview-0923"
MODEL_REVISION = "91a9d3e79fcdf75d233994c1a05296ad569d0ff4"
CHECKPOINT_SHA256 = "836622efe78fe757e2627aa6050c223d461c424ea430a1c110c15e6f42f5a012"
BASE_ID = "Qwen/Qwen3.5-2B"
BASE_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"


class _ValenWeights(nn.Module):
    def __init__(self, backbone, hidden_size: int, projection_dim: int):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Module()
        self.head.decision = nn.Linear(hidden_size, projection_dim, bias=False)
        self.head.candidate = nn.Linear(hidden_size, projection_dim, bias=False)


def export_valen(base: Path, checkpoint: Path, output: Path) -> dict:
    """Merge inference weights only; optimizer and training state are ignored."""
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((checkpoint / "config.json").read_text())
    if (
        config.get("stage") != "vision_top"
        or config.get("projection_dim") != 256
        or config.get("lora_rank") != 32
        or config.get("lora_alpha") != 64
    ):
        raise ValueError("unsupported Valen preview configuration")
    if sha256(checkpoint / "checkpoint.pt") != CHECKPOINT_SHA256:
        raise ValueError("Valen preview checkpoint hash mismatch")
    payload = torch.load(
        checkpoint / "checkpoint.pt", map_location="cpu", weights_only=True, mmap=True
    )
    provenance = payload.get("base_manifest") or {}
    if provenance.get("revision") != BASE_REVISION:
        raise ValueError("Valen base revision mismatch")
    base_weights = provenance.get("weight_sha256") or {}
    for filename, expected in base_weights.items():
        if sha256(base / filename) != expected:
            raise ValueError(f"Valen base weight hash mismatch: {filename}")

    container, info = Qwen3_5ForConditionalGeneration.from_pretrained(
        base,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        local_files_only=True,
        output_loading_info=True,
    )
    if any(
        info.get(key)
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    ):
        raise ValueError("incomplete Qwen3.5 base weight load")
    backbone = container.model
    del container
    targets = [
        name
        for name, module in backbone.language_model.named_modules()
        if isinstance(module, nn.Linear)
    ]
    backbone.language_model = get_peft_model(
        backbone.language_model,
        LoraConfig(
            r=32, lora_alpha=64, lora_dropout=0.0, target_modules=targets, bias="none"
        ),
    )
    model = _ValenWeights(backbone, 2048, 256).to(
        "cpu" if sys.platform == "darwin" else "cuda:0"
    )
    weights = payload["weights"]
    if set(weights) - set(model.state_dict()):
        raise ValueError("Valen checkpoint contains unknown inference weights")
    loaded = model.load_state_dict(weights, strict=False)
    if loaded.unexpected_keys:
        raise ValueError("Valen checkpoint contains unexpected weights")
    model.backbone.language_model = model.backbone.language_model.merge_and_unload(
        safe_merge=True
    )

    shards = {"vision.safetensors": {}, "language.safetensors": {}}
    for name, value in model.backbone.state_dict().items():
        filename = (
            "vision.safetensors"
            if name.startswith("visual.")
            else "language.safetensors"
        )
        shards[filename]["model." + name] = value.detach().cpu().contiguous()
    weight_map = {}
    total_size = 0
    for filename, tensors in shards.items():
        save_file(tensors, output / filename)
        weight_map.update({name: filename for name in tensors})
        total_size += sum(t.numel() * t.element_size() for t in tensors.values())
        tensors.clear()
    head = {
        "decision.weight": model.head.decision.weight.detach().cpu().contiguous(),
        "candidate.weight": model.head.candidate.weight.detach().cpu().contiguous(),
    }
    save_file(head, output / "valen_head.safetensors")
    (output / "model.safetensors.index.json").write_text(
        json.dumps(
            {"metadata": {"total_size": total_size}, "weight_map": weight_map},
            indent=2,
        )
        + "\n"
    )
    base_config = json.loads((base / "config.json").read_text())
    base_config["architectures"] = [VALEN_QWEN35_ARCHITECTURE]
    (output / "config.json").write_text(json.dumps(base_config, indent=2) + "\n")
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
        "vocab.json",
        "merges.txt",
    ):
        source = base / name
        if source.is_file():
            shutil.copy2(source, output / name)
    manifest = {
        "format": "vllm-jev-valen-v1",
        "architecture": VALEN_QWEN35_ARCHITECTURE,
        "prompt_protocol": "valen_qwen_v1",
        "source_repository": MODEL_ID,
        "source_revision": MODEL_REVISION,
        "source_weights_sha256": CHECKPOINT_SHA256,
        "foundation_repository": BASE_ID,
        "foundation_revision": BASE_REVISION,
        "media_kwargs": config.get("media_kwargs", {}),
        "max_length": config.get("max_length", 8192),
        "projection_dim": 256,
        "files": {
            path.name: sha256(path) for path in output.iterdir() if path.is_file()
        },
    }
    (output / "valen_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def verify_valen(path: Path, *, full: bool = True) -> dict:
    path = path.resolve()
    manifest = json.loads((path / "valen_manifest.json").read_text())
    if (
        manifest.get("format") != "vllm-jev-valen-v1"
        or manifest.get("architecture") != VALEN_QWEN35_ARCHITECTURE
        or manifest.get("source_repository") != MODEL_ID
        or manifest.get("source_revision") != MODEL_REVISION
        or manifest.get("source_weights_sha256") != CHECKPOINT_SHA256
        or manifest.get("prompt_protocol") != "valen_qwen_v1"
        or manifest.get("foundation_repository") != BASE_ID
        or manifest.get("foundation_revision") != BASE_REVISION
        or manifest.get("projection_dim") != 256
        or not isinstance(manifest.get("media_kwargs"), dict)
    ):
        raise ValueError("unrecognized Valen preview checkpoint")
    config = json.loads((path / "config.json").read_text())
    if config.get("architectures") != [VALEN_QWEN35_ARCHITECTURE]:
        raise ValueError("Valen model architecture mismatch")
    index = json.loads((path / "model.safetensors.index.json").read_text())
    checked_files = verify_files(
        path,
        manifest,
        index,
        (
            "config.json",
            "model.safetensors.index.json",
            "valen_head.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "preprocessor_config.json",
            "video_preprocessor_config.json",
        ),
        full=full,
    )
    with safe_open(
        path / "valen_head.safetensors", framework="pt", device="cpu"
    ) as file:
        if file.get_tensor("decision.weight").shape != (256, 2048):
            raise ValueError("invalid Valen decision head")
        if file.get_tensor("candidate.weight").shape != (256, 2048):
            raise ValueError("invalid Valen candidate head")
        if any(
            not torch.isfinite(file.get_tensor(name)).all()
            for name in ("decision.weight", "candidate.weight")
        ):
            raise ValueError("nonfinite Valen head")
    if not any(name.startswith("model.visual.") for name in index["weight_map"]):
        raise ValueError("Valen visual weights missing")
    if not any(
        name.startswith("model.language_model.") for name in index["weight_map"]
    ):
        raise ValueError("Valen language weights missing")
    return {
        "format": manifest["format"],
        "architecture": VALEN_QWEN35_ARCHITECTURE,
        "foundation_revision": BASE_REVISION,
        "checked_files": checked_files,
        "full": full,
    }
