"""Convert a full Tiny-Jev checkpoint to native vLLM token scoring."""

import json
import math
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from . import QWEN3_PROJECTED_ARCHITECTURE
from .checkpoint import sha256


def export_tiny(source: Path, output: Path, model_id: str, revision: str) -> dict:
    source, output = source.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(output)
    config = json.loads((source / "config.json").read_text())
    if (
        config.get("model_type") != "tiny_jev"
        or config.get("architectures") != ["TinyJevModel"]
        or type(config.get("hidden_size")) is not int
        or config["hidden_size"] <= 0
    ):
        raise ValueError("not a Tiny-Jev Qwen3 checkpoint")
    temperature = config.get("temperature")
    if (
        type(temperature) not in (int, float)
        or not math.isfinite(temperature)
        or temperature <= 0
    ):
        raise ValueError("invalid Tiny-Jev calibration temperature")

    with safe_open(source / "model.safetensors", framework="pt", device="cpu") as file:
        names = list(file.keys())
        if any(not name.startswith(("model.", "head.")) for name in names):
            raise ValueError("unexpected Tiny-Jev tensor names")
        tensors = {
            name: file.get_tensor(name) for name in names if name.startswith("model.")
        }
        head = {
            name.removeprefix("head."): file.get_tensor(name)
            for name in names
            if name.startswith("head.")
        }
    if (
        set(head) != {"weight", "bias"}
        or head["weight"].shape != (1, config["hidden_size"])
        or head["bias"].shape != (1,)
    ):
        raise ValueError("invalid Tiny-Jev decision head")
    if not tensors or any(not torch.isfinite(value).all() for value in head.values()):
        raise ValueError("missing Tiny-Jev backbone or nonfinite decision head")
    save_file(tensors, output / "model.safetensors")
    score = {f"score.{name}": value for name, value in head.items()}
    save_file(score, output / "score.safetensors")
    index = {
        "metadata": {
            "total_size": sum(
                t.numel() * t.element_size()
                for t in (*tensors.values(), *score.values())
            )
        },
        "weight_map": {
            **{name: "model.safetensors" for name in tensors},
            **{name: "score.safetensors" for name in score},
        },
    }
    (output / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n"
    )
    config.pop("auto_map", None)
    config["model_type"] = "qwen3"
    config["architectures"] = [QWEN3_PROJECTED_ARCHITECTURE]
    (output / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n"
    )
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        if (source / name).is_file():
            (output / name).write_bytes((source / name).read_bytes())
    manifest = {
        "format": "vllm-jev-pooling-v1",
        "architecture": QWEN3_PROJECTED_ARCHITECTURE,
        "prompt_protocol": "tiny_jev_marker",
        "source_repository": model_id,
        "source_revision": revision,
        "source_weights_sha256": sha256(source / "model.safetensors"),
        "calibration_temperature": float(temperature),
        "foundation_revision": revision,
        "files": {
            path.name: sha256(path) for path in output.iterdir() if path.is_file()
        },
    }
    (output / "jev_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
