"""Verify the exported Jev checkpoint before handing it to vLLM."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from safetensors import safe_open

from . import ARCHITECTURE, QWEN3_ARCHITECTURE, QWEN3_TOKEN_ARCHITECTURE


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: Path, *, full: bool = True) -> dict:
    path = path.resolve()
    manifest = json.loads((path / "jev_manifest.json").read_text())
    if manifest.get("format") != "vllm-jev-pooling-v1":
        raise ValueError("unrecognized Jev checkpoint format")
    architecture = manifest.get("architecture")
    if architecture not in (ARCHITECTURE, QWEN3_ARCHITECTURE, QWEN3_TOKEN_ARCHITECTURE):
        raise ValueError("Jev architecture mismatch")
    config = json.loads((path / "config.json").read_text())
    if config.get("architectures") != [architecture] or (
        architecture != QWEN3_TOKEN_ARCHITECTURE and config.get("num_labels") != 1
    ):
        raise ValueError("classifier config mismatch")
    hidden_size = config.get("hidden_size")
    if not isinstance(hidden_size, int) or hidden_size <= 0:
        raise ValueError("invalid hidden size")
    index = json.loads((path / "model.safetensors.index.json").read_text())
    if architecture != QWEN3_TOKEN_ARCHITECTURE:
        if index["weight_map"].get("score.weight") != "score.safetensors":
            raise ValueError("score weight not indexed")
        if index["weight_map"].get("score.bias") != "score.safetensors":
            raise ValueError("score bias not indexed")
    with safe_open(path / "score.safetensors", framework="pt", device="cpu") as file:
        weight_name = (
            "weight" if architecture == QWEN3_TOKEN_ARCHITECTURE else "score.weight"
        )
        bias_name = "bias" if architecture == QWEN3_TOKEN_ARCHITECTURE else "score.bias"
        if file.get_tensor(weight_name).shape != (1, hidden_size):
            raise ValueError("invalid score head shape")
        if file.get_tensor(bias_name).shape != (1,):
            raise ValueError("invalid score bias shape")
    names = (
        list(manifest["files"])
        if full
        else [
            "config.json",
            "model.safetensors.index.json",
            "score.safetensors",
            "tokenizer_config.json",
            "tokenizer.json",
        ]
    )
    for name in names:
        item = path / name
        if not item.is_file():
            raise FileNotFoundError(item)
        if sha256(item) != manifest["files"][name]:
            raise ValueError(f"checkpoint checksum mismatch: {name}")
    temperature = manifest.get("calibration_temperature", 1.0)
    if (
        not isinstance(temperature, (int, float))
        or not math.isfinite(temperature)
        or temperature <= 0
    ):
        raise ValueError("invalid calibration temperature")
    return {
        "format": manifest["format"],
        "architecture": architecture,
        "foundation_revision": manifest["foundation_revision"],
        "checked_files": len(names),
        "full": full,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    print(json.dumps(verify(args.model, full=not args.quick), indent=2))


if __name__ == "__main__":
    main()
