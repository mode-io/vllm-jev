"""Verify the exported Jev checkpoint before handing it to vLLM."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from safetensors import safe_open

from . import (
    ARCHITECTURE,
    QWEN3_ARCHITECTURE,
    QWEN3_PROJECTED_ARCHITECTURE,
    QWEN3_TOKEN_ARCHITECTURE,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_files(
    path: Path, manifest: dict, index: dict, required: tuple[str, ...], *, full: bool
) -> int:
    """Require all indexed shards, even when skipping their expensive checksums."""
    files, weights = manifest.get("files"), index.get("weight_map")
    if not isinstance(files, dict) or not isinstance(weights, dict) or not weights:
        raise ValueError("invalid checkpoint file manifest or weight index")
    if any(
        not isinstance(name, str) or not isinstance(filename, str)
        for name, filename in weights.items()
    ):
        raise ValueError("invalid checkpoint weight name")
    shards = set(weights.values())
    for name in (*files, *shards, *required):
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError("checkpoint filenames must stay within the checkpoint")
        item = path / name
        if item.resolve().parent != path or not item.is_file():
            raise ValueError(f"checkpoint file missing or outside checkpoint: {name}")
        digest = files.get(name)
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"checkpoint file checksum missing: {name}")
    names = list(files) if full else required
    for name in names:
        if sha256(path / name) != files[name]:
            raise ValueError(f"checkpoint checksum mismatch: {name}")
    if full:
        for shard in shards:
            with safe_open(path / shard, framework="pt", device="cpu") as file:
                if set(file.keys()) != {
                    name for name, filename in weights.items() if filename == shard
                }:
                    raise ValueError(f"checkpoint weight index mismatch: {shard}")
    return len(names)


def verify(path: Path, *, full: bool = True) -> dict:
    path = path.resolve()
    if (path / "vjev_manifest.json").is_file():
        from .vjev_export import verify_vjev

        return verify_vjev(path, full=full)
    if (path / "valen_manifest.json").is_file():
        from .valen_export import verify_valen

        return verify_valen(path, full=full)
    manifest = json.loads((path / "jev_manifest.json").read_text())
    if manifest.get("format") != "vllm-jev-pooling-v1":
        raise ValueError("unrecognized Jev checkpoint format")
    architecture = manifest.get("architecture")
    if architecture not in (
        ARCHITECTURE,
        QWEN3_ARCHITECTURE,
        QWEN3_TOKEN_ARCHITECTURE,
        QWEN3_PROJECTED_ARCHITECTURE,
    ):
        raise ValueError("Jev architecture mismatch")
    config = json.loads((path / "config.json").read_text())
    if config.get("architectures") != [architecture] or (
        architecture not in (QWEN3_TOKEN_ARCHITECTURE, QWEN3_PROJECTED_ARCHITECTURE)
        and config.get("num_labels") != 1
    ):
        raise ValueError("classifier config mismatch")
    hidden_size = config.get("hidden_size")
    if type(hidden_size) is not int or hidden_size <= 0:
        raise ValueError("invalid hidden size")
    index = json.loads((path / "model.safetensors.index.json").read_text())
    checked_files = verify_files(
        path,
        manifest,
        index,
        (
            "config.json",
            "model.safetensors.index.json",
            "score.safetensors",
            "tokenizer_config.json",
            "tokenizer.json",
        ),
        full=full,
    )
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
        if any(
            not torch.isfinite(file.get_tensor(name)).all()
            for name in (weight_name, bias_name)
        ):
            raise ValueError("nonfinite score head")
    temperatures = manifest.get("calibration_temperatures", {})
    if not isinstance(temperatures, dict) or (
        temperatures and set(temperatures) != {"choice", "noul", "score"}
    ):
        raise ValueError("invalid calibration temperatures")
    if any(
        type(value) not in (int, float) or not math.isfinite(value) or value <= 0
        for value in (
            manifest.get("calibration_temperature", 1.0),
            *temperatures.values(),
        )
    ):
        raise ValueError("invalid calibration temperature")
    return {
        "format": manifest["format"],
        "architecture": architecture,
        "foundation_revision": manifest["foundation_revision"],
        "checked_files": checked_files,
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
