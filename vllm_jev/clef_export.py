"""Prepare the released Clef-Flash checkpoint for native vLLM pooling.

The release already stores a merged bf16 ``Qwen3_5ForConditionalGeneration``
backbone. The export links its weight shards unchanged, names the pooling
architecture in ``config.json``, and keeps the joint schema head beside it.
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from safetensors import safe_open

from . import CLEF_QWEN35_ARCHITECTURE
from .checkpoint import sha256, verify_files
from .clef import PROTOCOL

MODELS = {"Cloudflare/clef-flash": "17f0b0ad64efb65d273590632833508766b2aae6"}
# sha256 of the released files, so a local copy is checked like a download.
RELEASE_HASHES = {
    "Cloudflare/clef-flash": {
        "model-00001-of-00004.safetensors": "8b45a8e968141cdcc58fb71c9adfc258e2c77b5f062bc636c1fd5bc5d916b565",
        "model-00002-of-00004.safetensors": "7590856c713eed844a2dcf48e6c43c4de165b788bc3f80e328311183cdbc7db8",
        "model-00003-of-00004.safetensors": "e6eac2467952c33361ed7dcb3c7959d1086bbe57201cd3749c3d769fdc17fe63",
        "model-00004-of-00004.safetensors": "9fcecc6556b39171238373a465f409794b7f821fb4cd1e6459e3a9c0fe317af7",
        "model.safetensors.index.json": "941305ff9f77551e145a6cea976ef456cd5cb208cbece99f168376c752fcf96c",
        "config.json": "66f87f6fb2616b46604daf2a9c67ddc87938296d07156efa34d59b5be49e3238",
        "joint_head.safetensors": "19cdcec8c81dc9212be320fff47462ab342fbc1278be4368fb3da71241cf5ba0",
        "joint_head_config.json": "77efe959a38b5b17b241543e129e695f3c77465ece55a25985279bd8176279a0",
        "tokenizer.json": "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
        "tokenizer_config.json": "91a08f825d370d085d692e04cf117cdd7faad7bf18e996f1e6031b6dab03db72",
        "processor_config.json": "d89ef49ce9cd37fbf510158e13c1ef063d9286411c1ec9049932dbe0487143b1",
        "chat_template.jinja": "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715",
    },
}
SOURCE_FILES = [*RELEASE_HASHES["Cloudflare/clef-flash"], "LICENSE"]
COPIED_FILES = (
    "joint_head.safetensors",
    "joint_head_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "processor_config.json",
    "chat_template.jinja",
)
MAX_LENGTH = 16384
HEAD_CONFIG = {
    "hidden_size": 4096,
    "width": 1024,
    "routing_layers": 2,
    "layers": 4,
    "heads": 16,
    "feedforward": 4096,
}


def _link(source: Path, target: Path) -> None:
    """Hard-link a large file when both sides share a file system."""
    try:
        os.link(source.resolve(), target)
    except OSError:
        shutil.copy2(source, target)


def export_clef(source: Path, output: Path, model_id: str) -> dict:
    if model_id not in MODELS:
        raise ValueError("unsupported Clef checkpoint")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    hashes = RELEASE_HASHES[model_id]
    for name, digest in hashes.items():
        if sha256(source / name) != digest:
            raise ValueError(f"Clef release checksum mismatch: {name}")
    if json.loads((source / "joint_head_config.json").read_text()) != HEAD_CONFIG:
        raise ValueError("unsupported Clef joint head configuration")
    config = json.loads((source / "config.json").read_text())
    if config.get("architectures") != ["Qwen3_5ForConditionalGeneration"]:
        raise ValueError("Clef backbone is not a Qwen3.5 vision-language checkpoint")
    output.mkdir(parents=True, exist_ok=True)
    index = json.loads((source / "model.safetensors.index.json").read_text())
    shards = sorted(set(index["weight_map"].values()))
    for shard in shards:
        if Path(shard).name != shard or shard not in hashes:
            raise ValueError("Clef shard is not part of the pinned release")
        _link(source / shard, output / shard)
    shutil.copy2(
        source / "model.safetensors.index.json", output / "model.safetensors.index.json"
    )
    for name in COPIED_FILES:
        shutil.copy2(source / name, output / name)
    config["architectures"] = [CLEF_QWEN35_ARCHITECTURE]
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    files = {
        name: hashes[name]
        for name in (*shards, "model.safetensors.index.json", *COPIED_FILES)
    }
    files["config.json"] = sha256(output / "config.json")
    manifest = {
        "format": "vllm-jev-clef-v1",
        "architecture": CLEF_QWEN35_ARCHITECTURE,
        "prompt_protocol": PROTOCOL,
        "source_repository": model_id,
        "source_revision": MODELS[model_id],
        "max_length": MAX_LENGTH,
        "files": files,
    }
    (output / "clef_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def verify_clef(path: Path, *, full: bool = True) -> dict:
    path = path.resolve()
    manifest = json.loads((path / "clef_manifest.json").read_text())
    model_id = manifest.get("source_repository")
    if (
        manifest.get("format") != "vllm-jev-clef-v1"
        or manifest.get("architecture") != CLEF_QWEN35_ARCHITECTURE
        or manifest.get("prompt_protocol") != PROTOCOL
        or model_id not in MODELS
        or manifest.get("source_revision") != MODELS[model_id]
        or manifest.get("max_length") != MAX_LENGTH
    ):
        raise ValueError("unrecognized Clef checkpoint")
    config = json.loads((path / "config.json").read_text())
    if config.get("architectures") != [CLEF_QWEN35_ARCHITECTURE]:
        raise ValueError("Clef model architecture mismatch")
    if json.loads((path / "joint_head_config.json").read_text()) != HEAD_CONFIG:
        raise ValueError("Clef joint head configuration mismatch")
    index = json.loads((path / "model.safetensors.index.json").read_text())
    if "lm_head.weight" not in index["weight_map"]:
        raise ValueError("Clef output embedding missing")
    checked_files = verify_files(
        path,
        manifest,
        index,
        ("config.json", "model.safetensors.index.json", *COPIED_FILES),
        full=full,
    )
    with safe_open(
        path / "joint_head.safetensors", framework="pt", device="cpu"
    ) as file:
        if any(not torch.isfinite(file.get_tensor(name)).all() for name in file.keys()):
            raise ValueError("nonfinite Clef joint head")
    return {
        "format": manifest["format"],
        "architecture": CLEF_QWEN35_ARCHITECTURE,
        "checked_files": checked_files,
        "full": full,
    }


def main() -> None:
    """Export a local copy of the release, e.g. to serve it without downloading."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", choices=sorted(MODELS), required=True)
    args = parser.parse_args()
    export_clef(args.source, args.output, args.model_id)
    verify_clef(args.output, full=True)
    print("JEV_CHECKPOINT_READY", args.output, flush=True)


if __name__ == "__main__":
    main()
