"""Prepare the released Clef checkpoints for native vLLM pooling.

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

MODELS = {
    "Cloudflare/clef-flash": "17f0b0ad64efb65d273590632833508766b2aae6",
    "Cloudflare/clef": "2f3de3dd85f379784083b0814d997ab627200f0c",
}
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
    "Cloudflare/clef": {
        "model-00001-of-00012.safetensors": "54d83c1d36631de231876217a8e0c2483eccee8746369a482b79442bdfc5d958",
        "model-00002-of-00012.safetensors": "464086af08be8e2ec14960a4dcff083ebc39974ade00d79d35497385f960ab3a",
        "model-00003-of-00012.safetensors": "092212d3a02fafacd6424723eda59d60e5282d2068d68f0e37cb891f63bbb658",
        "model-00004-of-00012.safetensors": "d06ff197668c782145fafa74bba61bbc296fb27e39afb15fd522918ce3514dc5",
        "model-00005-of-00012.safetensors": "cc693b8829614a72e0c2be303fb290cf05dbb4d7ded872a817b97bb222a78427",
        "model-00006-of-00012.safetensors": "e7cce15da2443cb8b84aaed66a9a71f0c87dc9d043f83b5f58f4a89f64ba60ad",
        "model-00007-of-00012.safetensors": "75fc7e76b57d5d17a5d85fff3e879d07dd33edc885a8ee04ad437a899bcd5307",
        "model-00008-of-00012.safetensors": "189b15cb6b1af48d5f118951446e15639bfeaf76081d5f20aed1f1b4253afe1d",
        "model-00009-of-00012.safetensors": "8101e2664bb14684fc7051f2e1f84903dbdf5489b17cae3212ac08a0af744a60",
        "model-00010-of-00012.safetensors": "a8e69016a1a8dab1c9ce8dd151c0d3224412ab864cf06d3a5e2b8a5775cfdfb8",
        "model-00011-of-00012.safetensors": "b328d21c36ab384696e40a30ac86a95aaf6dd82da89438ba009cb97d87c1d6b9",
        "model-00012-of-00012.safetensors": "7505eed910a84ea18e66953572476f8a8a6a6ef54192a6643d6e9cd21a3bb978",
        "model.safetensors.index.json": "a600c6626eb1eb653a3f078ab3dd1c7a8fab2e71185c7ef3aa923a39aaa637be",
        "config.json": "c42e88892bd3fd84e8276b2ad90df58c1c3b797676ea161035006a72ad468c58",
        "joint_head.safetensors": "a010ac04f078e699988e4049cbea5e62c962393f59fec366640b64e8d69a4953",
        "joint_head_config.json": "890be585d75b981201eb96a35f98a9967afe37bc8d220cfdea2e72d56507534f",
        "tokenizer.json": "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
        "tokenizer_config.json": "91a08f825d370d085d692e04cf117cdd7faad7bf18e996f1e6031b6dab03db72",
        "processor_config.json": "d89ef49ce9cd37fbf510158e13c1ef063d9286411c1ec9049932dbe0487143b1",
        "chat_template.jinja": "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041",
    },
}
SOURCE_FILES = sorted(
    {name for files in RELEASE_HASHES.values() for name in files} | {"LICENSE"}
)
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


def _head_config(model_id: str) -> dict:
    if model_id == "Cloudflare/clef":
        return {**HEAD_CONFIG, "hidden_size": 5120}
    return HEAD_CONFIG


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
    if json.loads((source / "joint_head_config.json").read_text()) != _head_config(
        model_id
    ):
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
    if json.loads((path / "joint_head_config.json").read_text()) != _head_config(
        model_id
    ):
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
