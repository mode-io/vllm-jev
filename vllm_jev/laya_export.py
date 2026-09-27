"""Prepare pinned Laya checkpoints for native vLLM pooling."""

import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from . import LAYA_MODERNBERT_ARCHITECTURE
from .checkpoint import sha256, verify_files

MODEL_ID = "convaiinnovations/laya"
MODEL_REVISION = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"
MODEL_SHA256 = "891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c"
MODELS = {
    MODEL_ID: (MODEL_REVISION, MODEL_SHA256, 512, 192),
    "convaiinnovations/laya-multilingual": (
        "e4e9ddf21a7b1903b7acffd8814ad4307bf63a67",
        "9d628fd971b700382ac6f65920a86f149777b2e748e0c955fb3b19695aa8f204",
        1024,
        256,
    ),
    "convaiinnovations/laya-typed-decisions": (
        "1a793eb568e6718f15941d08f85432581df534e3",
        "4fa56de72383a9d3efa9cfa78955733c81b9fc8067a587ca4beb82c78107a24e",
        1024,
        256,
    ),
}
PROTOCOL = "laya_markers_v1"
SOURCE_FILES = (
    "model.safetensors",
    "rl_agent_config.json",
    "encoder/config.json",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
)
SOURCE_HASHES = {
    MODEL_ID: (
        "bf3ab80598fdccf414855a2ce80f22859e4492d06ca8a62ddd1cfb63972f8979",
        "ae287b56bbcf5f8c4f4541ae9dfd00c914c4c48b940b8398c3058af37ba92bbd",
        "6c8aaa9a542084f2457eab775d4eeb51f92a70c0fd9de28d5edb0ddec3c08d30",
        "50044de60daaa73df97d262e15a40d4faf0160e7d742df64b377877a1320dd12",
    ),
    "convaiinnovations/laya-multilingual": (
        "83f6916d13ef0f556ac461f28308dc2bffa7ebeadee8ec9e2db5812020ea5bb4",
        "25061739243b617ad88d1219ba6f8a9c86c5881ca28df024fa2d9b3b2fcc30c6",
        "609d8f4c067cd3950f88594c5a802616cea245823836ef5848ee4fc40aab5b6f",
        "6c6b2d8e3c84ce0e671c129cd6b374b235d6f9863042a5836358d00a89bbb5a1",
    ),
    "convaiinnovations/laya-typed-decisions": (
        "5268d24ad3b77c8151de5dcb0762ba4391619aad9ab0bda33e36fb083cfeae6d",
        "ebf0cd524d92342a6be5e48e9fca3d7c2babfb5a56ccd79d2171ef5d8c7f7be8",
        "6c8aaa9a542084f2457eab775d4eeb51f92a70c0fd9de28d5edb0ddec3c08d30",
        "08d4cf3ac4dca381759441b85b91a6d40e688471dcd33d15d6649eb0a9a854d1",
    ),
}


def verify_source(source: Path, model_id: str) -> tuple[dict, dict]:
    _, weight_hash, max_len, head_max_len = MODELS[model_id]
    hashes = (weight_hash, *SOURCE_HASHES[model_id])
    names = (
        "model.safetensors",
        "encoder/config.json",
        "rl_agent_config.json",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
    )
    if any(sha256(source / name) != expected for name, expected in zip(names, hashes)):
        raise ValueError("Laya source files do not match the pinned release")
    tokenizer_files = {
        path.name for path in (source / "tokenizer").iterdir() if path.is_file()
    }
    if tokenizer_files != {"tokenizer.json", "tokenizer_config.json"}:
        raise ValueError("Laya source has unexpected tokenizer files")
    config = json.loads((source / "encoder/config.json").read_text())
    training = json.loads((source / "rl_agent_config.json").read_text())
    if (
        config.get("model_type") != "modernbert"
        or training.get("head_layers") != 2
        or training.get("max_len") != max_len
        or training.get("head_max_len") != head_max_len
    ):
        raise ValueError("unsupported Laya architecture")
    return config, training


def export_laya(source: Path, output: Path, model_id: str = MODEL_ID) -> dict:
    revision, expected_hash, _, _ = MODELS[model_id]
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    config, training = verify_source(source, model_id)
    output.mkdir(parents=True, exist_ok=True)
    weights = load_file(str(source / "model.safetensors"))
    encoder = {
        key.removeprefix("encoder."): value.contiguous()
        for key, value in weights.items()
        if key.startswith("encoder.")
    }
    head = {
        key: value.contiguous()
        for key, value in weights.items()
        if not key.startswith("encoder.")
    }
    if not encoder or set(head) - {
        key
        for key in weights
        if key.startswith(("head.", "type_emb.", "scorer.", "act_head."))
    } - {"temperature"}:
        raise ValueError("unexpected Laya decision-head weights")
    for prefix in (
        "head.layers.0.",
        "head.layers.1.",
        "type_emb.",
        "scorer.",
        "act_head.",
    ):
        if not any(key.startswith(prefix) for key in head):
            raise ValueError(f"incomplete Laya decision head: {prefix}")
    if head.get("type_emb.weight", torch.empty(0)).shape != (
        3,
        config["hidden_size"],
    ):
        raise ValueError("invalid Laya type embedding")
    save_file(encoder, output / "model.safetensors")
    save_file(head, output / "laya_head.safetensors")
    config["architectures"] = [LAYA_MODERNBERT_ARCHITECTURE]
    config["dtype"] = "float16"
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (output / "laya_config.json").write_text(json.dumps(training, indent=2) + "\n")
    index = {
        "metadata": {
            "total_size": sum(t.numel() * t.element_size() for t in encoder.values())
        },
        "weight_map": {key: "model.safetensors" for key in encoder},
    }
    (output / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2) + "\n"
    )
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        path = source / "tokenizer" / name
        if path.is_file():
            shutil.copy2(path, output / name)
    tokenizer_config = output / "tokenizer_config.json"
    metadata = json.loads(tokenizer_config.read_text())
    if metadata.get("tokenizer_class") in (None, "TokenizersBackend"):
        metadata["tokenizer_class"] = "PreTrainedTokenizerFast"
        metadata.pop("backend", None)
        metadata.pop("is_local", None)
    if isinstance(metadata.get("extra_special_tokens"), list):
        metadata["extra_special_tokens"] = {
            f"extra_{i}": token
            for i, token in enumerate(metadata["extra_special_tokens"])
        }
    tokenizer_config.write_text(json.dumps(metadata, indent=2) + "\n")
    manifest = {
        "format": "vllm-jev-laya-v1",
        "architecture": LAYA_MODERNBERT_ARCHITECTURE,
        "prompt_protocol": PROTOCOL,
        "source_repository": model_id,
        "source_revision": revision,
        "source_weights_sha256": expected_hash,
        "files": {
            path.name: sha256(path) for path in output.iterdir() if path.is_file()
        },
    }
    (output / "laya_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def verify_laya(path: Path, *, full: bool = True) -> dict:
    path = path.resolve()
    manifest = json.loads((path / "laya_manifest.json").read_text())
    model_id = manifest.get("source_repository")
    if model_id not in MODELS:
        raise ValueError("unrecognized Laya checkpoint")
    revision, expected_hash, _, _ = MODELS[model_id]
    if any(
        (
            manifest.get("format") != "vllm-jev-laya-v1",
            manifest.get("architecture") != LAYA_MODERNBERT_ARCHITECTURE,
            manifest.get("prompt_protocol") != PROTOCOL,
            manifest.get("source_revision") != revision,
            manifest.get("source_weights_sha256") != expected_hash,
        )
    ):
        raise ValueError("unrecognized Laya checkpoint")
    config = json.loads((path / "config.json").read_text())
    if config.get("architectures") != [LAYA_MODERNBERT_ARCHITECTURE]:
        raise ValueError("Laya architecture mismatch")
    index = json.loads((path / "model.safetensors.index.json").read_text())
    required = (
        "config.json",
        "laya_config.json",
        "model.safetensors.index.json",
        "model.safetensors",
        "laya_head.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
    )
    checked = verify_files(path, manifest, index, required, full=full)
    with safe_open(
        path / "laya_head.safetensors", framework="pt", device="cpu"
    ) as file:
        if "type_emb.weight" not in file.keys() or file.get_tensor(
            "type_emb.weight"
        ).shape != (3, config["hidden_size"]):
            raise ValueError("invalid Laya decision head")
    return {
        "format": manifest["format"],
        "architecture": LAYA_MODERNBERT_ARCHITECTURE,
        "checked_files": checked,
        "full": full,
    }
