"""Prepare the released Valen-Decision-1.5 checkpoints for native vLLM pooling.

A release stores a bf16 ``Qwen3_5Model`` backbone under ``backbone.``, the
FP32 Mixer head under ``head.`` and the FP32 grounding head under
``grounding.``. The export keeps the weight shards unchanged, names the
pooling architecture in a plain Qwen3.5 ``config.json``, and writes the two
heads beside it for the endpoint. The release's Python files are not copied
or executed.
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from . import VALEN_DECISION_QWEN35_ARCHITECTURE
from .checkpoint import sha256, verify_files
from .valen_decision import HEAD_FILE, MANIFEST, PROTOCOL
from .valen_decision_blocks import MASKS

FORMAT = "vllm-jev-valen-decision-v1"
MODELS = {
    "Valen-Team/Valen-Decision-1.5-0.8B": "e079395fe9b8e0763b384af403e636e91c1a1559",
    "Valen-Team/Valen-Decision-1.5-2B": "1b890606578ba0f40bb70ed46891ab65f778e76d",
    "Valen-Team/Valen-Decision-1.5-4B": "9b0cacc976d7fb72f252ee1adb1564f5f15e462c",
    "Valen-Team/Valen-Decision-1.5-9B": "5a512117ebc0f2b69f311becfcf47cdef8abf568",
    "Valen-Team/Valen-Decision-1.5-27B": "9f1b90bcb6841d9a4a44acfd84c6ecbd4c18fec6",
}
# sha256 of the released files, so a local copy is checked like a download.
RELEASE_HASHES = {
    "Valen-Team/Valen-Decision-1.5-0.8B": {
        "chat_template.jinja": "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80",
        "config.json": "13f0d828c5a39e18bd8329231e5ad259d087bd3b83326f2c3cd8e1b26e95af93",
        "model.safetensors": "d2808df20a2501debe47bd8bacac41d1cd33cc1a9d6f687d8dc436da9f3603bc",
        "processor_config.json": "d89ef49ce9cd37fbf510158e13c1ef063d9286411c1ec9049932dbe0487143b1",
        "tokenizer.json": "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
        "tokenizer_config.json": "7488550a894dcf741616126e322ea31d66cf9457b59ec5825f4c7efdced421f8",
    },
    "Valen-Team/Valen-Decision-1.5-2B": {
        "chat_template.jinja": "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80",
        "config.json": "97faa5441752eecb304a6ec8d17cf03971d5bb363481eb2020b347fe88072694",
        "model.safetensors": "1c9c1112aa0c417b081b0fa757c78be77e487b97de50adb4c83eecfbe03c35da",
        "processor_config.json": "d89ef49ce9cd37fbf510158e13c1ef063d9286411c1ec9049932dbe0487143b1",
        "tokenizer.json": "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
        "tokenizer_config.json": "7488550a894dcf741616126e322ea31d66cf9457b59ec5825f4c7efdced421f8",
    },
    "Valen-Team/Valen-Decision-1.5-4B": {
        "chat_template.jinja": "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715",
        "config.json": "fc6c60829e353a7c66bcf21072fe4a6f8a52490604867d91226aa740ec12698a",
        "model-00001-of-00002.safetensors": "e79499386c15eebd0caffe0126372e17a895e722c1c6f234db549ec7ace7365f",
        "model-00002-of-00002.safetensors": "18e13d4365d7181ab5eefad8de38bf526ca83792998c7623f99d287b57f40c34",
        "model.safetensors.index.json": "547249c00c48c710dbc21713d412da785e17e3a01a9ebf4e8d4db6403b598706",
        "processor_config.json": "d89ef49ce9cd37fbf510158e13c1ef063d9286411c1ec9049932dbe0487143b1",
        "tokenizer.json": "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
        "tokenizer_config.json": "7488550a894dcf741616126e322ea31d66cf9457b59ec5825f4c7efdced421f8",
    },
    "Valen-Team/Valen-Decision-1.5-9B": {
        "chat_template.jinja": "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715",
        "config.json": "25595cf89d8552fe33405fe6e8e5b44fa1a8ff379ab92f0d4dd7863038b07f59",
        "model-00001-of-00004.safetensors": "c8febff8ad3c029006b7443624becd5394daa0bd5281df6e959da91c59662a34",
        "model-00002-of-00004.safetensors": "e2e37962c95a303c63ea9d09534ceb7cbe5a0d6a0f186ebf4304f7df7a45fdb5",
        "model-00003-of-00004.safetensors": "5988fd49da5e85fe045898d2540ce587dc53cbeee420dde45670812f7b5373b8",
        "model-00004-of-00004.safetensors": "959a26154c0f432680036bfa041a6ea0058a6b34fb39d3ee8b7e1cc5e9c785a9",
        "model.safetensors.index.json": "5443f342888348373c769089ab6c62fff7411e8783a9260d4c75a7792afb8ea7",
        "processor_config.json": "d89ef49ce9cd37fbf510158e13c1ef063d9286411c1ec9049932dbe0487143b1",
        "tokenizer.json": "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
        "tokenizer_config.json": "7488550a894dcf741616126e322ea31d66cf9457b59ec5825f4c7efdced421f8",
    },
    "Valen-Team/Valen-Decision-1.5-27B": {
        "chat_template.jinja": "567255be85e0a113405f917423fd8ef5592b01616ce6bdfb6fc4122b9818ec0e",
        "config.json": "d30d71fc730984e6f0caf80a14eab83837fd88660224cf8699738b2371d3c830",
        "model-00001-of-00011.safetensors": "64db8c6fb6a142c4da7567f3d8a5c9e4f02459920311e544c7a64816b7ec2042",
        "model-00002-of-00011.safetensors": "fafb6729ed29e70ca6e485de9bc41d869f2a7c9d0bc1867cd9658316972fdb77",
        "model-00003-of-00011.safetensors": "3d2facb794092e90854ec37764a039586a4778fdcb49b54f41953d95a14de7cc",
        "model-00004-of-00011.safetensors": "6816b9136421aba4591fe29f85dada4c0bec2d86bb5b5c095c58b7e5cce8f83c",
        "model-00005-of-00011.safetensors": "43b5d9d7b222d96e258f6562a3db5b461acf837cfcd0df4d3d10363f5d188292",
        "model-00006-of-00011.safetensors": "63434f7882aab05d1420b3900a1919c3beccbb5af4538cb775399d5e4865115d",
        "model-00007-of-00011.safetensors": "6f3e9ce07026c073a0bc7cb369af1500b3a2e3fc4687007e255fd8cffd299c32",
        "model-00008-of-00011.safetensors": "ac5a0aa0358bb29d70939235435c18398946d09e42b52bb195b149f82cc00400",
        "model-00009-of-00011.safetensors": "57e92e32c3dd183adbd477f4f7b2eb9ab8b99700938a2e02034ef5305653fcde",
        "model-00010-of-00011.safetensors": "87e5033793296d842e6cccb304492a1a323f8b614e656ae55be9549c544d8cbc",
        "model-00011-of-00011.safetensors": "796ec1999b31b70eda19138a8d66e3da2cddcb4af09e0fe7c6c58503cda12d82",
        "model.safetensors.index.json": "df49cc26f6fb5b30359c0dc14cfe8716b5df40639bd809ff8aaeecedcf32be14",
        "processor_config.json": "d89ef49ce9cd37fbf510158e13c1ef063d9286411c1ec9049932dbe0487143b1",
        "tokenizer.json": "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
        "tokenizer_config.json": "7488550a894dcf741616126e322ea31d66cf9457b59ec5825f4c7efdced421f8",
    },
}
SOURCE_FILES = sorted(
    {name for files in RELEASE_HASHES.values() for name in files} | {"LICENSE"}
)
COPIED_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "processor_config.json",
    "chat_template.jinja",
)
INDEX_FILE = "model.safetensors.index.json"
HEAD_KEYS = (
    "head_width",
    "head_layers",
    "head_token_hidden_dim",
    "head_channel_hidden_dim",
)
# Keys of the release config that describe the decision model, not Qwen3.5.
VALEN_KEYS = (
    "auto_map",
    "qwen_execution",
    "valen_backbone_autocast_dtype",
    "valen_decision_version",
    "valen_full_attention_mask",
    "valen_grounding",
    "valen_head",
    "valen_max_length",
    "valen_media_kwargs",
)


def is_valen_decision(model_id: str) -> bool:
    return model_id in MODELS


def validate_template(template) -> None:
    if template.config not in ({}, {"layout": "native"}):
        raise ValueError("Valen-Decision-1.5 supports only its native template")


def _settings(config: dict) -> dict:
    """The decision settings of a release config, as manifest fields."""
    text = config.get("text_config") or {}
    head = config.get("valen_head") or {}
    grounding = config.get("valen_grounding")
    media = config.get("valen_media_kwargs")
    if (
        config.get("model_type") != "valen_qwen"
        or config.get("architectures") != ["ValenQwenForDecisionMaking"]
        or config.get("valen_decision_version") != "1.5"
        or config.get("qwen_execution") != "shared_state"
        or config.get("valen_full_attention_mask") not in MASKS
        or config.get("valen_backbone_autocast_dtype") is not None
        or config.get("dtype") != "bfloat16"
        or text.get("model_type") != "qwen3_5_text"
        or "full_attention" not in text.get("layer_types", [])
    ):
        raise ValueError("unsupported Valen-Decision-1.5 configuration")
    if head.get("head_type") != "mixer" or set(head) != {"head_type", *HEAD_KEYS}:
        raise ValueError("unsupported Valen-Decision-1.5 decision head")
    if grounding is not None and (
        not isinstance(grounding, dict) or set(grounding) != {"width", "hidden"}
    ):
        raise ValueError("unsupported Valen-Decision-1.5 grounding head")
    sizes = [
        text.get("hidden_size"),
        config.get("valen_max_length"),
        *(head[key] for key in HEAD_KEYS),
        *(grounding or {}).values(),
    ]
    if any(type(value) is not int or value <= 0 for value in sizes):
        raise ValueError("invalid Valen-Decision-1.5 dimensions")
    if not isinstance(media, dict) or not isinstance(
        media.get("images_kwargs", {}).get("size"), dict
    ):
        raise ValueError("Valen-Decision-1.5 release has no trained image size")
    return {
        "decision_version": "1.5",
        "full_attention_mask": config["valen_full_attention_mask"],
        "hidden_size": text["hidden_size"],
        "head": {key: head[key] for key in HEAD_KEYS},
        "grounding": dict(grounding) if grounding else None,
        "media_kwargs": media,
        "max_length": config["valen_max_length"],
    }


def _head_shapes(settings: dict) -> dict:
    """Shapes that tie the head file to the manifest's dimensions."""
    hidden, width = settings["hidden_size"], settings["head"]["head_width"]
    shapes = {
        "head.projection.weight": (width, hidden),
        "head.roles": (1, 4, width),
        "head.output.1.weight": (1, width),
    }
    if settings["grounding"]:
        shapes["grounding.key.weight"] = (settings["grounding"]["width"], hidden)
        shapes["grounding.regress.3.weight"] = (4, settings["grounding"]["hidden"])
    return shapes


def _place(source: Path, target: Path, link: bool) -> None:
    """Hard-link a large file when both sides share a file system."""
    if link:
        try:
            os.link(source.resolve(), target)
            return
        except OSError:
            pass
    shutil.copy2(source, target)


def export_valen_decision(
    source: Path, output: Path, model_id: str, *, link: bool = True
) -> dict:
    if model_id not in MODELS:
        raise ValueError("unsupported Valen-Decision-1.5 checkpoint")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    hashes = RELEASE_HASHES[model_id]
    for name, digest in hashes.items():
        if sha256(source / name) != digest:
            raise ValueError(f"Valen-Decision-1.5 release checksum mismatch: {name}")
    config = json.loads((source / "config.json").read_text())
    settings = _settings(config)
    output.mkdir(parents=True, exist_ok=True)
    shards = sorted(name for name in hashes if name.endswith(".safetensors"))
    weight_map, heads = {}, {}
    for shard in shards:
        with safe_open(source / shard, framework="pt", device="cpu") as file:
            for name in file.keys():
                if name.startswith(("head.", "grounding.")):
                    heads[name] = file.get_tensor(name)
                elif not name.startswith(
                    ("backbone.visual.", "backbone.language_model.")
                ):
                    raise ValueError(f"unexpected Valen-Decision-1.5 weight: {name}")
                weight_map[name] = shard
        _place(source / shard, output / shard, link)
    if INDEX_FILE in hashes:
        index = json.loads((source / INDEX_FILE).read_text())
        if index.get("weight_map") != weight_map:
            raise ValueError("Valen-Decision-1.5 shards disagree with their index")
        shutil.copy2(source / INDEX_FILE, output / INDEX_FILE)
    else:
        # A single-file release has no index; vLLM needs one to skip the head file.
        (output / INDEX_FILE).write_text(
            json.dumps(
                {"metadata": {"format": "pt"}, "weight_map": weight_map}, indent=2
            )
            + "\n"
        )
    if any(tensor.dtype != torch.float32 for tensor in heads.values()):
        raise ValueError("Valen-Decision-1.5 heads must be FP32")
    save_file(heads, str(output / HEAD_FILE), metadata={"format": "pt"})
    for name in COPIED_FILES:
        shutil.copy2(source / name, output / name)
    for key in VALEN_KEYS:
        config.pop(key, None)
    config["model_type"] = "qwen3_5"
    config["architectures"] = [VALEN_DECISION_QWEN35_ARCHITECTURE]
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    files = {name: hashes[name] for name in (*shards, *COPIED_FILES)}
    for name in ("config.json", INDEX_FILE, HEAD_FILE):
        files[name] = sha256(output / name)
    manifest = {
        "format": FORMAT,
        "architecture": VALEN_DECISION_QWEN35_ARCHITECTURE,
        "prompt_protocol": PROTOCOL,
        "source_repository": model_id,
        "source_revision": MODELS[model_id],
        **settings,
        "files": files,
    }
    (output / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def verify_valen_decision(path: Path, *, full: bool = True) -> dict:
    path = path.resolve()
    manifest = json.loads((path / MANIFEST).read_text())
    model_id = manifest.get("source_repository")
    if (
        manifest.get("format") != FORMAT
        or manifest.get("architecture") != VALEN_DECISION_QWEN35_ARCHITECTURE
        or manifest.get("prompt_protocol") != PROTOCOL
        or model_id not in MODELS
        or manifest.get("source_revision") != MODELS[model_id]
        or manifest.get("decision_version") != "1.5"
        or manifest.get("full_attention_mask") not in MASKS
        or not isinstance(manifest.get("media_kwargs"), dict)
        or type(manifest.get("max_length")) is not int
        or not isinstance(manifest.get("head"), dict)
        or set(manifest["head"]) != set(HEAD_KEYS)
    ):
        raise ValueError("unrecognized Valen-Decision-1.5 checkpoint")
    hashes = RELEASE_HASHES[model_id]
    released = (*(n for n in hashes if n.endswith(".safetensors")), *COPIED_FILES)
    if any(manifest.get("files", {}).get(name) != hashes[name] for name in released):
        raise ValueError("Valen-Decision-1.5 checkpoint differs from its release")
    config = json.loads((path / "config.json").read_text())
    if (
        config.get("architectures") != [VALEN_DECISION_QWEN35_ARCHITECTURE]
        or config.get("model_type") != "qwen3_5"
        or config.get("text_config", {}).get("hidden_size")
        != manifest.get("hidden_size")
    ):
        raise ValueError("Valen-Decision-1.5 model architecture mismatch")
    index = json.loads((path / INDEX_FILE).read_text())
    checked_files = verify_files(
        path,
        manifest,
        index,
        ("config.json", INDEX_FILE, HEAD_FILE, *COPIED_FILES),
        full=full,
    )
    for prefix in ("backbone.visual.", "backbone.language_model."):
        if not any(name.startswith(prefix) for name in index["weight_map"]):
            raise ValueError(f"Valen-Decision-1.5 weights missing: {prefix}*")
    with safe_open(path / HEAD_FILE, framework="pt", device="cpu") as file:
        names = set(file.keys())
        if bool(manifest["grounding"]) != any(
            n.startswith("grounding.") for n in names
        ):
            raise ValueError("Valen-Decision-1.5 grounding head mismatch")
        for name, shape in _head_shapes(manifest).items():
            if name not in names or tuple(file.get_tensor(name).shape) != shape:
                raise ValueError(f"invalid Valen-Decision-1.5 head tensor: {name}")
        if any(not torch.isfinite(file.get_tensor(name)).all() for name in names):
            raise ValueError("nonfinite Valen-Decision-1.5 head")
    return {
        "format": manifest["format"],
        "architecture": VALEN_DECISION_QWEN35_ARCHITECTURE,
        "checked_files": checked_files,
        "full": full,
    }


def main() -> None:
    """Export a local copy of the release, e.g. to serve it without downloading."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", choices=sorted(MODELS), required=True)
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy the weight shards instead of hard-linking them.",
    )
    args = parser.parse_args()
    export_valen_decision(args.source, args.output, args.model_id, link=not args.copy)
    verify_valen_decision(args.output, full=True)
    print("JEV_CHECKPOINT_READY", args.output, flush=True)


if __name__ == "__main__":
    main()
