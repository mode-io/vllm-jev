"""Prepare released RSI-Jev checkpoints for native vLLM pooling.

A release ships its fine-tuned Qwen3.5 text tower in float32 without the
embedding, a scorer, and a calibration. The export writes a bf16
``Qwen3_5ForConditionalGeneration`` checkpoint: the release's tower, plus the
embedding and the frozen vision tower of the pinned base model.
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from . import RSIJEV_QWEN35_ARCHITECTURE
from .checkpoint import sha256, verify_files
from .rsijev import PROTOCOL

BASE_ID = "Qwen/Qwen3.5-2B-Base"
BASE_REVISION = "b1485b2fa6dfa1287294f269f5fb618e03d52d7c"
MODELS = {
    "shgao/rsi-jev-v4.0-vl-qwen3.5-2b": "079ec5f4f421481a77609176dabc2b2ef46dc497",
    "shgao/rsi-jev-v3.0-qwen3.5-2b": "c778b68dd5c20e67ac7ca8d8ef1f9a1258a687a5",
}
# sha256 of the released files, so a local copy is checked like a download.
RELEASE_HASHES = {
    "shgao/rsi-jev-v4.0-vl-qwen3.5-2b": {
        "tower.safetensors": "35c147b8422d8729935831d4b23fdac04949a0b0fb69fd0af87b71d785fb2f48",
        "scorer.safetensors": "f1aafc26c2cea47dd16e55f231896fb2ebbdddf34e0d6f54bbb865f63ff4576b",
        "calibration.safetensors": "114e7c77a1f763a257a05fc9522c0a1ac98c9a89e35377fe7a3e2d035932ec43",
        "calibration.json": "dde535fb1f1324d8cf9c1b45de6826e1dcdde7778bd57c6756d7dc0698304f32",
        "meta.json": "02f18a4a6a9771504b452e7061f6ec61b3c90b1b2681b29da968fd84b15eefd1",
    },
    "shgao/rsi-jev-v3.0-qwen3.5-2b": {
        "tower.safetensors": "80b28dd9d202c4f497202dd95017c9a5e922db090180e71cd52c9369283f716b",
        "scorer.safetensors": "270bec61b9dc4649c473f38087e00cbc6af39265d30560ddc9d89f8543042d83",
        "calibration.safetensors": "f304625841aa591c6e88bb827caa478386b9239e7ec8b962aa6232ace9a0dc83",
        "calibration.json": "e8efd54d8c2a742c55c226bfa6dc729aa39b3689a8844875ec8a40f001c14338",
        "meta.json": "8b923c96613b27bcd791eb60ee49557ae21ffb92a861fafd39bc60ee71e1879d",
    },
}
SOURCE_FILES = [*RELEASE_HASHES["shgao/rsi-jev-v3.0-qwen3.5-2b"], "MANIFEST.json"]
BASE_FILES = [
    "config.json",
    "model.safetensors*",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
]
COPIED_BASE_FILES = BASE_FILES[2:]
TEXT_SHARD = "model-rsijev-text.safetensors"
VISION_SHARD = "model-qwen-vision.safetensors"


def readout_config(meta: dict, calibration: dict) -> dict:
    """The released readout settings this adapter implements, or an error."""
    spec = meta["spec"]
    extra = dict(spec.get("arch_extra") or {})
    supported = {"xattn_combine", "xattn_heads", "xattn_mlp_hidden", "xattn_dim"}
    if (
        meta.get("base_model") != BASE_ID
        or spec.get("readout") != "option_xattn"
        or spec.get("readout_layer") != -1
        or spec.get("option_pool") != "mean"
        or spec.get("layout") != "state_first"
        or spec.get("residual")
        or spec.get("logit_cap") is not None
        or spec.get("head_input_norm")
        or set(extra) - supported
        or extra.get("xattn_combine", "sum") not in ("sum", "mlp")
    ):
        raise ValueError("unsupported RSI-Jev readout")
    release = meta.get("release") or {}
    cal_mode = calibration.get("cal_mode")
    if cal_mode != release.get("cal_mode") or cal_mode not in (
        "none",
        "temp",
        "temp_mode",
        "oof_head",
        "oof_head_scorefloor",
    ):
        raise ValueError("unsupported RSI-Jev calibration")
    vision = release.get("vision")
    config = {
        "readout": "option_xattn",
        "xattn_combine": extra.get("xattn_combine", "sum"),
        "xattn_heads": int(extra.get("xattn_heads", 4)),
        "xattn_mlp_hidden": int(extra.get("xattn_mlp_hidden", 512)),
        "xattn_dim": extra.get("xattn_dim"),
        "max_options": int(spec["max_options"]),
        "max_length_text": int(release.get("max_length_text", 2048)),
        "cal_mode": cal_mode,
        "vision": None,
    }
    if vision:
        if (
            vision.get("model", BASE_ID) != BASE_ID
            or vision.get("revision", BASE_REVISION) != BASE_REVISION
        ):
            raise ValueError("RSI-Jev vision tower does not match the pinned base")
        config["vision"] = {
            "image_token_budget": int(vision.get("budget", 1024)),
            "min_tokens_per_image": int(vision.get("min_tokens_per_image", 64)),
        }
        config["max_length_image"] = int(
            vision.get("max_length_image", config["max_length_text"] + 1024)
        )
    return config


def export_rsijev(source: Path, base: Path, output: Path, model_id: str) -> dict:
    if model_id not in MODELS:
        raise ValueError("unsupported RSI-Jev checkpoint")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    for name, digest in RELEASE_HASHES[model_id].items():
        if sha256(source / name) != digest:
            raise ValueError(f"RSI-Jev release checksum mismatch: {name}")
    meta = json.loads((source / "meta.json").read_text())
    calibration = json.loads((source / "calibration.json").read_text())
    config = readout_config(meta, calibration)
    base_config = json.loads((base / "config.json").read_text())
    if base_config.get("architectures") != ["Qwen3_5ForConditionalGeneration"]:
        raise ValueError("RSI-Jev base is not a Qwen3.5 vision-language checkpoint")
    output.mkdir(parents=True, exist_ok=True)

    index = json.loads((base / "model.safetensors.index.json").read_text())
    base_weights = index["weight_map"]
    text, vision = {}, {}
    embedding = "model.language_model.embed_tokens.weight"
    with safe_open(source / "tower.safetensors", framework="pt", device="cpu") as file:
        for name in file.keys():
            # One round-to-nearest-even from float32, as RSI-Jev's bf16 serving does.
            text["model.language_model." + name] = (
                file.get_tensor(name).to(torch.bfloat16).contiguous()
            )
    expected = {
        name
        for name in base_weights
        if name.startswith("model.language_model.") and name != embedding
    }
    if set(text) != expected:
        raise ValueError("RSI-Jev tower does not match the Qwen3.5 base layout")
    for shard in sorted(set(base_weights.values())):
        if Path(shard).name != shard:
            raise ValueError("base shard path must stay within the checkpoint")
        with safe_open(base / shard, framework="pt", device="cpu") as file:
            for name in file.keys():
                if name == embedding:
                    text[name] = file.get_tensor(name).to(torch.bfloat16).contiguous()
                elif name.startswith("model.visual."):
                    vision[name] = file.get_tensor(name).contiguous()
    if embedding not in text or not vision:
        raise ValueError("Qwen3.5 base embedding or vision weights missing")
    save_file(text, str(output / TEXT_SHARD), metadata={"format": "pt"})
    save_file(vision, str(output / VISION_SHARD), metadata={"format": "pt"})
    weight_map = {
        **{name: TEXT_SHARD for name in text},
        **{name: VISION_SHARD for name in vision},
    }
    del text, vision
    (output / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map}, indent=2) + "\n"
    )
    base_config["architectures"] = [RSIJEV_QWEN35_ARCHITECTURE]
    (output / "config.json").write_text(json.dumps(base_config, indent=2) + "\n")
    for name in COPIED_BASE_FILES:
        if (base / name).is_file():
            shutil.copy2(base / name, output / name)
    shutil.copy2(source / "scorer.safetensors", output / "rsijev_scorer.safetensors")
    shutil.copy2(
        source / "calibration.safetensors", output / "rsijev_calibration.safetensors"
    )
    (output / "rsijev_config.json").write_text(json.dumps(config, indent=2) + "\n")
    manifest = {
        "format": "vllm-jev-rsijev-v1",
        "architecture": RSIJEV_QWEN35_ARCHITECTURE,
        "prompt_protocol": PROTOCOL,
        "source_repository": model_id,
        "source_revision": MODELS[model_id],
        "base_repository": BASE_ID,
        "base_revision": BASE_REVISION,
        "max_length": config.get("max_length_image", config["max_length_text"]),
        "vision": config["vision"] is not None,
        "files": {
            file.name: sha256(file) for file in output.iterdir() if file.is_file()
        },
    }
    (output / "rsijev_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def verify_rsijev(path: Path, *, full: bool = True) -> dict:
    path = path.resolve()
    manifest = json.loads((path / "rsijev_manifest.json").read_text())
    model_id = manifest.get("source_repository")
    if (
        manifest.get("format") != "vllm-jev-rsijev-v1"
        or manifest.get("architecture") != RSIJEV_QWEN35_ARCHITECTURE
        or manifest.get("prompt_protocol") != PROTOCOL
        or model_id not in MODELS
        or manifest.get("source_revision") != MODELS[model_id]
        or manifest.get("base_revision") != BASE_REVISION
        or type(manifest.get("max_length")) is not int
        or not 1 <= manifest["max_length"] <= 8192
    ):
        raise ValueError("unrecognized RSI-Jev checkpoint")
    config = json.loads((path / "config.json").read_text())
    if config.get("architectures") != [RSIJEV_QWEN35_ARCHITECTURE]:
        raise ValueError("RSI-Jev model architecture mismatch")
    readout = json.loads((path / "rsijev_config.json").read_text())
    if (readout.get("vision") is not None) != manifest.get("vision"):
        raise ValueError("RSI-Jev vision settings disagree with the manifest")
    hidden = config["text_config"]["hidden_size"]
    with safe_open(
        path / "rsijev_scorer.safetensors", framework="pt", device="cpu"
    ) as file:
        if file.get_slice("q.weight").get_shape()[1] != hidden:
            raise ValueError("RSI-Jev scorer does not match the tower width")
        if any(not torch.isfinite(file.get_tensor(name)).all() for name in file.keys()):
            raise ValueError("nonfinite RSI-Jev scorer")
    index = json.loads((path / "model.safetensors.index.json").read_text())
    checked_files = verify_files(
        path,
        manifest,
        index,
        (
            "config.json",
            "model.safetensors.index.json",
            "rsijev_config.json",
            "rsijev_scorer.safetensors",
            "rsijev_calibration.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "preprocessor_config.json",
        ),
        full=full,
    )
    return {
        "format": manifest["format"],
        "architecture": RSIJEV_QWEN35_ARCHITECTURE,
        "checked_files": checked_files,
        "full": full,
    }


def main() -> None:
    """Export a local copy of a release, e.g. to serve it without downloading."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", choices=sorted(MODELS), required=True)
    args = parser.parse_args()
    manifest = export_rsijev(args.source, args.base, args.output, args.model_id)
    verify_rsijev(args.output, full=True)
    print("JEV_CHECKPOINT_READY", args.output, manifest["max_length"], flush=True)


if __name__ == "__main__":
    main()
