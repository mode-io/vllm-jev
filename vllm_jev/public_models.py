"""Download and export released Open-Jev scalar-head checkpoints."""

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

from filelock import FileLock
from huggingface_hub import snapshot_download

from .checkpoint import sha256, verify

PROFILES = {
    "2b": (
        "ZefanCai/Open-Jev-2B",
        "0c7aa498b1627be8da4acf34c863ff0ee0a92785",
        "Qwen/Qwen3.5-2B",
        "15852e8c16360a2fea060d615a32b45270f8a8fc",
    ),
    "9b": (
        "ZefanCai/Open-Jev-9B",
        "47e966881e489511c0c7f5633a9e1960a676a551",
        "Qwen/Qwen3.5-9B",
        "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
    ),
}

RELEASE_HASHES = {
    "2b": (
        "2d23935b1a7380db444abac572c04646918ba794e59002d1588236182a3ca18f",
        "3532cd576c58d5ad5bf17c3e9f2df4be8c70e08c07fa6bb7fa673dcd7b401f2a",
        "090fb330a616338210a924d1c370c934e934f30f53f92c8a52a5cc7852cadd99",
    ),
    "9b": (
        "f85650a8fb97c6d0ac3e948cdca2f30a0ca6ace8b8a43aebca1c152fe38aeb75",
        "229fe9800384e824135e59d1af59bda7346da031aea61640be5f456b9be6ce70",
        "8cfe40a9f42cc2d0607a3381f2a4e0374cb3842506ac7263f7fff35facfca034",
    ),
}


def resolve_profile(name: str) -> str:
    for profile, (model_id, *_rest) in PROFILES.items():
        if name in (profile, model_id):
            return profile
    raise ValueError(f"unknown public model: {name}")


def prepare(profile: str, workspace: Path) -> Path:
    profile = resolve_profile(profile)
    workspace = workspace.resolve()
    output = workspace / "checkpoint" / PROFILES[profile][0]
    output.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output) + ".lock"):
        return _prepare(profile, workspace)


def _prepare(profile: str, workspace: Path) -> Path:
    adapter_id, adapter_revision, base_id, base_revision = PROFILES[profile]
    adapter_hash, head_hash, temperature_hash = RELEASE_HASHES[profile]
    workspace = workspace.resolve()
    output = workspace / "checkpoint" / adapter_id
    if output.exists():
        verify(output, full=True)
        manifest = json.loads((output / "jev_manifest.json").read_text())
        if (
            manifest.get("foundation_revision") != base_revision
            or manifest.get("adapter_weights_sha256") != adapter_hash
            or manifest.get("head_sha256") != head_hash
            or manifest.get("source_temperature_sha256") != temperature_hash
        ):
            raise ValueError("existing checkpoint does not match the public release")
        return output

    package = (
        Path(
            snapshot_download(
                repo_id=adapter_id,
                revision=adapter_revision,
                local_dir=workspace / "public" / adapter_id,
                allow_patterns=[
                    "package/checkpoint/*",
                    "release-manifest.json",
                    "LICENSE",
                    "UPSTREAM.md",
                ],
                token=False,
            )
        )
        / "package"
        / "checkpoint"
    )
    specification = json.loads((package / "model.json").read_text())
    if (specification.get("model_id"), specification.get("revision")) != (
        base_id,
        base_revision,
    ):
        raise ValueError("public checkpoint does not match its pinned base model")
    if (
        sha256(package / "adapter" / "adapter_model.safetensors") != adapter_hash
        or sha256(package / "head.pt") != head_hash
        or sha256(package / "temperature.json") != temperature_hash
    ):
        raise ValueError("public checkpoint files failed release hash verification")

    base = Path(
        snapshot_download(
            repo_id=base_id,
            revision=base_revision,
            cache_dir=os.environ.get("HF_HUB_CACHE"),
            token=False,
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".open-jev-{profile}-", dir=output.parent)
    )
    try:
        from .export import export

        export(base, package / "adapter", package / "head.pt", temporary)
        verify(temporary, full=True)
        temporary.replace(output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "profile", choices=sorted([*PROFILES, *(p[0] for p in PROFILES.values())])
    )
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    print("JEV_CHECKPOINT_READY", prepare(args.profile, args.workspace), flush=True)


if __name__ == "__main__":
    main()
