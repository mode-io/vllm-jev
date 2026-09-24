"""Select a verified native vLLM scoring protocol from a Hugging Face model ID."""

import argparse
import json
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.utils import validate_repo_id

from .checkpoint import sha256, verify
from .public_models import PROFILES
from .public_models import prepare as prepare_open_jev

BRANCH_PROTOCOL = "openjev_branch_v03"
CHOICE_PROTOCOL = "open_jev_choice"
TINY_PROTOCOL = "tiny_jev_marker"
BRANCH_FILES = {
    "MANIFEST.json",
    "adapter_config.json",
    "adapter_model.safetensors",
    "head.safetensors",
    "openjev_config.json",
    "calibration-v03.json",
}
SCALAR_FILES = {
    "package/checkpoint/model.json",
    "package/checkpoint/head.pt",
    "package/checkpoint/temperature.json",
    "package/checkpoint/adapter/adapter_config.json",
    "package/checkpoint/adapter/adapter_model.safetensors",
}


def _publish(
    output: Path,
    model_id: str,
    revision: str,
    protocol: str,
    build: Callable[[Path], dict],
) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".jev-", dir=output.parent))
    try:
        manifest = build(temporary)
        if manifest["prompt_protocol"] != protocol:
            raise ValueError("exported checkpoint protocol does not match the source")
        manifest.update(source_repository=model_id, source_revision=revision)
        (temporary / "jev_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        verify(temporary, full=True)
        temporary.replace(output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return output


def prepare(model_id: str, workspace: Path, protocol: str = "auto") -> Path:
    if model_id in PROFILES:
        model_id = PROFILES[model_id][0]
    validate_repo_id(model_id)
    if protocol not in ("auto", CHOICE_PROTOCOL, BRANCH_PROTOCOL, TINY_PROTOCOL):
        raise ValueError(f"unknown protocol: {protocol}")
    known = {spec[0]: name for name, spec in PROFILES.items()}
    if model_id in known:
        if protocol not in ("auto", CHOICE_PROTOCOL):
            raise ValueError(f"{model_id} uses {CHOICE_PROTOCOL}, not {protocol}")
        return prepare_open_jev(known[model_id], workspace)

    workspace = workspace.resolve()
    output = workspace / "checkpoint" / model_id
    if output.exists():
        manifest = json.loads((output / "jev_manifest.json").read_text())
        if manifest.get("source_repository") != model_id:
            raise ValueError("cached checkpoint belongs to another repository")
        if protocol not in ("auto", manifest.get("prompt_protocol")):
            raise ValueError("requested protocol differs from cached checkpoint")
        verify(output, full=True)
        return output

    info = HfApi().model_info(model_id, files_metadata=False, token=False)
    files = {item.rfilename for item in info.siblings}
    if SCALAR_FILES <= files:
        if protocol not in ("auto", CHOICE_PROTOCOL):
            raise ValueError(f"{model_id} uses {CHOICE_PROTOCOL}, not {protocol}")
        spec = json.loads(
            Path(
                hf_hub_download(
                    repo_id=model_id,
                    filename="package/checkpoint/model.json",
                    revision=info.sha,
                    token=False,
                )
            ).read_text()
        )
        base_id, base_revision = spec.get("model_id"), spec.get("revision")
        if (
            not isinstance(base_id, str)
            or not isinstance(base_revision, str)
            or len(base_revision) != 40
        ):
            raise ValueError("native scalar-head export requires a pinned base")
        base_config = json.loads(
            Path(
                hf_hub_download(
                    repo_id=base_id,
                    filename="config.json",
                    revision=base_revision,
                    token=False,
                )
            ).read_text()
        )
        if (
            base_config.get("model_type") != "qwen3_5"
            or base_config.get("text_config", {}).get("model_type") != "qwen3_5_text"
        ):
            raise ValueError(
                "base architecture is not supported by native Qwen3.5 pooling"
            )
        source = (
            Path(
                snapshot_download(
                    repo_id=model_id,
                    revision=info.sha,
                    token=False,
                    local_dir=workspace / "public" / model_id,
                    allow_patterns=sorted(SCALAR_FILES),
                )
            )
            / "package"
            / "checkpoint"
        )
        base = Path(
            snapshot_download(repo_id=base_id, revision=base_revision, token=False)
        )
        from .export import export

        return _publish(
            output,
            model_id,
            info.sha,
            CHOICE_PROTOCOL,
            lambda temporary: export(
                base, source / "adapter", source / "head.pt", temporary
            ),
        )
    if {"modeling_tiny_jev.py", "model.safetensors", "config.json"} <= files:
        if protocol not in ("auto", TINY_PROTOCOL):
            raise ValueError(f"{model_id} uses {TINY_PROTOCOL}, not {protocol}")
        from .tiny import export_tiny

        source = Path(
            snapshot_download(
                repo_id=model_id,
                revision=info.sha,
                token=False,
                local_dir=workspace / "public" / model_id,
                allow_patterns=[
                    "config.json",
                    "model.safetensors",
                    "tokenizer*",
                    "chat_template.jinja",
                ],
            )
        )
        return _publish(
            output,
            model_id,
            info.sha,
            TINY_PROTOCOL,
            lambda temporary: export_tiny(source, temporary, model_id, info.sha),
        )
    if not BRANCH_FILES <= files:
        if "openjet_runtime/candidate_projection.py" in files:
            raise ValueError(
                "this release uses a vocabulary-projection decision protocol"
            )
        if "open_jev_config.json" in files and "model.safetensors" in files:
            raise ValueError("this release uses an encoder/span decision head")
        raise ValueError(
            f"{model_id} has no supported native vLLM Jev checkpoint layout; "
            f"supported protocols: {CHOICE_PROTOCOL}, {BRANCH_PROTOCOL}, {TINY_PROTOCOL}"
        )
    if protocol not in ("auto", BRANCH_PROTOCOL):
        raise ValueError(f"{model_id} uses {BRANCH_PROTOCOL}, not {protocol}")

    package = Path(
        snapshot_download(
            repo_id=model_id,
            revision=info.sha,
            token=False,
            local_dir=workspace / "public" / model_id,
            allow_patterns=sorted(BRANCH_FILES),
        )
    )
    release = json.loads((package / "MANIFEST.json").read_text())
    config = json.loads((package / "openjev_config.json").read_text())
    adapter = json.loads((package / "adapter_config.json").read_text())
    if (
        config.get("architecture") != "SharedStateCandidateBranches"
        or config.get("model_name") != "Qwen/Qwen3-0.6B"
        or adapter.get("base_model_name_or_path") != config["model_name"]
        or adapter.get("revision") != config.get("model_revision")
        or release.get("base_model") != config["model_name"]
        or release.get("base_revision") != config.get("model_revision")
    ):
        raise ValueError(
            "repository is not a matching Qwen3 branch/scalar-head release"
        )
    for name in BRANCH_FILES - {"MANIFEST.json"}:
        if sha256(package / name) != release.get("files", {}).get(name):
            raise ValueError(f"release checksum mismatch: {name}")

    base = Path(
        snapshot_download(
            repo_id=config["model_name"],
            revision=config["model_revision"],
            token=False,
        )
    )
    from .export import export

    return _publish(
        output,
        model_id,
        info.sha,
        BRANCH_PROTOCOL,
        lambda temporary: export(
            base, package, package / "head.safetensors", temporary
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_id")
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        choices=("auto", CHOICE_PROTOCOL, BRANCH_PROTOCOL, TINY_PROTOCOL),
        default="auto",
    )
    args = parser.parse_args()
    print(
        "JEV_CHECKPOINT_READY",
        prepare(args.model_id, args.workspace, args.protocol),
        flush=True,
    )


if __name__ == "__main__":
    main()
