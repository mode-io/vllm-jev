"""Prepare a Jev checkpoint and hand serving over to the native vLLM CLI."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    if (
        len(sys.argv) > 1
        and sys.argv[1] == "serve"
        and any(value.startswith("--help=") for value in sys.argv[2:])
    ):
        os.execvpe(
            sys.executable,
            [sys.executable, "-m", "vllm.entrypoints.cli.main", *sys.argv[1:]],
            os.environ.copy(),
        )
    parser = argparse.ArgumentParser(prog="vllm-jev", allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser(
        "serve",
        allow_abbrev=False,
        help="Download, prepare, and serve a Jev model with vLLM.",
        epilog="Additional arguments are forwarded to vllm serve.",
    )
    serve.add_argument("model", help="Hugging Face model ID or exported checkpoint.")
    serve.add_argument(
        "--workspace",
        type=Path,
        default=Path(os.environ.get("VLLM_JEV_HOME", ".local")),
        help="Model and cache directory (default: VLLM_JEV_HOME or .local).",
    )
    serve.add_argument(
        "--protocol",
        choices=(
            "auto",
            "open_jev_choice",
            "openjev_branch_v03",
            "tiny_jev_marker",
            "valen_qwen_v1",
            "vjev_vision_v1",
        ),
        default="auto",
        help="Checkpoint protocol (default: auto).",
    )
    args, extra = parser.parse_known_args()
    workspace = args.workspace.resolve()
    environment = os.environ.copy()
    environment.setdefault("HF_HOME", str(workspace / ".cache"))
    environment.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    plugins = ["vllm_jev_model", "vllm_jev_endpoint"]
    plugins.extend(filter(None, environment.get("VLLM_PLUGINS", "").split(",")))
    environment["VLLM_PLUGINS"] = ",".join(dict.fromkeys(plugins))

    from .public_models import PROFILES

    model_id = PROFILES[args.model][0] if args.model in PROFILES else args.model
    checkpoint = Path(model_id)
    if checkpoint.is_dir():
        checkpoint = checkpoint.resolve()
        valen_manifest = checkpoint / "valen_manifest.json"
        vjev_manifest = checkpoint / "vjev_manifest.json"
        model_id = (
            json.loads(
                (
                    vjev_manifest if vjev_manifest.is_file() else valen_manifest
                ).read_text()
            )["source_repository"]
            if vjev_manifest.is_file() or valen_manifest.is_file()
            else "vllm-jev"
        )
        prepare = ["-m", "vllm_jev.checkpoint", "--model", str(checkpoint), "--quick"]
    else:
        from huggingface_hub.utils import validate_repo_id

        validate_repo_id(model_id)
        checkpoint = workspace / "checkpoint" / model_id
        prepare = [
            "-m",
            "vllm_jev.native_models",
            model_id,
            "--workspace",
            str(workspace),
            "--protocol",
            args.protocol,
        ]
    # Export can use CUDA; the child must exit before vLLM allocates GPU memory.
    try:
        subprocess.run([sys.executable, *prepare], env=environment, check=True)
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode) from None
    valen_manifest = checkpoint / "valen_manifest.json"
    vjev_manifest = checkpoint / "vjev_manifest.json"
    is_valen = valen_manifest.is_file()
    is_vjev = vjev_manifest.is_file()
    if args.protocol != "auto":
        manifest_path = (
            vjev_manifest
            if is_vjev
            else valen_manifest
            if is_valen
            else checkpoint / "jev_manifest.json"
        )
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("prompt_protocol", "open_jev_choice") != args.protocol:
            parser.error("requested protocol differs from the prepared checkpoint")

    defaults = (
        f"--runner pooling --convert none --max-model-len {8192 if is_valen else 4096} "
        "--enable-prefix-caching --mamba-cache-mode align "
        "--mamba-ssm-cache-dtype float32 --async-scheduling "
        "--gpu-memory-utilization 0.9 --host 127.0.0.1 --port 8795"
    ).split()
    if is_valen or is_vjev:
        defaults.extend(["--pooler-config", '{"task":"token_embed"}'])
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        str(checkpoint),
        *defaults,
        "--served-model-name",
        model_id,
        *extra,
    ]
    for name in (
        "VLLM_JEV_HOME",
        "VLLM_JEV_WORKSPACE",
        "VLLM_JEV_MODEL_ID",
        "VLLM_JEV_PORT",
    ):
        environment.pop(name, None)
    os.execvpe(sys.executable, command, environment)


if __name__ == "__main__":
    main()
