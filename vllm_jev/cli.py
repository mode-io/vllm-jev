"""Prepare a Jev checkpoint and hand serving over to the native vLLM CLI."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    if (
        sys.platform != "darwin"
        and len(sys.argv) > 1
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
        help="Download, prepare, and serve a Jev model.",
        epilog=(
            "Apple Silicon serving uses MLX or PyTorch MPS."
            if sys.platform == "darwin"
            else "Additional arguments are forwarded to vllm serve."
        ),
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
            "laya_markers_v1",
            "valen_qwen_v1",
            "vjev_vision_v1",
        ),
        default="auto",
        help="Checkpoint protocol (default: auto).",
    )
    arguments = sys.argv[1:]
    if sys.platform == "darwin":
        serve.add_argument("--host", default="127.0.0.1")
        serve.add_argument("--port", type=int, default=8795)
        arguments = [
            "--help" if value.startswith("--help=") else value for value in arguments
        ]
    args, extra = parser.parse_known_args(arguments)
    if sys.platform == "darwin":
        if extra:
            parser.error(f"unrecognized arguments: {' '.join(extra)}")
        if not 0 <= args.port <= 65535:
            parser.error("port must be between 0 and 65535")
    workspace = args.workspace.resolve()
    environment = os.environ.copy()
    environment.setdefault("HF_HOME", str(workspace / ".cache"))
    environment.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    plugins = ["vllm_jev_model", "vllm_jev_endpoint"]
    plugins.extend(filter(None, environment.get("VLLM_PLUGINS", "").split(",")))
    environment["VLLM_PLUGINS"] = ",".join(dict.fromkeys(plugins))

    from .laya_export import MODELS as LAYA_MODELS
    from .public_models import PROFILES

    model_id = PROFILES[args.model][0] if args.model in PROFILES else args.model
    checkpoint = Path(model_id)
    if (
        sys.platform == "darwin"
        and not checkpoint.is_dir()
        and model_id
        not in (
            "IamBusy/OpenJev-0.6B",
            "ZefanCai/Open-Jev-2B",
            "lostargon/Tiny-Jev",
            "Valen-Team/Valen-Preview-0923",
            *LAYA_MODELS,
        )
    ):
        parser.error(
            "macOS supports Open-Jev-2B, OpenJev-0.6B, Tiny-Jev, Valen, and Laya"
        )
    if checkpoint.is_dir():
        checkpoint = checkpoint.resolve()
        valen_manifest = checkpoint / "valen_manifest.json"
        vjev_manifest = checkpoint / "vjev_manifest.json"
        laya_manifest = checkpoint / "laya_manifest.json"
        if sys.platform == "darwin" and (checkpoint / "rl_agent_config.json").is_file():
            from .mac_laya import identify_source

            model_id = identify_source(checkpoint)
            if args.protocol not in ("auto", "laya_markers_v1"):
                parser.error(f"{model_id} uses laya_markers_v1, not {args.protocol}")
            prepare = None
        else:
            if sys.platform == "darwin" and laya_manifest.is_file():
                parser.error(
                    "macOS Laya requires its original checkpoint, not the Linux export"
                )
            model_id = (
                json.loads(
                    (
                        vjev_manifest
                        if vjev_manifest.is_file()
                        else valen_manifest
                        if valen_manifest.is_file()
                        else laya_manifest
                    ).read_text()
                )["source_repository"]
                if vjev_manifest.is_file()
                or valen_manifest.is_file()
                or laya_manifest.is_file()
                else "vllm-jev"
            )
            prepare = [
                "-m",
                "vllm_jev.checkpoint",
                "--model",
                str(checkpoint),
                "--quick",
            ]
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
    if prepare is not None:
        try:
            subprocess.run([sys.executable, *prepare], env=environment, check=True)
        except subprocess.CalledProcessError as error:
            raise SystemExit(error.returncode) from None
    valen_manifest = checkpoint / "valen_manifest.json"
    vjev_manifest = checkpoint / "vjev_manifest.json"
    laya_manifest = checkpoint / "laya_manifest.json"
    is_valen = valen_manifest.is_file()
    is_vjev = vjev_manifest.is_file()
    is_laya = laya_manifest.is_file()
    if args.protocol != "auto" and not (
        sys.platform == "darwin" and model_id in LAYA_MODELS
    ):
        manifest_path = (
            vjev_manifest
            if is_vjev
            else valen_manifest
            if is_valen
            else laya_manifest
            if is_laya
            else checkpoint / "jev_manifest.json"
        )
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("prompt_protocol", "open_jev_choice") != args.protocol:
            parser.error("requested protocol differs from the prepared checkpoint")
    if sys.platform == "darwin":
        from .mac import serve as serve_mac

        serve_mac(checkpoint, model_id, host=args.host, port=args.port)
        return

    if is_laya:
        max_len = json.loads((checkpoint / "laya_config.json").read_text())["max_len"]
        defaults = (
            f"--runner pooling --convert none --max-model-len {max_len} --dtype float16 "
            "--gpu-memory-utilization 0.9 --host 127.0.0.1 --port 8795"
        ).split()
    else:
        defaults = (
            f"--runner pooling --convert none --max-model-len {8192 if is_valen else 4096} "
            "--enable-prefix-caching --mamba-cache-mode align "
            "--mamba-ssm-cache-dtype float32 --async-scheduling "
            "--gpu-memory-utilization 0.9 --host 127.0.0.1 --port 8795"
        ).split()
    if is_valen or is_vjev or is_laya:
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
