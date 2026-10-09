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
        "--decision-template",
        help="native, instructions-first, or a decision-template JSON file.",
    )
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
            "rsijev_xattn_v1",
            "clef_joint_v1",
            "kev_pointer_v1",
            "decider_slot_v1",
            "task_json_v1",
            "thisthat_slot_v1",
            "mica_labels_v1",
            "jevk5_letters_v1",
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
    from .decision_template import ENVIRONMENT, current_template, load_template

    try:
        template = (
            load_template(args.decision_template)
            if args.decision_template is not None
            else current_template()
        )
    except (ValueError, OSError) as error:
        parser.error(str(error))
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
    if checkpoint.is_dir():
        checkpoint = checkpoint.resolve()
        decision_manifest = checkpoint / "decision_manifest.json"
        valen_manifest = checkpoint / "valen_manifest.json"
        vjev_manifest = checkpoint / "vjev_manifest.json"
        laya_manifest = checkpoint / "laya_manifest.json"
        rsijev_manifest = checkpoint / "rsijev_manifest.json"
        clef_manifest = checkpoint / "clef_manifest.json"
        if sys.platform == "darwin" and rsijev_manifest.is_file():
            parser.error("RSI-Jev serving requires Linux and native vLLM")
        if sys.platform == "darwin" and clef_manifest.is_file():
            parser.error("Clef serving requires Linux and native vLLM")
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
                        decision_manifest
                        if decision_manifest.is_file()
                        else rsijev_manifest
                        if rsijev_manifest.is_file()
                        else clef_manifest
                        if clef_manifest.is_file()
                        else vjev_manifest
                        if vjev_manifest.is_file()
                        else valen_manifest
                        if valen_manifest.is_file()
                        else laya_manifest
                    ).read_text()
                )["source_repository"]
                if decision_manifest.is_file()
                or rsijev_manifest.is_file()
                or clef_manifest.is_file()
                or vjev_manifest.is_file()
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
        if sys.platform == "darwin" and model_id.startswith("shgao/rsi-jev-"):
            parser.error("RSI-Jev serving requires Linux and native vLLM")
        if sys.platform == "darwin" and model_id.startswith("Cloudflare/clef"):
            parser.error("Clef serving requires Linux and native vLLM")
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
            prepare_environment = environment.copy()
            prepare_environment.pop(ENVIRONMENT, None)
            subprocess.run(
                [sys.executable, *prepare], env=prepare_environment, check=True
            )
        except subprocess.CalledProcessError as error:
            raise SystemExit(error.returncode) from None
    valen_manifest = checkpoint / "valen_manifest.json"
    vjev_manifest = checkpoint / "vjev_manifest.json"
    laya_manifest = checkpoint / "laya_manifest.json"
    decision_manifest = checkpoint / "decision_manifest.json"
    rsijev_manifest = checkpoint / "rsijev_manifest.json"
    clef_manifest = checkpoint / "clef_manifest.json"
    is_decision = decision_manifest.is_file()
    is_rsijev = rsijev_manifest.is_file()
    is_clef = clef_manifest.is_file()
    is_valen = valen_manifest.is_file()
    is_vjev = vjev_manifest.is_file()
    is_laya = laya_manifest.is_file()
    if args.protocol != "auto" and not (
        sys.platform == "darwin" and model_id in LAYA_MODELS
    ):
        manifest_path = (
            decision_manifest
            if is_decision
            else rsijev_manifest
            if is_rsijev
            else clef_manifest
            if is_clef
            else vjev_manifest
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
    if template.requires_open_jev:
        path = checkpoint / "jev_manifest.json"
        protocol = (
            json.loads(path.read_text()).get("prompt_protocol", "open_jev_choice")
            if path.is_file()
            else ""
        )
        try:
            template.validate_protocol(protocol)
        except ValueError as error:
            parser.error(str(error))
    environment[ENVIRONMENT] = json.dumps(
        template.config, ensure_ascii=False, separators=(",", ":")
    )
    if sys.platform == "darwin":
        os.environ[ENVIRONMENT] = environment[ENVIRONMENT]
        current_template.cache_clear()
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
    if is_decision:
        defaults[defaults.index("--convert") + 1] = "embed"
        defaults.extend(["--dtype", "bfloat16"])
    if is_rsijev:
        rsijev = json.loads(rsijev_manifest.read_text())
        # Leave the scheduler's final slot outside the model's input budget.
        defaults[defaults.index("--max-model-len") + 1] = str(rsijev["max_length"] + 1)
        # States come back in the tower's bf16, which halves the transfer and
        # changes no value; text-only releases skip the vision profile.
        defaults.extend(
            [
                "--dtype",
                "bfloat16",
                "--hf-overrides",
                '{"head_dtype":"model"}',
                "--limit-mm-per-prompt",
                json.dumps({"image": 4 if rsijev["vision"] else 0, "video": 0}),
            ]
        )
    if is_clef:
        clef = json.loads(clef_manifest.read_text())
        # vLLM schedules at most max_model_len - 1 prompt tokens, so a prompt
        # of exactly Clef's max_length would never finish.
        defaults[defaults.index("--max-model-len") + 1] = str(clef["max_length"] + 1)
        # The head reads every token's state, but a prefix-cache hit returns
        # states only for the tokens vLLM computed.
        defaults[defaults.index("--enable-prefix-caching")] = (
            "--no-enable-prefix-caching"
        )
        align = defaults.index("--mamba-cache-mode")
        del defaults[align : align + 2]
        defaults.extend(
            [
                "--dtype",
                "bfloat16",
                "--hf-overrides",
                '{"head_dtype":"model"}',
                "--limit-mm-per-prompt",
                json.dumps({"image": 8, "video": 1}),
            ]
        )
    if is_valen or is_vjev or is_laya or is_decision or is_rsijev or is_clef:
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
