"""Prepare a Jev checkpoint and select the platform's serving backend."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    if (
        sys.platform not in ("darwin", "win32")
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
            else "Windows CPU serving supports pinned Laya, Tiny-Jev 0.6B/1.7B, OpenJev-0.6B, Open-Jev-2B, Decider-0.8B/2B and 4B Q4_K_M, JevK5-2B/Lite and 4B/9B Q5_K_M, Kev-0.8B/0.6B, This-That 1.0/1.1/1.2, Mica-v0.1-4B Q5_K_M, Decision-0.8B Q8_0/4B Q4_K_M, Bruv1-0.8B, and RSI v1.0-0.8B. Use the model's documented isolated Windows dependency profile and GGUF CPU runtime."
            if sys.platform == "win32"
            else "Additional arguments are forwarded to vllm serve."
        ),
    )
    serve.add_argument("model", help="Hugging Face model ID or local checkpoint.")
    serve.add_argument(
        "--decision-template",
        help="Decision-template JSON file, or native/instructions-first compatibility preset.",
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
            "jevany_pointer_v1",
            "kev_pointer_v1",
            "decider_slot_v1",
            "task_json_v1",
            "thisthat_slot_v1",
            "mica_labels_v1",
            *(
                (
                    "mica_gguf_v1",
                    "jevk5_lite_v1",
                    "jevk5_gguf_v1",
                    "decider_gguf_v1",
                    "decision_gguf_v1",
                    "bruv_direct_options_v1",
                    "rsi_option_xattn_v1",
                )
                if sys.platform == "win32"
                else ()
            ),
            "jevk5_letters_v1",
        ),
        default="auto",
        help="Checkpoint protocol (default: auto).",
    )
    from .radixflow.launch import arguments as radixflow_arguments

    radixflow_arguments(serve)
    arguments = sys.argv[1:]
    if sys.platform in ("darwin", "win32"):
        serve.add_argument("--host", default="127.0.0.1")
        serve.add_argument("--port", type=int, default=8795)
        arguments = [
            "--help" if value.startswith("--help=") else value for value in arguments
        ]
    if sys.platform == "win32":
        serve.add_argument(
            "--cpu-threads",
            type=int,
            default=min(4, os.cpu_count() or 1),
            help="CPU threads for Windows serving (default: up to 4).",
        )
    args, extra = parser.parse_known_args(arguments)
    from .radixflow.launch import selected_policy

    try:
        args.radixflow_policy = selected_policy(args)
    except ValueError as error:
        parser.error(str(error))
    from .decision_template import ENVIRONMENT, current_template, load_template

    try:
        template = (
            load_template(args.decision_template)
            if args.decision_template is not None
            else current_template()
        )
    except (ValueError, OSError) as error:
        parser.error(str(error))
    if sys.platform in ("darwin", "win32"):
        if args.radixflow_policy != "off":
            parser.error("RadixFlow requires the native Linux vLLM backend")
        if extra:
            parser.error(f"unrecognized arguments: {' '.join(extra)}")
        if not 0 <= args.port <= 65535:
            parser.error("port must be between 0 and 65535")
    workspace = args.workspace.resolve()
    if sys.platform == "win32":
        from .windows_dependencies import require_windows_dependencies
        from .windows_models import identify_source, prepare, protocol_for

        if not 1 <= args.cpu_threads <= (os.cpu_count() or 1):
            parser.error("cpu-threads must be between 1 and the logical CPU count")
        try:
            checkpoint = Path(args.model)
            if checkpoint.is_dir():
                checkpoint = checkpoint.resolve()
                model_id = identify_source(checkpoint)
            else:
                model_id = args.model
            protocol = protocol_for(model_id)
            if args.protocol not in ("auto", protocol):
                raise ValueError(f"{model_id} uses {protocol}, not {args.protocol}")
            template.validate_protocol(protocol)
            if protocol == "mica_gguf_v1":
                from .windows_mica import validate_template

                validate_template(template)
                if args.cpu_threads > 4:
                    raise ValueError("Mica CPU threads must be between 1 and 4")
            if protocol == "jevk5_lite_v1":
                from .windows_jevk5_lite import validate_template

                validate_template(template)
                if args.cpu_threads > 4:
                    raise ValueError("JevK5-Lite CPU threads must be between 1 and 4")
            if protocol == "jevk5_gguf_v1":
                from .windows_jevk5_gguf import validate_template

                validate_template(template)
                if args.cpu_threads > 4:
                    raise ValueError("JevK5 GGUF CPU threads must be between 1 and 4")
            if protocol == "decider_gguf_v1":
                from .windows_decider_gguf import validate_template, verify_runtime

                validate_template(template)
                if args.cpu_threads > 4:
                    raise ValueError("Decider GGUF CPU threads must be between 1 and 4")
                verify_runtime()
            if protocol in (
                "decision_gguf_v1",
                "bruv_direct_options_v1",
                "rsi_option_xattn_v1",
            ):
                from .windows_models import _adapter_model_ids, _additional_adapters

                adapter = next(
                    item
                    for item in _additional_adapters()
                    if model_id in _adapter_model_ids(item)
                )
                adapter.validate_template(template)
                if args.cpu_threads > 4:
                    raise ValueError("Windows CPU threads must be between 1 and 4")
                if protocol == "decision_gguf_v1":
                    adapter.verify_runtime()
            require_windows_dependencies(model_id)
            # Preparation can validate the current template too; publish the
            # explicit CLI override before that check, not only before serving.
            os.environ[ENVIRONMENT] = json.dumps(template.config, ensure_ascii=False)
            current_template.cache_clear()
            if not checkpoint.is_dir():
                checkpoint = prepare(model_id, workspace)
        except (ValueError, OSError) as error:
            parser.error(str(error))
        if protocol == "laya_markers_v1":
            from .windows_laya import serve as serve_windows
        else:
            from .windows_qwen import serve as serve_windows

        serve_windows(
            checkpoint,
            model_id,
            host=args.host,
            port=args.port,
            threads=args.cpu_threads,
        )
        return
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
        jevany_pointer_manifest = checkpoint / "jevany_pointer_manifest.json"
        if sys.platform == "darwin" and rsijev_manifest.is_file():
            parser.error("RSI-Jev serving requires Linux and native vLLM")
        if sys.platform == "darwin" and clef_manifest.is_file():
            parser.error("Clef serving requires Linux and native vLLM")
        if sys.platform == "darwin" and jevany_pointer_manifest.is_file():
            parser.error("JevAny Pointer serving requires Linux and native vLLM")
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
                        else jevany_pointer_manifest
                        if jevany_pointer_manifest.is_file()
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
                or jevany_pointer_manifest.is_file()
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
        if sys.platform == "darwin" and model_id.startswith("SimpleJev/JevAny-"):
            parser.error("JevAny Pointer serving requires Linux and native vLLM")
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
    jevany_pointer_manifest = checkpoint / "jevany_pointer_manifest.json"
    is_decision = decision_manifest.is_file()
    is_jevany_pointer = jevany_pointer_manifest.is_file()
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
            else jevany_pointer_manifest
            if is_jevany_pointer
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
    if is_jevany_pointer:
        jevany_pointer = json.loads(jevany_pointer_manifest.read_text())
        defaults[defaults.index("--max-model-len") + 1] = str(
            jevany_pointer["max_length"] + 1
        )
        defaults[defaults.index("--enable-prefix-caching")] = (
            "--no-enable-prefix-caching"
        )
        defaults.extend(
            [
                "--dtype",
                "bfloat16",
                "--hf-overrides",
                '{"head_dtype":"model"}',
                "--limit-mm-per-prompt",
                json.dumps({"image": jevany_pointer["max_images"], "video": 0}),
                "--mm-processor-kwargs",
                json.dumps(
                    {"size": {"shortest_edge": 32 * 32, "longest_edge": 512 * 512}}
                ),
            ]
        )
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
    if (
        is_valen
        or is_vjev
        or is_laya
        or is_decision
        or is_jevany_pointer
        or is_rsijev
        or is_clef
    ):
        defaults.extend(["--pooler-config", '{"task":"token_embed"}'])
    from .radixflow.launch import configure as configure_radixflow

    if args.radixflow_policy != "off":
        defaults = [x for x in defaults if x != "--async-scheduling"]
    try:
        radixflow_flags = configure_radixflow(args, [*defaults, *extra], environment)
    except ValueError as error:
        parser.error(str(error))
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
        *radixflow_flags,
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
