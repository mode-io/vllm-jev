"""Shared opt-in settings for Jev and ordinary generative model serving."""

import argparse
import os
import sys
from pathlib import Path

from .policy import POLICIES


def boolean(value):
    if value.lower() not in ("true", "false"):
        raise argparse.ArgumentTypeError("expected true or false")
    return value.lower() == "true"


def selected_policy(args):
    switch = getattr(args, "radixflow", None)
    policy = args.radixflow_policy
    if switch is False and policy not in (None, "off"):
        raise ValueError("--radixflow false conflicts with an enabled policy")
    if switch is True and policy == "off":
        raise ValueError("--radixflow true conflicts with --radixflow-policy off")
    return policy or ("radixflow" if switch else "off")


def arguments(parser):
    switch = parser.add_mutually_exclusive_group()
    switch.add_argument(
        "--radixflow",
        type=boolean,
        nargs="?",
        const=True,
        default=None,
        metavar="true|false",
        help="Enable model-independent RadixFlow scheduling",
    )
    switch.add_argument("--no-radixflow", dest="radixflow", action="store_false")
    parser.add_argument("--radixflow-policy", choices=("off", *POLICIES), default=None)
    parser.add_argument("--radixflow-library", type=Path)
    parser.add_argument("--radixflow-trace-dir", type=Path)
    parser.add_argument("--radixflow-wait-ms", type=float, default=5)


def configure(args, forwarded, environment):
    policy = selected_policy(args)
    from ..backend import resolve_backend

    if resolve_backend(environment=environment) == "ascend" and policy != "off":
        raise ValueError("RadixFlow NPU state-copy adapter is not yet validated")
    if policy == "off":
        for key in list(environment):
            if key.startswith("VLLM_JEV_RADIXFLOW_"):
                environment.pop(key)
        return []
    library = args.radixflow_library
    if library is None:
        configured = environment.get("VLLM_JEV_RADIXFLOW_LIBRARY")
        library = (
            Path(configured)
            if configured
            else (Path(__file__).resolve().parents[2] / "libjev_radixflow.so")
        )
    if not library.is_file():
        raise ValueError(
            "build the Rust index once; set --radixflow-library or "
            "VLLM_JEV_RADIXFLOW_LIBRARY to its path"
        )
    if not 0 <= args.radixflow_wait_ms <= 100:
        raise ValueError("--radixflow-wait-ms must be within [0,100]")
    if any(
        x.startswith("--scheduler-cls")
        or x == "--async-scheduling"
        or x.startswith("--async-scheduling=")
        or x == "--no-enable-prefix-caching"
        for x in forwarded
    ):
        raise ValueError(
            "RadixFlow conflicts with the requested scheduler/cache settings"
        )
    if environment.get("VLLM_JEV_ONLINE_PREFIX_CACHE") == "1":
        raise ValueError("disable online checkpoint creation when using RadixFlow")
    plugins = list(filter(None, environment.get("VLLM_PLUGINS", "").split(",")))
    environment["VLLM_PLUGINS"] = ",".join(dict.fromkeys([*plugins, "vllm_jev_model"]))
    environment["VLLM_JEV_RADIXFLOW_POLICY"] = policy
    environment["VLLM_JEV_RADIXFLOW_LIBRARY"] = str(library.resolve())
    environment["VLLM_JEV_RADIXFLOW_WAIT_MS"] = str(args.radixflow_wait_ms)
    if args.radixflow_trace_dir:
        environment["VLLM_JEV_RADIXFLOW_TRACE_DIR"] = str(
            args.radixflow_trace_dir.resolve()
        )
    return [
        "--scheduler-cls",
        "vllm_jev.radixflow.scheduler.RadixFlowScheduler",
        "--no-async-scheduling",
        "--enable-prefix-caching",
        "--enable-prompt-tokens-details",
    ]


def main():
    parser = argparse.ArgumentParser(prog="vllm-jev-radixflow", allow_abbrev=False)
    parser.add_argument("model", help="Ordinary generative model; no Jev export")
    arguments(parser)
    parser.add_argument("--backend", choices=("auto", "cuda", "ascend"), default=None)
    args, extra = parser.parse_known_args()
    environment = os.environ.copy()
    from ..backend import BACKEND_ENV, resolve_backend, server_module

    try:
        backend = resolve_backend(args.backend)
        server_module(backend)
        environment[BACKEND_ENV] = backend
        if backend == "ascend":
            plugins = list(filter(None, environment.get("VLLM_PLUGINS", "").split(",")))
            environment["VLLM_PLUGINS"] = ",".join(dict.fromkeys([*plugins, "ascend"]))
        flags = configure(args, extra, environment)
    except ValueError as e:
        parser.error(str(e))
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        args.model,
        *flags,
        *extra,
    ]
    os.execvpe(sys.executable, command, environment)


if __name__ == "__main__":
    main()
