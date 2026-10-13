"""Shared opt-in settings for Jev and ordinary generative model serving."""

import argparse
import os
import sys
from pathlib import Path

from .policy import POLICIES


def arguments(parser):
    parser.add_argument("--radixflow-policy", choices=("off", *POLICIES), default="off")
    parser.add_argument("--radixflow-library", type=Path)
    parser.add_argument("--radixflow-trace-dir", type=Path)
    parser.add_argument("--radixflow-wait-ms", type=float, default=5)


def configure(args, forwarded, environment):
    if args.radixflow_policy == "off":
        for key in list(environment):
            if key.startswith("VLLM_JEV_RADIXFLOW_"):
                environment.pop(key)
        return []
    if not args.radixflow_library or not args.radixflow_library.is_file():
        raise ValueError("--radixflow-library must point to the built Rust library")
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
    environment["VLLM_JEV_RADIXFLOW_POLICY"] = args.radixflow_policy
    environment["VLLM_JEV_RADIXFLOW_LIBRARY"] = str(args.radixflow_library.resolve())
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
    args, extra = parser.parse_known_args()
    environment = os.environ.copy()
    try:
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
