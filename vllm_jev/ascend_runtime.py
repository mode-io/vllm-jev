"""Declarative upstream runtime rows, not NPU model acceptance claims."""

import os
from pathlib import Path

RUNTIMES = {
    "0.22.1": {
        "vllm-ascend": "0.22.1rc1",
        "torch": "2.10.0",
        "torch-npu": "2.10.0",
        "transformers": "5.5.4",
        "triton-ascend": "3.2.1",
        "cann": "9.0.0",
    },
    "0.30.0": {
        "vllm-ascend": "0.30.0rc1",
        "torch": "2.10.0",
        "torch-npu": "2.10.0.post4",
        "transformers": "5.14.1",
        "triton-ascend": "3.2.2",
        "cann": "9.2.0",
    },
}


def cann_version(environment=None):
    env = os.environ if environment is None else environment
    roots = [env.get("ASCEND_HOME_PATH"), env.get("ASCEND_TOOLKIT_HOME")]
    for root in filter(None, roots):
        for relative in ("opp/version.info", "version.info"):
            path = Path(root) / relative
            if path.is_file():
                for line in path.read_text().splitlines():
                    if line.startswith("Version="):
                        return line.split("=", 1)[1].strip()
    return None


def runtime_issues(packages, cann):
    issues = []
    for name in (
        "torch",
        "torch-npu",
        "vllm",
        "vllm-ascend",
        "transformers",
        "triton-ascend",
    ):
        if not packages.get(name):
            issues.append(f"{name} is not installed")
    native = (packages.get("vllm") or "").split("+", 1)[0]
    row = RUNTIMES.get(native)
    if row is None:
        if packages.get("vllm"):
            issues.append("Native vLLM version is not an integration target")
        return issues
    for name, expected in row.items():
        actual = cann if name == "cann" else packages.get(name)
        if name == "cann" and actual is None:
            issues.append("CANN version unavailable; source its set_env.sh first")
        elif actual and actual.split("+", 1)[0] != expected:
            issues.append(
                f"{name}={actual}; expected {expected} for vLLM {packages['vllm']}"
            )
    return issues
