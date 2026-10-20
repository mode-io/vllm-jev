"""Read-only Ascend environment diagnostics; smoke is an explicit option."""

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path

PACKAGES = (
    "torch",
    "torch-npu",
    "vllm",
    "vllm-ascend",
    "transformers",
    "triton-ascend",
)


def version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def diagnose(smoke=False):
    packages = {name: version(name) for name in PACKAGES}
    result = {
        "system": platform.platform(),
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "packages": packages,
        "device_nodes": [str(p) for p in Path("/dev").glob("davinci*")],
        "environment": {
            k: os.environ.get(k)
            for k in (
                "VLLM_JEV_BACKEND",
                "ASCEND_HOME_PATH",
                "ASCEND_RT_VISIBLE_DEVICES",
                "ASCEND_VISIBLE_DEVICES",
                "HF_HOME",
                "TMPDIR",
            )
        },
        "issues": [],
        "validation": "diagnostics only; not model compatibility certification",
    }
    if shutil.which("npu-smi"):
        p = subprocess.run(
            ["npu-smi", "info"], capture_output=True, text=True, timeout=20
        )
        result["npu_smi"] = {
            "returncode": p.returncode,
            "output": (p.stdout + p.stderr)[-16000:],
        }
        if p.returncode:
            result["issues"].append(
                "npu-smi failed; load toolkit and driver library paths"
            )
    else:
        result["issues"].append("npu-smi not found")
    for name in ("torch-npu", "vllm", "vllm-ascend"):
        if not packages[name]:
            result["issues"].append(f"{name} is not installed")
    if packages["vllm"] and packages["vllm"] not in ("0.22.1", "0.30.0"):
        result["issues"].append(
            "Ascend integration targets vLLM 0.22.1 or 0.30.0; this version is not covered"
        )
    if smoke:
        import torch
        import torch_npu  # noqa: F401

        if not torch.npu.is_available():
            raise RuntimeError("No usable Ascend NPU")
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="npu:0")
        value = (x @ x).cpu()
        torch.testing.assert_close(value, torch.tensor([[7.0, 10.0], [15.0, 22.0]]))
        result["smoke"] = {"matmul": "passed", "devices": torch.npu.device_count()}
    result["prerequisites_present"] = not result["issues"]
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(prog="vllm-jev doctor")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(diagnose(args.smoke), indent=2))


if __name__ == "__main__":
    main()
