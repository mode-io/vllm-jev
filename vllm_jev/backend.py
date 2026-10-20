"""Hardware selection shared by exporters, readout heads and launchers.

Selection does not initialize a device. Actual availability is checked only when
an exporter or readout asks for a device; native vLLM owns backbone placement.
"""

import importlib.util
import os
import sys
from pathlib import Path

BACKEND_ENV = "VLLM_JEV_BACKEND"
BACKENDS = ("auto", "cuda", "ascend", "cpu")


def resolve_backend(value=None, environment=None):
    env = os.environ if environment is None else environment
    selected = value or env.get(BACKEND_ENV, "auto")
    if selected not in BACKENDS:
        raise ValueError(f"{BACKEND_ENV} must be one of {BACKENDS}")
    if selected != "auto":
        if selected == "ascend" and sys.platform != "linux":
            raise ValueError("Ascend requires Linux and torch-npu")
        return selected
    if sys.platform == "linux" and Path("/dev/davinci_manager").exists():
        if importlib.util.find_spec("torch_npu") is not None:
            return "ascend"
    return "auto"


def readout_device(override=None):
    import torch

    backend = resolve_backend()
    requested = override or os.environ.get("VLLM_JEV_HEAD_DEVICE")
    if requested:
        if str(requested).split(":", 1)[0] == "npu":
            import torch_npu  # noqa: F401
        return torch.device(requested)
    if backend == "ascend":
        import torch_npu  # noqa: F401

        if not torch.npu.is_available():
            raise RuntimeError("Ascend selected but no usable NPU is visible")
        return torch.device("npu:0")
    if backend == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if backend == "cuda":
        raise RuntimeError("CUDA selected but no usable CUDA device is visible")
    return torch.device("cpu")


def export_device():
    # HF/PEFT merge is a preparation step, not NPU inference. CPU avoids invoking
    # CUDA-only HF GDN kernels or consuming the native engine's NPU memory.
    if resolve_backend() in ("ascend", "cpu") or sys.platform == "darwin":
        return "cpu"
    return str(readout_device())


def server_module(backend):
    if backend != "ascend":
        return "vllm.entrypoints.cli.main"
    import importlib.metadata

    try:
        native = importlib.metadata.version("vllm")
        plugin = importlib.metadata.version("vllm-ascend")
    except importlib.metadata.PackageNotFoundError as error:
        raise ValueError(
            "Install a matched vLLM/Ascend stack; run vllm-jev doctor"
        ) from error
    expected = {"0.22.1": "0.22.1rc1", "0.30.0": "0.30.0rc1"}
    if expected.get(native) != plugin:
        raise ValueError(
            f"Uncovered Ascend version pair: vLLM {native}, vllm-ascend {plugin}"
        )
    return (
        "vllm_jev.ascend_server" if native == "0.22.1" else "vllm.entrypoints.cli.main"
    )
