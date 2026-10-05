"""Opt-in BF16 GEMM settings applied before worker warmup and compilation."""

import os

MATMUL_ENV = "VLLM_JEV_BF16_MATMUL"
_configured_mode: str | None = None


def matmul_mode() -> str:
    mode = os.environ.get(MATMUL_ENV, "default")
    if mode not in ("default", "no_splitk"):
        raise ValueError(f"{MATMUL_ENV} must be 'default' or 'no_splitk', got {mode!r}")
    return mode


def configure_matmul() -> None:
    """Called by the general plugin in each process, including spawn workers.

    vLLM 0.29 hashes registered environment getters for its compile cache.
    Include this setting even in default mode: compiled graphs must not cross
    numerical modes. Registering the getter does not initialize CUDA.
    """
    global _configured_mode

    from vllm import envs

    mode = matmul_mode()
    if _configured_mode is not None and mode != _configured_mode:
        raise RuntimeError(f"Restart the server to change {MATMUL_ENV}.")
    if mode == "default":
        _configured_mode = mode
        envs.environment_variables[MATMUL_ENV] = lambda: mode
        return

    import torch

    matmul = torch.backends.cuda.matmul
    if torch.version.cuda is None or not hasattr(
        matmul, "allow_bf16_reduced_precision_reduction_split_k"
    ):
        raise RuntimeError(
            f"{MATMUL_ENV}=no_splitk requires a CUDA PyTorch build with "
            "BF16 split-K control (validated with PyTorch 2.13)."
        )
    # Disabling split-K requires cuBLASLt. Setting False alone still allows
    # split-K and leaves shape-dependent rounding in some BF16 projections.
    previous_backend = torch.backends.cuda.preferred_blas_library()
    previous_reduction = (
        matmul.allow_bf16_reduced_precision_reduction,
        matmul.allow_bf16_reduced_precision_reduction_split_k,
    )
    try:
        torch.backends.cuda.preferred_blas_library("cublaslt")
        matmul.allow_bf16_reduced_precision_reduction = (False, False)
    except (RuntimeError, TypeError) as error:
        torch.backends.cuda.preferred_blas_library(previous_backend)
        matmul.allow_bf16_reduced_precision_reduction = previous_reduction
        raise RuntimeError(f"Cannot enable {MATMUL_ENV}=no_splitk: {error}") from error
    _configured_mode = mode
    # Hash the applied mode, even if callers later modify their environment.
    envs.environment_variables[MATMUL_ENV] = lambda: mode
