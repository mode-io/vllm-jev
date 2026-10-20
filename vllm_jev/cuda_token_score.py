"""CUDA-only Tiny-Jev score kernel, imported lazily on CUDA tensors."""

import torch
import triton
import triton.language as tl


@triton.jit
def _token_score_kernel(
    hidden,
    weight,
    bias,
    scores,
    size: tl.constexpr,
    row_stride: tl.constexpr,
    col_stride: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, block)
    value = tl.load(
        hidden + row * row_stride + col * col_stride, col < size, other=0
    ).to(tl.float32)
    scale = tl.load(weight + col, col < size, other=0)
    mean = tl.sum(value, 0) / size
    centered = tl.where(col < size, value - mean, 0.0)
    variance = tl.sum(centered * centered, 0) / size
    score = tl.sum(centered * tl.rsqrt(variance + 1e-5) * scale, 0)
    tl.store(scores + row, score + tl.load(bias))


def project(hidden, weight, bias):
    rows, size = hidden.shape
    result = torch.empty((rows, 1), device=hidden.device, dtype=torch.float32)
    _token_score_kernel[(rows,)](
        hidden,
        weight,
        bias,
        result,
        size,
        hidden.stride(0),
        hidden.stride(1),
        triton.next_power_of_2(size),
        num_warps=4,
    )
    return result
