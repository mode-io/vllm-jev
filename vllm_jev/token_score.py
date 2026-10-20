"""Portable FP32 Tiny-Jev projection with a lazy CUDA optimization."""

import torch


def project_token_scores(hidden, weight, bias):
    if hidden.shape[0] < 256:
        return hidden
    if hidden.device.type == "cuda":
        from .cuda_token_score import project

        return project(hidden, weight, bias)
    value = hidden.float()
    centered = value - value.mean(-1, keepdim=True)
    variance = (centered * centered).mean(-1, keepdim=True)
    normalized = centered * torch.rsqrt(variance + 1e-5)
    return (normalized * weight.float()).sum(-1, keepdim=True) + bias.float()
