#!/usr/bin/env bash
# Install this plugin into an existing vendor-matched NPU runtime, without
# replacing torch/vLLM/CANN or changing the platform-managed driver.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workspace="${1:?usage: install_ascend.sh /mnt/workspace/jev [runtime-python]}"
python="${2:-python3}"
export TMPDIR="$workspace/tmp" XDG_CACHE_HOME="$workspace/cache"
export HF_HOME="$workspace/cache/hf" TORCH_HOME="$workspace/cache/torch"
export PIP_CACHE_DIR="$workspace/cache/pip" VLLM_CACHE_ROOT="$workspace/cache/vllm"
export VLLM_JEV_BACKEND=ascend PYTHONDONTWRITEBYTECODE=1
mkdir -p "$TMPDIR" "$HF_HOME" "$TORCH_HOME" "$PIP_CACHE_DIR" "$VLLM_CACHE_ROOT"
"$python" - <<'PY'
import importlib.metadata as m
pairs = {'0.22.1': '0.22.1rc1', '0.30.0': '0.30.0rc1'}
a, b = m.version('vllm'), m.version('vllm-ascend')
if pairs.get(a) != b:
    raise SystemExit(f'Uncovered runtime pair {a}/{b}; see docs/ascend.md')
import torch, torch_npu
if not torch.npu.is_available():
    raise SystemExit('NPU not usable; load CANN and driver library paths first')
PY
"$python" -m pip install --no-deps -e "$repo_root"
"$python" -m vllm_jev.doctor --smoke
