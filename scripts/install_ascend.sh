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
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$TMPDIR" "$HF_HOME" "$TORCH_HOME" "$PIP_CACHE_DIR" "$VLLM_CACHE_ROOT"
"$python" - <<'PY'
import importlib.metadata as m
from vllm_jev.ascend_runtime import cann_version, runtime_issues
packages = {}
for name in ['vllm','vllm-ascend','torch','torch-npu','transformers','triton-ascend']:
    try: packages[name] = m.version(name)
    except m.PackageNotFoundError: packages[name] = None
issues = runtime_issues(packages, cann_version())
if issues:
    raise SystemExit('Runtime prerequisites failed before installation: ' + '; '.join(issues))
import torch, torch_npu
if not torch.npu.is_available():
    raise SystemExit('NPU not usable; load CANN and driver library paths first')
PY
"$python" -m pip install --no-deps -e "$repo_root"
"$python" -m vllm_jev.doctor --smoke
