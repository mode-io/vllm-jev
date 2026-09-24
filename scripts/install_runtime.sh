#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workspace="${1:-$repo_root}"
export UV_CACHE_DIR="$workspace/.cache/uv"
export TMPDIR="$workspace/.tmp"
mkdir -p "$UV_CACHE_DIR" "$TMPDIR"
if [[ ! -x "$workspace/runtime/bin/python" ]]; then
  uv venv --python 3.12 "$workspace/runtime"
fi
uv pip install \
  --python "$workspace/runtime/bin/python" \
  --no-config --system-certs --index-url https://pypi.org/simple \
  vllm==0.29.0 transformers==5.10.4 peft==0.20.0
"$workspace/runtime/bin/python" - <<'PY'
import torch, transformers, vllm
print('RUNTIME_READY', vllm.__version__, torch.__version__, transformers.__version__)
PY
