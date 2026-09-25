#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workspace="${1:-$repo_root/.local}"
export UV_CACHE_DIR="$workspace/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$workspace/.python"
export TMPDIR="$workspace/.tmp"
mkdir -p "$workspace" "$TMPDIR"

if [[ ! -x "$workspace/runtime/bin/python" ]]; then
  uv venv --python 3.12 "$workspace/runtime"
fi
uv pip install --python "$workspace/runtime/bin/python" \
  vllm==0.29.0 transformers==5.10.4 peft==0.20.0
uv pip install --python "$workspace/runtime/bin/python" \
  --no-deps -e "$repo_root"
