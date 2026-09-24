#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model_id="${1:-ZefanCai/Open-Jev-2B}"
if (($#)); then shift; fi
: "${CUDA_VISIBLE_DEVICES:?Select a GPU with CUDA_VISIBLE_DEVICES=0}"
command -v uv >/dev/null || { echo "Install uv first: https://docs.astral.sh/uv/" >&2; exit 1; }

workspace="${VLLM_JEV_HOME:-$repo_root/.local}"
bash "$repo_root/scripts/install_runtime.sh" "$workspace"
exec "$workspace/runtime/bin/vllm-jev" serve "$model_id" \
  --workspace "$workspace" --port "${VLLM_JEV_PORT:-8795}" "$@"
