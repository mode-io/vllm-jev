#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
profile="${1:-2b}"
case "$profile" in
  2b|9b) ;;
  *) echo "Usage: bash scripts/quickstart.sh [2b|9b]" >&2; exit 2 ;;
esac
: "${CUDA_VISIBLE_DEVICES:?Select a GPU with CUDA_VISIBLE_DEVICES=0}"
command -v uv >/dev/null || { echo "Install uv first: https://docs.astral.sh/uv/" >&2; exit 1; }

workspace="${VLLM_JEV_HOME:-$repo_root/.local}"
export HF_HOME="${HF_HOME:-$workspace/.cache}"

bash "$repo_root/scripts/install_runtime.sh" "$workspace"
"$workspace/runtime/bin/python" -m vllm_jev.public_models "$profile" \
  --workspace "$workspace"

export VLLM_JEV_WORKSPACE="$workspace"
exec bash "$repo_root/scripts/serve.sh" \
  "$workspace/checkpoint/open-jev-$profile"
