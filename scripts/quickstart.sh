#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
case "${1:-ZefanCai/Open-Jev-2B}" in
  2b|ZefanCai/Open-Jev-2B) model_id=ZefanCai/Open-Jev-2B ;;
  9b|ZefanCai/Open-Jev-9B) model_id=ZefanCai/Open-Jev-9B ;;
  *) echo "Usage: bash scripts/quickstart.sh [ZefanCai/Open-Jev-2B|ZefanCai/Open-Jev-9B]" >&2; exit 2 ;;
esac
: "${CUDA_VISIBLE_DEVICES:?Select a GPU with CUDA_VISIBLE_DEVICES=0}"
command -v uv >/dev/null || { echo "Install uv first: https://docs.astral.sh/uv/" >&2; exit 1; }

workspace="${VLLM_JEV_HOME:-$repo_root/.local}"
export HF_HOME="${HF_HOME:-$workspace/.cache}"

bash "$repo_root/scripts/install_runtime.sh" "$workspace"
"$workspace/runtime/bin/python" -m vllm_jev.public_models "$model_id" \
  --workspace "$workspace"

export VLLM_JEV_WORKSPACE="$workspace"
export VLLM_JEV_MODEL_ID="$model_id"
checkpoint="$workspace/checkpoint/$model_id"
exec bash "$repo_root/scripts/serve.sh" \
  "$checkpoint"
