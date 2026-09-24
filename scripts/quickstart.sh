#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model_id="${1:-ZefanCai/Open-Jev-2B}"
if (($#)); then shift; fi
case "$model_id" in
  2b) model_id=ZefanCai/Open-Jev-2B ;;
  9b) model_id=ZefanCai/Open-Jev-9B ;;
esac
protocol=auto
if (($#)); then
  if [[ "$1" != --protocol || $# -ne 2 ]]; then
    echo "Usage: bash scripts/quickstart.sh [HuggingFace/model-id] [--protocol NAME]" >&2
    exit 2
  fi
  protocol="$2"
fi
: "${CUDA_VISIBLE_DEVICES:?Select a GPU with CUDA_VISIBLE_DEVICES=0}"
command -v uv >/dev/null || { echo "Install uv first: https://docs.astral.sh/uv/" >&2; exit 1; }

workspace="${VLLM_JEV_HOME:-$repo_root/.local}"
export HF_HOME="${HF_HOME:-$workspace/.cache}"

bash "$repo_root/scripts/install_runtime.sh" "$workspace"
"$workspace/runtime/bin/python" -m vllm_jev.native_models "$model_id" \
  --workspace "$workspace" --protocol "$protocol"

export VLLM_JEV_WORKSPACE="$workspace"
export VLLM_JEV_MODEL_ID="$model_id"
exec bash "$repo_root/scripts/serve.sh" "$workspace/checkpoint/$model_id"
