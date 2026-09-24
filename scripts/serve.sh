#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workspace="${VLLM_JEV_WORKSPACE:-${VLLM_JEV_HOME:-$repo_root/.local}}"
checkpoint="${1:?Usage: bash scripts/serve.sh /path/to/checkpoint [vllm args...]}"
shift
: "${CUDA_VISIBLE_DEVICES:?Select a GPU with CUDA_VISIBLE_DEVICES=0}"
model_id="${VLLM_JEV_MODEL_ID:-vllm-jev}"
port="${VLLM_JEV_PORT:-8795}"

exec "$workspace/runtime/bin/vllm-jev" serve "$checkpoint" \
  --workspace "$workspace" --port "$port" \
  --served-model-name "$model_id" "$@"
