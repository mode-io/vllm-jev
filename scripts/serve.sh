#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workspace="${VLLM_JEV_WORKSPACE:-${VLLM_JEV_HOME:-$repo_root/.local}}"
checkpoint="${1:?Usage: bash scripts/serve.sh /path/to/checkpoint [vllm args...]}"
shift
: "${CUDA_VISIBLE_DEVICES:?Select a GPU with CUDA_VISIBLE_DEVICES=0}"
model_id="${VLLM_JEV_MODEL_ID:-vllm-jev}"
port="${VLLM_JEV_PORT:-8795}"

"$workspace/runtime/bin/python" -m vllm_jev.checkpoint \
  --model "$checkpoint" --quick
export VLLM_PLUGINS="vllm_jev_model,vllm_jev_endpoint${VLLM_PLUGINS:+,$VLLM_PLUGINS}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
unset VLLM_JEV_WORKSPACE VLLM_JEV_HOME VLLM_JEV_MODEL_ID VLLM_JEV_PORT

exec "$workspace/runtime/bin/vllm" serve "$checkpoint" \
  --runner pooling --convert none --max-model-len 4096 \
  --enable-prefix-caching --mamba-cache-mode align \
  --mamba-ssm-cache-dtype float32 --async-scheduling \
  --gpu-memory-utilization 0.9 \
  --host 127.0.0.1 --port "$port" \
  --served-model-name "$model_id" "$@"
