#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workspace="${VLLM_JEV_WORKSPACE:-$repo_root}"
runtime="${VLLM_JEV_RUNTIME:-$workspace/runtime}"
checkpoint="${VLLM_JEV_CHECKPOINT:-$workspace/checkpoint/qwen35_4b_jev_pooling_v3}"
port="${VLLM_JEV_PORT:-8795}"
kv_bytes="${VLLM_JEV_KV_BYTES:-auto}"
gpu_memory_utilization="${VLLM_JEV_GPU_MEMORY_UTILIZATION:-0.9}"

if [[ ! -f "$checkpoint/jev_manifest.json" ]]; then
  echo "Missing Jev checkpoint manifest: $checkpoint" >&2
  exit 1
fi
if [[ ! -x "$runtime/bin/vllm" ]]; then
  echo "Missing isolated vLLM runtime: $runtime" >&2
  exit 1
fi
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to an available GPU before serving}"

export VLLM_PLUGINS=vllm_jev_model,vllm_jev_endpoint
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HOME="${HF_HOME:-$workspace/.cache/hub}"

if [[ "${VLLM_JEV_VERIFY_QUICK:-0}" == 1 ]]; then
  "$runtime/bin/python" -m vllm_jev.checkpoint --model "$checkpoint" --quick
else
  "$runtime/bin/python" -m vllm_jev.checkpoint --model "$checkpoint"
fi

extra=()
kv_args=()
if [[ "$kv_bytes" == auto ]]; then
  kv_args+=(--gpu-memory-utilization "$gpu_memory_utilization")
elif [[ "$kv_bytes" =~ ^[0-9]+$ ]] && (( kv_bytes > 0 )); then
  kv_args+=(--kv-cache-memory-bytes "$kv_bytes")
else
  echo "VLLM_JEV_KV_BYTES must be a positive byte count or auto" >&2
  exit 1
fi
if [[ "${VLLM_JEV_EAGER:-0}" == 1 ]]; then
  extra+=(--enforce-eager)
fi
if [[ "${VLLM_JEV_ASYNC:-1}" == 1 ]]; then
  extra+=(--async-scheduling)
fi

exec "$runtime/bin/vllm" serve "$checkpoint" \
  --runner pooling --convert none --max-model-len 4096 \
  --enable-prefix-caching --mamba-cache-mode align \
  --mamba-ssm-cache-dtype float32 "${kv_args[@]}" \
  --host 127.0.0.1 --port "$port" \
  --served-model-name vllm-jev "${extra[@]}" "$@"
