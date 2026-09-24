#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
profile=2b
if (($#)) && [[ "$1" != -* ]]; then
  profile="$1"
  shift
fi
case "$profile" in
  2b|9b) ;;
  *) echo "Use: bash scripts/quickstart.sh [2b|9b] [vllm serve args...]" >&2; exit 2 ;;
esac

if [[ -n "${VLLM_JEV_HOME:-}" ]]; then
  workspace="$VLLM_JEV_HOME"
elif [[ -x "$repo_root/runtime/bin/vllm" ]]; then
  workspace="$repo_root"
else
  workspace="$repo_root/.local"
fi
port="${VLLM_JEV_PORT:-8795}"
max_idle_mib=256
mkdir -p "$workspace"
if ! command -v nvidia-smi >/dev/null; then
  echo "A visible NVIDIA GPU and nvidia-smi are required" >&2
  exit 1
fi
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  CUDA_VISIBLE_DEVICES="$(nvidia-smi --query-gpu=index,memory.used \
    --format=csv,noheader,nounits | awk -F, -v pool="${JEV_GPU_POOL:-}" -v limit="$max_idle_mib" '
      BEGIN { allowed="," pool "," }
      $2+0 < limit && selected=="" {
        gpu=$1; gsub(/ /,"",gpu)
        if (pool=="" || index(allowed,"," gpu ",")>0) selected=gpu
      }
      END { print selected }')"
  if [[ -z "$CUDA_VISIBLE_DEVICES" ]]; then
    echo "No free GPU found; set CUDA_VISIBLE_DEVICES after freeing a card" >&2
    exit 1
  fi
  export CUDA_VISIBLE_DEVICES
fi
for gpu in ${CUDA_VISIBLE_DEVICES//,/ }; do
  used="$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
  if ((used >= max_idle_mib)); then
    echo "GPU $gpu already uses ${used} MiB; choose a free card" >&2
    exit 1
  fi
done
if python3 - "$port" <<'PY'
import socket,sys
with socket.socket() as sock:
    sys.exit(0 if sock.connect_ex(("127.0.0.1",int(sys.argv[1]))) == 0 else 1)
PY
then
  echo "Port $port is already in use" >&2
  exit 1
fi

if ! command -v uv >/dev/null; then
  python3 -m venv "$workspace/bootstrap"
  "$workspace/bootstrap/bin/python" -m pip install uv
  export PATH="$workspace/bootstrap/bin:$PATH"
fi
bash "$repo_root/scripts/install_runtime.sh" "$workspace"
uv pip install --python "$workspace/runtime/bin/python" \
  --no-deps --no-build-isolation -e "$repo_root"

export HF_HOME="${HF_HOME:-$workspace/.cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
mkdir -p "$HF_HUB_CACHE"

echo "Preparing public Open-Jev $profile on GPU $CUDA_VISIBLE_DEVICES" >&2
"$workspace/runtime/bin/python" -m vllm_jev.public_models "$profile" \
  --workspace "$workspace"

export VLLM_JEV_WORKSPACE="$workspace"
export VLLM_JEV_CHECKPOINT="$workspace/checkpoint/open-jev-$profile"
export VLLM_JEV_PORT="$port"
echo "Serving Open-Jev $profile at http://127.0.0.1:$port" >&2
exec bash "$repo_root/scripts/serve.sh" "$@"
