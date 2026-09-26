#!/usr/bin/env bash
# From the repository root: source scripts/install_mac.sh

_vllm_jev_install_mac() {
  if [ "$(uname -s)" != Darwin ] || [ "$(uname -m)" != arm64 ]; then
    echo "This installer requires an Apple Silicon Mac." >&2
    return 1
  fi
  local macos_major
  macos_major=$(sw_vers -productVersion)
  macos_major=${macos_major%%.*}
  if [ "$macos_major" -lt 15 ]; then
    echo "This installer requires macOS 15 or newer." >&2
    return 1
  fi
  if [ ! -f pyproject.toml ] || [ ! -d vllm_jev ]; then
    echo "Run this command from the vllm-jev repository directory." >&2
    return 1
  fi
  if ! command -v uv >/dev/null 2>&1; then
    echo "Install uv first: https://docs.astral.sh/uv/" >&2
    return 1
  fi

  mkdir -p .local/wheels
  local log="$PWD/.local/install-mac.log"
  local venv="${VLLM_JEV_MAC_VENV:-$PWD/.venv}"
  local core="vllm-0.29.0+cpu-cp312-cp312-macosx_11_0_arm64.whl"
  local metal="vllm_metal-0.29.0-cp312-cp312-macosx_15_0_arm64.whl"
  local core_url="https://github.com/vllm-project/vllm/releases/download/v0.29.0/vllm-0.29.0%2Bcpu-cp312-cp312-macosx_11_0_arm64.whl"
  local metal_url="https://github.com/vllm-project/vllm-metal/releases/download/v0.29.0/$metal"
  echo "Preparing Python 3.12 and vLLM-Metal (details: $log)"
  uv python install 3.12 >> "$log" 2>&1 || { tail -30 "$log"; return 1; }
  if [ ! -x "$venv/bin/python" ]; then
    uv venv --python 3.12 "$venv" >> "$log" 2>&1 || { tail -30 "$log"; return 1; }
  fi
  if ! "$venv/bin/python" -c 'import platform, sys; assert sys.version_info[:2] == (3, 12) and platform.machine() == "arm64" and platform.python_implementation() == "CPython"' >/dev/null 2>&1; then
    echo "The existing environment at $venv must use arm64 CPython 3.12. Choose a new VLLM_JEV_MAC_VENV directory." >&2
    return 1
  fi
  . "$venv/bin/activate" || return 1

  if ! "$venv/bin/python" -c 'from importlib.metadata import version; assert version("vllm").startswith("0.29.0")' >/dev/null 2>&1; then
    if uv pip install --offline "$core_url" >> "$log" 2>&1; then
      :
    elif command -v gh >/dev/null 2>&1 &&
      gh release download v0.29.0 --repo vllm-project/vllm \
        --pattern "$core" --dir .local/wheels --skip-existing >> "$log" 2>&1 &&
      unzip -tq ".local/wheels/$core" >> "$log" 2>&1; then
      uv pip install ".local/wheels/$core" >> "$log" 2>&1 || { tail -30 "$log"; return 1; }
    else
      rm -f ".local/wheels/$core"
      uv pip install "$core_url" >> "$log" 2>&1 || { tail -30 "$log"; return 1; }
    fi
  fi
  if ! "$venv/bin/python" -c 'from importlib.metadata import version; assert version("vllm-metal") == "0.29.0"' >/dev/null 2>&1; then
    if uv pip install --offline "$metal_url" >> "$log" 2>&1; then
      :
    elif command -v gh >/dev/null 2>&1 &&
      gh release download v0.29.0 --repo vllm-project/vllm-metal \
        --pattern "$metal" --dir .local/wheels --skip-existing >> "$log" 2>&1 &&
      unzip -tq ".local/wheels/$metal" >> "$log" 2>&1; then
      uv pip install ".local/wheels/$metal" >> "$log" 2>&1 || { tail -30 "$log"; return 1; }
    else
      rm -f ".local/wheels/$metal"
      uv pip install "$metal_url" >> "$log" 2>&1 || { tail -30 "$log"; return 1; }
    fi
  fi
  if ! uv pip install --offline . >> "$log" 2>&1; then
    uv pip install . >> "$log" 2>&1 || { tail -30 "$log"; return 1; }
  fi
  echo "Ready. Run: vllm-jev serve IamBusy/OpenJev-0.6B"
  echo "For text and images: vllm-jev serve Valen-Team/Valen-Preview-0923"
}

_vllm_jev_install_mac
