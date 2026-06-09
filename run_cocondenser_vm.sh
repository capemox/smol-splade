#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

export HF_HOME="${HF_HOME:-$ROOT/.hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export TOKENIZERS_PARALLELISM="false"
export PYTHONUNBUFFERED="1"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed; installing it in the current user account..."
  if ! command -v curl >/dev/null 2>&1; then
    echo "curl is required to install uv. Install curl and rerun this script." >&2
    exit 1
  fi
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

echo "Syncing Python dependencies..."
uv sync --no-dev

exec uv run python scripts/run_cocondenser_vm.py "$@"
