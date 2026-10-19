#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"
ENV_FILE="${ENV_FILE:-configs/runtime.env}"
if [[ ! -f "$ENV_FILE" ]]; then
  echo "Copy configs/runtime.env.example to configs/runtime.env and set the persistent paths." >&2
  exit 2
fi
set -a
source "$ENV_FILE"
set +a
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
: "${WORKSPACE:?WORKSPACE is required}"
