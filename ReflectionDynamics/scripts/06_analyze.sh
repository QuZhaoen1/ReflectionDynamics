#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "$#" -lt 2 ]]; then
  echo "Usage: bash scripts/06_analyze.sh output_dir eval1.jsonl eval2.jsonl ..." >&2
  exit 2
fi
OUTPUT="$1"
shift
rd analyze --input "$@" --output-dir "$OUTPUT" --bootstrap 1000
