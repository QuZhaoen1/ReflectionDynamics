#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
TARGET="${1:-600}"
rd curate --candidate-states data/states/train_candidates.jsonl \
  --probe-results data/evaluations/train_probe/train_0.jsonl \
                  data/evaluations/train_probe/train_1.jsonl \
  --target "$TARGET" --output data/states/train.jsonl
