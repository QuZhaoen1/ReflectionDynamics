#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
rd gate --train-states data/states/train.jsonl \
  --curation-report data/states/train.curation.json \
  --validation-states data/states/validation.jsonl \
  --validation-results data/evaluations/base/validation_0.jsonl data/evaluations/base/validation_1.jsonl \
  --model Qwen/Qwen2.5-32B --output data/selection/train_gate.json
