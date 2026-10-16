#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
LIMIT="${1:-1000}"
REPETITIONS="${PROBE_REPETITIONS:-4}"
MODE=(--overwrite)
if [[ "${RESUME:-0}" == "1" ]]; then MODE=(--resume); fi
mkdir -p data/evaluations/train_probe
CUDA_VISIBLE_DEVICES=0,1,2,3 rd evaluate \
  --states data/states/train_candidates.jsonl --model Qwen/Qwen2.5-32B \
  --checkpoint base --repetitions "$REPETITIONS" --max-new-tokens 1024 \
  --output data/evaluations/train_probe/train_0.jsonl --limit "$LIMIT" \
  --retry-control "${MODE[@]}" --num-shards 2 --shard-index 0 &
P0=$!
CUDA_VISIBLE_DEVICES=4,5,6,7 rd evaluate \
  --states data/states/train_candidates.jsonl --model Qwen/Qwen2.5-32B \
  --checkpoint base --repetitions "$REPETITIONS" --max-new-tokens 1024 \
  --output data/evaluations/train_probe/train_1.jsonl --limit "$LIMIT" \
  --retry-control "${MODE[@]}" --num-shards 2 --shard-index 1 &
P1=$!
FAILED=0
wait "$P0" || FAILED=1
wait "$P1" || FAILED=1
exit "$FAILED"
