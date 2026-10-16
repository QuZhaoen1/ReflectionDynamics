#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
SPLIT="${1:-validation}"
CHECKPOINT="${2:-base}"
ADAPTER="${3:-}"
LIMIT="${EVAL_LIMIT:-0}"
REPETITIONS="${EVAL_REPETITIONS:-4}"
mkdir -p "data/evaluations/${CHECKPOINT}"
if [[ "$CHECKPOINT" == "base" && -n "$ADAPTER" ]]; then
  echo "The base checkpoint cannot include an adapter" >&2
  exit 2
fi
EXTRA=()
if [[ -n "$ADAPTER" ]]; then EXTRA=(--adapter "$ADAPTER"); fi
MODE=(--overwrite)
if [[ "${RESUME:-0}" == "1" ]]; then MODE=(--resume); fi
CUDA_VISIBLE_DEVICES=0,1,2,3 rd evaluate \
  --states "data/states/${SPLIT}.jsonl" \
  --model Qwen/Qwen2.5-32B "${EXTRA[@]}" \
  --checkpoint "$CHECKPOINT" --repetitions "$REPETITIONS" --max-new-tokens 1024 \
  --output "data/evaluations/${CHECKPOINT}/${SPLIT}_0.jsonl" \
  --limit "$LIMIT" --retry-control "${MODE[@]}" --num-shards 2 --shard-index 0 &
P0=$!
CUDA_VISIBLE_DEVICES=4,5,6,7 rd evaluate \
  --states "data/states/${SPLIT}.jsonl" \
  --model Qwen/Qwen2.5-32B "${EXTRA[@]}" \
  --checkpoint "$CHECKPOINT" --repetitions "$REPETITIONS" --max-new-tokens 1024 \
  --output "data/evaluations/${CHECKPOINT}/${SPLIT}_1.jsonl" \
  --limit "$LIMIT" --retry-control "${MODE[@]}" --num-shards 2 --shard-index 1 &
P1=$!
FAILED=0
wait "$P0" || FAILED=1
wait "$P1" || FAILED=1
exit "$FAILED"
