#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
SPLIT="${1:-validation}"
LIMIT="${2:-0}"
SAMPLES="${DRAFT_SAMPLES:-8}"
if [[ ! -s "data/corpus/${SPLIT}.jsonl" ]]; then
  echo "No questions for split ${SPLIT}" >&2
  exit 2
fi
mkdir -p data/drafts data/calibration data/states
OPTIONS=()
if [[ "${RESUME:-0}" == "1" ]]; then OPTIONS=(--resume); else OPTIONS=(--overwrite); fi
CUDA_VISIBLE_DEVICES=0,1,2,3 rd draft \
  --questions "data/corpus/${SPLIT}.jsonl" \
  --model Qwen/Qwen2.5-32B --output "data/drafts/${SPLIT}_0.jsonl" \
  --samples "$SAMPLES" --limit "$LIMIT" --max-new-tokens 1536 "${OPTIONS[@]}" \
  --shard-index 0 --num-shards 2 &
P0=$!
CUDA_VISIBLE_DEVICES=4,5,6,7 rd draft \
  --questions "data/corpus/${SPLIT}.jsonl" \
  --model Qwen/Qwen2.5-32B --output "data/drafts/${SPLIT}_1.jsonl" \
  --samples "$SAMPLES" --limit "$LIMIT" --max-new-tokens 1536 "${OPTIONS[@]}" \
  --shard-index 1 --num-shards 2 &
P1=$!
FAILED=0
wait "$P0" || FAILED=1
wait "$P1" || FAILED=1
if [[ "$FAILED" != 0 ]]; then
  echo "At least one drafting shard failed; no selection performed" >&2
  exit 1
fi
rd screen --drafts "data/drafts/${SPLIT}_0.jsonl" "data/drafts/${SPLIT}_1.jsonl" \
  --split "$SPLIT" --minimum-samples "$SAMPLES" --output-dir data/calibration
FILE="data/states/${SPLIT}.jsonl"
if [[ "$SPLIT" == "train" ]]; then FILE="data/states/train_candidates.jsonl"; fi
rd select --drafts "data/drafts/${SPLIT}_0.jsonl" "data/drafts/${SPLIT}_1.jsonl" \
  --split "$SPLIT" --minimum-samples "$SAMPLES" \
  --output-dir data/states --output-file "$FILE"
