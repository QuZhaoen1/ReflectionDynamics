#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
bash scripts/00_preflight.sh
SEED="${1:-41}"
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
torchrun --standalone --nproc_per_node=8 -m refldyn.train_grpo \
  --config configs/grpo_32b_a10040.yaml \
  --gate-report data/selection/train_gate.json \
  --seed-override "$SEED" \
  --output-dir-override "runs/grpo_32b_seed${SEED}"
