#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
bash scripts/00_preflight.sh
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
torchrun --standalone --nproc_per_node=8 -m refldyn.train_grpo \
  --config configs/grpo_32b_a10040.yaml --pilot
