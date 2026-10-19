#!/usr/bin/env bash
set -euo pipefail
source /personal_fast_storage/reflectiondynamics/activate.sh
ROOT=/personal_fast_storage/reflectiondynamics/runs/seed41-20261017-formal
cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/source/src"
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false NCCL_DEBUG=WARN
mkdir -p logs
exec 9>run.lock
flock -n 9 || { echo 'Run already active'; exit 1; }
nvidia-smi --query-gpu=timestamp,index,name,memory.used,utilization.gpu --format=csv -l 30 >> logs/gpu.csv &
MONITOR=$!
trap 'kill "$MONITOR" 2>/dev/null || true' EXIT
python -u pipeline.py 2>&1 | tee -a logs/pipeline.log
