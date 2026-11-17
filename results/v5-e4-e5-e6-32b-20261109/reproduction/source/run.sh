#!/usr/bin/env bash
set -euo pipefail
umask 077
RUN_ROOT=${RUN_ROOT}
RUN_ENV=${LEGACY_ROOT}/envs/py311-torch291
export PATH="$RUN_ROOT/tools:$RUN_ENV/bin:$PATH"
export PYTHONPATH="$RUN_ROOT/code/src"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HF_HOME=${LEGACY_ROOT}/cache/huggingface
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TORCH_EXTENSIONS_DIR=${LEGACY_ROOT}/cache/torch_extensions
export TRITON_CACHE_DIR=${LEGACY_ROOT}/cache/triton
export LD_LIBRARY_PATH="/usr/local/cuda/compat:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TMPDIR="$RUN_ROOT/tmp" XDG_CACHE_HOME="$RUN_ROOT/cache"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
cd "$RUN_ROOT"
{
  date -Is
  grep -E 'CapEff|Seccomp|NoNewPrivs' /proc/self/status
} > "$RUN_ROOT/startup.log" 2>&1
exec 9>"$RUN_ROOT/task.lock"
flock -n 9 || { echo 'Another task holds the experiment lock'; exit 1; }
nvidia-smi --query-gpu=timestamp,index,name,memory.used,utilization.gpu --format=csv -l 10 >> "$RUN_ROOT/gpu.csv" &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID" 2>/dev/null || true' EXIT
python -u "$RUN_ROOT/pipeline.py" 2>&1 | tee -a "$RUN_ROOT/task.log"
