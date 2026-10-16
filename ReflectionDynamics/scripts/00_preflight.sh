#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m refldyn.checks --config configs/grpo_32b_a10040.yaml --check-gpu
python - <<'PY'
import importlib, os, shutil
items = ('torch','trl','transformers','datasets','peft','deepspeed','math_verify')
for item in items:
    importlib.import_module(item)
print('Training modules import successfully')
memory = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / (1024**3)
cache_dir = os.environ.get('HF_HOME', '.')
os.makedirs(cache_dir, exist_ok=True)
free = shutil.disk_usage(cache_dir).free / (1024**3)
print(f'Host RAM: {memory:.0f} GiB; cache volume free: {free:.0f} GiB')
if memory < 160:
    raise SystemExit('Stop: less than 160 GiB RAM is risky with CPU parameter offloading')
if free < 150:
    raise SystemExit('Stop: less than 150 GiB free disk for 32B weights and experiments')
PY
nvidia-smi topo -m
