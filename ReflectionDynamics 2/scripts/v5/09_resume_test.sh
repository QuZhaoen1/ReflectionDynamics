#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
WORLD=$(python -c 'import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))["hardware"]["world_size"])' "$TRAIN_PROFILE")
OUT="$WORKSPACE/runs/resume_test"
if [[ -d "$OUT" ]]; then echo "Use a fresh resume_test directory for this integration test." >&2; exit 2; fi
CUDA_VISIBLE_DEVICES="$GPU_IDS" python -m torch.distributed.run --standalone --nproc_per_node="$WORLD" \
  -m cavr.train --config "$TRAIN_PROFILE" --states "$WORKSPACE/states/gate_train.jsonl" --output "$OUT" --pilot --stop-at-step 2
python -c 'import json,sys; s=json.load(open(sys.argv[1])); assert s["global_step"]==2 and s["interrupted"]' "$OUT/status.json"
CUDA_VISIBLE_DEVICES="$GPU_IDS" python -m torch.distributed.run --standalone --nproc_per_node="$WORLD" \
  -m cavr.train --config "$TRAIN_PROFILE" --states "$WORKSPACE/states/gate_train.jsonl" --output "$OUT" --pilot --resume auto
python -c 'import json,sys; s=json.load(open(sys.argv[1])); assert s["global_step"]==8 and s["success"]' "$OUT/status.json"
python -m cavr.cli export-adapter --checkpoint-path "$OUT/checkpoint-8" --output "$WORKSPACE/adapters/resume_test"
