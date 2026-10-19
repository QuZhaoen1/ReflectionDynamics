#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
WORLD=$(python -c 'import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))["hardware"]["world_size"])' "$TRAIN_PROFILE")
for SEED in 41 73 107; do
  OUT="$WORKSPACE/runs/standard_v5_seed${SEED}"
  RESUME=()
  if compgen -G "$OUT/checkpoint-*" > /dev/null; then RESUME=(--resume auto); fi
  if [[ -f "$OUT/status.json" ]] && python -c 'import json,sys;sys.exit(0 if json.load(open(sys.argv[1])).get("success") else 1)' "$OUT/status.json"; then continue; fi
  CUDA_VISIBLE_DEVICES="$GPU_IDS" python -m torch.distributed.run --standalone --nproc_per_node="$WORLD" \
    -m cavr.train --config "$TRAIN_PROFILE" --states "$WORKSPACE/states/gate_train.jsonl" \
    --output "$OUT" --mode standard --seed "$SEED" "${RESUME[@]}"
  python -m cavr.cli export-adapter --checkpoint-path "$OUT/checkpoint-400" --output "$WORKSPACE/adapters/standard_seed${SEED}"
done
