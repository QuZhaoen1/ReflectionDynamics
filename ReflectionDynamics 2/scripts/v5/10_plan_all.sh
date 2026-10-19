#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
for STAGE in frozen measurement crossed online; do
  python -m cavr.cli workflow --experiment "$WORKSPACE/experiment.json" --stage "$STAGE" --gpus "$GPU_IDS" --profile "$TRAIN_PROFILE"
done
