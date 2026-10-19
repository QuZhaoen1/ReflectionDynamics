#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
python -m cavr.cli workflow --experiment "$WORKSPACE/experiment.json" --stage measurement --gpus "$GPU_IDS" --execute --checkpoints base step40 step80 step120 step160 step200 step240 step280 step320 step360 step400 "$@"
