#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
python -m cavr.cli workflow --experiment "$WORKSPACE/experiment.json" --stage crossed --gpus "$GPU_IDS" --execute  "$@"
