#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
KIND="${1:-aime}"
python -m cavr.cli workflow --experiment "$WORKSPACE/experiment.json" --stage ood --kind "$KIND" --gpus "$GPU_IDS" --execute
