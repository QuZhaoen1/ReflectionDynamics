#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
: "${LEGACY_RUN:?LEGACY_RUN is required}"
python -m cavr.cli init --legacy-root "$LEGACY_RUN" --workspace "$WORKSPACE"
CUDA_VISIBLE_DEVICES="$GPU_IDS" python -m cavr.cli preflight --profile "$TRAIN_PROFILE" --output "$WORKSPACE/preflight.json"
