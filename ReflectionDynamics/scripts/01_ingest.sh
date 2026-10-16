#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
rd ingest --output data/corpus --include-aime
