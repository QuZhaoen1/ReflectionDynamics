#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
docker build -t reflection-code-runner:1 docker/
docker image inspect --format '{{.Id}}' reflection-code-runner:1
