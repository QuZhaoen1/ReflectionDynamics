#!/usr/bin/env bash
set -euo pipefail
umask 077
VLLM_RPC_BASE_PATH=$(mktemp -d /tmp/rd-ipc.XXXXXXXX)
export VLLM_RPC_BASE_PATH
trap 'rmdir "$VLLM_RPC_BASE_PATH" 2>/dev/null || true' EXIT
${LEGACY_ROOT}/envs/vllm0141-test/bin/python - <<'PY'
import zmq
from vllm.utils.network_utils import get_open_zmq_ipc_path
with zmq.Context() as ctx:
    with ctx.socket(zmq.PAIR) as server, ctx.socket(zmq.PAIR) as client:
        path=get_open_zmq_ipc_path()
        server.bind(path)
        client.connect(path)
        client.send(b'ready')
        assert server.poll(3000) and server.recv()==b'ready'
        print('vLLM IPC verified:',path,flush=True)
PY
if [[ ${1:-} != --check-only ]]; then
    bash ${RUN_ROOT}/run.sh
fi
