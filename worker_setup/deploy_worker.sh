#!/usr/bin/env bash
# Copy worker code to the Windows GPU box and (re)start it.  First time: also run setup.ps1 there.
set -euo pipefail
cd "$(dirname "$0")/.."
GPU_HOST="${GPU_HOST:-cisiwen@192.168.1.120}"
GPU_PORT="${GPU_PORT:-2200}"
SSH=(ssh -o BatchMode=yes -p "$GPU_PORT" "$GPU_HOST")
REMOTE=D:/dubbing-worker

"${SSH[@]}" "powershell -NoProfile -Command \"New-Item -ItemType Directory -Force D:\\dubbing-worker\\dubbing | Out-Null\""
scp -q -P "$GPU_PORT" worker_setup/*.ps1 "$GPU_HOST:/$REMOTE/"
scp -q -P "$GPU_PORT" dubbing/__init__.py dubbing/config.py dubbing/fit.py dubbing/engines.py dubbing/tts_batch.py dubbing/t3_graph.py dubbing/worker.py "$GPU_HOST:/$REMOTE/dubbing/"
if [[ "${1:-}" == "--setup" ]]; then
  "${SSH[@]}" "powershell -NoProfile -ExecutionPolicy Bypass -File D:\\dubbing-worker\\setup.ps1"
fi
"${SSH[@]}" "powershell -NoProfile -ExecutionPolicy Bypass -File D:\\dubbing-worker\\stop_worker.ps1"
# Launch detached from the SSH session so it keeps running after we disconnect.
"${SSH[@]}" "powershell -NoProfile -Command \"Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='powershell -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File D:\\dubbing-worker\\start_worker.ps1'} | Out-Null\""
echo "worker restarted on $GPU_HOST (localhost:8765 there; ./run_studio.sh tunnels to it)"
