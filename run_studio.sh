#!/usr/bin/env bash
# Start the dubbing studio on this Mac, using the GPU worker on the Windows box through an SSH tunnel.
#   ./run_studio.sh            -> http://localhost:8000
# Set DUB_WORKER_URL= (empty) to run every model locally instead.
set -euo pipefail
cd "$(dirname "$0")"

GPU_HOST="${GPU_HOST:-cisiwen@192.168.1.120}"
GPU_PORT="${GPU_PORT:-2200}"
export DUB_WORKER_URL="${DUB_WORKER_URL-http://127.0.0.1:8765}"

if [[ -n "$DUB_WORKER_URL" ]]; then
  pkill -f "ssh .*-L 8765:127.0.0.1:8765" 2>/dev/null || true
  # Keep the tunnel up across sleep / Wi-Fi drops: reconnect whenever ssh exits.
  (
    set +e  # ssh exits non-zero whenever the tunnel drops; errexit would end this loop for good
    while true; do
      ssh -o BatchMode=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
          -N -L 8765:127.0.0.1:8765 -p "$GPU_PORT" "$GPU_HOST" 2>/dev/null
      rc=$?
      echo "$(date '+%F %T') worker tunnel closed (ssh exit $rc); reconnecting"
      sleep 3
    done
  ) &
  TUNNEL_PID=$!
  trap 'kill $TUNNEL_PID 2>/dev/null; pkill -f "ssh .*-L 8765:127.0.0.1:8765"' EXIT
  for _ in 1 2 3 4 5; do curl -s --max-time 3 "$DUB_WORKER_URL/health" && break; sleep 2; done ||
    echo "Worker not answering yet. (Re)start it with ./worker_setup/deploy_worker.sh"
  echo
fi

.venv/bin/python -m uvicorn dubbing.server:app --host 127.0.0.1 --port "${PORT:-8000}"
