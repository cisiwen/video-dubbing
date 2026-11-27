#!/usr/bin/env bash
# Start the dubbing studio on this Mac -> http://localhost:8000
#   ./run_studio.sh            the models run on the GPU worker on the Windows box (SSH tunnel)
#   ./run_studio.sh --local    every model runs on this Mac (set up with worker_setup/setup_mac.sh)
set -euo pipefail
cd "$(dirname "$0")"

if [[ "${1:-}" == "--local" ]]; then
  export DUB_WORKER_URL=
  # Ops Metal lacks fall back to the CPU. The GPU shares main memory: cap Metal at the recommended
  # working set so a job too big fails with out-of-memory instead of swapping the Mac to a standstill
  # (the low watermark must not be above the high one).
  export PYTORCH_ENABLE_MPS_FALLBACK=1 PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.0 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.8
  export PYTHONUNBUFFERED=1
  exec .venv/bin/python -m uvicorn dubbing.server:app --host 127.0.0.1 --port "${PORT:-8000}"
fi

# The GPU box on the home LAN, else over Tailscale (when away from home). Set GPU_HOST to force one.
GPU_USER="${GPU_USER:-cisiwen}"
GPU_ADDRS="${GPU_ADDRS:-192.168.1.120 100.115.157.50}"
if [[ -z "${GPU_HOST:-}" ]]; then
  for a in $GPU_ADDRS; do
    if nc -z -G 3 "$a" "${GPU_PORT:-22}" 2>/dev/null; then GPU_HOST="$GPU_USER@$a"; break; fi
  done
  GPU_HOST="${GPU_HOST:-$GPU_USER@${GPU_ADDRS%% *}}"
  echo "GPU worker host: $GPU_HOST"
fi
GPU_PORT="${GPU_PORT:-22}"
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
