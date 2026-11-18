#!/usr/bin/env bash
# Copy the studio (server + UI) to the Windows GPU box and (re)start it there, next to the worker.
# HomeLLM then makes it reachable at https://homellm.eiuoo.com/dubbing/ (HomeLLM runs with
# -app dubbing=http://127.0.0.1:8000). First time: --setup (Python env, ffmpeg, boot tasks).
#   ./worker_setup/deploy_studio.sh [--setup]
set -euo pipefail
cd "$(dirname "$0")/.."
GPU_USER="${GPU_USER:-cisiwen}"
GPU_ADDRS="${GPU_ADDRS:-192.168.1.120 100.115.157.50}"
if [[ -z "${GPU_HOST:-}" ]]; then
  for a in $GPU_ADDRS; do
    if nc -z -G 3 "$a" "${GPU_PORT:-22}" 2>/dev/null; then GPU_HOST="$GPU_USER@$a"; break; fi
  done
  GPU_HOST="${GPU_HOST:-$GPU_USER@${GPU_ADDRS%% *}}"
fi
GPU_PORT="${GPU_PORT:-22}"
SSH=(ssh -o BatchMode=yes -p "$GPU_PORT" "$GPU_HOST")
REMOTE=D:/dubbing-studio
ps() { "${SSH[@]}" "powershell -NoProfile -ExecutionPolicy Bypass -Command \"$1\""; }

ps "New-Item -ItemType Directory -Force D:\\dubbing-studio\\dubbing, D:\\dubbing-studio\\static, D:\\dubbing-studio\\projects | Out-Null"
scp -q -P "$GPU_PORT" requirements-studio.txt worker_setup/start_studio.ps1 worker_setup/setup_studio.ps1 "$GPU_HOST:/$REMOTE/"
scp -q -P "$GPU_PORT" dubbing/*.py "$GPU_HOST:/$REMOTE/dubbing/"
scp -q -P "$GPU_PORT" static/* "$GPU_HOST:/$REMOTE/static/"
# The worker's code too, so both run the same version.
scp -q -P "$GPU_PORT" worker_setup/*.ps1 "$GPU_HOST:/D:/dubbing-worker/"
scp -q -P "$GPU_PORT" dubbing/__init__.py dubbing/config.py dubbing/fit.py dubbing/engines.py dubbing/tts_batch.py dubbing/t3_graph.py dubbing/worker.py "$GPU_HOST:/D:/dubbing-worker/dubbing/"
if [[ "${1:-}" == "--setup" ]]; then
  "${SSH[@]}" "powershell -NoProfile -ExecutionPolicy Bypass -File D:\\dubbing-studio\\setup_studio.ps1"
fi
# Restart both: stop the loops and their servers, then start the boot tasks again.
ps "Get-CimInstance Win32_Process | Where-Object { \$_.ProcessId -ne \$PID -and \$_.Name -ne 'cmd.exe' } | Where-Object { \$_.CommandLine -like '*dubbing.server*' -or \$_.CommandLine -like '*start_studio.ps1*' -or \$_.CommandLine -like '*dubbing.worker*' -or \$_.CommandLine -like '*start_worker.ps1*' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force -ErrorAction SilentlyContinue }"
ps "Start-ScheduledTask DubbingWorker; Start-ScheduledTask DubbingStudio"
echo "studio restarted on $GPU_HOST (127.0.0.1:8000 there; reach it at https://homellm.eiuoo.com/dubbing/)"
