#!/usr/bin/env bash
# Sets up the studio to run every model on this Mac (Apple silicon, Metal), with no GPU worker:
#   ./worker_setup/setup_mac.sh            install / update the models' Python packages in .venv here;
#                                          then ./run_studio.sh --local serves http://127.0.0.1:8000
#   ./worker_setup/setup_mac.sh --service  install (or update) a copy in ~/Library/Application Support/
#                                          DubbingStudio that starts at login and keeps running (launchd)
# --service uses its own folder because macOS doesn't let background services read ~/Documents. Its
# projects live there too (projects/). Run it again after changing the code to update and restart it.
# The Windows worker setup (setup.ps1, deploy_worker.sh) is unchanged; run_studio.sh without --local
# still uses it.
set -euo pipefail
cd "$(dirname "$0")/.."
[[ "$(uname -s)" == Darwin ]] || { echo "This is for macOS; on Windows use setup.ps1." >&2; exit 1; }
command -v uv >/dev/null || { echo "Needs uv: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2; exit 1; }
command -v ffmpeg >/dev/null || { echo "Needs ffmpeg: brew install ffmpeg" >&2; exit 1; }

dest="$PWD"
if [[ "${1:-}" == "--service" ]]; then
  dest="$HOME/Library/Application Support/DubbingStudio"
  mkdir -p "$dest/projects"
  rsync -a --delete --exclude __pycache__ dubbing static "$dest/"
  cp run_studio.sh requirements-studio.txt "$dest/"
fi

[[ -x "$dest/.venv/bin/python" ]] || uv venv --python 3.11 "$dest/.venv"
# Same model versions as the Windows worker (setup.ps1), minus the CUDA-only bitsandbytes (4-bit
# Qwen): on a Mac translation goes through Ollama. mlx-whisper runs Whisper on Apple's MLX.
overrides="$(mktemp)"
printf 'torch==2.8.0\ntorchaudio==2.8.0\nnumpy>=2.0\n' > "$overrides"
uv pip install -q --python "$dest/.venv/bin/python" --override "$overrides" -r requirements-studio.txt \
  torch==2.8.0 torchaudio==2.8.0 chatterbox-tts==0.1.7 "pyannote.audio>=4.0" torchcodec==0.7.0 demucs \
  funasr mlx-whisper sentencepiece "setuptools<81" accelerate
rm -f "$overrides"
echo "Packages ready in $dest/.venv. The models download on first use (about 10 GB)."
[[ -f "$HOME/.cache/huggingface/token" || -n "${HF_TOKEN:-}" ]] || {
  echo "Speaker detection uses pyannote's gated community-1 model: accept its terms at"
  echo "https://huggingface.co/pyannote/speaker-diarization-community-1, then sign in: $dest/.venv/bin/hf auth login"
}

if [[ "${1:-}" == "--service" ]]; then
  label=com.dubbing.studio
  plist="$HOME/Library/LaunchAgents/$label.plist"
  log="$HOME/Library/Logs/dubbing-studio.log"
  mkdir -p "$(dirname "$plist")" "$(dirname "$log")"
  cat > "$plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$label</string>
  <key>ProgramArguments</key><array><string>/bin/bash</string><string>$dest/run_studio.sh</string><string>--local</string></array>
  <key>WorkingDirectory</key><string>$dest</string>
  <key>EnvironmentVariables</key><dict>
    <key>PATH</key><string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string>
    <key>DUB_PROJECTS_DIR</key><string>$dest/projects</string>
  </dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ThrottleInterval</key><integer>10</integer>
  <key>StandardOutPath</key><string>$log</string>
  <key>StandardErrorPath</key><string>$log</string>
</dict></plist>
EOF
  launchctl bootout "gui/$(id -u)/$label" 2>/dev/null || true
  # bootout returns before the job is gone; bootstrap fails while it's still there.
  for _ in $(seq 20); do launchctl print "gui/$(id -u)/$label" >/dev/null 2>&1 || break; sleep 0.5; done
  launchctl bootstrap "gui/$(id -u)" "$plist"
  echo "Studio runs at login: http://127.0.0.1:8000 (log: $log, projects: $dest/projects)."
  echo "For https://homellm.eiuoo.com/dubbing/, add \"app\": [\"dubbing=http://127.0.0.1:8000\"] to HomeLLM's config.json."
fi
