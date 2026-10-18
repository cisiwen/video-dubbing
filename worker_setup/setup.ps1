$ErrorActionPreference = "Continue"
New-Item -ItemType Directory -Force D:\dubbing-worker | Out-Null
Set-Location D:\dubbing-worker
"torch==2.8.0`ntorchaudio==2.8.0" | Set-Content -Encoding ascii overrides.txt
uv venv --python 3.11 .venv
uv pip install --python .venv\Scripts\python.exe --override overrides.txt --index https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match torch==2.8.0 torchaudio==2.8.0 chatterbox-tts==0.1.7 "pyannote.audio>=4.0" torchcodec==0.7.0 demucs faster-whisper funasr fastapi "uvicorn[standard]" python-multipart sentencepiece soundfile "setuptools<81" bitsandbytes accelerate
"EXIT=$LASTEXITCODE"
