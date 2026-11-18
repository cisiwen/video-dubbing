# Runs the studio (review UI + orchestration) on the GPU box, next to the worker, so it's
# reachable from anywhere through HomeLLM (homellm.eiuoo.com/dubbing/) without the Mac.
# Started at boot by the DubbingStudio scheduled task (deploy_studio.sh); restarts if it dies.
Set-Location D:\dubbing-studio
$env:PATH = "D:\dubbing-studio\ffmpeg\bin;$env:PATH"
$env:DUB_WORKER_URL = "http://127.0.0.1:8765"
$env:DUB_PROJECTS_DIR = "D:\dubbing-studio\projects"
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONUTF8 = "1"
while ($true) {
    .venv\Scripts\python.exe -m uvicorn dubbing.server:app --host 127.0.0.1 --port 8000 *>> D:\dubbing-studio\studio.log
    "$(Get-Date -Format s) studio exited with code $LASTEXITCODE; restarting in 5 s" *>> D:\dubbing-studio\studio.log
    Start-Sleep 5
}
