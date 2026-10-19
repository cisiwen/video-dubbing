Set-Location D:\dubbing-worker
$env:DUB_WORKER_CACHE = "D:\dubbing-worker\cache"
$env:PYTHONUNBUFFERED = "1"
$env:DUB_LLM_CACHE = "D:\dubbing-worker\hf"
$env:HF_HUB_DISABLE_PROGRESS_BARS = "1"
$env:TRANSFORMERS_VERBOSITY = "error"
$env:PYTHONUTF8 = "1"
$env:DUB_WORKER_CRASH_LOG = "D:\dubbing-worker\worker_crash.log"
Remove-Item D:\dubbing-worker\worker.log -ErrorAction SilentlyContinue
# Restart the worker if it dies (e.g. a native crash in torch); stop_worker.ps1 ends this loop too.
while ($true) {
    .venv\Scripts\python.exe -m uvicorn dubbing.worker:app --host 127.0.0.1 --port 8765 *>> D:\dubbing-worker\worker.log
    "$(Get-Date -Format s) worker exited with code $LASTEXITCODE; restarting in 5 s" *>> D:\dubbing-worker\worker.log
    Add-Content D:\dubbing-worker\worker_crash.log "$(Get-Date -Format s) worker exited with code $LASTEXITCODE"
    Start-Sleep 5
}
