Set-Location D:\dubbing-worker
$env:DUB_WORKER_CACHE = "D:\dubbing-worker\cache"
$env:PYTHONUNBUFFERED = "1"
$env:DUB_LLM_CACHE = "D:\dubbing-worker\hf"
$env:HF_HUB_DISABLE_PROGRESS_BARS = "1"
$env:TRANSFORMERS_VERBOSITY = "error"
$env:PYTHONUTF8 = "1"
.venv\Scripts\python.exe -m uvicorn dubbing.worker:app --host 127.0.0.1 --port 8765 *> D:\dubbing-worker\worker.log
