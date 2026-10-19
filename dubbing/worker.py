"""GPU worker: serves engines.py over HTTP. Run on the GPU box:

    python -m uvicorn dubbing.worker:app --host 0.0.0.0 --port 8765

Files are content-addressed (sha256) so the orchestrator uploads each input only once.
"""
import hashlib
import os
import tempfile
import traceback
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from . import engines

CACHE = Path(os.environ.get("DUB_WORKER_CACHE", Path(tempfile.gettempdir()) / "dub-worker"))
CACHE.mkdir(parents=True, exist_ok=True)
TOKEN = os.environ.get("DUB_WORKER_TOKEN")

app = FastAPI(title="dubbing-worker")


@app.middleware("http")
async def auth(request: Request, call_next):
    if TOKEN and request.headers.get("x-worker-token") != TOKEN:
        return JSONResponse({"detail": "bad token"}, status_code=401)
    return await call_next(request)


def _path(fid: str) -> Path:
    if not fid.isalnum():
        raise HTTPException(400, "bad file id")
    return CACHE / f"{fid}.wav"


def _need(fid: str) -> Path:
    p = _path(fid)
    if not p.exists():
        raise HTTPException(404, f"file {fid} not uploaded")
    return p


def _store(tmp: Path) -> str:
    fid = hashlib.sha256(tmp.read_bytes()).hexdigest()
    tmp.replace(_path(fid))
    return fid


def _new_tmp() -> Path:
    fd, name = tempfile.mkstemp(suffix=".wav", dir=CACHE)
    os.close(fd)
    return Path(name)


@app.exception_handler(Exception)
async def errors(_, exc):
    traceback.print_exc()
    return JSONResponse({"detail": f"{type(exc).__name__}: {exc}"}, status_code=500)


@app.get("/health")
def health():
    import torch
    return {"ok": True, "cuda": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


@app.head("/files/{fid}")
def has_file(fid: str):
    _need(fid)
    return {}


@app.put("/files/{fid}")
async def put_file(fid: str, request: Request):
    data = await request.body()
    if hashlib.sha256(data).hexdigest() != fid:
        raise HTTPException(400, "sha256 mismatch")
    _path(fid).write_bytes(data)
    return {"id": fid}


@app.get("/files/{fid}")
def get_file(fid: str):
    return FileResponse(_need(fid), media_type="audio/wav")


class FileReq(BaseModel):
    file: str
    language: str | None = None
    clips: list[list[float]] = []
    num_speakers: int | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None


@app.post("/separate")
def separate(r: FileReq):
    v, b = _new_tmp(), _new_tmp()
    info = engines.separate(_need(r.file), v, b)
    return {**info, "vocals": _store(v), "background": _store(b)}


@app.post("/transcribe")
def transcribe(r: FileReq):
    return engines.transcribe(_need(r.file), r.language)


@app.post("/detect_languages")
def detect_languages(r: FileReq):
    return engines.detect_languages(_need(r.file), r.clips)


@app.post("/diarize")
def diarize(r: FileReq):
    return engines.diarize(_need(r.file), r.num_speakers, r.min_speakers, r.max_speakers)


@app.post("/emotions")
def emotions(r: FileReq):
    return engines.emotions(_need(r.file), r.clips)


class TranslateReq(BaseModel):
    texts: list[str]
    src: str
    tgt: str
    context: list[dict] | None = None
    provider: str | None = None  # llm | nllb | ollama (default: the worker's config)


@app.post("/translate")
def translate(r: TranslateReq):
    return engines.translate(r.texts, r.src, r.tgt, r.context, r.provider)


class SuggestNamesReq(BaseModel):
    texts: list[str]
    src: str
    tgt: str
    known: list[str] = []
    provider: str | None = None


@app.post("/suggest_names")
def suggest_names(r: SuggestNamesReq):
    return engines.suggest_names(r.texts, r.src, r.tgt, r.known, r.provider)


class TTSReq(BaseModel):
    text: str
    language: str
    ref: str
    exaggeration: float = 0.5
    cfg_weight: float = 0.5
    temperature: float = 0.8
    max_tokens: int | None = None  # decode budget (only used by /tts_batch)
    seed: int | None = None  # per-line sampling seed (only used by /tts_batch)
    group: str | None = None  # best-of-N: takes of one line share a group ...
    slot: list[float] | None = None  # ... and its [seg_len, room]; only the best take is vocoded


class TTSBatchReq(BaseModel):
    items: list[TTSReq]


@app.post("/tts_batch")
def tts_batch(r: TTSBatchReq):
    outs = [_new_tmp() for _ in r.items]
    items = [{**it.model_dump(), "ref": _need(it.ref), "out": o} for it, o in zip(r.items, outs)]
    infos = engines.tts_batch(items)
    res = []
    for info, o in zip(infos, outs):
        if info.get("vocoded", True):
            res.append({**info, "file": _store(o)})
        else:
            o.unlink(missing_ok=True)
            res.append({**info, "file": None})
    return res


@app.post("/tts")
def tts(r: TTSReq):
    out = _new_tmp()
    info = engines.tts(r.text, r.language, _need(r.ref), out, r.exaggeration, r.cfg_weight, r.temperature)
    return {**info, "file": _store(out)}
