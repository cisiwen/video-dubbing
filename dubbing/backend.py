"""Routes each heavy stage to the remote GPU worker (DUB_WORKER_URL) or runs it in-process.

DUB_LOCAL_STAGES lists stages to keep on this machine even when a worker is set,
e.g. DUB_LOCAL_STAGES=transcribe,detect_languages to use mlx-whisper on the Mac.
"""
import hashlib
import json
import os
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path

WORKER_URL = os.environ.get("DUB_WORKER_URL", "").rstrip("/")
WORKER_TOKEN = os.environ.get("DUB_WORKER_TOKEN")
LOCAL_STAGES = {s.strip() for s in os.environ.get("DUB_LOCAL_STAGES", "").split(",") if s.strip()}


def _remote(stage):
    return bool(WORKER_URL) and stage not in LOCAL_STAGES


def where(stage):
    return f"worker {WORKER_URL}" if _remote(stage) else "local"


# ---------------------------------------------------------------- http helpers
def _req(method, path, body=None, raw=None, timeout=7200):
    headers = {}
    if WORKER_TOKEN:
        headers["x-worker-token"] = WORKER_TOKEN
    data = raw
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(WORKER_URL + path, data=data, method=method, headers=headers)
    try:
        for attempt in range(20):
            try:
                return urllib.request.urlopen(req, timeout=timeout)
            except urllib.error.URLError as e:
                # Tunnel reconnecting or worker restarting: wait up to ~1 min before giving up.
                if isinstance(e, urllib.error.HTTPError) or attempt == 19:
                    raise
                time.sleep(3)
    except urllib.error.HTTPError as e:
        if e.code == 404 and method == "HEAD":
            return None
        detail = e.read().decode(errors="replace")
        raise RuntimeError(f"worker {path} failed ({e.code}): {detail}") from None


def _post(path, body):
    with _req("POST", path, body) as r:
        return json.loads(r.read())


def _upload(path):
    data = Path(path).read_bytes()
    fid = hashlib.sha256(data).hexdigest()
    if _req("HEAD", f"/files/{fid}") is None:
        _req("PUT", f"/files/{fid}", raw=data).close()
    return fid


def _download(fid, dest):
    with _req("GET", f"/files/{fid}") as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)


def health():
    if not WORKER_URL:
        return {"ok": True, "worker": None}
    with _req("GET", "/health", timeout=10) as r:
        return {**json.loads(r.read()), "worker": WORKER_URL}


# ---------------------------------------------------------------- stage API
def _engines():
    from . import engines
    return engines


def separate(audio, vocals_out, background_out):
    if not _remote("separate"):
        return _engines().separate(audio, vocals_out, background_out)
    res = _post("/separate", {"file": _upload(audio)})
    _download(res["vocals"], vocals_out)
    _download(res["background"], background_out)
    return res


def transcribe(audio, language=None):
    if not _remote("transcribe"):
        return _engines().transcribe(audio, language)
    return _post("/transcribe", {"file": _upload(audio), "language": language})


def detect_languages(audio, clips):
    if not _remote("detect_languages"):
        return _engines().detect_languages(audio, clips)
    return _post("/detect_languages", {"file": _upload(audio), "clips": clips})


def diarize(audio, num_speakers=None, min_speakers=None, max_speakers=None):
    if not _remote("diarize"):
        return _engines().diarize(audio, num_speakers, min_speakers, max_speakers)
    return _post("/diarize", {"file": _upload(audio), "num_speakers": num_speakers,
                              "min_speakers": min_speakers, "max_speakers": max_speakers})


def emotions(audio, clips):
    if not _remote("emotions"):
        return _engines().emotions(audio, clips)
    return _post("/emotions", {"file": _upload(audio), "clips": clips})


def translate(texts, src, tgt, context=None):
    if not texts:
        return []
    from .config import TRANSLATION_PROVIDER
    if not _remote("translate"):
        return _engines().translate(texts, src, tgt, context, TRANSLATION_PROVIDER)
    return _post("/translate", {"texts": texts, "src": src, "tgt": tgt, "context": context,
                                "provider": TRANSLATION_PROVIDER})


def suggest_names(texts, src, tgt, known=()):
    from .config import TRANSLATION_PROVIDER
    if not _remote("translate"):
        return _engines().suggest_names(texts, src, tgt, list(known), TRANSLATION_PROVIDER)
    return _post("/suggest_names", {"texts": texts, "src": src, "tgt": tgt, "known": list(known),
                                    "provider": TRANSLATION_PROVIDER})


def tts_batch(items):
    """items: [{text, language, ref (path), out (path), exaggeration, cfg_weight, max_tokens?, seed?}]
    -> writes each out; returns per-item info."""
    if not _remote("tts"):
        return _engines().tts_batch(items)
    keys = ("text", "language", "exaggeration", "cfg_weight")
    body = {"items": [{**{k: it[k] for k in keys}, "temperature": it.get("temperature", 0.8),
                       "max_tokens": it.get("max_tokens"), "seed": it.get("seed"), "group": it.get("group"),
                       "slot": it.get("slot"),
                       "ref": _upload(it["ref"])} for it in items]}
    res = _post("/tts_batch", body)
    for it, r in zip(items, res):
        if r.get("file"):
            _download(r["file"], it["out"])
    return res


def tts(text, language, ref, out_path, exaggeration=0.5, cfg_weight=0.5, temperature=0.8):
    if not _remote("tts"):
        return _engines().tts(text, language, ref, out_path, exaggeration, cfg_weight, temperature)
    res = _post("/tts", {"text": text, "language": language, "ref": _upload(ref),
                         "exaggeration": exaggeration, "cfg_weight": cfg_weight, "temperature": temperature})
    _download(res["file"], out_path)
    return res
