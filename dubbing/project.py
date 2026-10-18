"""Project storage: one folder per video with a project.json holding all reviewable state."""
import json
import threading
import time
import uuid
from pathlib import Path

from . import config

STAGES = ["extract", "separate", "transcribe", "diarize", "segment", "emotion", "translate",
          "synthesize", "mix"]

_locks: dict[str, threading.RLock] = {}


def lock(pid):
    return _locks.setdefault(pid, threading.RLock())


def pdir(pid) -> Path:
    return config.PROJECTS_DIR / pid


def new_id():
    return time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]


def load(pid) -> dict:
    with lock(pid):
        return json.loads((pdir(pid) / "project.json").read_text())


def save(p: dict):
    with lock(p["id"]):
        p["updated"] = time.time()
        path = pdir(p["id"]) / "project.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(p, ensure_ascii=False, indent=1))
        tmp.replace(path)


def update(pid, fn):
    """Atomically load, mutate with fn(project) and save. Returns fn's result."""
    with lock(pid):
        p = load(pid)
        result = fn(p)
        save(p)
        return result


def list_all():
    out = []
    for d in sorted(config.PROJECTS_DIR.iterdir(), reverse=True):
        f = d / "project.json"
        if f.exists():
            p = json.loads(f.read_text())
            out.append({k: p.get(k) for k in ("id", "name", "created", "status", "source_lang",
                                              "target_lang", "duration")}
                       | {"segments": len(p.get("segments", []))})
    return out


def create(name, video_ext, target_lang, settings, source_url=None):
    pid = new_id()
    d = pdir(pid)
    for sub in ("clips", "refs", "tts", "out"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    p = {"id": pid, "name": name, "created": time.time(), "source_video": f"source{video_ext}",
         "target_lang": target_lang, "source_lang": None, "duration": None,
         "settings": settings, "stages": {s: "pending" for s in STAGES},
         "status": {"state": "queued", "stage": None, "progress": 0, "message": "", "error": None},
         "speakers": {}, "segments": [], "log": []}
    if source_url:  # downloaded by the pipeline's extract stage
        p["source_url"] = source_url
    save(p)
    return p


def seg_index(p, sid):
    for i, s in enumerate(p["segments"]):
        if s["id"] == sid:
            return i
    raise KeyError(sid)


def next_seg_id(p):
    n = max([int(s["id"][1:]) for s in p["segments"]] + [0])
    return f"s{n + 1:05d}"


def mark(seg, *needs):
    seg["needs"] = sorted(set(seg.get("needs", [])) | set(needs))
