"""Review UI + API server.  Run:  python -m uvicorn dubbing.server:app --port 8000"""
import re
import shutil
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import backend, captions, config, glossary, pipeline
from . import project as P

app = FastAPI(title="Video dubbing studio")
STATIC = config.ROOT / "static"


@app.middleware("http")
async def no_stale_ui(request, call_next):
    resp = await call_next(request)
    if not request.url.path.startswith(("/api", "/media")):
        resp.headers["Cache-Control"] = "no-cache"
    return resp


def _get(pid):
    try:
        return P.load(pid)
    except FileNotFoundError:
        raise HTTPException(404, "project not found")


def _busy(p):
    if p["status"]["state"] in ("running", "queued"):
        raise HTTPException(409, "pipeline is running; wait for it to finish")


# ---------------------------------------------------------------- projects
@app.get("/api/meta")
def meta():
    try:
        worker = backend.health()
    except Exception as e:
        worker = {"ok": False, "error": str(e), "worker": backend.WORKER_URL}
    return {"languages": config.TTS_LANGUAGES, "emotions": config.EMOTIONS, "stages": P.STAGES,
            "worker": worker, "routing": {s: backend.where(s) for s in
                                          ["separate", "transcribe", "diarize", "emotions", "translate", "tts"]}}


@app.get("/api/projects")
def projects():
    return P.list_all()


@app.post("/api/projects")
async def create_project(video: UploadFile | None = File(None), url: str = Form(""),
                         target_lang: str = Form(...),
                         source_lang: str = Form(""), num_speakers: int = Form(0),
                         pause_after: str = Form("segment"), text_source: str = Form("auto"),
                         caption_file: UploadFile | None = File(None)):
    """Either upload `video`, or pass a `url` (YouTube etc.) that the extract stage downloads with yt-dlp."""
    if target_lang not in config.TTS_LANGUAGES:
        raise HTTPException(400, f"unsupported target language {target_lang}")
    url = url.strip()
    if video and video.filename:
        url = ""
    elif not url:
        raise HTTPException(400, "choose a video file or paste a video link")
    elif not url.lower().startswith(("http://", "https://")):
        raise HTTPException(400, "the video link must start with http:// or https://")
    if text_source not in ("auto", "captions", "asr"):
        raise HTTPException(400, "text_source must be auto, captions or asr")
    has_caption_file = bool(caption_file and caption_file.filename)
    if has_caption_file and Path(caption_file.filename).suffix.lower() not in captions.CAPTION_EXTS:
        raise HTTPException(400, "captions must be an .srt or .vtt file")
    settings = {"source_lang": source_lang or None, "num_speakers": num_speakers or None,
                "pause_after": None if pause_after in ("", "none") else pause_after, "text_source": text_source}
    if url:
        p = P.create(url, ".mp4", target_lang, settings, source_url=url)  # renamed to the title once known
    else:
        ext = Path(video.filename).suffix.lower() or ".mp4"
        p = P.create(Path(video.filename).stem, ext, target_lang, settings)
        with open(P.pdir(p["id"]) / p["source_video"], "wb") as f:
            shutil.copyfileobj(video.file, f, 1 << 22)
    if has_caption_file:
        p = _save_caption_file(p["id"], caption_file, source_lang or None)
    pipeline.enqueue(p["id"])
    return p


def _save_caption_file(pid, f: UploadFile, lang):
    """Store an uploaded .srt/.vtt as the spoken-language captions (lang None = whatever is spoken)."""
    d = P.pdir(pid) / "captions"
    d.mkdir(exist_ok=True)
    path = d / f"upload{len(list(d.glob('upload*')))}{Path(f.filename).suffix.lower()}"
    path.write_bytes(f.file.read())
    if not captions.parse(path):
        path.unlink()
        raise HTTPException(400, "no caption lines found in that file (expected SRT or WebVTT)")
    def g(p):
        p["captions"] = [t for t in p.get("captions", []) if not (t["origin"] == "upload" and t.get("lang") == lang)]
        p["captions"].insert(0, {"lang": lang, "file": f"captions/{path.name}", "origin": "upload"})
        return p
    return P.update(pid, g)


@app.post("/api/projects/{pid}/captions")
async def add_captions(pid: str, file: UploadFile = File(...), lang: str = Form("")):
    """Add a caption file to an existing project; re-run from "transcribe" to use it."""
    _busy(_get(pid))
    if Path(file.filename or "").suffix.lower() not in captions.CAPTION_EXTS:
        raise HTTPException(400, "captions must be an .srt or .vtt file")
    return _save_caption_file(pid, file, lang or None)


@app.get("/api/projects/{pid}")
def get_project(pid: str):
    return _get(pid)


@app.delete("/api/projects/{pid}")
def delete_project(pid: str):
    p = _get(pid)
    _busy(p)
    shutil.rmtree(P.pdir(pid))
    return {"ok": True}


class RunReq(BaseModel):
    from_stage: str = "emotion"
    full: bool = False


@app.post("/api/projects/{pid}/run")
def run(pid: str, r: RunReq):
    p = _get(pid)
    _busy(p)
    if r.from_stage not in P.STAGES:
        raise HTTPException(400, "unknown stage")
    pipeline.enqueue(pid, r.from_stage, r.full)
    return {"ok": True}


class ProjectPatch(BaseModel):
    target_lang: str | None = None


@app.patch("/api/projects/{pid}")
def patch_project(pid: str, r: ProjectPatch):
    def f(p):
        _busy(p)
        if r.target_lang and r.target_lang != p["target_lang"]:
            if r.target_lang not in config.TTS_LANGUAGES:
                raise HTTPException(400, "unsupported language")
            p["target_lang"] = r.target_lang
            for s in p["segments"]:
                s["translation_locked"] = False
                P.mark(s, "translate", "tts")
        return p
    return P.update(pid, f)


# ---------------------------------------------------------------- name glossary
class GlossaryEntry(BaseModel):
    source: str
    target: str = ""
    aliases: list[str] = []
    status: str = "approved"  # "suggested" entries are not used until a reviewer approves them
    example: str = ""


class GlossaryPut(BaseModel):
    entries: list[GlossaryEntry]


@app.put("/api/projects/{pid}/glossary")
def put_glossary(pid: str, r: GlossaryPut):
    def f(p):
        _busy(p)
        old = glossary.pairs(p)
        entries, seen = [], set()
        for e in r.entries:
            e.source = " ".join(e.source.split())
            if not e.source or e.source.lower() in seen:
                continue
            seen.add(e.source.lower())
            e.aliases = [a for a in (" ".join(x.split()) for x in e.aliases) if a]
            e.status = "approved" if e.status == "approved" else "suggested"
            entries.append(e.model_dump())
        p["glossary"] = entries
        glossary.apply_change(p, old)
        return p
    return P.update(pid, f)


@app.post("/api/projects/{pid}/glossary/suggest")
def suggest_glossary(pid: str):
    """One LLM pass that drafts entries for names not in the glossary yet; they stay "suggested"."""
    p = _get(pid)
    _busy(p)
    if p["source_lang"] == p["target_lang"]:
        raise HTTPException(400, "source and target language are the same")
    known = [x for e in p.get("glossary", []) for x in [e["source"], *e.get("aliases", [])]]
    found = backend.suggest_names([s["text"] for s in p["segments"] if s["text"].strip()],
                                  p["source_lang"], p["target_lang"], known)

    def f(p):
        have = {x.lower() for e in p.setdefault("glossary", []) for x in [e["source"], *e.get("aliases", [])]}
        added = [{"source": e["source"], "target": e["target"], "aliases": [], "status": "suggested",
                  "example": e.get("example", "")} for e in found if e["source"].lower() not in have]
        p["glossary"].extend(added)
        return {"added": len(added), "glossary": p["glossary"]}
    return P.update(pid, f)


# ---------------------------------------------------------------- segments
class SegPatch(BaseModel):
    text: str | None = None
    translation: str | None = None
    speaker: str | None = None
    emotion: str | None = None
    lang: str | None = None
    start: float | None = None
    end: float | None = None
    reviewed: bool | None = None
    exaggeration: float | None = None
    cfg_weight: float | None = None


@app.patch("/api/projects/{pid}/segments/{sid}")
def patch_segment(pid: str, sid: str, r: SegPatch):
    def f(p):
        _busy(p)
        s = p["segments"][P.seg_index(p, sid)]
        ch = r.model_dump(exclude_unset=True)
        if "text" in ch and ch["text"] != s["text"]:
            s["text"] = ch["text"]
            if not s.get("translation_locked"):
                P.mark(s, "translate")
            P.mark(s, "tts")
        if "lang" in ch and ch["lang"] != s["lang"]:
            s["lang"] = ch["lang"]
            if not s.get("translation_locked"):
                P.mark(s, "translate")
        if "translation" in ch and ch["translation"] != s["translation"]:
            s["translation"] = ch["translation"]
            s["translation_locked"] = True
            s["needs"] = [n for n in s.get("needs", []) if n != "translate"]
            P.mark(s, "tts")
        if "speaker" in ch and ch["speaker"] != s["speaker"]:
            if ch["speaker"] not in p["speakers"]:
                raise HTTPException(400, "unknown speaker")
            s["speaker"] = ch["speaker"]
            P.mark(s, "tts")
        if "emotion" in ch and ch["emotion"] != s["emotion"]:
            s["emotion"] = ch["emotion"]
            s["emotion_locked"] = True
            s["needs"] = [n for n in s.get("needs", []) if n != "emotion"]
            P.mark(s, "tts")
        for k in ("start", "end"):
            if k in ch and ch[k] is not None and abs(ch[k] - s[k]) > 1e-3:
                s[k] = round(ch[k], 3)
                P.mark(s, "tts")
        if s["end"] <= s["start"]:
            raise HTTPException(400, "end must be after start")
        for k in ("exaggeration", "cfg_weight"):
            if k in ch:
                s[k] = ch[k]
                P.mark(s, "tts")
        if "reviewed" in ch:
            s["reviewed"] = ch["reviewed"]
        p["segments"].sort(key=lambda x: x["start"])
        return s
    return P.update(pid, f)


class SplitReq(BaseModel):
    at: float
    text_split: int | None = None  # character index into text where the second half starts


@app.post("/api/projects/{pid}/segments/{sid}/split")
def split_segment(pid: str, sid: str, r: SplitReq):
    def f(p):
        _busy(p)
        i = P.seg_index(p, sid)
        s = p["segments"][i]
        if not s["start"] < r.at < s["end"]:
            raise HTTPException(400, "split point must be inside the segment")
        frac = (r.at - s["start"]) / (s["end"] - s["start"])
        cut = r.text_split if r.text_split is not None else _word_cut(s["text"], frac)
        b = {**s, "id": P.next_seg_id(p), "start": round(r.at, 3), "text": s["text"][cut:].strip(),
             "translation": "", "tts": None, "translation_locked": False, "reviewed": False}
        s.update(end=round(r.at, 3), text=s["text"][:cut].strip(), translation="", tts=None,
                 translation_locked=False, reviewed=False)
        for x in (s, b):
            x["needs"] = ["emotion", "translate", "tts"]
        p["segments"].insert(i + 1, b)
        return [s, b]
    return P.update(pid, f)


def _word_cut(text, frac):
    target = int(len(text) * frac)
    spaces = [i for i, c in enumerate(text) if c == " "]
    return min(spaces, key=lambda i: abs(i - target)) if spaces else target


_NO_SPACE = re.compile(r"[\u3000-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff00-\uffef]")


def _join(parts):
    """Join line texts the way the language writes them: no space between Chinese/Japanese text
    (including full-width punctuation), one space otherwise."""
    out = ""
    for t in (x.strip() for x in parts):
        if not t:
            continue
        cjk_edge = out and _NO_SPACE.match(out[-1]) and _NO_SPACE.match(t[0])
        out += t if not out or cjk_edge else " " + t
    return out


@app.get("/api/projects/{pid}/segments/{sid}/captions")
def segment_captions(pid: str, sid: str):
    """Caption cues around one line from every caption track, offset-corrected, with the text each can
    supply as this line's transcript or translation (the review UI's "find in caption")."""
    p = _get(pid)
    try:
        s = p["segments"][P.seg_index(p, sid)]
    except KeyError:
        raise HTTPException(404, "line not found")
    spoken = s["lang"] or p["source_lang"]
    out = []
    for t in p.get("captions") or []:
        path = P.pdir(pid) / t["file"]
        if path.exists():
            out.append(captions.cue_matches(t, captions.load(path), s, spoken, p["target_lang"], p["segments"]))
    return {"spoken": spoken, "target": p["target_lang"], "tracks": out}


class MergeReq(BaseModel):
    ids: list[str]


@app.post("/api/projects/{pid}/segments/merge")
def merge_segments(pid: str, r: MergeReq):
    def f(p):
        _busy(p)
        segs = sorted((p["segments"][P.seg_index(p, i)] for i in r.ids), key=lambda s: s["start"])
        if len(segs) < 2:
            raise HTTPException(400, "pick at least two segments")
        a = segs[0]
        # The merged line belongs to whoever speaks most of it, not to whichever part came first.
        talk = {}
        for s in segs:
            talk[s["speaker"]] = talk.get(s["speaker"], 0) + s["end"] - s["start"]
        speaker = max(talk, key=lambda k: (talk[k], k == a["speaker"]))
        a.update(end=max(s["end"] for s in segs), speaker=speaker, text=_join([s["text"] for s in segs]),
                 translation=_join([s["translation"] for s in segs if s["translation"]]),
                 tts=None, reviewed=False, translation_locked=False, needs=["emotion", "translate", "tts"])
        drop = {s["id"] for s in segs[1:]}
        p["segments"] = [s for s in p["segments"] if s["id"] not in drop]
        return a
    return P.update(pid, f)


@app.delete("/api/projects/{pid}/segments/{sid}")
def delete_segment(pid: str, sid: str):
    def f(p):
        _busy(p)
        p["segments"].pop(P.seg_index(p, sid))
        return {"ok": True}
    return P.update(pid, f)


@app.post("/api/projects/{pid}/segments/{sid}/retranslate")
def retranslate(pid: str, sid: str):
    p = _get(pid)
    _busy(p)
    s = p["segments"][P.seg_index(p, sid)]
    src = s["lang"] or p["source_lang"]
    out = s["text"] if src == p["target_lang"] else backend.translate(
        [s["text"]], src, p["target_lang"], [pipeline.translation_context(p, s)])[0]

    def f(p):
        seg = p["segments"][P.seg_index(p, sid)]
        seg.update(translation=out, translation_locked=False)
        seg["needs"] = [n for n in seg.get("needs", []) if n != "translate"]
        P.mark(seg, "tts")
        return seg
    return P.update(pid, f)


@app.post("/api/projects/{pid}/segments/{sid}/synthesize")
def synthesize(pid: str, sid: str):
    p = _get(pid)
    _busy(p)
    pipeline.synth_one(pid, sid)
    return _get(pid)["segments"][P.seg_index(_get(pid), sid)]


# ---------------------------------------------------------------- speakers
class SpeakerPatch(BaseModel):
    name: str | None = None
    ref_segments: list[str] | None = None
    ref_mode: str | None = None  # "segment" (clone each line's own delivery) | "speaker"
    exaggeration: float | None = None
    cfg_weight: float | None = None


@app.patch("/api/projects/{pid}/speakers/{spk}")
def patch_speaker(pid: str, spk: str, r: SpeakerPatch):
    ch = r.model_dump(exclude_unset=True)

    def f(p):
        _busy(p)
        info = p["speakers"][spk]
        info.update(ch)
        if set(ch) - {"name"}:
            for s in p["segments"]:
                if s["speaker"] == spk:
                    P.mark(s, "tts")
        return info
    res = P.update(pid, f)
    if "ref_segments" in ch:
        pipeline.build_refs(pid, [spk])
    return res


@app.post("/api/projects/{pid}/speakers")
def add_speaker(pid: str):
    def f(p):
        n = len(p["speakers"])
        spk = f"SPEAKER_{n:02d}"
        while spk in p["speakers"]:
            n += 1
            spk = f"SPEAKER_{n:02d}"
        p["speakers"][spk] = {"id": spk, "name": f"Speaker {len(p['speakers']) + 1}",
                              "color": config.PALETTE[n % len(config.PALETTE)], "ref_segments": [],
                              "ref_mode": "segment", "exaggeration": None, "cfg_weight": None}
        return p["speakers"][spk]
    return P.update(pid, f)


class SpeakerMerge(BaseModel):
    source: str
    into: str


@app.post("/api/projects/{pid}/speakers/merge")
def merge_speakers(pid: str, r: SpeakerMerge):
    def f(p):
        _busy(p)
        if r.source == r.into or r.into not in p["speakers"]:
            raise HTTPException(400, "bad merge")
        for s in p["segments"]:
            if s["speaker"] == r.source:
                s["speaker"] = r.into
                P.mark(s, "tts")
        p["speakers"].pop(r.source, None)
        p["speakers"][r.into]["ref_segments"] = []
        return p["speakers"]
    res = P.update(pid, f)
    pipeline.build_refs(pid, [r.into])
    return res


# ---------------------------------------------------------------- media & exports
@app.get("/media/{pid}/{path:path}")
def media(pid: str, path: str):
    base = P.pdir(pid).resolve()
    f = (base / path).resolve()
    if base not in f.parents or not f.is_file():
        raise HTTPException(404)
    return FileResponse(f)


@app.get("/api/projects/{pid}/peaks")
def peaks(pid: str, per_sec: int = 50):
    """Downsampled dialogue waveform for the timeline (cached)."""
    import json

    import numpy as np
    import soundfile as sf
    d = P.pdir(pid)
    cache = d / f"peaks{per_sec}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    src = d / "vocals16k.wav"
    if not src.exists():
        raise HTTPException(404, "not separated yet")
    data, sr = sf.read(str(src), dtype="float32")
    hop = sr // per_sec
    n = len(data) // hop
    pk = np.abs(data[:n * hop]).reshape(n, hop).max(axis=1)
    pk = (pk / (pk.max() or 1) * 255).astype(int).tolist()
    res = {"per_sec": per_sec, "peaks": pk}
    cache.write_text(json.dumps(res))
    return res


@app.on_event("startup")
def _recover():
    """Jobs don't survive a restart; flag them so the user can resume."""
    for item in P.list_all():
        if item["status"]["state"] in ("running", "queued"):
            P.update(item["id"], lambda p: p["status"].update(
                state="error", error="interrupted by server restart — use Run from stage to resume"))


@app.get("/api/projects/{pid}/export/{kind}")
def export(pid: str, kind: str):
    p = _get(pid)
    files = {"video": "out/dubbed.mp4", "audio": "out/dub.wav", "srt_source": "out/source.srt",
             "srt_target": f"out/{p['target_lang']}.srt", "project": "project.json"}
    if kind not in files or not (P.pdir(pid) / files[kind]).exists():
        raise HTTPException(404, "not available yet — run the mix stage")
    path = P.pdir(pid) / files[kind]
    return FileResponse(path, filename=f"{p['name']}.{p['target_lang']}{path.suffix}"
                        if kind != "project" else f"{p['name']}.json")


app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
