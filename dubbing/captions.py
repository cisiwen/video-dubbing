"""Manual captions: fetch, parse, clean, and align to Whisper's word timings.

Captions give the wording (names, terms, punctuation); Whisper's word timestamps give the timing, since
cue times are padded and a cue often spans two speakers. Only creator-made captions are used — auto-
generated ones are no better than Whisper.

project["captions"] = [{"lang": "en" | None, "file": "captions/en.vtt", "origin": "link" | "embedded" | "upload"}]
"""
import difflib
import html
import re
import subprocess
import unicodedata
from pathlib import Path

TEXT_CODECS = {"subrip", "srt", "mov_text", "webvtt", "ass", "ssa", "text"}
CAPTION_EXTS = {".srt", ".vtt"}

# A caption track whose words match less than this share of Whisper's is probably for a different cut
# or language; "auto" then falls back to Whisper's text.
MIN_MATCH = 0.5


# ---------------------------------------------------------------- languages
def base_lang(code):
    """'en-US' / 'en_GB' / 'eng' -> 'en'; None stays None."""
    if not code:
        return None
    code = code.lower().replace("_", "-").split("-")[0]
    return _ISO3.get(code, code)


_ISO3 = {"eng": "en", "zho": "zh", "chi": "zh", "jpn": "ja", "kor": "ko", "spa": "es", "fra": "fr", "fre": "fr",
         "deu": "de", "ger": "de", "ita": "it", "por": "pt", "rus": "ru", "ara": "ar", "hin": "hi", "nld": "nl",
         "dut": "nl", "pol": "pl", "tur": "tr", "swe": "sv", "dan": "da", "fin": "fi", "nor": "no", "ell": "el",
         "gre": "el", "heb": "he", "msa": "ms", "may": "ms", "sw": "sw", "swa": "sw"}


def pick(tracks, lang):
    """The track for spoken language `lang`; a track with no language counts as the spoken one."""
    lang = base_lang(lang)
    ok = [t for t in tracks if not t.get("lang") or base_lang(t["lang"]) == lang]
    # a file the reviewer uploaded first, then a track tagged with the language ('en' before 'en-GB')
    return min(ok, key=lambda t: (t["origin"] != "upload", not t.get("lang"), len(t.get("lang") or "")),
               default=None)


# ---------------------------------------------------------------- fetch
def from_ytdlp(info, ydl, out_dir: Path, want_langs):
    """Save the video's manual subtitle tracks (yt-dlp `subtitles`, never `automatic_captions`) for the
    languages in `want_langs` (base codes; None = every track, capped at 8). Returns track records."""
    subs = {k: v for k, v in (info.get("subtitles") or {}).items() if k not in ("live_chat", "rechat") and v}
    keys = [k for k in subs if want_langs is None or base_lang(k) in want_langs][:8]
    out_dir.mkdir(exist_ok=True)
    tracks = []
    for k in keys:
        fmt = next((f for ext in ("vtt", "srt") for f in subs[k] if f.get("ext") == ext and f.get("url")), None)
        if not fmt:
            continue
        data = ydl.urlopen(fmt["url"]).read()
        path = out_dir / f"{_safe(k)}.{fmt['ext']}"
        path.write_bytes(data)
        if parse(path):
            tracks.append({"lang": k, "file": f"{out_dir.name}/{path.name}", "origin": "link"})
        else:
            path.unlink()
    return tracks


def from_container(video: Path, out_dir: Path):
    """Extract text subtitle streams muxed into an uploaded video (image subtitles like PGS are skipped)."""
    p = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "s", "-show_entries",
                        "stream=index,codec_name:stream_tags=language,title:stream_disposition=hearing_impaired",
                        "-of", "json", str(video)], capture_output=True, text=True)
    if p.returncode:
        return []
    import json
    streams = json.loads(p.stdout or "{}").get("streams", [])
    tracks = []
    for n, s in enumerate(streams):
        if s.get("codec_name") not in TEXT_CODECS:
            continue
        lang = (s.get("tags") or {}).get("language")
        lang = None if lang in (None, "und") else lang
        out_dir.mkdir(exist_ok=True)
        path = out_dir / f"embedded{n}_{_safe(lang or 'und')}.srt"
        r = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video),
                            "-map", f"0:{s['index']}", "-f", "srt", str(path)], capture_output=True, text=True)
        if r.returncode == 0 and parse(path):
            tracks.append({"lang": lang, "file": f"{out_dir.name}/{path.name}", "origin": "embedded"})
    return tracks


def _safe(s):
    return re.sub(r"[^A-Za-z0-9_-]", "_", s)


# ---------------------------------------------------------------- parse + clean
_TIME = r"(?:(\d+):)?(\d{1,2}):(\d{2})[.,](\d{1,3})"
_CUE = re.compile(_TIME + r"\s*-->\s*" + _TIME)


def _secs(h, m, s, ms):
    return int(h or 0) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")) / 1000


def parse(path: Path):
    """SRT or WebVTT -> [{start, end, text, turns:[char offsets where a new speaker starts]}], cleaned,
    with empty cues dropped and roll-up duplicates removed."""
    raw = Path(path).read_text(encoding="utf-8-sig", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
    cues = []
    for block in re.split(r"\n\s*\n", raw):
        lines = block.strip("\n").split("\n")
        for i, line in enumerate(lines):
            m = _CUE.search(line)
            if m:
                start, end = _secs(*m.groups()[:4]), _secs(*m.groups()[4:])
                text, turns = clean("\n".join(lines[i + 1:]))
                if text and end > start:
                    cues.append({"start": round(start, 3), "end": round(end, 3), "text": text, "turns": turns})
                break
    return _dedupe(cues)


_cache: dict = {}


def load(path: Path):
    """parse(), cached until the file changes."""
    path = Path(path)
    key = (str(path), path.stat().st_mtime)
    if key not in _cache:
        _cache[key] = parse(path)
    return _cache[key]


_SPEAKER = re.compile(r"^(?:>>\s*|-\s+|[A-Z][A-Z0-9 .'&-]{0,30}:\s+|\[[^\]]{1,30}\]:\s*)")


def clean(text):
    """Strip markup, sound descriptions, lyrics and speaker labels from one cue. Returns (text, turns):
    `turns` are offsets in the cleaned text where a speaker label or dialogue dash started a new speaker."""
    text = re.sub(r"<\d{1,2}:\d{2}[^>]*>", "", text)  # VTT karaoke timestamps
    text = re.sub(r"</?[a-zA-Z][^>]*>", "", text)  # <i>, <c.color>, <v Name>
    text = re.sub(r"\{\\[^}]*\}", "", text)  # ASS overrides {\an8}
    text = html.unescape(text).replace("\\N", "\n").replace("\\n", "\n")
    out, turns = [], []
    for line in text.split("\n"):
        line = line.strip()
        if not line or "♪" in line or "♫" in line:
            continue
        m = _SPEAKER.match(line)
        if m:
            line = line[m.end():]
        line = re.sub(r"\[[^\]]*\]|\([^)]*\)|（[^）]*）|【[^】]*】", " ", line)  # [MUSIC] (laughs)
        line = re.sub(r"\s+", " ", line).strip()
        if not line:
            continue
        pos = sum(len(x) + 1 for x in out)
        if m and pos:
            turns.append(pos)
        out.append(line)
    return " ".join(out), turns


def _dedupe(cues):
    """Roll-up captions repeat the previous line at the start of the next cue; keep only the new part."""
    out = []
    for c in cues:
        if out:
            prev = out[-1]["text"]
            if c["text"] == prev:
                out[-1]["end"] = max(out[-1]["end"], c["end"])
                continue
            for k in range(min(len(prev), len(c["text"])), 9, -1):
                if c["text"].startswith(prev[-k:]):
                    c = {**c, "text": c["text"][k:].strip(), "turns": [t - k for t in c["turns"] if t > k]}
                    break
            if not c["text"]:
                continue
        out.append(c)
    return out


# ---------------------------------------------------------------- tokenise
_CJK = "\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\uff00"
_PIECE = re.compile(f"[{_CJK}]|[^{_CJK}]+")


def _norm(s):
    return "".join(ch for ch in unicodedata.normalize("NFKC", s).lower()
                   if not unicodedata.category(ch).startswith(("P", "S", "Z", "C")))


def _units(text):
    """Alignment units: words for spaced scripts, single characters for Chinese/Japanese. Each unit is
    (display, norm, space_before); punctuation is kept on the unit before it."""
    out = []
    for word in text.split():
        for k, piece in enumerate(_PIECE.findall(word)):
            n = _norm(piece)
            if not n:
                if out:  # a standalone symbol ("&", "—") keeps its space
                    out[-1] = (out[-1][0] + (" " if k == 0 else "") + piece, out[-1][1], out[-1][2])
                continue
            out.append((piece, n, k == 0 and not re.match(f"[{_CJK}]", piece)))
    return out


# ---------------------------------------------------------------- align
def align(asr, cues, require=False):
    """Replace Whisper's words with the caption's, keeping Whisper's timing.

    Returns (new_asr, stats) — or (asr, stats) unchanged when the caption matches less than MIN_MATCH of
    Whisper's words and `require` is false. Every word gets `src`: "cc" (caption text on a Whisper match),
    "cc~" (caption text, timing interpolated) or "asr" (Whisper-only: filler, crosstalk, missing captions),
    plus `turn` where the caption marks a new speaker.
    """
    a = []  # Whisper units: (norm, start, end, prob, display, space)
    for seg in asr["segments"]:
        for w in seg.get("words") or []:
            us = _units(w["word"]) or []
            dur = (w["end"] - w["start"]) / max(1, len(us))
            for k, (disp, n, space) in enumerate(us):
                a.append({"n": n, "start": w["start"] + k * dur, "end": w["start"] + (k + 1) * dur,
                          "prob": w.get("prob", 1.0), "word": (" " if space else "") + disp})
    c = []  # caption units: (norm, display, space, cue index, turn)
    for i, cue in enumerate(cues):
        pos = 0
        for disp, n, space in _units(cue["text"]):
            at = cue["text"].find(disp, pos)
            at = pos if at < 0 else at
            turn = any(pos <= t <= at for t in cue["turns"])  # first unit after a speaker label / dash
            pos = at + len(disp)
            c.append({"n": n, "word": (" " if space else "") + disp, "cue": i, "turn": turn})
    if not a or not c:
        return asr, {"match": 0.0, "used": False, "caption_units": len(c), "asr_units": len(a)}

    sm = difflib.SequenceMatcher(None, [u["n"] for u in a], [u["n"] for u in c], autojunk=False)
    matched = sum(b.size for b in sm.get_matching_blocks())
    match = matched / len(a)
    stats = {"match": round(match, 3), "caption_units": len(c), "asr_units": len(a)}
    if match < MIN_MATCH and not require:
        return asr, {**stats, "used": False}

    words = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            for x, y in zip(a[i1:i2], c[j1:j2]):
                words.append(_w(y["word"], x["start"], x["end"], x["prob"], "cc", y["turn"]))
        elif op == "delete":
            for x in a[i1:i2]:
                words.append(_w(x["word"], x["start"], x["end"], x["prob"], "asr"))
        elif op == "replace" and i2 - i1 == j2 - j1:
            for x, y in zip(a[i1:i2], c[j1:j2]):
                words.append(_w(y["word"], x["start"], x["end"], x["prob"], "cc~", y["turn"]))
        else:  # replace with different counts, or caption-only words: spread over the available time
            if op == "replace":
                t0, t1 = a[i1]["start"], a[i2 - 1]["end"]
                prob = sum(x["prob"] for x in a[i1:i2]) / (i2 - i1)
            else:
                t0 = a[i1 - 1]["end"] if i1 > 0 else cues[c[j1]["cue"]]["start"]
                t1 = a[i1]["start"] if i1 < len(a) else cues[c[j2 - 1]["cue"]]["end"]
                t0, t1 = _clip_to_cues(t0, t1, cues[c[j1]["cue"]], cues[c[j2 - 1]["cue"]])
                prob = 0.5  # Whisper heard nothing here: worth a look
            ys = c[j1:j2]
            total = sum(len(y["n"]) for y in ys)
            t = t0
            for y in ys:
                dt = (t1 - t0) * len(y["n"]) / total
                words.append(_w(y["word"], t, t + dt, prob, "cc~", y["turn"]))
                t += dt
    for w in words:
        w["start"], w["end"] = round(w["start"], 3), round(max(w["end"], w["start"]), 3)
    return {"language": asr["language"], "text_source": "captions", "caption_match": stats["match"],
            "segments": [{"start": words[0]["start"], "end": words[-1]["end"],
                          "text": "".join(w["word"] for w in words).strip(), "words": words}]}, \
        {**stats, "used": True}


def _w(word, start, end, prob, src, turn=False):
    w = {"word": word, "start": start, "end": end, "prob": prob, "src": src}
    if turn:
        w["turn"] = True
    return w


def _clip_to_cues(t0, t1, first, last):
    """Caption-only words sit between their Whisper neighbours; when that gap is far larger than the cues
    themselves (Whisper skipped a long stretch), use the cue times instead."""
    lo, hi = max(t0, first["start"]), min(t1, last["end"])
    return (lo, hi) if hi - lo > 0.05 else (t0, max(t1, t0 + 0.05))


def overlapping(cues, start, end):
    """Text of the cues that mostly fall inside [start, end] — a translation reference from dub-language
    captions. A cue that only grazes the line is left out: the translator would carry its meaning into
    this line and the next line would say it again."""
    def inside(c):
        ov = min(c["end"], end) - max(c["start"], start)
        return ov > 0 and ov >= 0.5 * min(c["end"] - c["start"], end - start)
    return " ".join(c["text"] for c in cues if inside(c))


# ---------------------------------------------------------------- offset + per-line lookup
def local_offset(cues, segments, t, window=120.0, max_shift=15.0, step=0.1):
    """Seconds to add to cue times so they line up with the speech around time `t`: the shift that best
    overlaps cue intervals with the project's line intervals within ±window. Caption files are often
    offset by a few seconds (and drift), which would otherwise pair a line with its neighbour's caption.
    Returns (shift, overlap_share); shift is 0 unless it clearly beats no shift."""
    import numpy as np
    lo, n, pad = t - window, int(2 * window / step), int(max_shift / step)
    speech, cc = np.zeros(n), np.zeros(n + 2 * pad)
    for s in segments:
        a, b = int((s["start"] - lo) / step), int((s["end"] - lo) / step)
        if b > 0 and a < n:
            speech[max(a, 0):min(b, n)] = 1
    for c in cues:
        a, b = int((c["start"] - lo) / step) + pad, int((c["end"] - lo) / step) + pad
        if b > 0 and a < n + 2 * pad:
            cc[max(a, 0):min(b, n + 2 * pad)] = 1
    total = speech.sum()
    if total < 5 / step:  # under 5 s of speech nearby: nothing to measure against
        return 0.0, None
    # overlap for shift k (cue time + k*step): speech[i] * cc[i + pad - k]
    scores = np.correlate(cc, speech, mode="valid")[::-1]  # index j <-> shift (j - pad) steps
    best = int(scores.argmax())
    if scores[best] - scores[pad] < 0.05 * total:  # less than 5 points better than no shift
        return 0.0, round(float(scores[pad] / total), 3)
    return round((best - pad) * step, 2), round(float(scores[best] / total), 3)


_SCRIPT = {"zh": f"[{_CJK}]", "ja": f"[{_CJK}]", "ko": "[가-힯ᄀ-ᇿ]"}


def script_part(text, lang, keep):
    """From a two-language cue ("Heroes are born 乱世则必出英雄"), keep the words written in `lang`'s script
    (keep=True) or the rest (keep=False). Text in languages without a distinct script is returned whole."""
    pat = _SCRIPT.get(base_lang(lang))
    if not pat:
        return text
    words = [w for w in text.split() if bool(re.search(pat, w)) == keep]
    return " ".join(words)


def cue_matches(track, cues, line, spoken, target, segments, around=20.0):
    """Cues of one track near a line, each with the text it can supply: as_text (transcript, spoken
    language) and/or as_translation (dub language); `match` marks cues inside the line's time slot after
    correcting the track's local offset; `sim` is text similarity to the line for spoken-language text."""
    spoken, target = base_lang(spoken), base_lang(target)
    lang = base_lang(track.get("lang")) or spoken
    shift, quality = local_offset(cues, segments, (line["start"] + line["end"]) / 2)
    distinct = _SCRIPT.get(spoken) != _SCRIPT.get(target)
    out = []
    for i, c in enumerate(cues):
        a, b = c["start"] + shift, c["end"] + shift
        if b < line["start"] - around or a > line["end"] + around:
            continue
        if lang == spoken:
            as_text, as_tr = c["text"], None
        elif lang == target:
            as_tr = script_part(c["text"], spoken, False) if distinct else c["text"]
            as_text = (script_part(c["text"], spoken, True) or None) if distinct else None
        else:
            as_text = as_tr = None
        ov = min(b, line["end"]) - max(a, line["start"])
        match = ov > 0 and ov >= 0.5 * min(b - a, line["end"] - line["start"])
        sim = (round(difflib.SequenceMatcher(None, _norm(as_text), _norm(line["text"])).ratio(), 2)
               if as_text and line["text"] else None)
        out.append({"i": i, "start": round(a, 2), "end": round(b, 2), "text": c["text"], "match": match,
                    "sim": sim, "as_text": as_text, "as_translation": as_tr or None})
    return {"lang": track.get("lang"), "origin": track["origin"], "file": track["file"], "shift": shift,
            "quality": quality, "cues": out}
