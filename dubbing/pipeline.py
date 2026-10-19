"""Dubbing pipeline orchestrator.

Stages run in order and each one is re-runnable. Human edits in the review UI flag segments with
`needs` (emotion / translate / tts); re-running from a stage only touches flagged segments
unless `full=True`.
"""
import queue
import random
import shutil
import threading
import time
import traceback

from . import audio, backend, config, fit, glossary
from . import project as P

# ---------------------------------------------------------------- job runner (one GPU, one job)
_jobs: "queue.Queue[tuple[str, str, bool]]" = queue.Queue()


def enqueue(pid, from_stage="extract", full=False):
    def f(p):
        p["status"].update(state="queued", error=None, message=f"queued from {from_stage}")
    P.update(pid, f)
    _jobs.put((pid, from_stage, full))


def _worker():
    while True:
        pid, from_stage, full = _jobs.get()
        try:
            run(pid, from_stage, full)
        except Exception as e:
            traceback.print_exc()
            P.update(pid, lambda p: p["status"].update(state="error", error=f"{type(e).__name__}: {e}"))


threading.Thread(target=_worker, daemon=True).start()


def _status(pid, stage, progress=None, message=""):
    def f(p):
        p["status"].update(state="running", stage=stage, message=message)
        if progress is not None:
            p["status"]["progress"] = round(progress, 3)
    P.update(pid, f)


def _log(pid, msg):
    print(f"[{pid}] {msg}", flush=True)

    def f(p):
        p["log"] = (p["log"] + [{"t": time.time(), "msg": msg}])[-200:]
    P.update(pid, f)


def run(pid, from_stage="extract", full=False):
    stages = P.STAGES[P.STAGES.index(from_stage):]
    pause_after = P.load(pid)["settings"].get("pause_after")
    for st in stages:
        _status(pid, st, 0, f"{st} ({backend.where(st)})")
        t0 = time.time()
        STAGE_FUNCS[st](pid, full)
        P.update(pid, lambda p: p["stages"].__setitem__(st, "done"))
        _log(pid, f"{st} done in {time.time() - t0:.1f}s")
        if st == pause_after and from_stage == "extract" and st != P.STAGES[-1]:
            P.update(pid, lambda p: p["status"].update(
                state="review", stage=st, progress=1,
                message="Paused for review. Fix speakers/text, then press Apply & continue."))
            return
    P.update(pid, lambda p: p["status"].update(state="done", stage=None, progress=1, message="Dub ready"))


# ---------------------------------------------------------------- stages
DOWNLOAD_SHARE = 0.6  # of the extract stage's progress bar


def download(pid, url):
    """Fetch a URL with yt-dlp (≤1080p, audio merged) into source.<ext>; name the project after the title."""
    import re

    import yt_dlp

    d = P.pdir(pid)
    done_bytes, last = {}, [0.0]

    def hook(h):
        if h["status"] != "downloading" or time.time() - last[0] < 1:
            return
        last[0] = time.time()
        total = h.get("total_bytes") or h.get("total_bytes_estimate") or 0
        done_bytes[h.get("filename")] = (h.get("downloaded_bytes") or 0, total)
        got, size = map(sum, zip(*done_bytes.values()))
        frac = got / size if size else 0
        msg = f"downloading {frac:.0%} of {size / 1e6:.0f} MB" if size else f"downloading {got / 1e6:.0f} MB"
        _status(pid, "extract", DOWNLOAD_SHARE * min(frac, 1), msg)

    opts = {"format": "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/bv*[height<=1080]+ba/b[height<=1080]/bv*+ba/b",
            "merge_output_format": "mp4/mkv", "outtmpl": str(d / "source.%(ext)s"), "noplaylist": True,
            "quiet": True, "no_warnings": True, "noprogress": True, "progress_hooks": [hook],
            "socket_timeout": 30, "retries": 3}
    _status(pid, "extract", 0, "looking up video")
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
            if info.get("_type") == "playlist" or "entries" in info:
                raise RuntimeError("that link is a playlist or channel; paste the link of a single video")
            if info.get("is_live"):
                raise RuntimeError("live streams can't be dubbed; wait until it has finished")
            title = (info.get("title") or "").strip()
            if title:
                P.update(pid, lambda p: p.update(name=title[:120]))
            _log(pid, f"downloading {title or url} ({info.get('extractor_key')}, "
                      f"{info.get('height') or '?'}p, {info.get('duration') or '?'}s)")
            info = ydl.process_ie_result(info, download=True)
    except yt_dlp.utils.DownloadError as e:
        msg = re.sub(r"\x1b\[[0-9;]*m", "", str(e)).removeprefix("ERROR: ").strip()
        raise RuntimeError(f"download failed: {msg}") from None
    path = next((r.get("filepath") for r in info.get("requested_downloads") or [] if r.get("filepath")), None)
    path = path and P.pdir(pid).joinpath(path)  # filepath is absolute
    if not path or not path.exists():
        path = next((f for f in d.glob("source.*") if f.suffix not in (".part", ".ytdl")), None)
    if not path:
        raise RuntimeError("download failed: yt-dlp produced no file")
    P.update(pid, lambda p: p.update(source_video=path.name))


def extract(pid, full):
    d = P.pdir(pid)
    p = P.load(pid)
    base = 0.0
    if p.get("source_url") and not (d / p["source_video"]).exists():
        download(pid, p["source_url"])
        p = P.load(pid)
        base = DOWNLOAD_SHARE
        _status(pid, "extract", base, "extracting audio")
    src = d / p["source_video"]
    info = audio.probe(src)
    if not info["has_audio"]:
        raise RuntimeError("video has no audio track")
    audio.extract(src, d / "audio.wav")
    _status(pid, "extract", base + (1 - base) * 0.5, "making browser preview")
    audio.make_preview(src, d / "preview.mp4")
    P.update(pid, lambda p: p.update(duration=info["duration"]))


def separate(pid, full):
    d = P.pdir(pid)
    backend.separate(d / "audio.wav", d / "vocals.wav", d / "background.wav")
    audio.to_mono16k(d / "vocals.wav", d / "vocals16k.wav")


def transcribe(pid, full):
    d = P.pdir(pid)
    p = P.load(pid)
    res = backend.transcribe(d / "vocals16k.wav", p["settings"].get("source_lang"))
    (d / "asr.json").write_text(__import__("json").dumps(res, ensure_ascii=False))
    P.update(pid, lambda p: p.update(source_lang=res["language"]))


def diarize(pid, full):
    d = P.pdir(pid)
    s = P.load(pid)["settings"]
    turns = backend.diarize(d / "vocals16k.wav", s.get("num_speakers"), s.get("min_speakers"),
                            s.get("max_speakers"))
    (d / "diarization.json").write_text(__import__("json").dumps(turns))


def _speaker_at(turns, a, b):
    """Speaker with the most overlap with [a, b]; nearest turn if none overlap."""
    best, best_ov = None, 0.0
    for t in turns:
        ov = min(b, t["end"]) - max(a, t["start"])
        if ov > best_ov:
            best, best_ov = t["speaker"], ov
    if best is None and turns:
        mid = (a + b) / 2
        best = min(turns, key=lambda t: min(abs(t["start"] - mid), abs(t["end"] - mid)))["speaker"]
    return best or "SPEAKER_00"


def _ends_sentence(w):
    return w["word"].strip()[-1:] in ".!?。！？"


def _merge_side(frag, prev, nxt, same_speaker):
    """Which neighbour ("prev"/"next"/None) a fragment [frag[0]..frag[-1]] should join: one within
    MERGE_GAP_SEC, preferring the side the sentence runs on to, then the same speaker, then the closer."""
    cands = []
    for side, nb, gap in (("prev", prev, prev and frag[0]["start"] - prev[-1]["end"]),
                          ("next", nxt, nxt and nxt[0]["start"] - frag[-1]["end"])):
        if not nb or gap >= config.MERGE_GAP_SEC or (same_speaker and nb[0]["speaker"] != frag[0]["speaker"]):
            continue
        # A fragment ending a sentence belongs with what came before; one after a sentence end, with what follows.
        breaks = _ends_sentence(frag[-1]) if side == "next" else _ends_sentence(prev[-1])
        cands.append((breaks, nb[0]["speaker"] != frag[0]["speaker"], gap, side == "next", side))
    return min(cands)[-1] if cands else None


def _smooth_speakers(words):
    """Diarization flips the speaker for a word or two mid-phrase (turn edges land inside words).
    A run of <= 2 words under MERGE_FRAGMENT_SEC (between pauses) touching another speaker's words
    takes that speaker."""
    runs = []
    for w in words:
        if (runs and runs[-1][-1]["speaker"] == w["speaker"]
                and w["start"] - runs[-1][-1]["end"] <= config.SPLIT_PAUSE_SEC):
            runs[-1].append(w)
        else:
            runs.append([w])
    for i, run in enumerate(runs):
        if len(run) > 2 or run[-1]["end"] - run[0]["start"] >= config.MERGE_FRAGMENT_SEC:
            continue
        prev, nxt = (runs[i - 1] if i else None), (runs[i + 1] if i + 1 < len(runs) else None)
        side = _merge_side(run, prev, nxt, same_speaker=False)
        if side:
            spk = (prev[-1] if side == "prev" else nxt[0])["speaker"]
            for w in run:
                w["speaker"] = spk
    return words


def _merge_fragments(groups):
    """Fold short groups (< MERGE_FRAGMENT_SEC or 1-2 words) into an adjacent group of the same
    speaker across a small gap, keeping each line under the hard length cap."""
    def dur(g):
        return g[-1]["end"] - g[0]["start"]
    i = 0
    while i < len(groups):
        g = groups[i]
        if dur(g) < config.MERGE_FRAGMENT_SEC or len(g) <= 2:
            prev, nxt = (groups[i - 1] if i else None), (groups[i + 1] if i + 1 < len(groups) else None)
            fits_prev = prev and g[-1]["end"] - prev[0]["start"] <= config.MAX_SEGMENT_SEC * 1.6
            fits_next = nxt and nxt[-1]["end"] - g[0]["start"] <= config.MAX_SEGMENT_SEC * 1.6
            side = _merge_side(g, prev if fits_prev else None, nxt if fits_next else None, same_speaker=True)
            if side == "prev":
                groups[i - 1] = prev + g
                del groups[i]
                continue
            if side == "next":
                groups[i + 1] = g + nxt
                del groups[i]
                continue
        i += 1
    return groups


def _group_words(asr, turns):
    """Assign each word a speaker, then cut into dubbing segments at speaker changes, pauses,
    sentence ends past the length limit, and a hard max length; too-short fragments are folded
    into a neighbouring line."""
    words = []
    for seg in asr["segments"]:
        ws = seg["words"] or [{"start": seg["start"], "end": seg["end"], "word": " " + seg["text"], "prob": 1}]
        for w in ws:
            words.append({**w, "speaker": _speaker_at(turns, w["start"], w["end"])})
    words = _smooth_speakers(words)
    groups, cur = [], []
    for w in words:
        if cur:
            prev = cur[-1]
            dur = w["end"] - cur[0]["start"]
            end_sentence = _ends_sentence(prev)
            if (w["speaker"] != prev["speaker"] or w["start"] - prev["end"] > config.SPLIT_PAUSE_SEC
                    or (dur > config.MAX_SEGMENT_SEC and end_sentence)
                    or dur > config.MAX_SEGMENT_SEC * 1.6):
                groups.append(cur)
                cur = []
        cur.append(w)
    if cur:
        groups.append(cur)
    return _merge_fragments(groups)


def segment(pid, full):
    """Build segments + speakers from ASR words and diarization, then per-segment language ID."""
    import json
    d = P.pdir(pid)
    asr = json.loads((d / "asr.json").read_text())
    turns = json.loads((d / "diarization.json").read_text())
    groups = _group_words(asr, turns)
    primary = asr["language"]

    segs = []
    for i, g in enumerate(groups, 1):
        segs.append({"id": f"s{i:05d}", "start": round(g[0]["start"], 3), "end": round(g[-1]["end"], 3),
                     "speaker": g[0]["speaker"], "lang": primary, "lang_prob": None,
                     "text": "".join(w["word"] for w in g).strip(),
                     "confidence": round(sum(w["prob"] for w in g) / len(g), 3),
                     "emotion": None, "emotion_score": None, "translation": "", "tts": None,
                     "reviewed": False, "needs": ["emotion", "translate", "tts"]})

    # Spoken language per segment: catches code-switching / foreign-language lines.
    _status(pid, "segment", 0.4, "detecting language per segment")
    idx = [i for i, s in enumerate(segs) if s["end"] - s["start"] >= 1.0]
    langs = backend.detect_languages(d / "vocals16k.wav", [[segs[i]["start"], segs[i]["end"]] for i in idx])
    for i, r in zip(idx, langs):
        segs[i]["lang_prob"] = round(r["prob"], 3)
        if r["language"] != primary and r["prob"] > 0.75:
            segs[i]["lang"] = r["language"]
            clip = d / "clips" / f"{segs[i]['id']}_16k.wav"
            audio.cut(d / "vocals16k.wav", segs[i]["start"], segs[i]["end"], clip)
            redo = backend.transcribe(clip, r["language"])
            text = " ".join(s["text"] for s in redo["segments"]).strip()
            if text:
                segs[i]["text"] = text

    speakers = {}
    for n, spk in enumerate(sorted({s["speaker"] for s in segs})):
        speakers[spk] = {"id": spk, "name": f"Speaker {n + 1}", "color": config.PALETTE[n % len(config.PALETTE)],
                         "ref_segments": [], "ref_mode": "segment", "exaggeration": None, "cfg_weight": None}

    def f(p):
        p["segments"] = segs
        p["speakers"] = speakers
    P.update(pid, f)
    build_refs(pid)


def build_refs(pid, speaker_ids=None):
    """Create each speaker's voice-clone reference from their cleanest, longest lines
    (or the lines a reviewer picked)."""
    d = P.pdir(pid)
    p = P.load(pid)
    for spk, info in p["speakers"].items():
        if speaker_ids and spk not in speaker_ids:
            continue
        mine = [s for s in p["segments"] if s["speaker"] == spk]
        chosen = [s for s in mine if s["id"] in info["ref_segments"]]
        if not chosen:
            others = [s for s in p["segments"] if s["speaker"] != spk]

            def clean(s):
                return not any(o["start"] < s["end"] and o["end"] > s["start"] for o in others)
            cands = sorted((s for s in mine if clean(s) and 2.0 <= s["end"] - s["start"] <= 15),
                           key=lambda s: (s["confidence"], s["end"] - s["start"]), reverse=True)
            total = 0.0
            for s in cands:
                chosen.append(s)
                total += s["end"] - s["start"]
                if total >= 10:
                    break
            chosen = chosen or sorted(mine, key=lambda s: s["end"] - s["start"], reverse=True)[:1]
        clips = []
        for s in chosen:
            c = d / "clips" / f"{s['id']}.wav"
            audio.cut(d / "vocals.wav", s["start"], s["end"], c)
            clips.append(c)
        if clips:
            audio.concat(clips, d / "refs" / f"{spk}.wav")
        P.update(pid, lambda p, spk=spk, ids=[s["id"] for s in chosen]:
                 p["speakers"][spk].__setitem__("ref_segments", ids))


def emotion(pid, full):
    d = P.pdir(pid)
    p = P.load(pid)
    todo = [s for s in p["segments"] if full or "emotion" in s.get("needs", [])]
    if not todo:
        return
    res = backend.emotions(d / "vocals16k.wav", [[s["start"], s["end"]] for s in todo])
    found = {s["id"]: r for s, r in zip(todo, res)}

    def f(p):
        for s in p["segments"]:
            if s["id"] in found:
                if not s.get("emotion_locked"):
                    s["emotion"] = found[s["id"]]["emotion"]
                    s["emotion_score"] = round(found[s["id"]]["score"], 3)
                s["needs"] = [n for n in s["needs"] if n != "emotion"]
    P.update(pid, f)


def translate(pid, full):
    speaker_rates(pid)  # from lines voiced earlier, if any: sizes each speaker's length budget
    p = P.load(pid)
    tgt = p["target_lang"]
    todo = [s for s in p["segments"] if (full and not s.get("translation_locked"))
            or "translate" in s.get("needs", [])]
    by_lang: dict[str, list] = {}
    for s in todo:
        by_lang.setdefault(s["lang"] or p["source_lang"], []).append(s)
    done = {}
    for n, (lang, segs) in enumerate(by_lang.items()):
        _status(pid, "translate", n / max(1, len(by_lang)), f"{lang} → {tgt}: {len(segs)} lines")
        if lang == tgt:
            out = [s["text"] for s in segs]
        else:
            out = backend.translate([s["text"] for s in segs], lang, tgt, [translation_context(p, s) for s in segs])
        done.update({s["id"]: t for s, t in zip(segs, out)})

    def f(p):
        for s in p["segments"]:
            if s["id"] in done:
                s["translation"] = done[s["id"]]
                s["needs"] = sorted(set(n for n in s["needs"] if n != "translate") | {"tts"})
    P.update(pid, f)


def translation_context(p, s):
    """What the translator needs to fit a line: its slot, a length budget, tone, neighbours and the
    glossary renderings of the names it mentions."""
    i = P.seg_index(p, s["id"])
    room = _room(p, s)
    return {"duration": round(s["end"] - s["start"], 2), "room": round(room, 2),
            "budget": config.speech_budget(p["target_lang"], room, _speaker_rate(p, s["speaker"])),
            "emotion": s["emotion"],
            "prev": [x["text"] for x in p["segments"][max(0, i - 2):i]],
            "next": [x["text"] for x in p["segments"][i + 1:i + 2]],
            "names": glossary.names_in(p, s["text"])}


def _speaker_rate(p, spk):
    """This speaker's measured TTS pace in the project's target language, or None (use the default)."""
    r = (p["speakers"].get(spk) or {}).get("speech_rate") or {}
    return r.get("rate") if r.get("lang") == p["target_lang"] else None


def speaker_rates(pid):
    """Measure each speaker's natural TTS pace — spoken_units(translation) per second of voiced speech before
    any tempo change — from their up-to-date voiced lines, and store it on project.speakers[*].speech_rate.
    The mean of a line's takes is used, not the kept take (which best-of-N biases toward whatever fit).
    Speakers with fewer than RATE_MIN_LINES lines get rate None (the language default applies)."""
    p = P.load(pid)
    tgt = p["target_lang"]
    default = config.speech_rate(tgt)
    acc: dict[str, list] = {}
    for s in p["segments"]:
        t = s.get("tts")
        if not t or "tts" in s.get("needs", []) or t.get("text", s["translation"]) != s["translation"]:
            continue
        units = config.spoken_units(s["translation"])
        durs = [c["duration"] for c in t.get("takes", []) if not c.get("capped")]
        natural = sum(durs) / len(durs) if durs else t["duration"] * t["speed"]
        # One-unit lines are mostly onset/offset; runaway or truncated takes say nothing about pace.
        if units < 2 or not natural or not 0.5 <= units / natural / default <= 2:
            continue
        a = acc.setdefault(s["speaker"], [0, 0.0, 0])
        a[0], a[1], a[2] = a[0] + units, a[1] + natural, a[2] + 1
    rates = {}
    for spk in p["speakers"]:
        units, sec, n = acc.get(spk, (0, 0.0, 0))
        rate = None
        if n >= config.RATE_MIN_LINES:
            rate = round((units + default * config.RATE_PRIOR_SEC) / (sec + config.RATE_PRIOR_SEC), 2)
        rates[spk] = {"rate": rate, "lines": n, "seconds": round(sec, 1), "lang": tgt}

    def f(p):
        for spk, r in rates.items():
            if spk in p["speakers"]:
                p["speakers"][spk]["speech_rate"] = r
    P.update(pid, f)
    return rates


def _room(p, s):
    """Seconds this line may occupy: its slot, plus up to 0.5 s of silence before the next line."""
    i = P.seg_index(p, s["id"])
    seg_len = s["end"] - s["start"]
    nxt = p["segments"][i + 1]["start"] if i + 1 < len(p["segments"]) else (p["duration"] or s["end"] + 2)
    return max(seg_len, min(nxt - s["start"], seg_len + 0.5))


def _tts_item(d, p, s):
    """Pick the clone reference and voice style for one line."""
    spk = p["speakers"].get(s["speaker"], {})
    seg_len = s["end"] - s["start"]
    # Own clip carries this line's emotion and prosody; fall back to the speaker's reference.
    if spk.get("ref_mode", "segment") == "segment" and seg_len >= config.MIN_REF_SEC:
        ref = d / "clips" / f"{s['id']}.wav"
        audio.cut(d / "vocals.wav", s["start"], s["end"], ref)
    else:
        ref = d / "refs" / f"{s['speaker']}.wav"
    style = dict(config.EMOTION_STYLE.get(s["emotion"] or "neutral", config.EMOTION_STYLE["neutral"]))
    for k in ("exaggeration", "cfg_weight"):
        if spk.get(k) is not None:
            style[k] = spk[k]
        if s.get(k) is not None:
            style[k] = s[k]
    # Speech beyond room x max speed-up can never fit; stop decoding well after that (25 tokens/s).
    max_tokens = int(25 * _room(p, s) * config.MAX_SPEEDUP * 1.5) + 50
    return {"id": s["id"], "text": s["translation"], "language": p["target_lang"], "ref": ref,
            "out": d / "tts" / f"{s['id']}_raw.wav", "max_tokens": max_tokens, **style}


def _take_cfg(cfg, k):
    """cfg_weight for take k: the line's own, then alternately lower / higher by TAKE_CFG_STEP, 2x, ..."""
    if k == 0:
        return cfg
    step = config.TAKE_CFG_STEP * ((k + 1) // 2)
    return round(min(1.0, max(0.05, cfg - step if k % 2 else cfg + step)), 3)


def _fit(pid, d, p, sid, raw, dur, style, gen_sec, extra=None):
    """Tempo-fit a generated, silence-trimmed line (`dur` s) into its slot, then record it on the segment."""
    i = P.seg_index(p, sid)
    s = p["segments"][i]
    seg_len = s["end"] - s["start"]
    out = d / "tts" / f"{sid}.wav"
    room = _room(p, s)
    factor = fit.tempo(dur, seg_len, room)
    factor = min(factor, config.MAX_SPEEDUP) if factor > 1 else max(factor, config.MAX_SLOWDOWN)
    if abs(factor - 1) > 0.02:
        audio.stretch(raw, out, factor)
    else:
        out.write_bytes(raw.read_bytes())
    final = audio.duration(out)
    info = {"file": f"tts/{sid}.wav", "duration": round(final, 3), "speed": round(factor, 3),
            "overflow": round(max(0.0, final - room), 3), "style": style, "gen_sec": round(gen_sec, 2),
            "made": time.time(), "text": s["translation"], **(extra or {})}

    def f(p):
        seg = p["segments"][P.seg_index(p, sid)]
        seg["tts"] = info
        seg["needs"] = [n for n in seg["needs"] if n != "tts"]
    P.update(pid, f)
    return info


def _synth_batch(pid, sids, takes=None):
    """Voice lines best-of-N: every line gets `takes` (default TTS_TAKES) candidates decoded in the same
    batch, each with its own seed and a nudged cfg_weight; the one that best fits the slot is kept. The
    worker picks by token count and vocodes only that take; the others' durations are estimates."""
    takes = takes or config.TTS_TAKES
    d = P.pdir(pid)
    p = P.load(pid)
    segs = [p["segments"][P.seg_index(p, sid)] for sid in sids]
    lines = [_tts_item(d, p, s) for s in segs if s["translation"].strip()]
    if not lines:
        return []
    seed = random.randrange(2 ** 31 - 1000)
    items = []
    for n, ln in enumerate(lines):
        s = p["segments"][P.seg_index(p, ln["id"])]
        slot = [s["end"] - s["start"], _room(p, s)]
        items += [{**ln, "seed": seed + n * takes + k, "cfg_weight": _take_cfg(ln["cfg_weight"], k),
                   "group": ln["id"], "slot": slot, "out": d / "tts" / f"{ln['id']}_take{k}.wav"}
                  for k in range(takes)]
    for it in items:
        it["out"].unlink(missing_ok=True)  # a leftover take must not pass for this run's
    t0 = time.time()
    res = backend.tts_batch(items) or []
    res = list(res) + [{}] * (len(items) - len(res))
    per_line = (time.time() - t0) / len(lines)  # batch wall time shared across its lines
    p = P.load(pid)
    out = []
    for n, ln in enumerate(lines):
        s = p["segments"][P.seg_index(p, ln["id"])]
        cands = []
        for k in range(takes):
            it, r = items[n * takes + k], res[n * takes + k]
            c = {"take": k, "seed": it["seed"], "cfg_weight": it["cfg_weight"], "capped": r.get("capped", False)}
            if r.get("vocoded", True) and it["out"].exists() and it["out"].stat().st_size:
                c["duration"] = round(audio.trim_silence(it["out"]), 3)
            else:
                c.update(duration=r.get("est_duration", 0.0), est=True)
            cands.append(c)
        best = min((c for c in cands if not c.get("est")),
                   key=lambda c: fit.take_score(c, s["end"] - s["start"], _room(p, s)))
        for k in range(takes):
            take_wav = items[n * takes + k]["out"]
            if k == best["take"]:
                take_wav.replace(ln["out"])
            else:
                take_wav.unlink(missing_ok=True)
        style = {"exaggeration": ln["exaggeration"], "cfg_weight": best["cfg_weight"]}
        extra = {"take": best["take"], "takes": cands} if takes > 1 else {"seed": best["seed"]}
        out.append(_fit(pid, d, p, ln["id"], ln["out"], best["duration"], style, per_line, extra))
    return out


def synth_one(pid, sid):
    """Voice-clone one segment, best-of-N (review UI's regenerate button)."""
    res = _synth_batch(pid, [sid])
    return res[0] if res else None


def synthesize(pid, full):
    p = P.load(pid)
    todo = [s for s in p["segments"] if s["translation"].strip()
            and (full or "tts" in s.get("needs", []) or not s.get("tts"))]
    # Similar-length lines finish decoding together, so less of each batch is wasted padding.
    todo.sort(key=lambda s: len(s["translation"]))
    n = config.TTS_BATCH
    t0 = time.time()
    for k in range(0, len(todo), n):
        eta = (time.time() - t0) / k * (len(todo) - k) if k else None
        _status(pid, "synthesize", k / max(1, len(todo)),
                f"voices {k + 1}-{min(k + n, len(todo))}/{len(todo)}" + (f", ~{eta / 60:.0f} min left" if eta else ""))
        _synth_batch(pid, [s["id"] for s in todo[k:k + n]])
    rates = speaker_rates(pid)
    _log(pid, "speaker rates (units/s): " + ", ".join(
        f"{p['speakers'][spk]['name']} {r['rate'] or '-'} ({r['lines']})" for spk, r in rates.items()
        if spk in p["speakers"]))
    _shorten_overflows(pid, {s["id"] for s in todo})


def _overflowing(p, sids):
    return [s for s in p["segments"] if s["id"] in sids and s.get("tts")
            and s["tts"]["overflow"] > config.RETRY_OVERFLOW_SEC]


def _shorten_overflows(pid, sids):
    """Lines still too long after the tempo cap get re-translated shorter and re-voiced (at most
    RETRY_ROUNDS rounds); each line keeps whichever attempt overflows least."""
    if config.TRANSLATION_PROVIDER not in ("llm", "ollama"):
        return
    d = P.pdir(pid)
    best: dict[str, dict] = {}
    start = len(_overflowing(P.load(pid), sids))
    for rnd in range(1, config.RETRY_ROUNDS + 1):
        p = P.load(pid)
        tgt = p["target_lang"]
        over = [s for s in _overflowing(p, sids) if not s.get("translation_locked")
                and (s["lang"] or p["source_lang"]) != tgt]
        if not over:
            break
        _status(pid, "synthesize", 1, f"shortening {len(over)} overlong lines (round {rnd}/{config.RETRY_ROUNDS})")
        by_lang: dict[str, list] = {}
        for s in over:
            keep = d / "tts" / f"{s['id']}_try{rnd - 1}.wav"
            shutil.copyfile(d / s["tts"]["file"], keep)
            if s["id"] not in best or s["tts"]["overflow"] < best[s["id"]]["tts"]["overflow"]:
                best[s["id"]] = {"translation": s["translation"], "tts": s["tts"], "wav": keep}
            by_lang.setdefault(s["lang"] or p["source_lang"], []).append(s)
        new = {}
        for lang, segs in by_lang.items():
            ctxs = []
            for s in segs:
                ctx = translation_context(p, s)
                # This voice's own pace on this line, unless it ran on (far slower than normal speech).
                units = config.spoken_units(s["translation"])
                natural = s["tts"]["duration"] * s["tts"]["speed"]
                rate = max(units / natural if natural else 0,
                           0.75 * (_speaker_rate(p, s["speaker"]) or config.speech_rate(tgt)))
                fit = int(rate * ctx["room"] * config.MAX_SPEEDUP * 0.9)
                ctx.update(budget=max(1, min(units - 1, fit)), attempt=s["translation"],
                           attempt_over=s["tts"]["overflow"])
                ctxs.append(ctx)
            new.update(zip([s["id"] for s in segs], backend.translate([s["text"] for s in segs], lang, tgt, ctxs)))

        def f(p):
            for s in p["segments"]:
                if new.get(s["id"]):
                    s.setdefault("translation_attempts", []).append(
                        {"text": s["translation"], "overflow": s["tts"]["overflow"]})
                    s["translation"] = new[s["id"]]
        P.update(pid, f)
        ids = sorted(new, key=lambda sid: len(new[sid]))
        for k in range(0, len(ids), config.TTS_BATCH):
            _synth_batch(pid, ids[k:k + config.TTS_BATCH])

    def f(p):
        for sid, b in best.items():
            s = p["segments"][P.seg_index(p, sid)]
            if s["tts"]["overflow"] > b["tts"]["overflow"]:
                shutil.copyfile(b["wav"], d / b["tts"]["file"])
                s.update(translation=b["translation"], tts=b["tts"])
    P.update(pid, f)
    for w in (d / "tts").glob("*_try*.wav"):
        w.unlink()
    if best:
        _log(pid, f"shortened overlong lines: {start} -> {len(_overflowing(P.load(pid), sids))} "
                  f"over {config.RETRY_OVERFLOW_SEC}s")


def mix(pid, full):
    d = P.pdir(pid)
    p = P.load(pid)
    placements = [(d / s["tts"]["file"], s["start"]) for s in p["segments"]
                  if s.get("tts") and (d / s["tts"]["file"]).exists()]
    audio.mix(d / "background.wav", placements, d / "out" / "dub.wav", p["duration"])
    _status(pid, "mix", 0.6, "muxing video")
    audio.mux(d / p["source_video"], d / "out" / "dub.wav", d / "out" / "dubbed.mp4", p["target_lang"])
    audio.srt(p["segments"], "text", d / "out" / "source.srt")
    audio.srt(p["segments"], "translation", d / "out" / f"{p['target_lang']}.srt")


STAGE_FUNCS = {"extract": extract, "separate": separate, "transcribe": transcribe, "diarize": diarize,
               "segment": segment, "emotion": emotion, "translate": translate,
               "synthesize": synthesize, "mix": mix}
