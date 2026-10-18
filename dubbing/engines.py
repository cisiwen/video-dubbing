"""Heavy model wrappers. Imported by the GPU worker (or in-process when no worker is configured).

Every function takes plain file paths / python values and returns JSON-serialisable data or
writes wav files, so the same calls can be served over HTTP by worker.py.
Only one model family is kept resident at a time to fit in a 12 GB GPU.
"""
import gc
import sys
import threading

import numpy as np
import soundfile as sf

from . import config

_lock = threading.RLock()
_resident: dict = {}


def _model(name, loader):
    with _lock:
        if name not in _resident:
            for other in list(_resident):
                del _resident[other]
            gc.collect()
            _empty_cache()
            _resident[name] = loader()
        return _resident[name]


def _empty_cache():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


def load_audio(path, sr=None, mono=True):
    """Load wav as float32 (channels-first when not mono)."""
    data, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    data = data.T  # (C, N)
    if mono:
        data = data.mean(axis=0)
    if sr and sr != file_sr:
        import librosa
        data = librosa.resample(data, orig_sr=file_sr, target_sr=sr)
        file_sr = sr
    return data, file_sr


# ---------------------------------------------------------------- source separation
def separate(audio_path, vocals_out, background_out):
    """Split dialogue from music/effects with Demucs (two stems)."""
    import torch
    from demucs.api import Separator

    sep = _model("demucs", lambda: Separator(model=config.DEMUCS_MODEL, device=config.device(),
                                             segment=7.8 if config.DEMUCS_MODEL == "htdemucs" else None))
    wav, sr = load_audio(audio_path, mono=False)
    if wav.shape[0] == 1:
        wav = np.repeat(wav, 2, axis=0)
    with _lock, torch.inference_mode():
        _, stems = sep.separate_tensor(torch.from_numpy(wav), sr)
    vocals = stems["vocals"].cpu().numpy()
    background = sum(v for k, v in stems.items() if k != "vocals").cpu().numpy()
    sf.write(str(vocals_out), vocals.T, sep.samplerate)
    sf.write(str(background_out), background.T, sep.samplerate)
    return {"sample_rate": sep.samplerate}


# ---------------------------------------------------------------- speech recognition
def _use_mlx():
    return sys.platform == "darwin"


def _fw_model():
    from faster_whisper import WhisperModel
    name = config.WHISPER_MODEL
    if name.startswith("mlx-community/"):
        name = "large-v3"
    return WhisperModel(name, device="cuda" if config.device() == "cuda" else "cpu",
                        compute_type="float16" if config.device() == "cuda" else "int8")


def transcribe(audio_path, language=None):
    """Return {language, segments:[{start,end,text,words:[{start,end,word,prob}]}]}."""
    audio, _ = load_audio(audio_path, sr=16000)
    if _use_mlx():
        import mlx_whisper
        with _lock:
            res = mlx_whisper.transcribe(audio, path_or_hf_repo=config.WHISPER_MODEL, language=language,
                                         word_timestamps=True, condition_on_previous_text=False)
        segs = [{"start": s["start"], "end": s["end"], "text": s["text"].strip(),
                 "words": [{"start": w["start"], "end": w["end"], "word": w["word"],
                            "prob": w.get("probability", 1.0)} for w in s.get("words", [])]}
                for s in res["segments"]]
        return {"language": res.get("language"), "segments": segs}

    model = _model("whisper", _fw_model)
    with _lock:
        it, info = model.transcribe(audio, language=language, word_timestamps=True, vad_filter=True,
                                    condition_on_previous_text=False)
        segs = [{"start": s.start, "end": s.end, "text": s.text.strip(),
                 "words": [{"start": w.start, "end": w.end, "word": w.word, "prob": w.probability}
                           for w in (s.words or [])]} for s in it]
    return {"language": info.language, "segments": segs}


def detect_languages(audio_path, clips):
    """Spoken-language ID per clip [[start,end],...] -> [{language, prob}]."""
    audio, sr = load_audio(audio_path, sr=16000)
    out = []
    if _use_mlx():
        import mlx.core as mx
        from mlx_whisper.audio import N_FRAMES, log_mel_spectrogram, pad_or_trim
        from mlx_whisper.decoding import detect_language
        from mlx_whisper.load_models import load_model
        model = _model("mlx_whisper", lambda: load_model(config.WHISPER_MODEL, dtype=mx.float16))
        for s, e in clips:
            mel = log_mel_spectrogram(audio[int(s * sr):int(e * sr)], n_mels=model.dims.n_mels)
            mel = pad_or_trim(mel, N_FRAMES, axis=-2).astype(mx.float16)
            with _lock:
                _, probs = detect_language(model, mel)
            probs = probs[0] if isinstance(probs, list) else probs
            lang = max(probs, key=probs.get)
            out.append({"language": lang, "prob": float(probs[lang])})
        return out

    model = _model("whisper", _fw_model)
    for s, e in clips:
        with _lock:
            lang, prob, _ = model.detect_language(audio[int(s * sr):int(e * sr)])
        out.append({"language": lang, "prob": float(prob)})
    return out


# ---------------------------------------------------------------- speaker diarization
def diarize(audio_path, num_speakers=None, min_speakers=None, max_speakers=None):
    """Return [{start,end,speaker}] speaker turns."""
    import torch
    from pyannote.audio import Pipeline

    def load():
        import pyannote.audio
        v4 = int(pyannote.audio.__version__.split(".")[0]) >= 4
        name = config.DIARIZATION_MODEL or ("pyannote/speaker-diarization-community-1" if v4
                                            else "pyannote/speaker-diarization-3.1")
        auth = {"token": config.HF_TOKEN} if v4 else {"use_auth_token": config.HF_TOKEN}
        try:
            p = Pipeline.from_pretrained(name, **auth)
        except Exception:
            # Gated model already in the local HF cache: load it without a token.
            from huggingface_hub import constants
            constants.HF_HUB_OFFLINE = True
            try:
                p = Pipeline.from_pretrained(name)
            finally:
                constants.HF_HUB_OFFLINE = False
        if p is None:
            raise RuntimeError("Could not load pyannote pipeline: set HF_TOKEN and accept the model terms")
        p.to(torch.device(config.device()))
        return p

    pipe = _model("pyannote", load)
    audio, sr = load_audio(audio_path, sr=16000)
    kwargs = {k: v for k, v in dict(num_speakers=num_speakers, min_speakers=min_speakers,
                                    max_speakers=max_speakers).items() if v}
    with _lock:
        out = pipe({"waveform": torch.from_numpy(audio)[None], "sample_rate": sr}, **kwargs)
    # pyannote 4 returns DiarizeOutput; prefer the overlap-free track for word assignment.
    ann = getattr(out, "exclusive_speaker_diarization", None) or getattr(out, "speaker_diarization", out)
    return [{"start": t.start, "end": t.end, "speaker": spk}
            for t, _, spk in ann.itertracks(yield_label=True)]


# ---------------------------------------------------------------- emotion
def emotions(audio_path, clips):
    """Speech emotion per clip -> [{emotion, score, scores}] using emotion2vec+."""
    from funasr import AutoModel
    model = _model("emotion2vec", lambda: AutoModel(model=config.EMOTION_MODEL, hub="hf",
                                                     disable_update=True, device=config.device()
                                                     if config.device() == "cuda" else "cpu"))
    audio, sr = load_audio(audio_path, sr=16000)
    out = []
    for s, e in clips:
        chunk = audio[int(s * sr):int(e * sr)]
        if len(chunk) < sr * 0.3:
            out.append({"emotion": "neutral", "score": 0.0, "scores": {}})
            continue
        with _lock:
            r = model.generate(chunk, granularity="utterance", extract_embedding=False,
                               disable_pbar=True)[0]
        scores = {}
        for label, score in zip(r["labels"], r["scores"]):
            name = label.split("/")[-1]
            name = {"<unk>": "other", "unknown": "other"}.get(name, name)
            scores[name] = scores.get(name, 0.0) + float(score)
        best = max(scores, key=scores.get)
        out.append({"emotion": best, "score": scores[best], "scores": scores})
    return out


# ---------------------------------------------------------------- translation
def translate(texts, src, tgt, context=None, provider=None):
    """context[i] (optional): {duration, room, budget, emotion, prev:[...], next:[...],
    attempt, attempt_over} - used by the LLM providers to keep each line inside its time slot -
    and names: [[source, target], ...], the project glossary entries that occur in the line."""
    provider = provider or config.TRANSLATION_PROVIDER
    out = None
    if provider in ("llm", "ollama"):
        try:
            out = (_translate_llm if provider == "llm" else _translate_ollama)(texts, src, tgt, context)
        except Exception as e:
            print(f"{provider} translation failed ({type(e).__name__}: {e}); falling back to NLLB", flush=True)
    if out is None:
        out = _translate_nllb(texts, src, tgt)
    return [_apply_names(o, (c or {}).get("names")) for o, c in zip(out, context or [None] * len(out))]


def _apply_names(out, names):
    """Replace any glossary name the translator left in its source spelling with the agreed rendering."""
    import re
    for src_name, tgt_name in sorted(names or [], key=lambda n: -len(n[0])):
        if src_name and tgt_name:
            out = re.sub(rf"(?<![A-Za-z]){re.escape(src_name)}(?![A-Za-z])", lambda _: tgt_name, out, flags=re.I)
    return out


def _missing_names(out, names):
    return [(a, b) for a, b in names or [] if b and b not in out]


def _translate_nllb(texts, src, tgt):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    def load():
        tok = AutoTokenizer.from_pretrained(config.NLLB_MODEL)
        mdl = AutoModelForSeq2SeqLM.from_pretrained(config.NLLB_MODEL).to(config.device()).eval()
        return tok, mdl

    tok, mdl = _model("nllb", load)
    src_code = config.NLLB_CODES.get(src, "eng_Latn")
    tgt_code = config.NLLB_CODES[tgt]
    out = []
    for i in range(0, len(texts), 16):
        batch = texts[i:i + 16]
        tok.src_lang = src_code
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=400).to(mdl.device)
        with _lock, torch.inference_mode():
            gen = mdl.generate(**enc, forced_bos_token_id=tok.convert_tokens_to_ids(tgt_code),
                               max_new_tokens=400, num_beams=4)
        out.extend(tok.batch_decode(gen, skip_special_tokens=True))
    return out


_LANG_NAMES = {"en": "English", **config.TTS_LANGUAGES, "zh": "Simplified Chinese (Mandarin)"}
_CJK = {"zh", "ja", "ko"}
_META = __import__("re").compile(r"\b(the user|translat\w*|let me think|time slot)\b|用户|翻译", __import__("re").I)
_HAN = r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]"  # kana, CJK ideographs, hangul


def _messages(text, src, tgt, ctx):
    """Chat prompt for one dubbing line with its time budget, emotion and neighbouring lines."""
    tgt_name = _LANG_NAMES.get(tgt, tgt)
    unit = "characters (each digit counts as one)" if tgt in _CJK else "syllables"
    script = (f"Use only {tgt_name} characters, digits and punctuation - no Latin letters; transliterate "
              f"foreign names." if tgt in _CJK else f"Write it as a {tgt_name} voice actor would read it, using only "
              f"{tgt_name} words and Latin letters; romanise names and titles (pinyin for Chinese).")
    system = (
        f"You are a professional dubbing translator for broadcast TV. You translate one line of "
        f"{_LANG_NAMES.get(src, src)} dialogue at a time into natural, spoken {tgt_name}; a voice actor "
        f"reads it in the same time slot as the original. Obey the length limit strictly: condense, "
        f"paraphrase and drop filler or repetition, but keep names, numbers and the key meaning. "
        f"Lines are cut from continuous speech, so the line may be a sentence fragment: translate only "
        f"that fragment, so the dubbed lines still flow in order. The surrounding lines are context only: "
        f"never translate or add their content. {script} "
        f"Reply with the translation only - no quotes, notes, romanisation or explanations.")
    before = [t for t in ctx.get("prev", []) if t]
    after = [t for t in ctx.get("next", []) if t]
    user = ""
    if before or after:
        user = "Context only, do NOT translate:\n"
        user += "".join(f"  before: {t}\n" for t in before) + "".join(f"  after: {t}\n" for t in after) + "\n"
    user += f"Translate ONLY this line: {text}"
    if ctx.get("names"):
        user += ("\nNames in this line - write each exactly like this: "
                 + "; ".join(f"{a} = {b}" for a, b in ctx["names"]) + ".")
    if ctx.get("emotion"):
        user += f"\nTone: {ctx['emotion']}."
    if ctx.get("budget"):
        hi = ctx["budget"]
        lo = int(hi * (0.6 if ctx.get("attempt") else 0.75))
        size = f"about {lo}-{hi} {unit}, never more than {hi}" if lo >= 3 else f"at most {hi} {unit}"
        user += (f"\nTime slot: {ctx.get('room') or ctx.get('duration', 0):.1f} s. Length: {size}. "
                 f"Use the room to keep the full meaning; only cut what does not fit.")
    if ctx.get("attempt"):
        user += (f"\nThe previous version \"{ctx['attempt']}\" was {ctx.get('attempt_over', 0):.1f} s too "
                 f"long when spoken. Make it clearly shorter.")
    if ctx.get("fix"):
        user += f"\nYour previous answer \"{ctx['fix_text']}\" {ctx['fix']} Try again."
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _clean(out):
    out = out.strip().split("\n")[0].strip()
    for pre in ("Translation:", "翻译：", "翻译:"):
        if out.startswith(pre):
            out = out[len(pre):].strip()
    return out.strip("\"'“”「」 ")


def _llm_generate(prompts, max_new, clean=True):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    def load():
        tok = AutoTokenizer.from_pretrained(config.LLM_MODEL, cache_dir=config.LLM_CACHE)
        tok.padding_side = "left"
        mdl = AutoModelForCausalLM.from_pretrained(config.LLM_MODEL, cache_dir=config.LLM_CACHE,
                                                   device_map=config.device(), dtype=torch.float16).eval()
        return tok, mdl

    tok, mdl = _model("llm", load)
    out = []
    for i in range(0, len(prompts), 16):
        chunk = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)
                 for m in prompts[i:i + 16]]
        enc = tok(chunk, return_tensors="pt", padding=True).to(mdl.device)
        with _lock, torch.inference_mode():
            gen = mdl.generate(**enc, max_new_tokens=max(max_new[i:i + 16]), do_sample=False, temperature=None,
                               top_p=None, top_k=None, pad_token_id=tok.pad_token_id or tok.eos_token_id)
        out.extend((_clean(t) if clean else t.strip()) for t in tok.batch_decode(gen[:, enc["input_ids"].shape[1]:],
                                                       skip_special_tokens=True))
    return out


def _problem(out, budget, tgt, names=None):
    """Why an LLM translation should be redone before any audio is made, or None."""
    import re
    if not out:
        return "was empty."
    if _META.search(out):
        return "talks about the task instead of translating; reply with the spoken line only."
    if tgt in _CJK and re.search(r"[A-Za-z]", out):
        return "contains Latin letters; write every word, including names, in the target script."
    if tgt not in _CJK and re.search(_HAN, out):
        return f"contains Chinese/Japanese characters; write every word, including names, in {_LANG_NAMES.get(tgt, tgt)}."
    missing = _missing_names(out, names)
    if missing:
        return "does not use the required name spelling: " + "; ".join(f"write {a} as {b}" for a, b in missing) + "."
    if budget:
        n = config.spoken_units(out)
        if n > budget * 1.25 + 1:
            return f"is {n} units long, over the limit of {budget}; condense it."
        if budget >= 12 and n < budget * 0.45:
            return f"is only {n} units long and drops meaning; use more of the {budget} available."
    return None


def _translate_llm(texts, src, tgt, context=None):
    ctxs = [dict(c or {}) for c in (context or [None] * len(texts))]
    max_new = [min(400, 3 * (c.get("budget") or 60) + 40) for c in ctxs]
    out = _llm_generate([_messages(t, src, tgt, c) for t, c in zip(texts, ctxs)], max_new)
    out = [_apply_names(o, c.get("names")) for o, c in zip(out, ctxs)]
    # Up to two self-correction passes (overlong, far too short, wrong script, name spelling, notes instead of
    # a translation) before any audio is made; only lines that still have a problem go round again.
    for _ in range(2):
        redo = [i for i, (o, c) in enumerate(zip(out, ctxs)) if _problem(o, c.get("budget"), tgt, c.get("names"))]
        if not redo:
            break
        for i in redo:
            ctxs[i].update(fix=_problem(out[i], ctxs[i].get("budget"), tgt, ctxs[i].get("names")), fix_text=out[i])
            if _META.search(out[i]) or _wrong_script(out[i], tgt):
                # Source-language neighbours get copied into the output (or discussed); retry without them.
                ctxs[i].update(prev=[], next=[])
        again = _llm_generate([_messages(texts[i], src, tgt, ctxs[i]) for i in redo], [max_new[i] for i in redo])
        for i, o in zip(redo, again):
            o = _apply_names(o, ctxs[i].get("names"))
            c = ctxs[i]
            if o and _badness(o, c.get("budget"), tgt, c.get("names")) < _badness(out[i], c.get("budget"), tgt,
                                                                                c.get("names")):
                out[i] = o
    return out


def _wrong_script(out, tgt):
    import re
    return bool(re.search(r"[A-Za-z]", out) if tgt in _CJK else re.search(_HAN, out))


def _badness(out, budget, tgt, names=None):
    """Ranks two candidate translations: empty, then Latin letters in a CJK target, then glossary names
    not spelled as agreed, then length overshoot."""
    import re
    if not out:
        return (1, 0, 0, 0, 0)
    latin = len(re.findall(r"[A-Za-z]", out) if tgt in _CJK else re.findall(_HAN, out))
    return (0, bool(_META.search(out)), latin, len(_missing_names(out, names)),
            max(0, config.spoken_units(out) - budget) if budget else 0)


def _chat(prompts, max_new, provider=None):
    """Free-form completions (unlike translate(), keeps every output line)."""
    import json
    import urllib.request
    provider = provider or config.TRANSLATION_PROVIDER
    if provider == "llm":
        return _llm_generate(prompts, max_new, clean=False)
    if provider != "ollama":
        raise RuntimeError(f"name suggestions need an LLM translation provider, not {provider}")
    out = []
    for m in prompts:
        req = urllib.request.Request(f"{config.OLLAMA_URL}/api/chat", method="POST",
                                     data=json.dumps({"model": config.OLLAMA_MODEL, "stream": False, "messages": m,
                                                      "options": {"temperature": 0}}).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            out.append(json.loads(r.read())["message"]["content"].strip())
    return out


def suggest_names(texts, src, tgt, known=(), provider=None):
    """Draft glossary: people's names in the transcript with a target-language rendering each.
    Whole names are rendered with the sentence they occur in (single words out of context come out
    badly with a small model), so every suggestion carries that sentence for the reviewer.
    Returns [{source, target, example}]; target is "" when the model gave nothing usable."""
    import re
    src_name, tgt_name = _LANG_NAMES.get(src, src), _LANG_NAMES.get(tgt, tgt)

    def has(text, name):
        return re.search(rf"(?<![A-Za-z]){re.escape(name)}(?![A-Za-z])", text, re.I)

    # 1. Find the names, in chunks of transcript.
    chunks = [texts[i:i + 40] for i in range(0, len(texts), 40)]
    asks = [[{"role": "system", "content":
              f"You find people's names in {src_name} TV transcripts. The transcript comes from speech "
              f"recognition, so names may be lower-case or misspelled. List every person's name - people only, "
              f"not places, countries, teams or events. Use the full name (first and last) whenever the "
              f"transcript gives it, spelled exactly as in the transcript. One name per line, no numbering, "
              f"no other text. Write NONE if there are no names."},
             {"role": "user", "content": "\n".join(c)}] for c in chunks]
    found = []
    for reply in _chat(asks, [300] * len(asks), provider):
        for line in reply.splitlines():
            name = re.sub(r"^[\s\-*\d.)]+", "", line).strip(" .,;:\"'")
            if (name and name.upper() != "NONE" and len(name) <= 40 and re.fullmatch(r"[^\W\d_][\w' .-]*", name)
                    and any(has(t, name) for t in texts)):
                name = re.sub(r"^(coach|mr|mrs|ms|miss|dr|sir) ", "", name, flags=re.I)
                found.append(" ".join(w[:1].upper() + w[1:] for w in name.split()))
    # Whole names first; a single word already covered by a whole name is not suggested again.
    names, seen = [], {k.lower() for k in known}
    for name in sorted(set(found), key=lambda n: (-len(n.split()), n)):
        parts = {w.lower() for n in names + list(known) for w in n.split()}
        if name.lower() in seen or (" " not in name and name.lower() in parts):
            continue
        seen.add(name.lower())
        names.append(name)

    # 2. Render each whole name inside the sentence it occurs in.
    script = (f"Write it in {tgt_name} characters only, as {tgt_name} news media would: transliterate "
              f"every part by sound, keep the original order (given name first), and separate the parts "
              f"with ·." + (" For example Michael Phelps = 迈克尔·菲尔普斯, Emma McKeon = 艾玛·麦基翁, "
                           "Kyle Chalmers = 凯尔·查默斯." if tgt == "zh" else "") if tgt in _CJK
              else f"Write it as {tgt_name} news media would.")
    examples = [next(t for t in texts if has(t, n)) for n in names]
    asks = [[{"role": "system", "content": f"You are a broadcast translator. {script} Reply with the name only."},
             {"role": "user", "content": f"Sentence: {ex}\nHow is the person's name \"{n}\" in this sentence "
                                         f"written in {tgt_name}?"}] for n, ex in zip(names, examples)]
    out = []
    for n, ex, reply in zip(names, examples, _chat(asks, [24] * len(asks), provider) if asks else []):
        r = _clean(reply).strip("。.，, ")
        parts = [x for x in re.split(r"[·・•‧\s]+", r) if x]
        if (not r or len(r) > 4 * len(n) or (tgt in _CJK and (re.search(r"[A-Za-z]", r)
                                                               or len(parts) != len(n.split())))):
            r = ""  # e.g. a dropped given name: leave it blank for the reviewer rather than half right
        out.append({"source": n, "target": r, "example": ex})
    return out


def _translate_ollama(texts, src, tgt, context=None):
    """Same length-aware prompt, served by an Ollama instance."""
    import json
    import urllib.request
    out = []
    for i, text in enumerate(texts):
        ctx = (context or [None] * len(texts))[i] or {}
        req = urllib.request.Request(f"{config.OLLAMA_URL}/api/chat", method="POST",
                                     data=json.dumps({"model": config.OLLAMA_MODEL, "stream": False,
                                                      "messages": _messages(text, src, tgt, ctx),
                                                      "options": {"temperature": 0}}).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            out.append(_clean(json.loads(r.read())["message"]["content"]))
    return out


# ---------------------------------------------------------------- voice-cloned speech
def tts(text, language, ref_path, out_path, exaggeration=0.5, cfg_weight=0.5, temperature=0.8):
    """Speak `text` in `language` with the voice (and delivery) of `ref_path`."""
    import torch
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    model = _model("chatterbox", lambda: ChatterboxMultilingualTTS.from_pretrained(device=config.device()))
    with _lock, torch.inference_mode():
        wav = model.generate(text, language_id=language, audio_prompt_path=str(ref_path),
                             exaggeration=exaggeration, cfg_weight=cfg_weight, temperature=temperature)
    wav = wav.squeeze(0).cpu().numpy()
    sf.write(str(out_path), wav, model.sr)
    return {"duration": len(wav) / model.sr, "sample_rate": model.sr}


_cond_cache: "dict" = {}


def tts_batch(items):
    """Generate several lines in one decode loop. items: [{text, language, ref, out, exaggeration,
    cfg_weight, temperature}]; writes each `out` wav. Falls back to one-by-one on failure."""
    import torch
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    from .tts_batch import generate_batch

    model = _model("chatterbox", lambda: ChatterboxMultilingualTTS.from_pretrained(device=config.device()))
    if len(_cond_cache) > 256:
        _cond_cache.clear()
    try:
        if config.TTS_IMPL != "batch":
            raise RuntimeError("stock TTS selected")
        with _lock:
            wavs = generate_batch(model, [{"temperature": 0.8, **it} for it in items], cond_cache=_cond_cache)
    except Exception as e:
        if config.TTS_IMPL == "batch":
            print(f"batched TTS failed ({type(e).__name__}: {e}); falling back to one line at a time", flush=True)
        return [tts(it["text"], it["language"], it["ref"], it["out"], it["exaggeration"], it["cfg_weight"],
                    it.get("temperature", 0.8)) for it in items]
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    out = []
    for it, wav in zip(items, wavs):
        sf.write(str(it["out"]), wav, model.sr)
        out.append({"duration": len(wav) / model.sr, "sample_rate": model.sr})
    return out
