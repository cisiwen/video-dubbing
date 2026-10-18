"""ffmpeg / numpy audio helpers used by the orchestrator."""
import json
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf


def run(*args):
    p = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *map(str, args)],
                       capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError(f"ffmpeg failed: {p.stderr.strip()[-2000:]}")


def probe(path):
    p = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams",
                        str(path)], capture_output=True, text=True, check=True)
    info = json.loads(p.stdout)
    video = next((s for s in info["streams"] if s["codec_type"] == "video"), None)
    return {"duration": float(info["format"]["duration"]),
            "has_audio": any(s["codec_type"] == "audio" for s in info["streams"]),
            "width": video and video.get("width"), "height": video and video.get("height")}


def extract(video, out_wav, sr=44100, channels=2):
    run("-i", video, "-vn", "-ac", channels, "-ar", sr, "-c:a", "pcm_s16le", out_wav)


def to_mono16k(src, out_wav):
    run("-i", src, "-ac", 1, "-ar", 16000, "-c:a", "pcm_s16le", out_wav)


def make_preview(video, out_mp4):
    """Small H.264 copy the browser can always play (hardware encoder on macOS when available)."""
    common = ["-i", video, "-vf", "scale=-2:'min(540,ih)'", "-c:a", "aac", "-b:a", "128k",
              "-movflags", "+faststart"]
    try:
        run(*common, "-c:v", "h264_videotoolbox", "-b:v", "1500k", out_mp4)
    except RuntimeError:
        run(*common, "-c:v", "libx264", "-preset", "veryfast", "-crf", "28", out_mp4)


def cut(src, start, end, out_wav, sr=None):
    data, file_sr = sf.read(str(src), dtype="float32", always_2d=True)
    seg = data[int(start * file_sr):int(end * file_sr)].mean(axis=1)
    if sr and sr != file_sr:
        import librosa
        seg = librosa.resample(seg, orig_sr=file_sr, target_sr=sr)
        file_sr = sr
    sf.write(str(out_wav), seg, file_sr)
    return len(seg) / file_sr


def concat(clips, out_wav, gap=0.25):
    parts, sr = [], None
    for c in clips:
        d, sr = sf.read(str(c), dtype="float32")
        parts += [d, np.zeros(int(gap * sr), dtype="float32")]
    sf.write(str(out_wav), np.concatenate(parts[:-1]), sr)


def trim_silence(path, top_db=40):
    import librosa
    d, sr = sf.read(str(path), dtype="float32")
    if d.ndim > 1:
        d = d.mean(axis=1)
    _, (a, b) = librosa.effects.trim(d, top_db=top_db)
    pad = int(0.03 * sr)
    d = d[max(0, a - pad):b + pad]
    sf.write(str(path), d, sr)
    return len(d) / sr


def duration(path):
    return sf.info(str(path)).duration


def stretch(src, out_wav, factor):
    """Change tempo without changing pitch. factor > 1 = faster/shorter."""
    run("-i", src, "-filter:a", f"atempo={factor:.4f}", out_wav)


def mix(background, placements, out_wav, total_dur, dub_gain=1.0):
    """Overlay [(wav_path, start_sec)] onto the background stem."""
    import librosa
    bg, sr = sf.read(str(background), dtype="float32", always_2d=True)
    n = max(len(bg), int(total_dur * sr))
    out = np.zeros((n, bg.shape[1]), dtype="float32")
    out[:len(bg)] = bg
    for path, start in placements:
        d, dsr = sf.read(str(path), dtype="float32")
        if d.ndim > 1:
            d = d.mean(axis=1)
        if dsr != sr:
            d = librosa.resample(d, orig_sr=dsr, target_sr=sr)
        i = int(start * sr)
        j = min(n, i + len(d))
        out[i:j] += dub_gain * d[:j - i, None]
    peak = np.abs(out).max()
    if peak > 0.99:
        out *= 0.99 / peak
    sf.write(str(out_wav), out, sr)


ISO639_2 = {"ar": "ara", "da": "dan", "de": "deu", "el": "ell", "en": "eng", "es": "spa", "fi": "fin",
            "fr": "fra", "he": "heb", "hi": "hin", "it": "ita", "ja": "jpn", "ko": "kor", "ms": "msa",
            "nl": "nld", "no": "nor", "pl": "pol", "pt": "por", "ru": "rus", "sv": "swe", "sw": "swa",
            "tr": "tur", "zh": "zho"}


def mux(video, dub_audio, out_mp4, language="und", keep_original=True):
    language = ISO639_2.get(language, language)  # MP4 drops 2-letter codes
    args = ["-i", video, "-i", dub_audio, "-map", "0:v:0", "-map", "1:a:0"]
    if keep_original:
        args += ["-map", "0:a:0?"]
    args += ["-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
             "-metadata:s:a:0", f"language={language}", "-metadata:s:a:0", "title=Dubbed",
             "-metadata:s:a:0", "handler_name=Dubbed", "-disposition:a:0", "default"]
    if keep_original:
        args += ["-metadata:s:a:1", "title=Original", "-metadata:s:a:1", "handler_name=Original",
                 "-disposition:a:1", "0"]
    run(*args, "-shortest", out_mp4)


def srt(segments, key, out_path):
    def ts(t):
        ms = int(round(t * 1000))
        return f"{ms // 3600000:02}:{ms // 60000 % 60:02}:{ms // 1000 % 60:02},{ms % 1000:03}"
    lines = []
    for i, s in enumerate((s for s in segments if s.get(key)), 1):
        lines += [str(i), f"{ts(s['start'])} --> {ts(s['end'])}", s[key], ""]
    Path(out_path).write_text("\n".join(lines), encoding="utf-8")
