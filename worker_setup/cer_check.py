"""Whisper round-trip check for generated dub lines: character error rate vs. the intended text.

    python worker_setup/cer_check.py <bench dir> out_stock out_batch8 ...
"""
import json
import re
import sys
import unicodedata
from pathlib import Path

import mlx_whisper

try:
    from zhconv import convert
except ImportError:
    def convert(s, _):
        return s

root = Path(sys.argv[1])
items = json.loads((root / "items.json").read_text())
skip = set(filter(None, __import__("os").environ.get("SKIP", "").split(",")))
items = [it for it in items if it["id"] not in skip]


def norm(s, lang):
    s = unicodedata.normalize("NFKC", s)
    if lang == "zh":
        s = convert(s, "zh-cn")
    s = re.sub(r"[\W_]+", "", s.lower())
    return s if lang in ("zh", "ja") else s  # character-level either way


def edit(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


for mode in sys.argv[2:]:
    errs = chars = bad = 0
    worst = []
    for it in items:
        wav = root / mode / f"{it['id']}.wav"
        hyp = mlx_whisper.transcribe(str(wav), path_or_hf_repo="mlx-community/whisper-large-v3-turbo",
                                     language=it["language"])["text"]
        ref_n, hyp_n = norm(it["text"], it["language"]), norm(hyp, it["language"])
        e = edit(ref_n, hyp_n)
        errs += e
        chars += len(ref_n)
        cer = e / max(1, len(ref_n))
        bad += cer > 0.3
        worst.append((cer, it["id"], it["text"], hyp.strip()))
    worst.sort(reverse=True)
    print(f"{mode:12s} CER={errs / chars:.3f}  lines with CER>30%: {bad}/{len(items)}")
    for cer, sid, ref, hyp in worst[:2]:
        print(f"    worst {sid} {cer:.2f}  want: {ref}  heard: {hyp}")
