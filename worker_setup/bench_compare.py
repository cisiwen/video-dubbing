"""Stock one-line-at-a-time vs batched Chatterbox on real lines (run on the GPU worker).

    python bench_compare.py <dir with items.json + refs> [batch sizes, default 8,16]

items.json: [{"id", "text", "language", "ref", "exaggeration", "cfg_weight"}] (ref relative to dir).
Writes <dir>/out_<mode>/<id>.wav and prints wall time and real-time factor per mode.
"""
import json
import sys
import time
from pathlib import Path

import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chatterbox.mtl_tts import ChatterboxMultilingualTTS  # noqa: E402

from dubbing.tts_batch import generate_batch  # noqa: E402

root = Path(sys.argv[1])
sizes = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "8,16").split(",")]
items = json.loads((root / "items.json").read_text(encoding="utf-8"))
for it in items:
    it["ref"] = str(root / it["ref"])
    it["temperature"] = 0.8
    it.setdefault("max_tokens", 1000)

model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")


def save(mode, wavs):
    out = root / f"out_{mode}"
    out.mkdir(exist_ok=True)
    total = 0.0
    for it, w in zip(items, wavs):
        sf.write(str(out / f"{it['id']}.wav"), w, model.sr)
        total += len(w) / model.sr
    return total


def report(mode, secs, audio):
    print(f"{mode:10s} lines={len(items)} wall={secs:6.1f}s audio={audio:6.1f}s RTF={secs / audio:.2f} "
          f"per-line={secs / len(items):.2f}s", flush=True)


# warm-up so neither mode pays for CUDA init
model.generate(items[0]["text"], language_id=items[0]["language"], audio_prompt_path=items[0]["ref"])

for b in sizes:
    torch.cuda.synchronize()
    t = time.time()
    order = sorted(range(len(items)), key=lambda i: len(items[i]["text"]))
    wavs = [None] * len(items)
    cache = {}
    for k in range(0, len(order), b):
        idx = order[k:k + b]
        for i, w in zip(idx, generate_batch(model, [items[i] for i in idx], cond_cache=cache)):
            wavs[i] = w
    torch.cuda.synchronize()
    report(f"batch{b}", time.time() - t, save(f"batch{b}", wavs))

t = time.time()
wavs = []
for it in items:
    w = model.generate(it["text"], language_id=it["language"], audio_prompt_path=it["ref"],
                       exaggeration=it["exaggeration"], cfg_weight=it["cfg_weight"])
    wavs.append(w.squeeze(0).numpy())
torch.cuda.synchronize()
report("stock", time.time() - t, save("stock", wavs))
