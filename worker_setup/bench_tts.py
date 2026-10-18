"""Profile Chatterbox Multilingual on the GPU worker.

    .venv\\Scripts\\python.exe worker_setup\\bench_tts.py <ref.wav> [lang]

Times each phase (conditioning, T3 token generation, S3Gen vocoder, watermark) under a few
speed settings, and prints the real-time factor (generation seconds / audio seconds).
"""
import sys
import time

import torch

from chatterbox.mtl_tts import ChatterboxMultilingualTTS

REF = sys.argv[1]
LANG = sys.argv[2] if len(sys.argv) > 2 else "fr"
LINES = [
    "Bonsoir et bon retour ! Ce soir, un invité très spécial nous rejoint.",
    "Merci beaucoup de m'avoir accueilli, c'est un vrai plaisir d'être ici.",
    "Maintenant, je dois vous demander, est-il vrai que vous avez grimpé le mont Everest sans oxygène ?",
    "Oui, c'est tout à fait vrai, je n'arrive toujours pas à croire que j'ai atteint le sommet.",
    "Quelle histoire merveilleuse, nous serons de retour après cette courte pause.",
]


def sync():
    torch.cuda.synchronize()


model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")
timings = {}


def timed(obj, name, key):
    fn = getattr(obj, name)

    def wrap(*a, **k):
        sync()
        t = time.perf_counter()
        out = fn(*a, **k)
        sync()
        timings[key] = timings.get(key, 0) + time.perf_counter() - t
        return out
    setattr(obj, name, wrap)


timed(model, "prepare_conditionals", "cond")
timed(model.t3, "inference", "t3")
timed(model.s3gen, "inference", "s3gen")
timed(model.watermarker, "apply_watermark", "watermark")


def run(label, reuse_cond=False, autocast=None):
    timings.clear()
    audio = 0.0
    sync()
    t0 = time.perf_counter()
    for i, line in enumerate(LINES):
        ref = None if (reuse_cond and i) else REF
        ctx = torch.autocast("cuda", dtype=autocast) if autocast else torch.autocast("cuda", enabled=False)
        with ctx:
            wav = model.generate(line, language_id=LANG, audio_prompt_path=ref, exaggeration=0.5, cfg_weight=0.5)
        audio += wav.shape[-1] / model.sr
    total = time.perf_counter() - t0
    parts = "  ".join(f"{k}={v:.1f}s" for k, v in timings.items())
    print(f"{label:28s} total={total:5.1f}s audio={audio:5.1f}s RTF={total / audio:.2f}  {parts}", flush=True)


run("warmup")
run("baseline fp32")
run("reuse conditionals", reuse_cond=True)
run("bf16 autocast", autocast=torch.bfloat16)
run("fp16 autocast", autocast=torch.float16)
print("GPU peak mem GB", round(torch.cuda.max_memory_allocated() / 1e9, 2))
