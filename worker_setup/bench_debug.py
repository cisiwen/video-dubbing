"""Run one debug batch of the benchmark lines on the GPU worker: bench_debug.py <dir> <batch> [n_lines]"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from dubbing.tts_batch import generate_batch
root = Path(sys.argv[1]); b = int(sys.argv[2]); n = int(sys.argv[3]) if len(sys.argv) > 3 else b
items = json.loads((root / "items.json").read_text(encoding="utf-8"))[:n]
for it in items:
    it["ref"] = str(root / it["ref"]); it["temperature"] = 0.8
m = ChatterboxMultilingualTTS.from_pretrained(device="cuda")
t = time.time()
wavs = generate_batch(m, items[:b], debug=True)
print(f"batch {b}: {time.time() - t:.1f}s, audio {[round(len(w) / m.sr, 1) for w in wavs]}, "
      f"peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)
