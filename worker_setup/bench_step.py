"""Micro-benchmark one T3 decode step vs batch size and attention settings (GPU worker)."""
import time

import torch

from chatterbox.mtl_tts import ChatterboxMultilingualTTS

model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")
tfmr = model.t3.tfmr
dim = model.t3.cfg.hidden_size
CTX = 300  # conditioning + text tokens already in the KV cache


@torch.inference_mode()
def bench(batch, attentions, steps=60):
    x = torch.randn(batch, CTX, dim, device="cuda")
    out = tfmr(inputs_embeds=x, use_cache=True, output_attentions=attentions)
    past = out.past_key_values
    tok = torch.randn(batch, 1, dim, device="cuda")
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(steps):
        out = tfmr(inputs_embeds=tok, past_key_values=past, use_cache=True,
                   output_attentions=attentions, output_hidden_states=attentions)
        past = out.past_key_values
        out.last_hidden_state[0, 0, 0].item()  # the real loop syncs every step too
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / steps * 1000


print("attn impl:", tfmr.config._attn_implementation)
for att in (True, False):
    for b in (2, 4, 8, 16, 32):
        ms = bench(b, att)
        print(f"output_attentions={att!s:5} batch={b:2d} (={b // 2} lines w/ CFG): {ms:6.1f} ms/step"
              f"  -> {b // 2 * 1000 / ms:6.1f} line-tokens/s", flush=True)
if tfmr.config._attn_implementation == "eager":
    tfmr.config._attn_implementation = "sdpa"
    for b in (2, 16):
        print(f"sdpa batch={b}: {bench(b, False):.1f} ms/step", flush=True)
