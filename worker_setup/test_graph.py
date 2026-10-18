"""Parity + speed of StaticT3 (CUDA graph) vs HF LlamaModel on the GPU worker."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from dubbing.t3_graph import StaticT3
from dubbing.tts_batch import ALIGNED_HEADS, _install_spy_attention, _spy

m = ChatterboxMultilingualTTS.from_pretrained(device="cuda")
t3, tf = m.t3, m.t3.tfmr
_install_spy_attention(); tf.config._attn_implementation = "spy_sdpa"
R, T, C = 16, 120, tf.config.hidden_size
torch.manual_seed(0)
x = torch.randn(R, T, C, device="cuda") * 0.5
mask = torch.ones(R, T, dtype=torch.long, device="cuda"); mask[1::3, :17] = 0  # some left padding
pos = (mask.cumsum(-1) - 1).clamp(min=0)
steps = [torch.randn(R, 1, C, device="cuda") * 0.5 for _ in range(20)]

with torch.inference_mode():
    _spy["on"] = True
    out = tf(inputs_embeds=x, attention_mask=mask, position_ids=pos, use_cache=True)
    ref = [t3.speech_head(out.last_hidden_state[:, -1])]; ref_att = [_spy["attn"][9].clone()]
    past, mk, ps = out.past_key_values, mask, pos
    for e in steps:
        mk = torch.cat([mk, torch.ones(R, 1, dtype=mk.dtype, device="cuda")], 1); ps = ps[:, -1:] + 1
        out = tf(inputs_embeds=e, attention_mask=mk, position_ids=ps, past_key_values=past, use_cache=True)
        past = out.past_key_values
        ref.append(t3.speech_head(out.last_hidden_state[:, -1])); ref_att.append(_spy["attn"][9][:, 0].clone())
    _spy["on"] = False

    st = StaticT3(tf, t3.speech_head, R, 256, ALIGNED_HEADS)
    lg, spy = st.prefill(x, mask, pos)
    got = [lg.clone()]; got_att = [spy[2].clone()]  # ALIGNED_HEADS[2] is layer 9
    for e in steps:
        lg, so = st.step(e)
        got.append(lg.clone()); got_att.append(so[2, :, :T + len(got) - 1].clone())

real = mask.bool()
err = max((a - b).abs().max().item() for a, b in zip(ref, got))
scale = max(a.abs().max().item() for a in ref)
aerr = max((a[:, -1] if a.dim() == 3 else a).sub(b[:, -1] if b.dim() == 3 else b).abs().max().item()
           for a, b in zip(ref_att[1:], got_att[1:]))
print(f"logits max abs diff {err:.2e} (max |logit| {scale:.1f}); spy attn max diff {aerr:.2e}")
print("spy attn mean", float(got_att[-1].mean()), "rows sum", float(got_att[-1].sum(-1).mean()))
print("argmax agree:", all(torch.equal(a.argmax(-1), b.argmax(-1)) for a, b in zip(ref, got)))

with torch.inference_mode():
    e = steps[0]
    for name, fn in [("hf", lambda: tf(inputs_embeds=e, attention_mask=mk, position_ids=ps, past_key_values=past, use_cache=True)),
                     ("graph", lambda: st.step(e))]:
        for _ in range(3): fn()
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(30):
            r = fn()
            (r.last_hidden_state if hasattr(r, "last_hidden_state") else r[0])[0, 0].sum().item()
        print(f"{name:6s} {(time.perf_counter() - t) / 30 * 1000:.1f} ms/step at {R} rows", flush=True)
