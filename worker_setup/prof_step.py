"""Where does one T3 decode step spend its time? (GPU worker)"""
import time
import torch
from torch.profiler import ProfilerActivity, profile
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

m = ChatterboxMultilingualTTS.from_pretrained(device="cuda")
tfmr = m.t3.tfmr
print("attn:", tfmr.config._attn_implementation, "dtype:", next(tfmr.parameters()).dtype,
      "layers:", tfmr.config.num_hidden_layers, "hidden:", tfmr.config.hidden_size, flush=True)
x = torch.randn(8, 300, tfmr.config.hidden_size, device="cuda")
with torch.inference_mode():
    out = tfmr(inputs_embeds=x, use_cache=True)
    past = out.past_key_values
    tok = torch.randn(8, 1, tfmr.config.hidden_size, device="cuda")
    for _ in range(5):
        out = tfmr(inputs_embeds=tok, past_key_values=past, use_cache=True); past = out.past_key_values
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(20):
        out = tfmr(inputs_embeds=tok, past_key_values=past, use_cache=True); past = out.past_key_values
    t_launch = (time.perf_counter() - t) / 20 * 1000
    torch.cuda.synchronize()
    t_total = (time.perf_counter() - t) / 20 * 1000
    print(f"per step: CPU-side (launch) {t_launch:.1f} ms, with GPU sync {t_total:.1f} ms", flush=True)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(10):
            out = tfmr(inputs_embeds=tok, past_key_values=past, use_cache=True); past = out.past_key_values
        torch.cuda.synchronize()
print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=12, max_name_column_width=45))
print(prof.key_averages().table(sort_by="self_device_time_total", row_limit=6, max_name_column_width=45))
import platform, os
print("CPU:", platform.processor(), "cores:", os.cpu_count())
