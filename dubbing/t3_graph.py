"""Static-KV-cache T3 (Llama) decoder with the per-token step captured as one CUDA graph.

Profiling on the RTX 5070 / Windows worker showed a decode step = ~10 ms of GPU work but ~90 ms of CPU
time launching ~1100 small kernels (WDDM launches cost ~40 µs each). Replaying a captured graph turns
those launches into one. Math mirrors HF LlamaModel (RMSNorm, llama3 RoPE via the model's own
rotary_emb, SDPA attention, SiLU MLP) using the model's own weights; nothing is copied.
"""
import torch
import torch.nn.functional as F


def _rms(norm, x):
    v = x.pow(2).mean(-1, keepdim=True)
    return norm.weight * (x * torch.rsqrt(v + norm.variance_epsilon))


def _rot(x, cos, sin):
    h = x.shape[-1] // 2
    return x * cos + torch.cat((-x[..., h:], x[..., :h]), dim=-1) * sin


class StaticT3:
    def __init__(self, tfmr, head, rows, max_len, spy_heads):
        """tfmr: LlamaModel; head: speech logits Linear; rows: batch rows (2 per line for CFG);
        spy_heads: [(layer, head)] whose attention weights are exported for the alignment guard."""
        p = next(tfmr.parameters())
        cfg = tfmr.config
        self.tfmr, self.head, self.R, self.L = tfmr, head, rows, max_len
        self.H, self.D = cfg.num_attention_heads, cfg.head_dim
        self.spy = {layer: (i, h) for i, (layer, h) in enumerate(spy_heads)}
        n = len(tfmr.layers)
        self.k = torch.zeros(n, rows, self.H, max_len, self.D, device=p.device, dtype=p.dtype)
        self.v = torch.zeros_like(self.k)
        self.mask = torch.full((rows, 1, 1, max_len), float("-inf"), device=p.device, dtype=p.dtype)
        # decode-step inputs/outputs (graph-static buffers)
        self.in_emb = torch.zeros(rows, 1, cfg.hidden_size, device=p.device, dtype=p.dtype)
        self.in_pos = torch.zeros(rows, 1, device=p.device, dtype=torch.long)
        self.col = torch.zeros(1, device=p.device, dtype=torch.long)
        self.logits = None
        self.spy_out = torch.zeros(len(spy_heads), rows, max_len, device=p.device, dtype=p.dtype)
        self.graph = None

    # ------------------------------------------------------------ shared layer math
    def _layers(self, x, pos, attend):
        """x: (R, T, C). attend(l, q, k, v) -> attention output (R, H, T, D)."""
        cos, sin = self.tfmr.rotary_emb(x, pos)
        cos, sin = cos[:, None], sin[:, None]
        R, T, _ = x.shape
        for li, layer in enumerate(self.tfmr.layers):
            a = layer.self_attn
            h = _rms(layer.input_layernorm, x)
            q = a.q_proj(h).view(R, T, self.H, self.D).transpose(1, 2)
            k = a.k_proj(h).view(R, T, self.H, self.D).transpose(1, 2)
            v = a.v_proj(h).view(R, T, self.H, self.D).transpose(1, 2)
            q, k = _rot(q, cos, sin), _rot(k, cos, sin)
            o = attend(li, q, k, v)
            x = x + a.o_proj(o.transpose(1, 2).reshape(R, T, -1))
            h = _rms(layer.post_attention_layernorm, x)
            m = layer.mlp
            x = x + m.down_proj(m.act_fn(m.gate_proj(h)) * m.up_proj(h))
        return _rms(self.tfmr.norm, x)

    # ------------------------------------------------------------ prefill (eager, variable length)
    @torch.inference_mode()
    def prefill(self, x, pad_mask, pos):
        """x: (R, T, C) left-padded prompt; pad_mask: (R, T) 1=real. Returns (logits_last, spy[(R, T, T)])."""
        R, T, _ = x.shape
        keep = pad_mask.bool()
        allowed = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()[None] & keep[:, None, :]
        allowed = allowed | ~allowed.any(-1, keepdim=True)  # padding queries: avoid all-masked rows
        add = torch.zeros(R, 1, T, T, device=x.device, dtype=x.dtype).masked_fill(~allowed[:, None], float("-inf"))
        spy = {}

        def attend(li, q, k, v):
            self.k[li, :, :, :T] = k
            self.v[li, :, :, :T] = v
            if li in self.spy:
                i, hd = self.spy[li]
                w = torch.softmax(q[:, hd] @ k[:, hd].transpose(-1, -2) * self.D ** -0.5 + add[:, 0], -1)
                spy[i] = w
            return F.scaled_dot_product_attention(q, k, v, attn_mask=add)

        h = self._layers(x, pos, attend)
        self.mask.fill_(float("-inf"))
        self.mask[:, 0, 0, :T] = torch.where(keep, 0.0, float("-inf")).to(self.mask.dtype)
        self.col.fill_(T)
        self.in_pos.copy_(pos[:, -1:] + 1)
        return self.head(h[:, -1]), [spy[i] for i in range(len(self.spy))]

    # ------------------------------------------------------------ decode step (graph-captured)
    def _step(self):
        self.mask.index_fill_(3, self.col, 0.0)

        def attend(li, q, k, v):
            self.k[li].index_copy_(2, self.col, k)
            self.v[li].index_copy_(2, self.col, v)
            K, V = self.k[li], self.v[li]
            if li in self.spy:
                i, hd = self.spy[li]
                w = torch.softmax(q[:, hd] @ K[:, hd].transpose(-1, -2) * self.D ** -0.5 + self.mask[:, 0], -1)
                self.spy_out[i].copy_(w[:, 0])
            return F.scaled_dot_product_attention(q, K, V, attn_mask=self.mask)

        h = self._layers(self.in_emb, self.in_pos, attend)
        out = self.head(h[:, -1])
        if self.logits is None:
            self.logits = out
        else:
            self.logits.copy_(out)

    @torch.inference_mode()
    def step(self, emb):
        """emb: (R, 1, C) next-token embeddings. Returns logits (R, V) and spy (n, R, L) — graph outputs,
        valid until the next call."""
        self.in_emb.copy_(emb)
        if self.graph is None:
            self._capture()
        self.graph.replay()
        self.col += 1
        self.in_pos += 1
        return self.logits, self.spy_out

    def _capture(self):
        # Warm up on a side stream (allocator + cuBLAS handles), restoring state, then capture.
        saved = (self.k.clone(), self.v.clone(), self.mask.clone())
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                self._step()
        torch.cuda.current_stream().wait_stream(s)
        self.k.copy_(saved[0]); self.v.copy_(saved[1]); self.mask.copy_(saved[2])
        del saved
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._step()
