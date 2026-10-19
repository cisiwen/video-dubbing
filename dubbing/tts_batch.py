"""Batched Chatterbox Multilingual generation.

Stock `ChatterboxMultilingualTTS.generate` decodes one line at a time. Each step is dominated by fixed
per-step overhead (≈ the same cost for 1 or 8 lines), so decoding several lines together is several
times faster. Upstream inference has two more problems this module avoids:
  * it registers new attention hooks on every call and never removes them, so long jobs keep slowing down;
  * it switches all 30 layers to eager attention when only the 3 alignment layers need their weights.

The per-line logic (CFG, alignment-based hallucination guard, repetition penalty, min-p/top-p sampling)
follows `T3.inference` / `AlignmentStreamAnalyzer` from chatterbox-tts 0.1.7.
"""
import logging
import time

import torch
import torch.nn.functional as F
from transformers.generation.logits_process import (MinPLogitsWarper, RepetitionPenaltyLogitsProcessor,
                                                    TopPLogitsWarper)

log = logging.getLogger(__name__)

ALIGNED_HEADS = [(12, 15), (13, 11), (9, 2)]  # (layer, head) pairs that track text position
NEG = -(2 ** 15)
KV_BUDGET_GB = float(__import__("os").environ.get("DUB_TTS_KV_GB", 3.0))  # static KV cache limit per batch

_spy = {"on": False, "attn": {}}


def _install_spy_attention():
    """Register an attention implementation that is plain SDPA except on the alignment layers, where it
    computes softmax weights explicitly and keeps the watched head for the hallucination guard."""
    if _spy.get("installed"):
        return
    _spy["installed"] = True
    from transformers import AttentionInterface, AttentionMaskInterface
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    from transformers.masking_utils import sdpa_mask
    from transformers.models.llama.modeling_llama import repeat_kv

    heads = dict(ALIGNED_HEADS)

    def spy_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kw):
        idx = getattr(module, "layer_idx", None)
        if not _spy["on"] or idx not in heads:
            return sdpa_attention_forward(module, query, key, value, attention_mask, scaling=scaling,
                                          dropout=dropout, **kw)
        k = repeat_kv(key, module.num_key_value_groups)
        v = repeat_kv(value, module.num_key_value_groups)
        scores = torch.matmul(query, k.transpose(2, 3)) * (scaling or query.shape[-1] ** -0.5)
        q_len, k_len = query.shape[-2], k.shape[-2]
        if attention_mask is not None:
            mask = attention_mask[:, :, :, :k_len]
            scores = scores.masked_fill(~mask, float("-inf")) if mask.dtype == torch.bool else scores + mask
        elif q_len > 1:
            causal = torch.ones(q_len, k_len, dtype=torch.bool, device=scores.device).tril(k_len - q_len)
            scores = scores.masked_fill(~causal, float("-inf"))
        # Left-padding rows attend to nothing -> NaN; zero them so NaN never reaches the KV cache.
        w = torch.softmax(scores.float(), dim=-1).nan_to_num(0.0).to(query.dtype)
        _spy["attn"][idx] = w[:, heads[idx]]  # (B, q_len, k_len), stays on GPU
        out = torch.matmul(w, v).transpose(1, 2).contiguous()
        return out, None

    def spy_mask(*args, **kwargs):
        mask = sdpa_mask(*args, **kwargs)
        if mask is not None and mask.dtype == torch.bool:
            # Left-padding query rows see no keys -> SDPA returns NaN that poisons the KV cache.
            # Let them see everything instead; no real token ever attends to a padding position.
            mask = mask | ~mask.any(dim=-1, keepdim=True)
        return mask

    AttentionInterface.register("spy_sdpa", spy_attention)
    AttentionMaskInterface.register("spy_sdpa", spy_mask)


class _Guard:
    """Per-line port of AlignmentStreamAnalyzer (works on CPU copies of one line's alignment rows)."""

    def __init__(self, text_slice):
        self.i, self.j = text_slice
        self.alignment = torch.zeros(0, self.j - self.i)
        self.frame = 0
        self.text_position = 0
        self.started = False
        self.complete = False
        self.completed_at = None
        self.tokens = []

    def step(self, chunk, last_token):
        """chunk: (T, S) alignment rows for this frame. Returns 'suppress_eos', 'force_eos' or None."""
        chunk = chunk.clone()
        chunk[:, self.frame + 1:] = 0
        self.alignment = torch.cat((self.alignment, chunk), dim=0)
        A = self.alignment
        T, S = A.shape
        pos = int(chunk[-1].argmax())
        if -4 < pos - self.text_position < 7:
            self.text_position = pos
        false_start = (not self.started) and (A[-2:, -2:].max() > 0.1 or A[:, :4].max() < 0.5)
        self.started = not false_start
        self.complete = self.complete or self.text_position >= S - 3
        if self.complete and self.completed_at is None:
            self.completed_at = T
        long_tail = self.complete and A[self.completed_at:, -3:].sum(dim=0).max() >= 5
        # S <= 5 (one-word lines): nothing left of the text to repeat, and max() over it would crash.
        repetition = self.complete and S > 5 and A[self.completed_at:, :-5].max(dim=1).values.sum() > 5
        if last_token is not None:
            self.tokens = (self.tokens + [last_token])[-8:]
        token_rep = len(self.tokens) >= 3 and len(set(self.tokens[-2:])) == 1
        self.frame += 1
        if long_tail or repetition or token_rep:
            return "force_eos"
        if pos < S - 3 and S > 5:
            return "suppress_eos"
        return None


def _left_pad(seqs):
    """Stack (2, L_i, D) tensors into (2B, Lmax, D) with left padding; returns embeds, mask, pads."""
    lmax = max(s.shape[1] for s in seqs)
    out, mask, pads = [], [], []
    for s in seqs:
        p = lmax - s.shape[1]
        out.append(F.pad(s, (0, 0, p, 0)))
        m = torch.ones(s.shape[0], lmax, dtype=torch.long, device=s.device)
        m[:, :p] = 0
        mask.append(m)
        pads.append(p)
    return torch.cat(out), torch.cat(mask), pads


@torch.inference_mode()
def generate_batch(model, items, max_new_tokens=1000, repetition_penalty=2.0, min_p=0.05, top_p=1.0,
                   cond_cache=None, debug=False, use_graph=True, stats=None):
    """items: [{text, language, ref, exaggeration, cfg_weight, temperature, max_tokens?, seed?, group?, slot?}]
    -> 1-D numpy wavs. A row with a seed samples from its own RNG, so a take is the same whatever it is batched
    with (up to float noise). Takes of one line share a `group` and carry its `slot` [seg_len, room]: only the
    take whose length (from its token count) best fits the slot is vocoded; the others get wav None.
    use_graph: decode with the CUDA-graph static decoder (t3_graph.py) instead of HF LlamaModel.
    stats: optional list, extended with {tokens, capped, est_duration} per item (capped = cut at max_tokens,
    so speech is truncated; est_duration = trimmed length estimated from the token count)."""
    tfmr = model.t3.tfmr
    _install_spy_attention()
    saved_impl = tfmr.config._attn_implementation
    tfmr.config._attn_implementation = "spy_sdpa"
    try:
        toks, conds, st = [None] * len(items), [None] * len(items), [None] * len(items)
        for idx in _kv_chunks(model, items, max_new_tokens, use_graph):
            res = _generate(model, [items[i] for i in idx], max_new_tokens, repetition_penalty, min_p, top_p,
                            cond_cache, debug, use_graph)
            for i, (t, c, s) in zip(idx, zip(*res)):
                toks[i], conds[i], st[i] = t, c, s
        keep = _pick_takes(items, st)
        t0 = time.perf_counter()
        wavs = [_vocode(model, toks[i], conds[i]) if keep[i] else None for i in range(len(items))]
        print(f"TTS vocoded {sum(keep)}/{len(items)} takes in {time.perf_counter() - t0:.1f}s", flush=True)
        if stats is not None:
            stats.extend(st)
        return wavs
    finally:
        # Upstream ChatterboxMultilingualTTS.generate (our fallback) needs the original attention setting.
        tfmr.config._attn_implementation = saved_impl


def _pick_takes(items, stats):
    """Which items to vocode: every ungrouped item, and per group the take that best fits its slot."""
    from .fit import TOKEN_SEC, TRIM_SEC, take_score
    best = {}
    for i, (it, st) in enumerate(zip(items, stats)):
        st["est_duration"] = round(max(0.0, st["tokens"] * TOKEN_SEC - TRIM_SEC), 3)
        if it.get("group") is None or not it.get("slot"):
            continue
        score = take_score({"duration": st["est_duration"], "capped": st["capped"]}, *it["slot"])
        if it["group"] not in best or score < best[it["group"]][0]:
            best[it["group"]] = (score, i)
    chosen = {i for _, i in best.values()}
    return [it.get("group") is None or not it.get("slot") or i in chosen for i, it in enumerate(items)]


def _vocode(model, toks, cond):
    from chatterbox.models.s3tokenizer import SPEECH_VOCAB_SIZE
    # Upstream runs drop_invalid_tokens, then keeps ids < 6561. Filtering alone does both (it is cut at the
    # first EOS) and also drops any sampled special/unused id, which would crash S3Gen.
    toks = toks[toks < SPEECH_VOCAB_SIZE].to(model.device)
    if toks.numel() == 0:
        return torch.zeros(int(model.sr * 0.1)).numpy()
    wav, _ = model.s3gen.inference(speech_tokens=toks, ref_dict=cond.gen)
    wav = wav.squeeze(0).detach().cpu().numpy()
    return model.watermarker.apply_watermark(wav, sample_rate=model.sr)


def _kv_chunks(model, items, max_new_tokens, use_graph):
    """Split item indices into batches whose static KV cache fits DUB_TTS_KV_GB (Windows would silently spill
    an oversized cache to system RAM and crawl). Items are packed greedily in order of decode budget, so
    short lines share a large batch instead of being sized (and padded) for the longest line."""
    if not (use_graph and torch.cuda.is_available() and str(model.device).startswith("cuda")) or len(items) < 2:
        return [list(range(len(items)))]
    from chatterbox.mtl_tts import punc_norm

    cfgm = model.t3.tfmr.config
    per_tok = cfgm.num_hidden_layers * 2 * 2 * cfgm.num_attention_heads * cfgm.head_dim * 4 / 1e9
    lens = {}

    def rows_len(i):
        # Same L as _generate: conditioning (34 with the perceiver; +16 slack) + start/stop text tokens + BOS
        # + text tokens (zh ≈ 4.5 per character) + decode budget, rounded up to 256.
        if i not in lens:
            it = items[i]
            tt = model.tokenizer.text_to_tokens(punc_norm(it["text"]), language_id=it["language"].lower())
            n_text = tt.shape[-1]
            cap = min(max_new_tokens, int(it.get("max_tokens") or max_new_tokens))
            lens[i] = -(-(50 + n_text + 3 + cap + 2) // 256) * 256
        return lens[i]

    chunks, cur, cur_len = [], [], 0
    for i in sorted(range(len(items)), key=rows_len):
        L = max(cur_len, rows_len(i))
        if cur and (len(cur) + 1) * L * per_tok > KV_BUDGET_GB:
            chunks.append(cur)
            cur, L = [], rows_len(i)
        cur.append(i)
        cur_len = L
    chunks.append(cur)
    if len(chunks) > 1:
        print(f"TTS batch of {len(items)} split into {[len(c) for c in chunks]} for the KV budget", flush=True)
    return chunks


def _generate(model, items, max_new_tokens, repetition_penalty, min_p, top_p, cond_cache, debug, use_graph):
    from chatterbox.mtl_tts import punc_norm

    t3 = model.t3
    tfmr = t3.tfmr
    dev = model.device

    conds, embeds, slices = [], [], []
    for it in items:
        key = (str(it["ref"]), round(float(it["exaggeration"]), 3))
        if cond_cache is not None and key in cond_cache:
            c = cond_cache[key]
        else:
            model.prepare_conditionals(str(it["ref"]), exaggeration=it["exaggeration"])
            c = model.conds
            if cond_cache is not None:
                cond_cache[key] = c
        conds.append(c)
        tt = model.tokenizer.text_to_tokens(punc_norm(it["text"]), language_id=it["language"].lower()).to(dev)
        tt = torch.cat([tt, tt], dim=0)
        tt = F.pad(F.pad(tt, (1, 0), value=t3.hp.start_text_token), (0, 1), value=t3.hp.stop_text_token)
        bos = torch.full((2, 1), t3.hp.start_speech_token, dtype=torch.long, device=dev)
        emb, len_cond = t3.prepare_input_embeds(t3_cond=c.t3, text_tokens=tt, speech_tokens=bos,
                                                cfg_weight=it["cfg_weight"])
        bos_emb = t3.speech_emb(bos[:1]) + t3.speech_pos_emb.get_fixed_embedding(0)
        embeds.append(torch.cat([emb, torch.cat([bos_emb, bos_emb])], dim=1))
        slices.append((len_cond, len_cond + tt.shape[-1]))

    B = len(items)
    x, mask, pads = _left_pad(embeds)
    pos = (mask.cumsum(-1) - 1).clamp(min=0)
    cfg = torch.tensor([it["cfg_weight"] for it in items], device=dev)[:, None]
    temp = torch.tensor([it["temperature"] for it in items], device=dev)[:, None]
    guards = [_Guard(s) for s in slices]
    eos = t3.hp.stop_speech_token
    rep = RepetitionPenaltyLogitsProcessor(penalty=float(repetition_penalty))
    minp, topp = MinPLogitsWarper(min_p=min_p), TopPLogitsWarper(top_p=top_p)
    cond_rows = torch.arange(0, 2 * B, 2, device=dev)
    gens = [None] * B
    for b, it in enumerate(items):
        if it.get("seed") is not None:
            gens[b] = torch.Generator(device=dev)
            gens[b].manual_seed(int(it["seed"]))

    caps = [min(max_new_tokens, int(it.get("max_tokens") or max_new_tokens)) for it in items]
    use_graph = use_graph and dev != "cpu" and torch.cuda.is_available() and str(dev).startswith("cuda")

    if use_graph:
        from .t3_graph import StaticT3
        T = x.shape[1]
        L = -(-(T + max(caps) + 2) // 256) * 256
        cfgm = tfmr.config
        kv_gb = cfgm.num_hidden_layers * 2 * 2 * B * cfgm.num_attention_heads * L * cfgm.head_dim * 4 / 1e9
        if kv_gb > KV_BUDGET_GB and B > 1:
            # Backstop if _kv_chunks underestimated the prompt: split rather than spill out of VRAM.
            half = B // 2
            kw = dict(max_new_tokens=max_new_tokens, repetition_penalty=repetition_penalty, min_p=min_p,
                      top_p=top_p, cond_cache=cond_cache, debug=debug, use_graph=use_graph)
            a, b = _generate(model, items[:half], **kw), _generate(model, items[half:], **kw)
            return tuple(x + y for x, y in zip(a, b))
        st = getattr(model, "_static_t3", None)
        if st is None or (st.R, st.L) != (2 * B, L):
            model._static_t3 = None
            torch.cuda.empty_cache()
            st = model._static_t3 = StaticT3(tfmr, t3.speech_head, 2 * B, L, ALIGNED_HEADS)

        def first_fwd():
            logits, spy = st.prefill(x, mask, pos)
            return logits, torch.stack([w[cond_rows] for w in spy]).mean(0)  # (B, T, T)

        def next_fwd(emb):
            logits, spy = st.step(emb)
            return logits, spy[:, cond_rows].mean(0)[:, None, :]  # (B, 1, L)
    else:
        state = {}

        def _hf(**kw):
            out = tfmr(use_cache=True, **kw)
            state["past"] = out.past_key_values
            att = torch.stack([_spy["attn"][layer][cond_rows] for layer, _ in ALIGNED_HEADS]).mean(0)
            return t3.speech_head(out.last_hidden_state[:, -1, :]), att

        def first_fwd():
            state["mask"], state["pos"] = mask, pos
            return _hf(inputs_embeds=x, attention_mask=mask, position_ids=pos)

        def next_fwd(emb):
            state["mask"] = torch.cat([state["mask"], torch.ones(2 * B, 1, dtype=mask.dtype, device=dev)], 1)
            state["pos"] = state["pos"][:, -1:] + 1
            return _hf(inputs_embeds=emb, attention_mask=state["mask"], position_ids=state["pos"],
                       past_key_values=state["past"])

    if debug:
        import time as _t
        debug = {"t": _t.perf_counter()}
    _spy["on"], _spy["attn"] = not use_graph, {}
    t_dec = time.perf_counter()
    try:
        logits_rows, att_gpu = first_fwd()
        generated = torch.full((B, 1), t3.hp.start_speech_token, dtype=torch.long, device=dev)
        finished = torch.zeros(B, dtype=torch.bool)
        capped = [False] * B
        lengths = [None] * B
        first = True
        for step in range(max(caps)):
            c, u = logits_rows[0::2], logits_rows[1::2]
            logits = c + cfg * (c - u)

            # Alignment guard: one GPU->CPU copy for the whole batch.
            att = att_gpu.float().cpu()  # (B, q, k)
            last = generated[:, -1].tolist()
            for b, g in enumerate(guards):
                if finished[b]:
                    continue
                i, j = slices[b][0] + pads[b], slices[b][1] + pads[b]
                # First frame holds the whole prompt; afterwards one row per step (KV cache).
                chunk = att[b, j:, i:j] if first else att[b, :, i:j]
                verdict = g.step(chunk, last[b])
                if verdict == "suppress_eos":
                    logits[b, eos] = NEG
                elif verdict == "force_eos":
                    logits[b] = NEG
                    logits[b, eos] = -NEG
            first = False

            logits = rep(generated, logits) / temp
            logits = topp(generated, minp(generated, logits))
            probs = torch.softmax(logits, dim=-1)
            if any(gens):
                # Exponential race (argmax p/E, E~Exp(1)) == multinomial sampling, with per-row RNGs.
                noise = torch.empty_like(probs).exponential_()
                for b, g in enumerate(gens):
                    if g is not None:
                        noise[b].exponential_(generator=g)
                nxt = (probs / noise).argmax(dim=-1, keepdim=True)
            else:
                nxt = torch.multinomial(probs, num_samples=1)  # (B, 1)
            generated = torch.cat([generated, nxt], dim=1)
            hit = (nxt[:, 0] == eos).cpu()
            for b in range(B):
                # Stop at EOS, or at the line's token budget (longer audio could never fit its slot).
                if not finished[b] and (hit[b] or generated.shape[1] - 1 >= caps[b]):
                    finished[b] = True
                    lengths[b] = generated.shape[1]
                    capped[b] = not hit[b]
            if finished.all():
                break
            if debug and step % 50 == 0:
                import time as _t
                now = _t.perf_counter()
                print(f"  step {step}: {(now - debug['t']) / 50 * 1000:.0f} ms/step, "
                      f"unfinished {int((~finished).sum())}/{B}, "
                      f"text pos {[g.text_position for g in guards]}/{[g.j - g.i for g in guards]}, "
                      f"mem {torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0:.1f} GB",
                      flush=True)
                debug["t"] = now

            emb = t3.speech_emb(nxt) + t3.speech_pos_emb.get_fixed_embedding(step + 1)
            logits_rows, att_gpu = next_fwd(emb.repeat_interleave(2, dim=0))  # cond + uncond rows per line
    finally:
        _spy["on"], _spy["attn"] = False, {}

    toks = [generated[b, 1:lengths[b] or generated.shape[1]] for b in range(B)]
    stats = [{"tokens": int(t.numel()), "capped": bool(capped[b] or lengths[b] is None)}
             for b, t in enumerate(toks)]
    print(f"TTS {B} takes: decoded {generated.shape[1] - 1} steps in {time.perf_counter() - t_dec:.1f}s",
          flush=True)
    return toks, conds, stats
