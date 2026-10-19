# Dubbing Studio

Local, open-source video dubbing with a human review UI.

```
video ─► extract ─► separate ─► transcribe ─► diarize ─► segment ─► emotion ─► translate ─► synthesize ─► mix
          ffmpeg     Demucs      Whisper       pyannote   words→lines  emotion2vec+  Qwen3-8B   Chatterbox     ffmpeg
                    (dialogue /  large-v3      community-1 + per-line                (NLLB fallback) Multilingual
                     music+fx)   word times                 language ID                          voice clone
```

* **Speakers**: pyannote diarization, assigned per word, so lines are cut wherever the speaker changes.
* **Language**: Whisper picks the main language, then each line gets its own language check. A line in
  another language is transcribed again in that language and translated from it.
* **Emotion**: emotion2vec+ large labels each line (neutral/happy/sad/angry/fearful/surprised/disgusted).
* **Voice clone**: by default each line uses its own original audio as the clone prompt, so the delivery
  carries over. The detected emotion sets Chatterbox's `exaggeration` and `cfg_weight`
  (see `EMOTION_STYLE` in `dubbing/config.py`). Lines shorter than 3 s use the speaker's reference clip.
* **Timing**: each line is voiced 3 times in the same batch (different seed, `cfg_weight` nudged ±0.1) and the
  take that fits its slot with the least tempo change is kept; it is then trimmed and sped up (max ×1.35) or
  slowed down (min ×0.9) to fit. The translator's length budget uses each speaker's own measured speaking
  rate (`speakers[*].speech_rate`, from their voiced lines). If a line is still too long, the UI flags it
  as "+Ns long".
* **Music & effects** are kept: the dub is mixed over the Demucs background stem.
* **Output**: an MP4 with the dubbed track (default) and the original track, plus source and target `.srt` files.

## Running

The Mac runs the UI and orchestration. The Windows box (RTX 5070, `D:\dubbing-worker`) runs the models.
The two talk through an SSH tunnel, so no firewall port has to be opened.

```bash
./worker_setup/deploy_worker.sh            # copy code + (re)start the GPU worker  (--setup = reinstall env)
./run_studio.sh                            # opens tunnel, serves http://localhost:8000
```

`start_worker.ps1` restarts the worker if it dies, and the studio waits for it and retries the interrupted
request once. A native crash writes every thread's Python stack to `D:\dubbing-worker\worker_crash.log`.

Video links are downloaded with yt-dlp in the studio's `.venv`. Sites change often, so if downloads start
failing, update it: `uv pip install --python .venv/bin/python -U yt-dlp`.

Environment knobs (all optional):

| var | default | |
|---|---|---|
| `DUB_WORKER_URL` | `http://127.0.0.1:8765` | empty = run every model on this machine |
| `DUB_LOCAL_STAGES` | – | e.g. `transcribe,detect_languages` keeps ASR on the Mac (mlx-whisper) |
| `DUB_TRANSLATION_PROVIDER` | `llm` | `llm` = length-aware Qwen on the GPU worker (`DUB_LLM_MODEL`, default `unsloth/Qwen3-8B-bnb-4bit`, Apache-2.0); falls back to NLLB if it cannot load. `nllb` = NLLB-200 only. `ollama` = same prompt via Ollama (`DUB_OLLAMA_MODEL`) |
| `DUB_RETRY_OVERFLOW_SEC` / `DUB_RETRY_ROUNDS` | `0.3` / `2` | after voicing, lines still longer than their slot by this much are re-translated shorter and re-voiced (LLM providers only); each line keeps its best attempt |
| `DUB_MAX_SPEEDUP` / `DUB_MAX_SLOWDOWN` | 1.35 / 0.9 | timing-fit limits |
| `DUB_TTS_TAKES` / `DUB_TAKE_CFG_STEP` | `3` / `0.1` | best-of-N takes per line (1 = off) and the `cfg_weight` nudge between takes |
| `DUB_RATE_MIN_LINES` / `DUB_RATE_PRIOR_SEC` | `3` / `5` | a speaker's measured rate is used once they have this many voiced lines, shrunk toward the language default by this many seconds of default-pace speech |
| `HF_TOKEN` | – | only needed to download the gated pyannote model on a new machine |

## Voice-generation speed

Profiling on the RTX 5070 (Windows) showed each T3 decode step was ~10 ms of GPU work but ~90 ms of CPU
time launching ~1100 small kernels, so upstream Chatterbox ran at ~3.8× slower than real time.
`dubbing/tts_batch.py` decodes up to `DUB_TTS_BATCH` lines together (with the upstream hallucination guard
per line), and `dubbing/t3_graph.py` replays each step as one CUDA graph over a static KV cache
(logits match HF within 1.5e-5). Measured on 16 lines of a real episode: 389 s → 56 s (RTF 2.9 → 0.44),
Whisper round-trip CER unchanged (0.24 vs 0.23). Each line's decode budget is derived from its time slot, and a
batch is split if its KV cache would exceed `DUB_TTS_KV_GB` (default 3 GB) so Windows never spills VRAM to
system RAM (items are packed greedily by decode length). `DUB_TTS_IMPL=stock` switches back to upstream
generation.

Best-of-N takes are cheap because the worker picks the winning take from its token count (40 ms per token,
minus ~0.07 s of trimmed silence) and only vocodes that one: S3Gen costs ~1.8 s per line, more than the
batched decode. On the 6-min test episode, 3 takes cost about +25% synthesize wall time (two runs each: 299 s → 377 s on average).

Benchmarks: `worker_setup/bench_compare.py` (speed), `worker_setup/cer_check.py` (accuracy),
`worker_setup/test_graph.py` (graph-vs-HF parity), `worker_setup/prof_step.py` (per-step profile).

## Review workflow

1. **New project**: upload the video, or paste a link (YouTube or any site [yt-dlp](https://github.com/yt-dlp/yt-dlp)
   supports), and pick the dub language. A link is downloaded on this Mac as the first part of "extract"
   (best ≤1080p MP4/MKV with audio, progress in the status bar) and the project is named after the video
   title. If the download fails (private or geo-blocked video, bad link, playlist instead of a single video)
   the status shows why; re-run from "extract" to retry. Only download videos you have the rights to dub.
   "Pause after transcription" (on by default)
   stops before any voices are generated, so speakers and text can be fixed first. Voice generation is the
   slow part.
2. **Fix speakers**: rename them, merge duplicate clusters ("merge into"), reassign a line from its dropdown,
   or "+ add" a speaker the model missed.
3. **Fix lines**: edit the transcript, language, emotion or start/end. You can also drag segment edges on the
   timeline. Split at the playhead (S) and merge with the next line (M).
   **Names** (below the speakers): how each person's name is written in the dub, e.g. "Jenna Forrester" =
   "詹娜·福雷斯特". "suggest" drafts entries with the LLM (shown dashed with the line they came from; check each,
   fix it and tick ✓ — only ticked names are used). "also heard as" catches ASR misspellings. A whole name also
   covers its single words ("Jenna", "Forrester"). The translator is told the names in each line and any left in
   the source spelling are replaced; changing a name updates or flags every line that mentions it.
4. **Apply & continue / Apply changes**: only lines marked *pending* are re-processed (emotion → translate →
   voice), then the whole dub is remixed. Lines whose translation or emotion you edited by hand are kept as you
   left them.
5. **Polish**: edit a translation, then use "⟳ voice" to regenerate that one line and hear it straight away.
   For a speaker, choose a "fixed ref" voice (or "use selected as ref" for a clean line) and adjust "intensity".
6. Listen with **Original / Dubbed / Both**. Use the filter "Needs attention" to jump through low ASR
   confidence, foreign-language lines, uncertain emotion and lines that run too long.
7. **More → Download**: the dubbed MP4, the dub WAV, the SRT files or the project JSON.

Keys: Space play · J/K next/prev line · Enter mark reviewed & next · ⌘/Ctrl+Enter same from a text box ·
O/D play line original/dub.

## Layout

```
dubbing/engines.py   model wrappers (Demucs, Whisper, pyannote, emotion2vec, NLLB, Chatterbox)
dubbing/worker.py    FastAPI service on the GPU box exposing engines (content-addressed file cache)
dubbing/backend.py   per-stage routing: GPU worker vs in-process
dubbing/pipeline.py  stages, segmentation, voice references, timing fit, job queue
dubbing/server.py    review API + static UI;  static/  the UI (plain JS, no build)
projects/<id>/       project.json (all reviewable state) + audio stems, clips, tts, out/
```

## Licences — check before commercial use

Whisper, Demucs, pyannote code, Chatterbox: MIT. The pyannote `community-1` weights are CC-BY-4.0, and the
Hugging Face page requires accepting its terms. **NLLB-200 is CC-BY-NC 4.0 (non-commercial).** For commercial
output, keep the default `DUB_TRANSLATION_PROVIDER=llm` (Qwen3, Apache-2.0) or use `ollama` with another
Apache/MIT-licensed model; set it to `nllb` only for non-commercial work. Chatterbox adds an inaudible Perth watermark to generated audio. Only clone
voices you have the rights to use.
