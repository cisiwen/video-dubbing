"""Central configuration. Override any value with an environment variable of the same name."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROJECTS_DIR = Path(os.environ.get("DUB_PROJECTS_DIR", ROOT / "projects"))
PROJECTS_DIR.mkdir(parents=True, exist_ok=True)

# Models
WHISPER_MODEL = os.environ.get("DUB_WHISPER_MODEL", "mlx-community/whisper-large-v3-turbo")
DIARIZATION_MODEL = os.environ.get("DUB_DIARIZATION_MODEL")  # default: community-1 on pyannote 4, else 3.1
EMOTION_MODEL = os.environ.get("DUB_EMOTION_MODEL", "emotion2vec/emotion2vec_plus_large")
# llm: length-aware Qwen on the GPU worker (falls back to nllb if it cannot load) | nllb | ollama
TRANSLATION_PROVIDER = os.environ.get("DUB_TRANSLATION_PROVIDER", "llm")
# Apache-2.0 (commercial use OK). Pre-quantised 4-bit Qwen3-8B, ~8 GB peak; evicted before TTS loads.
LLM_MODEL = os.environ.get("DUB_LLM_MODEL", "unsloth/Qwen3-8B-bnb-4bit")
LLM_CACHE = os.environ.get("DUB_LLM_CACHE")  # HF cache dir for the LLM weights (default HF cache)
NLLB_MODEL = os.environ.get("DUB_NLLB_MODEL", "facebook/nllb-200-distilled-600M")
OLLAMA_URL = os.environ.get("DUB_OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("DUB_OLLAMA_MODEL", "qwen2.5:14b")
DEMUCS_MODEL = os.environ.get("DUB_DEMUCS_MODEL", "htdemucs")
HF_TOKEN = os.environ.get("HF_TOKEN")

# Segmentation
MAX_SEGMENT_SEC = float(os.environ.get("DUB_MAX_SEGMENT_SEC", 12))
SPLIT_PAUSE_SEC = float(os.environ.get("DUB_SPLIT_PAUSE_SEC", 0.6))
# Fragments shorter than this (or of 1-2 words) are folded into the adjacent line when the gap is
# under MERGE_GAP_SEC: TTS cannot voice even one target-language word in a sub-second slot.
MERGE_FRAGMENT_SEC = float(os.environ.get("DUB_MERGE_FRAGMENT_SEC", 1.0))
MERGE_GAP_SEC = float(os.environ.get("DUB_MERGE_GAP_SEC", 0.6))

# Timing fit: how much we are allowed to speed up / slow down synthesized speech
MAX_SPEEDUP = float(os.environ.get("DUB_MAX_SPEEDUP", 1.35))
MAX_SLOWDOWN = float(os.environ.get("DUB_MAX_SLOWDOWN", 0.9))
TTS_BATCH = int(os.environ.get("DUB_TTS_BATCH", 8))  # lines decoded together on the GPU
TTS_IMPL = os.environ.get("DUB_TTS_IMPL", "batch")  # batch | stock (upstream one-line generate)
MIN_REF_SEC = 3.0  # shortest clip usable as a voice-clone reference

# Natural TTS speaking rate in spoken_units()/s, used to give the translator a length budget.
# zh measured from Chatterbox output (median 4.0 over 35 lines); the rest are rough defaults.
SPEECH_RATE = {"zh": 4.0, "ja": 6.0, "ko": 5.0}
DEFAULT_SPEECH_RATE = 4.5  # syllables/s for alphabetic languages
BUDGET_SPEEDUP = float(os.environ.get("DUB_BUDGET_SPEEDUP", 1.1))  # tempo headroom assumed in budgets
RETRY_OVERFLOW_SEC = float(os.environ.get("DUB_RETRY_OVERFLOW_SEC", 0.3))  # re-translate lines over this
RETRY_ROUNDS = int(os.environ.get("DUB_RETRY_ROUNDS", 2))

PALETTE = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
           "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac"]

EMOTIONS = ["neutral", "happy", "sad", "angry", "fearful", "surprised", "disgusted", "other"]

# Chatterbox "exaggeration" (emotion intensity) and cfg_weight (pacing) per emotion.
EMOTION_STYLE = {
    "neutral":   {"exaggeration": 0.45, "cfg_weight": 0.5},
    "happy":     {"exaggeration": 0.7,  "cfg_weight": 0.4},
    "sad":       {"exaggeration": 0.55, "cfg_weight": 0.55},
    "angry":     {"exaggeration": 0.85, "cfg_weight": 0.3},
    "fearful":   {"exaggeration": 0.7,  "cfg_weight": 0.4},
    "surprised": {"exaggeration": 0.8,  "cfg_weight": 0.35},
    "disgusted": {"exaggeration": 0.7,  "cfg_weight": 0.4},
    "other":     {"exaggeration": 0.5,  "cfg_weight": 0.5},
}

# Whisper ISO code -> NLLB code
NLLB_CODES = {
    "en": "eng_Latn", "zh": "zho_Hans", "es": "spa_Latn", "fr": "fra_Latn", "de": "deu_Latn",
    "it": "ita_Latn", "ja": "jpn_Jpan", "ko": "kor_Hang", "pt": "por_Latn", "ru": "rus_Cyrl",
    "ar": "arb_Arab", "hi": "hin_Deva", "nl": "nld_Latn", "pl": "pol_Latn", "tr": "tur_Latn",
    "sv": "swe_Latn", "da": "dan_Latn", "fi": "fin_Latn", "no": "nob_Latn", "el": "ell_Grek",
    "he": "heb_Hebr", "ms": "zsm_Latn", "sw": "swh_Latn", "vi": "vie_Latn", "th": "tha_Thai",
    "id": "ind_Latn", "uk": "ukr_Cyrl", "cs": "ces_Latn", "ro": "ron_Latn", "hu": "hun_Latn",
}

# Languages Chatterbox Multilingual can speak (dub targets)
TTS_LANGUAGES = {
    "ar": "Arabic", "da": "Danish", "de": "German", "el": "Greek", "en": "English",
    "es": "Spanish", "fi": "Finnish", "fr": "French", "he": "Hebrew", "hi": "Hindi",
    "it": "Italian", "ja": "Japanese", "ko": "Korean", "ms": "Malay", "nl": "Dutch",
    "no": "Norwegian", "pl": "Polish", "pt": "Portuguese", "ru": "Russian", "sv": "Swedish",
    "sw": "Swahili", "tr": "Turkish", "zh": "Chinese",
}


def spoken_units(text: str) -> int:
    """Rough spoken length: one per CJK/kana/hangul character or digit, one per vowel group in
    alphabetic words (a syllable estimate)."""
    import re
    n = len(re.findall(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af\d]", text))
    for w in re.findall(r"[^\W\d_\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]+", text):
        n += max(1, len(re.findall(r"[aeiouyàáâãäåèéêëìíîïòóôõöùúûüýÿаеёиоуыэюяαεηιουωάέήίόύώ]+", w.lower())))
    return n


def speech_budget(lang: str, seconds: float) -> int:
    """How many spoken_units() fit in `seconds` at natural pace plus a little tempo headroom."""
    return max(1, int(SPEECH_RATE.get(lang, DEFAULT_SPEECH_RATE) * seconds * BUDGET_SPEEDUP))


def device() -> str:
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
