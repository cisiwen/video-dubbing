"""Timing fit of a voiced line into its slot, shared by the orchestrator and the GPU worker (which uses it
to pick the best take before vocoding)."""
import math

from . import config

TOKEN_SEC = 0.04  # one T3 speech token = 2 mel frames = 40 ms of S3Gen audio
TRIM_SEC = 0.07  # median leading/trailing silence that trim_silence() removes from a take


def tempo(dur, seg_len, room):
    """Tempo change (> 1 = faster) that would fit `dur` s of speech into the slot, before clamping."""
    if dur > room:
        return dur / room
    if dur < seg_len * 0.8:
        return dur / seg_len
    return 1.0


def take_score(take, seg_len, room):
    """Lower is better: a take not cut off at its token budget that fits within MAX_SPEEDUP, then the least
    overflow at the cap, then the smallest tempo change. (The decoder's alignment "complete" flag is not
    used: on zh it is set mostly on slow, drawn-out takes, so it would pick the worst fits.)"""
    dur = max(take["duration"], 0.05)
    want = tempo(dur, seg_len, room)
    return (bool(take.get("capped")), want > config.MAX_SPEEDUP, max(0.0, dur / config.MAX_SPEEDUP - room),
            abs(math.log(want)))
