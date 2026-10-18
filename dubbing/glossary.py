"""Per-project name glossary (project.json "glossary"): how each person's name is written in the
target language, so every line renders it the same way.

Entry: {source, target, aliases: [...], status: "approved" | "suggested", example}.
Only approved entries reach the translator. A whole name also covers its single words when both
sides split into the same number of parts ("Jenna Forrester" = "詹娜·福雷斯特" gives "Jenna" = "詹娜"
and "Forrester" = "福雷斯特"), unless two entries disagree on a word. Aliases catch ASR misspellings.
"""
import re

_SPLIT = re.compile(r"\s*[·・•‧]\s*|\s+")


def pairs(p):
    """Every (source spelling, target) the translator should know about, longest first."""
    out, derived = {}, {}
    for e in p.get("glossary", []):
        src, tgt = (e.get("source") or "").strip(), (e.get("target") or "").strip()
        if e.get("status") != "approved" or not src or not tgt:
            continue
        for s in [src] + [a.strip() for a in e.get("aliases", []) if a.strip()]:
            out.setdefault(s.lower(), (s, tgt))
        words, parts = src.split(), [x for x in _SPLIT.split(tgt) if x]
        if len(words) > 1 and len(words) == len(parts):
            for w, t in zip(words, parts):
                derived.setdefault(w.lower(), set()).add((w, t))
    for k, opts in derived.items():
        if k not in out and len({t for _, t in opts}) == 1:
            out[k] = sorted(opts)[0]
    return sorted(out.values(), key=lambda x: -len(x[0]))


def names_in(p, text, _pairs=None):
    """[[source, target], ...] for the glossary names in `text` (longest match wins), in order of appearance."""
    taken, found = [], []
    for src, tgt in (_pairs if _pairs is not None else pairs(p)):
        for m in re.finditer(rf"(?<![A-Za-z]){re.escape(src)}(?![A-Za-z])", text or "", re.I):
            if any(m.start() < b and a < m.end() for a, b in taken):
                continue
            taken.append((m.start(), m.end()))
            found.append((m.start(), [src, tgt]))
    out = []
    for _, pr in sorted(found, key=lambda x: x[0]):
        if pr not in out:
            out.append(pr)
    return out


def apply_change(p, old_pairs):
    """After the glossary changed: put a new rendering straight into translations that used the old one,
    and flag lines that still don't match for re-translation (locked lines are only string-replaced).
    Returns the number of lines touched."""
    from . import project as P
    new_pairs = pairs(p)
    old_map = {s.lower(): t for s, t in old_pairs}
    n = 0
    for s in p["segments"]:
        before, after = names_in(p, s["text"], old_pairs), names_in(p, s["text"], new_pairs)
        if before == after:
            continue
        tr = s.get("translation") or ""
        for src, tgt in after:
            old = old_map.get(src.lower())
            if old and old != tgt and old in tr:
                tr = tr.replace(old, tgt)
        if tr != s.get("translation"):
            s["translation"] = tr
            P.mark(s, "tts")
        if not s.get("translation_locked") and any(t not in tr for _, t in after):
            P.mark(s, "translate", "tts")
        n += 1
    return n
