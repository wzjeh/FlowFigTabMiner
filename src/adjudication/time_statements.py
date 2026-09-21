"""Residence-time statements in paper text (pure, regex, no model).

Scope tables often carry no residence-time column and no time in their
caption: the value sits in one sentence of the running text that describes
the flow system used for the table ("... shown in Scheme 10 (−78 °C,
Rt = 0.82 s) ... (Table 9)") or in the experimental section.  This module
lists every such statement with its verbatim sentence so the local-vars
builder can (a) constrain the model to these candidates and (b) fill the
value deterministically when the text offers exactly one.

Nothing is computed from reactor geometry or flow rates on purpose
(Zhao 2026-09-21): a long derivation accumulates errors that are invisible
downstream; an empty field is better than a plausible wrong number.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

_UNIT_S = {"ms": 1e-3, "s": 1.0, "sec": 1.0, "second": 1.0, "seconds": 1.0,
           "min": 60.0, "minute": 60.0, "minutes": 60.0}
_NUM = r"(\d+(?:\.\d+)?)"
_UNIT = r"(ms|s|sec|seconds?|min|minutes?)"
_TR_WORD = r"(?:residence\s+times?|t\s*_?\s*R\s*\d?|R\s*t\s*\d?|τ\s*\d?)"

_PATTERNS = [
    # "0.5 s residence time", "a residence time of ..." handled below; value-first form
    re.compile(r"\b" + _NUM + r"\s*" + _UNIT + r"\s+(?:of\s+)?residence\s+time", re.I),
    # "residence time (tR) in R1 of 0.82 s", "tR = 0.82 s", "Rt: 1.5 s", "residence time (0.055 s)"
    re.compile(_TR_WORD + r"(?:\s*\([^)\d]{0,20}\))?(?:\s+in\s+R\s*\d)?\s*(?:=|:|of|was|is|were|being)?\s*"
               r"(?:about|approximately|ca\.?|~|≈)?\s*\(?\s*" + _NUM + r"\s*" + _UNIT + r"\b", re.I),
    # "(−78 °C, Rt = 0.82 s)", "(T = -68 °C, tR1 = 0.5 s)"
    re.compile(r"\(\s*(?:T\s*=\s*)?[−–\-]?\s*\d+(?:\.\d+)?\s*°?\s*C\s*,\s*" + _TR_WORD + r"\s*=\s*" + _NUM + r"\s*" + _UNIT + r"\s*\)", re.I),
    # "the optimized conditions (−78 °C, 0.8 s)": a (temperature, time) pair is
    # the flow papers' shorthand for (T, tR); batch sources never receive it.
    re.compile(r"\(\s*[−–\-]?\s*\d+(?:\.\d+)?\s*°\s*C\s*,\s*" + _NUM + r"\s*" + _UNIT + r"\s*\)", re.I),
]
# A sweep of the residence time itself ("varying the residence time", "tR from
# 0.05 to 6.3 s", "residence times ranging between …") — not any sentence that
# merely contains "various" (substituents) or "between" (two reactors).
_VARIED_RE = re.compile(
    r"(?:vary|varied|varying)\s+(?:the\s+)?(?:\w+\s+){0,3}(?:residence|t\s*_?R\b|R\s*t\b)"
    r"|(?:residence\s+times?|t\s*_?R\d?|R\s*t)\b[^.;]{0,40}?\b(?:from|between|rang(?:e|ing)|varied|vary)\b"
    r"|\b(?:from|between)\s+\d+(?:\.\d+)?\s*(?:ms|s|min)?\s*(?:to|and|–|-)\s*\d+(?:\.\d+)?\s*(?:ms|s|min)\b",
    re.I)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")


def _sentence_around(text: str, start: int, end: int) -> str:
    lo = max(0, start - 220)
    hi = min(len(text), end + 220)
    chunk = text[lo:hi]
    # trim to the sentence that contains the match
    rel_s = start - lo
    left = chunk[:rel_s]
    right = chunk[rel_s:]
    m_left = list(_SENT_SPLIT.finditer(left))
    if m_left:
        left = left[m_left[-1].end():]
    m_right = _SENT_SPLIT.search(right)
    if m_right:
        right = right[:m_right.start()]
    return re.sub(r"\s+", " ", (left + right)).strip()


def find_residence_time_statements(text: str) -> List[Dict[str, Any]]:
    """Every residence-time value stated in ``text`` with its sentence.

    Returns ``[{value_s, raw, quote, varied}]`` ordered by position, one entry
    per distinct (value, sentence).  ``varied`` marks sentences that describe
    a sweep ("varying the residence time from 0.05 to 6 s"): such values are
    not fixed conditions and the caller must not use them as candidates.
    """
    out: List[Dict[str, Any]] = []
    seen = set()
    text = text or ""
    for pat in _PATTERNS:
        for m in pat.finditer(text):
            num, unit = m.group(1), m.group(2).lower()
            factor = _UNIT_S.get(unit) or _UNIT_S.get(unit.rstrip("s")) or None
            if factor is None:
                continue
            value_s = round(float(num) * factor, 6)
            quote = _sentence_around(text, m.start(), m.end())
            key = (value_s, quote[:80])
            if key in seen:
                continue
            seen.add(key)
            out.append({"value_s": value_s, "raw": re.sub(r"\s+", " ", m.group(0)).strip(),
                        "quote": quote[:400], "varied": bool(_VARIED_RE.search(quote)), "pos": m.start()})
    out.sort(key=lambda d: d["pos"])
    for d in out:
        d.pop("pos", None)
    return out


def fixed_candidates(statements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Statements usable as a fixed condition: not part of a sweep, one entry
    per distinct value (first quote kept)."""
    seen = set()
    cands = []
    for s in statements:
        if s.get("varied") or s["value_s"] in seen:
            continue
        seen.add(s["value_s"])
        cands.append(s)
    return cands
