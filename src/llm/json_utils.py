"""JSON sanitization helpers for messy LLM responses.

LLMs occasionally wrap their JSON in markdown code fences, leak inline
``//`` comments, scatter trailing commas, or write ``...`` ellipses
where a value would go.  ``sanitize_json_text`` does a deterministic
multi-pass clean-up so callers can ``json.loads`` the result.
"""

from __future__ import annotations

import re

__all__ = ["sanitize_json_text"]


def _string_mask(s: str) -> list:
    """True at every index that lies inside a JSON string literal."""
    mask = [False] * len(s)
    in_string = False
    escape = False
    for i, ch in enumerate(s):
        if in_string:
            mask[i] = True
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
                mask[i] = True
        elif ch == '"':
            in_string = True
            mask[i] = True
    return mask


def _strip_comments(s: str) -> str:
    """Remove ``//`` line comments and ``/* */`` blocks that sit outside strings."""
    mask = _string_mask(s)
    out = []
    i = 0
    n = len(s)
    while i < n:
        if not mask[i] and s.startswith("//", i):
            j = s.find("\n", i)
            i = n if j < 0 else j
            continue
        if not mask[i] and s.startswith("/*", i):
            j = s.find("*/", i + 2)
            i = n if j < 0 else j + 2
            continue
        out.append(s[i])
        i += 1
    return "".join(out)


def sanitize_json_text(text: str) -> str:
    """Best-effort cleanup of a JSON-bearing LLM response.

    The passes are intentionally conservative — each one only touches
    patterns that are LLM artefacts and would never appear in a
    well-formed JSON document.

    Steps:

    1. Strip markdown code fences.
    2. Remove ASCII control characters except tab / newline / CR.
    3. Extract the largest balanced object or array block.
    4. Remove ``//`` line and ``/* ... */`` block comments.
    5. Remove trailing commas before ``}`` or ``]``.
    6. Normalise ``...`` value placeholders to ``null``.
    """
    s = text or ""

    # 1. markdown fences
    s = re.sub(r"```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"```\s*", "", s)

    # 2. control characters
    s = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", s)

    # 3. largest balanced object or array block — string-aware: brackets
    #    inside JSON strings ("Yield [%]", "[a] GC yield", "https://doi…")
    #    are data, not structure.
    mask = _string_mask(s)
    candidates: list[str] = []
    for open_ch, close_ch in [("{", "}"), ("[", "]")]:
        depth = 0
        start = -1
        for i, ch in enumerate(s):
            if mask[i]:
                continue
            if ch == open_ch:
                if depth == 0:
                    start = i
                depth += 1
            elif ch == close_ch:
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start != -1:
                        candidates.append(s[start : i + 1])
                        start = -1
    # 3b. recovery for truncated arrays: prefer a reconstructed top-level
    # array over a single-record dict candidate.  If an opening `[` is
    # present but never closed cleanly, harvest every complete top-level
    # object inside it and wrap them as a fresh array (step 3 would
    # otherwise pick the longest balanced `{...}` — one inner record).
    depth_arr = 0
    depth_obj = 0
    arr_start = -1
    last_complete_close = -1
    for i, ch in enumerate(s):
        if mask[i]:
            continue
        if ch == "[":
            if depth_arr == 0 and depth_obj == 0:
                arr_start = i
            depth_arr += 1
        elif ch == "]":
            depth_arr -= 1
        elif ch == "{":
            depth_obj += 1
        elif ch == "}":
            depth_obj -= 1
            if depth_arr == 1 and depth_obj == 0:
                last_complete_close = i

    array_unbalanced = arr_start >= 0 and depth_arr > 0
    if array_unbalanced and last_complete_close > arr_start:
        s = s[arr_start : last_complete_close + 1] + "]"
    elif candidates:
        s = max(candidates, key=len)

    # 4. comments (outside strings only: "https://doi.org/…" is a value)
    s = _strip_comments(s)

    # 5. trailing commas
    s = re.sub(r",\s*([\}\]])", r"\1", s)

    # 6. ellipsis placeholders
    s = re.sub(r":\s*(\.\.\.|…)\s*([,\}\]])", r": null\2", s)

    return s.strip()
