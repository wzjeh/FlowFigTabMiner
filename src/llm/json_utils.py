"""JSON sanitization helpers for messy LLM responses.

LLMs occasionally wrap their JSON in markdown code fences, leak inline
``//`` comments, scatter trailing commas, or write ``...`` ellipses
where a value would go.  ``sanitize_json_text`` does a deterministic
multi-pass clean-up so callers can ``json.loads`` the result.
"""

from __future__ import annotations

import re

__all__ = ["sanitize_json_text"]


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

    # 3. largest balanced object or array block
    candidates: list[str] = []
    for open_ch, close_ch in [("{", "}"), ("[", "]")]:
        depth = 0
        start = -1
        for i, ch in enumerate(s):
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
    # array over a single-record dict candidate.  Walk the text once
    # tracking depth; if an opening `[` is present but never closed
    # cleanly, harvest every complete top-level object inside it and
    # wrap them as a fresh array.  Step 3 would otherwise pick the
    # longest balanced `{...}` (one inner record) — which is wrong when
    # 65 records were emitted before truncation.
    depth_arr = 0
    depth_obj = 0
    arr_start = -1
    in_string = False
    escape = False
    last_complete_close = -1
    for i, ch in enumerate(s):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
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

    # 4. comments
    s = re.sub(r"//.*?(?=\n|$)", "", s)
    s = re.sub(r"/\*[\s\S]*?\*/", "", s)

    # 5. trailing commas
    s = re.sub(r",\s*([\}\]])", r"\1", s)

    # 6. ellipsis placeholders
    s = re.sub(r":\s*(\.\.\.|…)\s*([,\}\]])", r": null\2", s)

    return s.strip()
