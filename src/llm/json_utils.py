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
    if candidates:
        s = max(candidates, key=len)

    # 4. comments
    s = re.sub(r"//.*?(?=\n|$)", "", s)
    s = re.sub(r"/\*[\s\S]*?\*/", "", s)

    # 5. trailing commas
    s = re.sub(r",\s*([\}\]])", r"\1", s)

    # 6. ellipsis placeholders
    s = re.sub(r":\s*(\.\.\.|…)\s*([,\}\]])", r": null\2", s)

    return s.strip()
