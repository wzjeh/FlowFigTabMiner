"""Helpers shared by every provider: retry with backoff and tolerant JSON recovery.

Kept free of vendor SDKs so each provider module imports only its own SDK.
"""

from __future__ import annotations

import logging
import random
import re
import time
from typing import Any, Callable, Optional

from src.llm.errors import RateLimitError, TransientError

logger = logging.getLogger(__name__)


def _backoff_call(
    call: Callable[[], Any],
    *,
    max_retries: int,
    label: str,
) -> tuple[Any, int]:
    """Run ``call`` with exponential backoff on RateLimitError / TransientError.

    Delays grow as 1 s, 2 s, 4 s, ... up to ``max_retries`` attempts.
    Other LLMProviderError types propagate immediately (no retry).

    Returns ``(result, retry_count)`` so callers can record how often a
    call was rate-limited.  ``retry_count`` is 0 on first-try success.
    """
    last_exc: Optional[BaseException] = None
    for attempt in range(max_retries + 1):
        try:
            return call(), attempt
        except (RateLimitError, TransientError) as exc:
            last_exc = exc
            if attempt >= max_retries:
                break
            sleep_s = (2 ** attempt) + random.uniform(0, 0.5)
            logger.warning(
                "%s %s (attempt %d/%d), sleeping %.1fs before retry",
                label, "rate-limited" if isinstance(exc, RateLimitError) else f"transient failure: {str(exc)[:80]}",
                attempt + 1,
                max_retries + 1,
                sleep_s,
            )
            time.sleep(sleep_s)
    assert last_exc is not None
    raise last_exc


def _strip_json_fence(text: str) -> str:
    """Remove the leading/trailing ```json fences Gemini sometimes emits."""
    out = text.strip()
    if out.startswith("```"):
        out = re.sub(r"^```(?:json)?\s*", "", out)
        out = re.sub(r"\s*```$", "", out)
    return out


def _extract_balanced_json(text: str) -> str | None:
    """Best-effort recovery: find the first balanced JSON object/array."""
    for opener, closer in (("{", "}"), ("[", "]")):
        i = text.find(opener)
        if i < 0:
            continue
        depth = 0
        for j in range(i, len(text)):
            ch = text[j]
            if ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    return text[i : j + 1]
    return None
