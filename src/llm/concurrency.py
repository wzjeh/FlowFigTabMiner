"""Module-level concurrency control for all LLM/VLM provider calls.

A single ``threading.Semaphore`` caps the number of in-flight Gemini
calls across the whole process — figure-track metadata calls, table-cell
calls, header-judge calls, and adjudication chats all queue through the
same gate.  This stops the pipeline from blowing Google's per-minute
quota when many figures / tables are processed in parallel (which the
inspection hooks already cause implicitly).

The semaphore is created lazily on first use and configured at startup by
``configure_concurrency()``.  Providers acquire a slot inside their
chat/inspect call:

    from src.llm.concurrency import acquire_llm_slot
    with acquire_llm_slot():
        response = self._client.models.generate_content(...)

A single global gate (rather than per-provider) is the right shape: an
external rate-limit boundary belongs at the process edge, not inside
each owner module.
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Default cap — overridable via configure_concurrency().  Five concurrent
# Gemini calls keep us well under the free-tier 15-RPM limit at typical
# 2-3 s latency per call; tune up if your billing tier is higher.
_DEFAULT_MAX_CONCURRENT = 5

_lock = threading.Lock()
_semaphore: threading.Semaphore | None = None
_configured_max: int = _DEFAULT_MAX_CONCURRENT


def configure_concurrency(max_concurrent: int) -> None:
    """Set (or reset) the global concurrency cap.

    Idempotent — safe to call multiple times; the latest call wins.
    Construct the semaphore eagerly so the cap is locked in even when
    no calls have happened yet.
    """
    global _semaphore, _configured_max
    if max_concurrent < 1:
        raise ValueError(f"max_concurrent must be ≥ 1, got {max_concurrent}")
    with _lock:
        _configured_max = max_concurrent
        _semaphore = threading.Semaphore(max_concurrent)
    logger.info("llm.concurrency configured max_concurrent=%d", max_concurrent)


def _ensure_semaphore() -> threading.Semaphore:
    global _semaphore
    if _semaphore is None:
        with _lock:
            if _semaphore is None:
                _semaphore = threading.Semaphore(_configured_max)
                logger.info(
                    "llm.concurrency lazily initialised max_concurrent=%d (default)",
                    _configured_max,
                )
    return _semaphore


@contextmanager
def acquire_llm_slot():
    """Block until a concurrency slot is free, then yield.

    Always releases on exit (including exceptions) thanks to the
    contextmanager protocol.  Holding time is the actual API call only;
    callers must not do unrelated heavy work inside the ``with`` block.
    """
    sem = _ensure_semaphore()
    sem.acquire()
    try:
        yield
    finally:
        sem.release()
