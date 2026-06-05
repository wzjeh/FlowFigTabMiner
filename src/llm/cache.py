"""Disk-backed response cache keyed on model + prompt + image hash.

Two real-world reasons this exists:

1. Spot-checking on the same PDF should not re-pay for the same Gemini
   call across iterations.
2. The same crop is sent for inspection twice (once when the figure track
   runs, once if the user re-runs the pipeline to debug) — caching keeps
   the marginal cost at zero.

Keys are SHA256 over the inputs that influence the response:
``(provider, model, system_prompt, user_prompt, image_bytes_hash)``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from diskcache import Cache

_DEFAULT_DIR = Path.home() / ".cache" / "flowfigtabminer" / "llm"


def _normalize(value: Any) -> Any:
    """Make ``value`` JSON-serializable for stable hashing."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return hashlib.sha256(value).hexdigest()
    if isinstance(value, (list, tuple)):
        return [_normalize(v) for v in value]
    if isinstance(value, dict):
        return {k: _normalize(v) for k, v in sorted(value.items())}
    return value


class ResponseCache:
    """Thin wrapper around ``diskcache.Cache``.

    The cache directory is user-scoped (``~/.cache/flowfigtabminer/llm``)
    so it survives ``git clean`` and is shared across repo checkouts.
    """

    def __init__(self, root: str | Path | None = None, size_limit_gb: float = 2.0):
        path = Path(root) if root else _DEFAULT_DIR
        path.mkdir(parents=True, exist_ok=True)
        self._cache = Cache(directory=str(path), size_limit=int(size_limit_gb * 1024**3))

    @staticmethod
    def key(*parts: Any) -> str:
        payload = json.dumps([_normalize(p) for p in parts], sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()

    def get(self, key: str) -> dict | None:
        return self._cache.get(key, default=None)

    def set(self, key: str, value: dict) -> None:
        self._cache.set(key, value)

    def __contains__(self, key: str) -> bool:
        return key in self._cache
