"""Provider implementations.

External callers should use the factory helpers ``get_llm_provider`` and
``get_vlm_provider`` rather than instantiating concrete classes — this
guarantees the rest of the codebase never imports a vendor SDK directly.
``llm.provider`` in config.yaml names the one to use.
"""

from __future__ import annotations

import importlib

from src.llm.config import LLMConfig, VLMConfig
from src.llm.providers._registry import (
    get_llm_provider,
    get_vlm_provider,
    register_llm_provider,
    register_vlm_provider,
)
from src.llm.providers.base import LLMProvider, VLMProvider

# Importing a provider module runs its @register_* decorators.  Each one needs
# only its own SDK, so a missing SDK leaves the other providers usable.
for _module in ("gemini", "claude", "openai"):
    try:
        importlib.import_module(f"src.llm.providers.{_module}")
    except ImportError:  # pragma: no cover
        pass

__all__ = [
    "LLMConfig",
    "VLMConfig",
    "LLMProvider",
    "VLMProvider",
    "register_llm_provider",
    "register_vlm_provider",
    "get_llm_provider",
    "get_vlm_provider",
]
