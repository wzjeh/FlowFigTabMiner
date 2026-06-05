"""Provider implementations.

External callers should use the factory helpers ``get_llm_provider`` and
``get_vlm_provider`` rather than instantiating concrete classes — this
guarantees the rest of the codebase never imports a vendor SDK directly.
"""

from __future__ import annotations

from src.llm.config import LLMConfig, VLMConfig
from src.llm.providers._registry import (
    get_llm_provider,
    get_vlm_provider,
    register_llm_provider,
    register_vlm_provider,
)
from src.llm.providers.base import LLMProvider, VLMProvider

# Importing the concrete provider modules triggers their @register_* decorators
# and makes them discoverable via the factory helpers above.
from src.llm.providers import gemini  # noqa: F401  (side-effect import)

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

# Re-export the concrete classes for type hints / tests that need them.
from src.llm.providers.gemini import GeminiProvider  # noqa: E402

__all__.append("GeminiProvider")
