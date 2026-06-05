"""Typed exceptions for LLM/VLM calls.

The contract:
- Provider implementations raise one of these exception types on failure.
- Callers can catch the narrow types they actually want to handle (e.g.,
  ``RateLimitError`` to back off; ``ParseError`` to mark a record
  ``review_needed``).
- Silent empty-string returns from the legacy ``llm_engine.py`` are gone:
  pipelines either get a valid ``LLMResponse`` or a typed exception.
"""

from __future__ import annotations


class LLMError(Exception):
    """Base class for all LLM/VLM failures."""


class LLMProviderError(LLMError):
    """The provider's underlying API failed (network, 5xx, etc.)."""


class RateLimitError(LLMProviderError):
    """Provider returned a rate-limit / quota error.

    Callers may back off and retry; the provider's own retry budget is
    governed by ``LLMConfig.max_retries``.
    """


class ContextLengthError(LLMError):
    """Request exceeded the model's context window."""


class ParseError(LLMError):
    """Response was received but not parseable as expected JSON/schema."""

    def __init__(self, message: str, raw_text: str = "") -> None:
        super().__init__(message)
        self.raw_text = raw_text


class ConfigError(LLMError):
    """The provided LLMConfig / VLMConfig is missing required fields."""
