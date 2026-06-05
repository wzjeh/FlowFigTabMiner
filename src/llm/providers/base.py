"""Abstract base classes for LLM and VLM providers."""

from __future__ import annotations

import abc
from typing import Optional, Type

from pydantic import BaseModel

from src.llm.config import LLMConfig, VLMConfig
from src.llm.types import ChatMessage, LLMResponse, VLMImage


class LLMProvider(abc.ABC):
    """Plain text-in / text-out provider.

    Implementations are expected to:
    - Accept the typed ``LLMConfig`` and apply temperature / token limits.
    - Raise typed exceptions from ``src.llm.errors`` on failure; never
      return an empty string silently.
    - Record latency, model id, and token counts in the returned
      ``LLMResponse`` so callers don't need to instrument call sites.
    """

    @abc.abstractmethod
    def chat(self, messages: list[ChatMessage], cfg: LLMConfig) -> LLMResponse:
        """Return a single completion for the given messages."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Stable identifier used in logs / cache keys (e.g. ``"gemini"``)."""


class VLMProvider(abc.ABC):
    """Multimodal (text + image) provider used by the inspection hooks.

    The contract is deliberately tighter than ``LLMProvider``:
    inspection prompts always return one structured JSON object that the
    inspector then validates against its own schema.  The provider parses
    the textual response; the inspector handles semantic interpretation.
    """

    @abc.abstractmethod
    def inspect(
        self,
        image: VLMImage,
        system_prompt: str,
        user_prompt: str,
        cfg: VLMConfig,
        response_schema: Optional[Type[BaseModel]] = None,
    ) -> tuple[LLMResponse, dict]:
        """Send ``image`` + prompts to the VLM and return the parsed JSON.

        Returns a tuple of ``(metadata, parsed_json)`` so that callers see
        both the raw transport stats (latency, model id, token counts) and
        the decoded JSON in one round-trip.

        When ``response_schema`` is provided, the provider enables the
        SDK's structured-output mode (Pydantic model class supplied
        verbatim).  In that mode the returned ``parsed_json`` is
        guaranteed to be schema-conformant and callers can immediately
        call ``response_schema.model_validate(parsed_json)``.
        """

    @property
    @abc.abstractmethod
    def name(self) -> str: ...
