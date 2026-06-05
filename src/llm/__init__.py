"""Unified LLM / VLM access layer for FlowFigTabMiner.

This subpackage replaces the legacy ``src/adjudication/llm_engine.py`` and
the manual VLM workflow scattered in ``src/extraction/table/vlm_table_pipeline.py``.

Architecture
------------
- ``providers``     — provider ABCs (LLMProvider, VLMProvider) + concrete impls
                      registered via decorator (currently: GeminiProvider).
- ``inspectors``    — figure / table inspectors that take a pipeline-produced
                      DataFrame, ask a VLMProvider to read the same crop
                      independently, and diff via pluggable matchers.
- ``types``         — Pydantic models for every message / response / report.
- ``config``        — typed LLMConfig / VLMConfig (loaded from config.yaml).
- ``errors``        — typed exceptions (LLMError, RateLimitError, …).
- ``cache``         — diskcache wrapper keyed on prompt + image hash so that
                      repeat calls (re-runs, spot-checks) hit local disk.

External callers see only ``LLMProvider`` / ``VLMProvider`` / ``Inspector``
abstractions.  No file outside ``src/llm/providers/`` may import a vendor
SDK (``google.genai``, etc.).
"""

from __future__ import annotations

from src.llm.config import LLMConfig, VLMConfig, load_llm_config, load_vlm_config
from src.llm.errors import (
    LLMError,
    LLMProviderError,
    ParseError,
    RateLimitError,
)
from src.llm.providers import GeminiProvider, get_llm_provider, get_vlm_provider
from src.llm.providers.base import LLMProvider, VLMProvider
from src.llm.types import (
    ChatMessage,
    InspectionReport,
    LLMResponse,
    MatchPair,
    Role,
    VLMImage,
)

__all__ = [
    # config
    "LLMConfig",
    "VLMConfig",
    "load_llm_config",
    "load_vlm_config",
    # types
    "ChatMessage",
    "InspectionReport",
    "LLMResponse",
    "MatchPair",
    "Role",
    "VLMImage",
    # providers
    "LLMProvider",
    "VLMProvider",
    "GeminiProvider",
    "get_llm_provider",
    "get_vlm_provider",
    # errors
    "LLMError",
    "LLMProviderError",
    "ParseError",
    "RateLimitError",
]
