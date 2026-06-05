"""Provider registration via decorator.

Concrete providers register themselves at import time, e.g.::

    @register_llm_provider("gemini")
    @register_vlm_provider("gemini")
    class GeminiProvider(LLMProvider, VLMProvider): ...

Callers look up providers by name via ``get_llm_provider`` /
``get_vlm_provider``.  This keeps the rest of the codebase decoupled from
vendor SDKs while still allowing dependency injection of a specific
instance from ``main.py``.
"""

from __future__ import annotations

from typing import Callable, TypeVar

from src.llm.errors import ConfigError
from src.llm.providers.base import LLMProvider, VLMProvider

_LLM_PROVIDERS: dict[str, type[LLMProvider]] = {}
_VLM_PROVIDERS: dict[str, type[VLMProvider]] = {}

L = TypeVar("L", bound=LLMProvider)
V = TypeVar("V", bound=VLMProvider)


def register_llm_provider(name: str) -> Callable[[type[L]], type[L]]:
    """Register an ``LLMProvider`` subclass under ``name``."""

    def _wrap(cls: type[L]) -> type[L]:
        _LLM_PROVIDERS[name] = cls
        return cls

    return _wrap


def register_vlm_provider(name: str) -> Callable[[type[V]], type[V]]:
    """Register a ``VLMProvider`` subclass under ``name``."""

    def _wrap(cls: type[V]) -> type[V]:
        _VLM_PROVIDERS[name] = cls
        return cls

    return _wrap


def get_llm_provider(name: str, **kwargs) -> LLMProvider:
    if name not in _LLM_PROVIDERS:
        raise ConfigError(
            f"unknown LLM provider '{name}'; registered: {sorted(_LLM_PROVIDERS)}"
        )
    return _LLM_PROVIDERS[name](**kwargs)


def get_vlm_provider(name: str, **kwargs) -> VLMProvider:
    if name not in _VLM_PROVIDERS:
        raise ConfigError(
            f"unknown VLM provider '{name}'; registered: {sorted(_VLM_PROVIDERS)}"
        )
    return _VLM_PROVIDERS[name](**kwargs)
