"""Typed LLM / VLM configuration.

Replaces the legacy ``cfg.get(...)`` pattern in
``src/adjudication/llm_engine.py``.  Pipeline code receives a frozen
``LLMConfig`` / ``VLMConfig`` instance — fields are validated at load time
and downstream consumers never see an untyped dict.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from src.llm.errors import ConfigError


class LLMConfig(BaseModel):
    """Configuration for plain (text-only) LLM calls."""

    model_config = {"frozen": True, "extra": "forbid"}

    provider: Literal["gemini"]
    model: str
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=8192, ge=1)
    max_retries: int = Field(default=3, ge=0)
    timeout_s: float = Field(default=120.0, gt=0.0)
    # Gemini 2.5 thinking budget — number of internal-reasoning tokens
    # the model may spend before emitting visible output.  Thinking tokens
    # count against ``max_output_tokens``, so leaving the default
    # ("dynamic", potentially 30k+) can starve a long structured-output
    # call like GlobalAssembly.  Set 0 for tasks that are essentially
    # translation/formatting; raise for tasks needing deep reasoning.
    thinking_budget: int = Field(default=0, ge=0)


class VLMConfig(BaseModel):
    """Configuration for vision-LLM (image+text) calls used by the
    inspection hooks (paper modules 6 and 12).
    """

    model_config = {"frozen": True, "extra": "forbid"}

    provider: Literal["gemini"]
    model: str
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=8192, ge=1)
    max_retries: int = Field(default=3, ge=0)
    timeout_s: float = Field(default=180.0, gt=0.0)
    # Match-rate threshold below which the InspectionReport sets
    # ``review_needed=True``; callers may consult or ignore this flag.
    match_threshold: float = Field(default=0.80, ge=0.0, le=1.0)
    # See LLMConfig.thinking_budget — same semantics for VLM calls.
    thinking_budget: int = Field(default=0, ge=0)


def _safe_get(d: dict, *keys: str) -> dict | None:
    """Walk a nested dict; return ``None`` rather than raising if missing."""
    cur: object = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur if isinstance(cur, dict) else None


def load_llm_config(yaml_path: str | Path) -> LLMConfig:
    """Parse ``llm.adjudication`` from a config.yaml into a typed model."""
    raw = yaml.safe_load(Path(yaml_path).read_text())
    section = _safe_get(raw, "llm", "adjudication")
    if section is None:
        raise ConfigError(f"missing llm.adjudication in {yaml_path}")
    return LLMConfig(**section)


def load_vlm_config(yaml_path: str | Path) -> VLMConfig:
    """Parse ``vlm.inspection`` from a config.yaml into a typed model."""
    raw = yaml.safe_load(Path(yaml_path).read_text())
    section = _safe_get(raw, "vlm", "inspection")
    if section is None:
        raise ConfigError(f"missing vlm.inspection in {yaml_path}")
    return VLMConfig(**section)
