"""Typed message / response / inspection-report models.

All structures used at the LLM-boundary are Pydantic models so the rest of
the codebase manipulates strongly-typed objects rather than dicts.  Adding
a new field is a one-line schema change and downstream consumers get a
clear type error if they're out of date.
"""

from __future__ import annotations

import enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator


class Role(str, enum.Enum):
    """Chat role.  Mirrors OpenAI / Anthropic / Gemini conventions."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class VLMImage(BaseModel):
    """A single image attachment to a multimodal chat message."""

    path: Path
    mime_type: str = "image/png"

    @field_validator("path")
    @classmethod
    def _exists(cls, value: Path) -> Path:
        if not value.exists():
            raise ValueError(f"image path does not exist: {value}")
        return value


class ChatMessage(BaseModel):
    """One message in an LLM conversation.

    ``content`` is the textual body; ``images`` is an optional list of
    image attachments for multimodal models (Gemini, GPT-4o, Claude).
    """

    role: Role
    content: str
    images: list[VLMImage] = Field(default_factory=list)


class LLMResponse(BaseModel):
    """Result of one LLM/VLM call.

    Carries enough metadata for the structured logger (model, token counts,
    latency) so callers don't need to instrument call sites themselves.
    """

    text: str
    model: str
    tokens_in: int | None = None
    tokens_out: int | None = None
    latency_ms: float
    cache_hit: bool = False
    finish_reason: str | None = None
    # Number of times this call was rate-limited and retried before
    # success.  0 means first-try success.  Populated by GeminiProvider;
    # other providers may leave default 0.
    retry_count: int = 0


class MatchPair(BaseModel):
    """One matched pair (pipeline record ↔ VLM record).

    ``distance`` is the matcher-specific normalized distance (e.g., relative
    coordinate distance for figures, edit distance for table cells).
    """

    pipeline_record: dict[str, Any]
    vlm_record: dict[str, Any]
    distance: float


class InspectionReport(BaseModel):
    """Output of one figure / table inspection round.

    ``source_id`` uniquely identifies the figure or table within the run
    (e.g. ``"fig_3a"`` or ``"tab_S5"``).  ``unmatched_*`` arrays surface
    records that neither side could pair off, which is exactly where OCR
    errors and missed data points hide.

    ``fused_records`` is the downstream-consumable view produced by a
    ``FusionPolicy`` (see ``src.llm.fusion``): one decision per cell or
    data-point, tagged with provenance (``pipeline``, ``vlm``, ``agreed``,
    or ``conflict``). When no policy is attached, this list is empty and
    consumers should fall back to the raw match arrays.
    """

    source_id: str
    kind: str  # "figure" or "table"
    pipeline_records: list[dict[str, Any]]
    vlm_records: list[dict[str, Any]]
    matches: list[MatchPair]
    unmatched_pipeline: list[dict[str, Any]]
    unmatched_vlm: list[dict[str, Any]]
    match_rate: float
    review_needed: bool
    vlm_model: str
    notes: list[str] = Field(default_factory=list)
    # Filled by the fusion policy if one is attached; otherwise empty and
    # consumers fall back to the raw ``matches`` / ``unmatched_*`` arrays.
    fused_records: list[Any] = Field(default_factory=list)
