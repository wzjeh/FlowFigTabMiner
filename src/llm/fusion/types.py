"""Typed output of fusion-layer 2."""

from __future__ import annotations

import enum
from typing import Any, Optional

from pydantic import BaseModel


class FusionSource(str, enum.Enum):
    """Which side a fused value comes from."""

    PIPELINE = "pipeline"        # picked from the local pipeline output
    VLM = "vlm"                  # picked from the VLM re-extraction
    AGREED = "agreed"            # both sides agreed within tolerance
    CONFLICT = "conflict"        # disagreement past tolerance; both kept
    PIPELINE_ONLY = "pipeline_only"  # VLM absent for this cell/point
    VLM_ONLY = "vlm_only"            # pipeline absent for this cell/point


class FusedRecord(BaseModel):
    """One downstream-ready row.

    ``value`` is the chosen value when ``source != CONFLICT``.  In
    conflict mode both ``pipeline_value`` and ``vlm_value`` are non-null
    and ``value`` is set to ``None`` to force the consumer to look.
    """

    record_key: str               # stable id: "fig3_pt12" or "tab2_r5_c3"
    role: str                     # e.g. "numeric_pipeline", "textual_vlm"
    value: Any = None
    source: FusionSource
    pipeline_value: Optional[Any] = None
    vlm_value: Optional[Any] = None
    distance: Optional[float] = None    # disagreement magnitude (if quantifiable)
