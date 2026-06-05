"""Typed evidence schema.

Two top-level models — ``FigureEvidence`` and ``TableEvidence`` — are
written by the new assemblers as the canonical ``_evidence.json`` files.
Each field carries a ``FieldValue`` wrapper that records which extractor
produced it, plus latency and any notes.  ``EvidenceStatus`` flags the
table when the new ``TableCellExtractor`` fails to align (R1 circuit
breaker): downstream consumers must skip rather than guess.
"""

from __future__ import annotations

import enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class FieldSource(str, enum.Enum):
    """Who produced a value.

    Order roughly follows the data-flow: visual detectors first, then
    OCR / structured-text models, then VLM and LLM.  ``MISSING`` /
    ``ALIGNMENT_FAILED`` are non-success states that downstream consumers
    use to decide whether to skip a record.
    """

    PIPELINE_YOLO = "pipeline_yolo"             # YOLO + RANSAC for data points
    PIPELINE_PADDLE = "pipeline_paddle"          # PaddleOCR for axis ticks / captions
    TATR = "tatr"                                # cell grid (row/col, bbox)
    MOLNEXTR = "molnextr"                        # image → SMILES
    VLM_METADATA = "vlm_metadata"                # Gemini metadata-only call (figure)
    VLM_CELL = "vlm_cell"                        # Gemini full-cell call (table)
    LLM_JUDGE = "llm_judge"                      # 1-shot LLM judgment (e.g. header row)
    MISSING = "missing"                          # extractor declined / failed gracefully
    ALIGNMENT_FAILED = "alignment_failed"        # R1: cell grid alignment circuit-broke


class EvidenceStatus(str, enum.Enum):
    """Top-level status of a single evidence record."""

    OK = "ok"
    ALIGNMENT_FAILED = "alignment_failed"        # R1: table cells did not align even after retry
    EXTRACTION_FAILED = "extraction_failed"      # an extractor threw and we couldn't recover


class FieldValue(BaseModel):
    """One field's value plus provenance.

    ``value`` is the actual datum (string, number, list — Pydantic
    accepts ``Any``).  When a field is unavailable, the wrapper still
    materializes with ``value=None`` and ``source=MISSING`` so consumers
    can tell "we tried and failed" from "this field was never collected".
    """

    model_config = ConfigDict(frozen=True)

    value: Any
    source: FieldSource
    model_id: Optional[str] = None
    latency_ms: Optional[float] = None
    notes: Optional[str] = None


# ─────────────────────────────────────────────────────────────── Figure schema


class DataPoint(BaseModel):
    """One (x, y) point on a figure.

    ``series_id`` is the marker-colour cluster id assigned by the
    pipeline; ``series_name`` is the chemistry label that the VLM
    metadata extractor associates with that colour (None until the
    metadata call resolves the legend).
    """

    model_config = ConfigDict(frozen=True)

    series_id: int
    x: float
    y: float
    y_secondary: Optional[float] = None          # for dual-y or heatmap z value
    series_name: Optional[str] = None
    series_color_hint: Optional[str] = None      # e.g. "red triangle"


class FigureEvidence(BaseModel):
    """All evidence extracted for one figure (one ``_evidence.json``)."""

    model_config = ConfigDict(frozen=True)

    source_id: str
    status: EvidenceStatus = EvidenceStatus.OK
    is_relevant: bool

    # ── metadata fields (owner: Gemini metadata-only call) ──────────
    title: FieldValue
    x_axis_label: FieldValue
    y_axis_label: FieldValue
    x_axis_unit: FieldValue
    y_axis_unit: FieldValue
    legend_series_names: FieldValue              # value: list[str]
    footnote: FieldValue

    # ── data points (owner: Pipeline YOLO + RANSAC) ─────────────────
    data_points: list[DataPoint] = Field(default_factory=list)
    data_points_source: FieldSource = FieldSource.PIPELINE_YOLO

    # ── caption (owner: PaddleOCR; used for relevance filter) ───────
    caption: FieldValue


# ─────────────────────────────────────────────────────────────── Table schema


class TableCell(BaseModel):
    """One cell in a table grid."""

    model_config = ConfigDict(frozen=True)

    row: int
    col: int
    text: str                                    # may be ""
    smiles: Optional[str] = None                 # populated by MolNexTR when applicable


class TableEvidence(BaseModel):
    """All evidence extracted for one table (one ``_evidence.json``)."""

    model_config = ConfigDict(frozen=True)

    source_id: str
    status: EvidenceStatus = EvidenceStatus.OK
    is_relevant: bool

    # ── grid (owner: TATR) ──────────────────────────────────────────
    row_count: int
    col_count: int

    # ── cells (owner: Gemini full-cell call; SMILES owner: MolNexTR) ─
    cells: list[TableCell] = Field(default_factory=list)
    cells_source: FieldSource = FieldSource.VLM_CELL

    # ── header row (owner: 1-shot LLM judge) ───────────────────────
    header_row_count: FieldValue                  # value: int ∈ {0, 1, 2}
    header_confidence: float = 0.0                # raw confidence from LLM

    # ── caption + footnote (owner: PaddleOCR) ───────────────────────
    caption: FieldValue
    footnote: FieldValue
