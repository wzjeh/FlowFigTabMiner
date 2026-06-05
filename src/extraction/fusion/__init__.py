"""Fusion layer 1 — pipeline-internal multi-source reconciliation.

Where layer 2 (``src/llm/fusion/``) compares the whole pipeline output
against an independent VLM re-extraction, this layer reconciles signals
*inside* the pipeline before any external comparison.

Sources currently in scope (signals that already exist in the pipeline
but are not cross-checked today):

- YOLO data-point detections (count, confidence, bbox spatial pattern)
- PaddleOCR tick-label / value-label parses (success rate, confidence)
- RANSAC axis-fit inlier ratio (gold confidence signal, currently ignored)
- TATR cell-grid resolution vs. PaddleOCR text-token count
- MolNexTR SMILES vs. SchemeParser compound-pool SMILES for the same label
- Legend matcher color distance vs. marker-shape match

Architecture
------------
- ``ConsistencyCheck`` ABC — receives a stage's outputs as a ``dict`` and
  returns a ``ConsistencyResult`` listing issues (with severity) plus a
  ``confidence_delta`` that downstream consumers can fold into a
  per-record ``pipeline_confidence`` field.
- ``ConsistencyResult`` — typed report; multiple checks can be combined.
- Concrete check classes live in this package; the bundled default is
  ``PointCountConsistency`` (figure track, YOLO vs OCR count guard) as a
  scaffold proving the architecture. Additional checks are independent
  units of work and ship in follow-up PRs.
"""

from __future__ import annotations

from src.extraction.fusion.base import (
    ConsistencyCheck,
    ConsistencyResult,
    Issue,
    Severity,
    run_checks,
)
from src.extraction.fusion.figure_checks import PointCountConsistency

__all__ = [
    "ConsistencyCheck",
    "ConsistencyResult",
    "Issue",
    "Severity",
    "run_checks",
    "PointCountConsistency",
]
