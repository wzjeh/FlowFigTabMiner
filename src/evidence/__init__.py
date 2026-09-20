"""Typed evidence schema and per-source assembly.

This subpackage replaces the dict-based ``_evidence.json`` produced by
the legacy ``src/assembly/evidence_assembler.py``.  Each field carries
explicit provenance (which tool / model produced it, latency, notes), so
downstream LocalVarsBuilder + GlobalAssembly can reason about quality
and skip fields that came from a failed extractor.

Architecture
------------
- ``types``        — Pydantic schemas (FieldValue, FigureEvidence,
                      FieldSource enum, EvidenceStatus).  Table evidence is
                      the plain dict written by ``src/extraction/table/pipeline.py``.

The per-field owner table — who's responsible for which field — lives in
this package's docstring rather than as code, because it's enforced by
the assemblers' constructors: each assembler takes exactly the inputs it
needs.  No registry of strategies is needed for this scale.
"""

from __future__ import annotations

from src.evidence.types import (
    DataPoint,
    EvidenceStatus,
    FieldSource,
    FieldValue,
    FigureEvidence,
)

__all__ = [
    "FieldSource",
    "FieldValue",
    "EvidenceStatus",
    "DataPoint",
    "FigureEvidence",
]
