"""Fusion layer 2 — route between pipeline and VLM per field type.

Layer 1 (``src/extraction/fusion/``) cleans up the pipeline's internal
multi-source disagreements first; layer 2 then asks the VLM to read the
same artifact independently and decides per-field who wins.

Public types
------------
- ``FusedRecord``      — one decision per cell / data-point.
- ``FusionPolicy``     — ABC: given the matcher's output, emit ``FusedRecord``s.
- ``ModalityRoutingPolicy`` — default impl: numeric ⇒ pipeline, textual ⇒ VLM,
  with explicit conflict tagging where the two sides disagree past tolerance.
- ``FIELD_ROLES`` / ``classify_field`` — the role catalogue and resolver.
"""

from __future__ import annotations

from src.llm.fusion.policies import (
    AgreementOnlyPolicy,
    FusionPolicy,
    ModalityRoutingPolicy,
)
from src.llm.fusion.roles import (
    FIELD_ROLES,
    FieldRole,
    classify_field,
)
from src.llm.fusion.types import FusedRecord, FusionSource

__all__ = [
    "FusedRecord",
    "FusionSource",
    "FusionPolicy",
    "AgreementOnlyPolicy",
    "ModalityRoutingPolicy",
    "FIELD_ROLES",
    "FieldRole",
    "classify_field",
]
