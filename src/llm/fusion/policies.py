"""FusionPolicy implementations.

A policy ingests the ``InspectionReport`` (which already carries matched +
unmatched records from the matcher) and emits ``list[FusedRecord]``.

Two policies ship by default:

- ``ModalityRoutingPolicy`` — the recommended default. For each matched
  pair, it asks ``classify_field`` what the field's role is and routes
  per the empirical winner from the paper benchmark. Disagreements past
  ``conflict_distance`` become CONFLICT records.
- ``AgreementOnlyPolicy``  — strict mode for downstream consumers that
  refuse to risk a single-side error. Only emits records where both
  sides agree; everything else is dropped (and surfaced through the
  report's ``unmatched_*`` arrays for human review).
"""

from __future__ import annotations

import abc
import logging
import math
from collections import Counter

from src.llm.fusion.roles import FieldRole, classify_field
from src.llm.fusion.types import FusedRecord, FusionSource
from src.llm.types import InspectionReport, MatchPair

logger = logging.getLogger(__name__)


class FusionPolicy(abc.ABC):
    """Map an ``InspectionReport`` to a list of ``FusedRecord``."""

    @abc.abstractmethod
    def fuse(self, report: InspectionReport) -> list[FusedRecord]: ...


def _key_for(record: dict, kind: str, index: int) -> str:
    """Build a stable key for a record, used by ``FusedRecord.record_key``."""
    if kind == "figure":
        series = record.get("series", "anon")
        return f"fig::{series}::{index}"
    # table
    row = record.get("row", "?")
    col = record.get("col", "?")
    return f"tab::r{row}::c{col}"


def _numeric_close(a, b, tol: float) -> bool:
    try:
        af, bf = float(a), float(b)
    except (TypeError, ValueError):
        return False
    scale = max(abs(af), abs(bf), 1.0)
    return abs(af - bf) / scale <= tol


class ModalityRoutingPolicy(FusionPolicy):
    """Default fusion: route per ``FieldRole``.

    The routing table is intentionally explicit (and matches the paper's
    benchmark winners). ``conflict_distance`` controls when matched pairs
    that *do* agree on coordinates but disagree on textual value get
    flagged as CONFLICT.
    """

    def __init__(self, *, numeric_tol: float = 0.05):
        self.numeric_tol = numeric_tol

    def fuse(self, report: InspectionReport) -> list[FusedRecord]:
        out: list[FusedRecord] = []

        # ── matched pairs: route per field role ────────────────────
        for idx, pair in enumerate(report.matches):
            out.extend(self._fuse_pair(pair, report.kind, idx))

        # ── pipeline-only: unmatched on the VLM side ─────────────
        offset = len(report.matches)
        for j, rec in enumerate(report.unmatched_pipeline):
            for field_key, field_value in rec.items():
                role = classify_field(field_key, field_value)
                if role == FieldRole.IDENTIFIER_AGREE:
                    # an identifier with no VLM counterpart cannot be trusted
                    out.append(
                        FusedRecord(
                            record_key=_key_for(rec, report.kind, offset + j),
                            role=role.value + f":{field_key}",
                            value=None,
                            source=FusionSource.CONFLICT,
                            pipeline_value=field_value,
                        )
                    )
                else:
                    out.append(
                        FusedRecord(
                            record_key=_key_for(rec, report.kind, offset + j),
                            role=role.value + f":{field_key}",
                            value=field_value,
                            source=FusionSource.PIPELINE_ONLY,
                            pipeline_value=field_value,
                        )
                    )

        # ── vlm-only: unmatched on the pipeline side ─────────────
        offset2 = offset + len(report.unmatched_pipeline)
        for j, rec in enumerate(report.unmatched_vlm):
            for field_key, field_value in rec.items():
                role = classify_field(field_key, field_value)
                if role == FieldRole.IDENTIFIER_AGREE:
                    out.append(
                        FusedRecord(
                            record_key=_key_for(rec, report.kind, offset2 + j),
                            role=role.value + f":{field_key}",
                            value=None,
                            source=FusionSource.CONFLICT,
                            vlm_value=field_value,
                        )
                    )
                else:
                    out.append(
                        FusedRecord(
                            record_key=_key_for(rec, report.kind, offset2 + j),
                            role=role.value + f":{field_key}",
                            value=field_value,
                            source=FusionSource.VLM_ONLY,
                            vlm_value=field_value,
                        )
                    )

        # Final source-tag breakdown logged once per inspection.
        breakdown = Counter(r.source.value for r in out)
        logger.info(
            "fusion.policy source=%s kind=%s matched_pairs=%d unmatched_pipe=%d unmatched_vlm=%d "
            "fused_records=%d sources=%s",
            report.source_id,
            report.kind,
            len(report.matches),
            len(report.unmatched_pipeline),
            len(report.unmatched_vlm),
            len(out),
            dict(breakdown),
        )
        return out

    def _fuse_pair(self, pair: MatchPair, kind: str, idx: int) -> list[FusedRecord]:
        """Per matched pair, emit one FusedRecord per shared field key."""
        p_rec, v_rec = pair.pipeline_record, pair.vlm_record
        keys = sorted(set(p_rec) | set(v_rec))

        out: list[FusedRecord] = []
        base_key = _key_for(p_rec, kind, idx)

        for k in keys:
            p_val, v_val = p_rec.get(k), v_rec.get(k)
            role = classify_field(k, p_val if p_val is not None else v_val)

            if p_val is None and v_val is None:
                continue

            if p_val is None:
                out.append(self._only(base_key, role, k, FusionSource.VLM_ONLY, vlm_value=v_val))
                continue
            if v_val is None:
                out.append(self._only(base_key, role, k, FusionSource.PIPELINE_ONLY, pipeline_value=p_val))
                continue

            agreed = self._agree(p_val, v_val, role)
            if agreed:
                out.append(
                    FusedRecord(
                        record_key=f"{base_key}::{k}",
                        role=role.value + f":{k}",
                        value=self._pick_when_agreed(p_val, v_val, role),
                        source=FusionSource.AGREED,
                        pipeline_value=p_val,
                        vlm_value=v_val,
                        distance=pair.distance,
                    )
                )
                continue

            # Disagreement — route per role.
            if role == FieldRole.NUMERIC_PIPELINE:
                out.append(
                    FusedRecord(
                        record_key=f"{base_key}::{k}",
                        role=role.value + f":{k}",
                        value=p_val,
                        source=FusionSource.PIPELINE,
                        pipeline_value=p_val,
                        vlm_value=v_val,
                        distance=pair.distance,
                    )
                )
            elif role in (FieldRole.TEXTUAL_VLM, FieldRole.STRUCTURAL_VLM):
                out.append(
                    FusedRecord(
                        record_key=f"{base_key}::{k}",
                        role=role.value + f":{k}",
                        value=v_val,
                        source=FusionSource.VLM,
                        pipeline_value=p_val,
                        vlm_value=v_val,
                        distance=pair.distance,
                    )
                )
            else:  # IDENTIFIER_AGREE or UNKNOWN — be conservative
                out.append(
                    FusedRecord(
                        record_key=f"{base_key}::{k}",
                        role=role.value + f":{k}",
                        value=None,
                        source=FusionSource.CONFLICT,
                        pipeline_value=p_val,
                        vlm_value=v_val,
                        distance=pair.distance,
                    )
                )
        return out

    # helpers
    @staticmethod
    def _only(base: str, role: FieldRole, field: str, src: FusionSource, **vals) -> FusedRecord:
        return FusedRecord(
            record_key=f"{base}::{field}",
            role=role.value + f":{field}",
            value=vals.get("pipeline_value", vals.get("vlm_value")),
            source=src,
            **vals,
        )

    def _agree(self, p_val, v_val, role: FieldRole) -> bool:
        if role == FieldRole.NUMERIC_PIPELINE:
            return _numeric_close(p_val, v_val, self.numeric_tol)
        # textual / structural / identifier: case-insensitive trimmed equality
        return str(p_val).strip().casefold() == str(v_val).strip().casefold()

    @staticmethod
    def _pick_when_agreed(p_val, v_val, role: FieldRole):
        # Prefer pipeline for numeric (preserves precision), VLM for textual
        # (preserves Unicode / typography), pipeline otherwise.
        if role in (FieldRole.TEXTUAL_VLM, FieldRole.STRUCTURAL_VLM):
            return v_val
        return p_val


class AgreementOnlyPolicy(FusionPolicy):
    """Strict mode: keep only fields where both sides agree.

    Useful for the *first* spot-check dataset where Zhao wants zero
    routing risk — everything else lands as ``unmatched_*`` for human
    review.
    """

    def __init__(self, *, numeric_tol: float = 0.05):
        self.numeric_tol = numeric_tol

    def fuse(self, report: InspectionReport) -> list[FusedRecord]:
        out: list[FusedRecord] = []
        for idx, pair in enumerate(report.matches):
            base = _key_for(pair.pipeline_record, report.kind, idx)
            for k in sorted(set(pair.pipeline_record) | set(pair.vlm_record)):
                p_val, v_val = pair.pipeline_record.get(k), pair.vlm_record.get(k)
                if p_val is None or v_val is None:
                    continue
                role = classify_field(k, p_val)
                if role == FieldRole.NUMERIC_PIPELINE:
                    if not _numeric_close(p_val, v_val, self.numeric_tol):
                        continue
                else:
                    if str(p_val).strip().casefold() != str(v_val).strip().casefold():
                        continue
                out.append(
                    FusedRecord(
                        record_key=f"{base}::{k}",
                        role=role.value + f":{k}",
                        value=p_val if role == FieldRole.NUMERIC_PIPELINE else v_val,
                        source=FusionSource.AGREED,
                        pipeline_value=p_val,
                        vlm_value=v_val,
                    )
                )
        return out
