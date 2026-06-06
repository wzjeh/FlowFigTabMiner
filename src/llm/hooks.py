"""Pipeline-hook implementations that bridge to ``src.llm`` inspectors.

These satisfy the ``src.pipeline.hooks.PipelineHook`` protocol so the
pipeline can call them without importing ``src.llm`` itself.  Each hook:

1. Runs the matching inspector (Figure or Table) on the crop + the
   pipeline-produced DataFrame.
2. Persists the resulting ``InspectionReport`` as JSON next to the
   pipeline's evidence file so the audit trail lives in the same folder.
3. Optionally runs ``src.extraction.fusion`` consistency checks **before**
   the VLM call so the report carries pipeline-internal confidence info
   alongside the cross-source comparison.

These hooks are constructed and injected by ``src/pipeline/main.py``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from src.extraction.fusion import ConsistencyCheck, run_checks
from src.llm.inspectors.figure import FigureInspector
from src.llm.inspectors.table import TableInspector
from src.pipeline.hooks import StageContext

logger = logging.getLogger(__name__)


def _inspection_path(ctx: StageContext) -> Path:
    """Where to drop the audit report next to the evidence file.

    Renamed from ``_vlm_inspection.json`` to ``_vlm_audit.json`` once the
    inspection track was demoted to forensic-only — downstream
    LocalVarsBuilder / GlobalAssembly no longer read this file; it is
    written purely for human / regression-tracking review.
    """
    return ctx.output_dir / f"{ctx.source_id}_vlm_audit.json"


def _write_report(report, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(report.model_dump_json(indent=2, exclude_none=False))
    logger.info(
        "pipeline.hook.report kind=%s source=%s path=%s match_rate=%.2f review_needed=%s",
        report.kind,
        report.source_id,
        dest,
        report.match_rate,
        report.review_needed,
    )


@dataclass
class FigureInspectionHook:
    """Post-extraction hook for the figure track (paper module 6).

    Workflow per figure:

    1. Run any provided ``consistency_checks`` against the pipeline's
       stage outputs (YOLO point count vs OCR value count, etc.).  These
       record per-stage issues to the log and contribute a
       ``pipeline_confidence`` to the StageContext.
    2. Hand the crop + the pipeline DataFrame to the ``FigureInspector``
       so the VLM re-extracts independently.
    3. Persist the ``InspectionReport`` (including fused records when a
       fusion policy is attached to the inspector) next to the evidence
       JSON the pipeline already wrote.
    """

    inspector: FigureInspector
    consistency_checks: Iterable[ConsistencyCheck] = ()
    name: str = "figure_vlm_inspection"

    def run(self, ctx: StageContext) -> StageContext:
        if ctx.kind != "figure":
            return ctx

        # Layer 1 — pipeline-internal consistency checks (logged + folded
        # into pipeline_confidence on the context for downstream).
        if list(self.consistency_checks):
            result = run_checks(
                self.consistency_checks,
                ctx.stage_outputs,
                stage_tag="figure_pipeline.assembly",
                source_id=ctx.source_id,
            )
            ctx.stage_outputs.setdefault("pipeline_confidence", 1.0)
            ctx.stage_outputs["pipeline_confidence"] += result.confidence_delta
            ctx.stage_outputs["fusion_issues"] = [i.model_dump() for i in result.issues]

        # Layer 2 — VLM cross-check via inspector
        report = self.inspector.inspect(
            image_path=ctx.artifact_path,
            pipeline_df=ctx.evidence_df,
            source_id=ctx.source_id,
        )
        _write_report(report, _inspection_path(ctx))
        return ctx


@dataclass
class TableInspectionHook:
    """Post-extraction hook for the table track (paper module 12)."""

    inspector: TableInspector
    consistency_checks: Iterable[ConsistencyCheck] = ()
    name: str = "table_vlm_inspection"

    def run(self, ctx: StageContext) -> StageContext:
        if ctx.kind != "table":
            return ctx

        if list(self.consistency_checks):
            result = run_checks(
                self.consistency_checks,
                ctx.stage_outputs,
                stage_tag="table_pipeline.assembly",
                source_id=ctx.source_id,
            )
            ctx.stage_outputs.setdefault("pipeline_confidence", 1.0)
            ctx.stage_outputs["pipeline_confidence"] += result.confidence_delta
            ctx.stage_outputs["fusion_issues"] = [i.model_dump() for i in result.issues]

        report = self.inspector.inspect(
            image_path=ctx.artifact_path,
            pipeline_df=ctx.evidence_df,
            source_id=ctx.source_id,
        )
        _write_report(report, _inspection_path(ctx))
        return ctx
