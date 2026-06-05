"""Pipeline hook protocol — extension points for post-extraction work.

This module sits at the centre of the pipeline / inspection seam: the
figure track and table track each finish their extraction, build a
``StageContext`` describing what was produced, and call the registered
hooks in sequence.  A hook may attach extra files (e.g. a VLM inspection
report), enrich the context for downstream hooks, or simply log.

Hooks are injected by ``src/pipeline/main.py``; pipelines have no
knowledge of which hooks (if any) are attached and never import the
``src.llm`` subpackage.  This is the dependency-inversion seam that lets
the same pipeline run with VLM inspection in production and without it in
local development / tests.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable, Literal, Protocol, runtime_checkable

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class StageContext(BaseModel):
    """Everything a post-extraction hook might need.

    Pydantic carries the typed fields; ``stage_outputs`` is an escape
    hatch for raw signals (YOLO confidences, RANSAC inlier ratio, OCR
    token counts) the hook may consume — most useful for the
    pipeline-internal consistency checks in ``src.extraction.fusion``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    pdf_basename: str
    source_id: str                          # figure_id or table_basename
    kind: Literal["figure", "table"]
    artifact_path: Path                     # path to the crop the VLM should see
    output_dir: Path                        # where hook outputs (e.g. inspection JSON) go
    evidence_df: pd.DataFrame = Field(default_factory=pd.DataFrame)
    stage_outputs: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class PipelineHook(Protocol):
    """One post-extraction extension point.

    The contract is intentionally minimal:
    - ``name`` is a stable identifier used in logs / dedupe.
    - ``run(ctx)`` returns the (possibly enriched) context; if the hook
      decides to abort downstream hooks it should still return the ctx
      and let the runner surface the situation through logs.
    """

    name: str

    def run(self, ctx: StageContext) -> StageContext: ...


def run_hooks(
    hooks: Iterable[PipelineHook],
    ctx: StageContext,
) -> StageContext:
    """Apply ``hooks`` in order, logging each invocation.

    Failures are caught per-hook so a broken inspection doesn't abort the
    whole pipeline run; the offending hook gets an ERROR log and the
    context propagates unchanged.
    """
    hooks_list = list(hooks)
    if not hooks_list:
        return ctx
    for hook in hooks_list:
        try:
            logger.info(
                "pipeline.hook stage=%s source=%s hook=%s",
                ctx.kind,
                ctx.source_id,
                getattr(hook, "name", type(hook).__name__),
            )
            ctx = hook.run(ctx)
        except Exception as exc:
            logger.exception(
                "pipeline.hook.error stage=%s source=%s hook=%s exc=%s",
                ctx.kind,
                ctx.source_id,
                getattr(hook, "name", type(hook).__name__),
                exc,
            )
    return ctx
