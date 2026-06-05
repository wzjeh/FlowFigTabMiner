"""Table inspector — paper module 12.

The pipeline's table track produces a row/column-indexed cell list; this
inspector asks the VLM to re-read the same crop and surfaces cells that
disagree.  Disagreement is the canonical OCR-error signature: the
pipeline reads ``"3.2 kV"`` while the VLM sees ``"32 kV"`` (decimal-point
dropped by PaddleOCR).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.llm.config import VLMConfig
from src.llm.errors import ParseError
from src.llm.fusion.policies import FusionPolicy
from src.llm.inspectors.matchers import CellMatcher, ExactCellMatcher
from src.llm.providers.base import VLMProvider
from src.llm.types import InspectionReport, VLMImage

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent / "prompts" / "table.md"
_PROMPT = _PROMPT_PATH.read_text()


def _df_to_cells(df: pd.DataFrame) -> list[dict]:
    """Convert a pipeline DataFrame into the inspector's cell schema.

    The pipeline table track emits one row per logical table row, with
    columns named by their headers; we melt that into ``{row, col, value}``
    tuples so the matcher can compare position-by-position.
    """
    if df.empty:
        return []
    headers = list(df.columns)
    out: list[dict] = []
    for row_idx, row in df.reset_index(drop=True).iterrows():
        for col_idx, header in enumerate(headers):
            out.append({"row": int(row_idx), "col": col_idx, "header": header, "value": row[header]})
    return out


class TableInspector:
    """Inspector for one extracted table."""

    def __init__(
        self,
        vlm: VLMProvider,
        cfg: VLMConfig,
        matcher: CellMatcher | None = None,
        policy: FusionPolicy | None = None,
    ):
        self.vlm = vlm
        self.cfg = cfg
        self.matcher = matcher or ExactCellMatcher()
        self.policy = policy

    def inspect(
        self,
        image_path: Path,
        pipeline_df: pd.DataFrame,
        source_id: str,
    ) -> InspectionReport:
        pipeline_records = _df_to_cells(pipeline_df)
        notes: list[str] = []

        image = VLMImage(path=image_path, mime_type=_guess_mime(image_path))
        try:
            meta, payload = self.vlm.inspect(
                image=image,
                system_prompt="",
                user_prompt=_PROMPT,
                cfg=self.cfg,
            )
        except ParseError as exc:
            notes.append(f"vlm response unparseable: {exc}")
            return InspectionReport(
                source_id=source_id,
                kind="table",
                pipeline_records=pipeline_records,
                vlm_records=[],
                matches=[],
                unmatched_pipeline=pipeline_records,
                unmatched_vlm=[],
                match_rate=0.0,
                review_needed=True,
                vlm_model=self.cfg.model,
                notes=notes,
            )

        vlm_records = _extract_cells(payload)
        matches, unmatched_p, unmatched_v = self.matcher.match(pipeline_records, vlm_records)

        denom = max(len(pipeline_records), len(vlm_records), 1)
        match_rate = len(matches) / denom
        review_needed = match_rate < self.cfg.match_threshold

        logger.info(
            "table inspect source=%s vlm=%s pipe=%d vlm_rec=%d matched=%d rate=%.2f review=%s",
            source_id,
            meta.model,
            len(pipeline_records),
            len(vlm_records),
            len(matches),
            match_rate,
            review_needed,
        )

        report = InspectionReport(
            source_id=source_id,
            kind="table",
            pipeline_records=pipeline_records,
            vlm_records=vlm_records,
            matches=matches,
            unmatched_pipeline=unmatched_p,
            unmatched_vlm=unmatched_v,
            match_rate=match_rate,
            review_needed=review_needed,
            vlm_model=meta.model,
            notes=notes,
        )
        if self.policy is not None:
            report.fused_records = [r.model_dump() for r in self.policy.fuse(report)]
        return report


def _extract_cells(payload: dict) -> list[dict]:
    cells = payload.get("cells") or []
    if not isinstance(cells, list):
        return []
    out: list[dict] = []
    for c in cells:
        if not isinstance(c, dict):
            continue
        if "row" not in c or "col" not in c or "value" not in c:
            continue
        out.append({"row": c["row"], "col": c["col"], "value": c["value"]})
    return out


def _guess_mime(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".")
    return {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "tif": "image/tiff",
        "tiff": "image/tiff",
        "bmp": "image/bmp",
        "webp": "image/webp",
    }.get(suffix, "image/png")
