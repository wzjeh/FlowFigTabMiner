"""Figure inspector — paper module 6.

Takes a figure crop + the pipeline-produced (series, x, y) DataFrame, asks
the VLM to read the same crop independently, then runs a ``PointMatcher``
to surface unmatched points.  The resulting ``InspectionReport`` flags
``review_needed`` when the match rate falls under the threshold.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.llm.config import VLMConfig
from src.llm.errors import ParseError
from src.llm.fusion.policies import FusionPolicy
from src.llm.inspectors.matchers import NearestPointMatcher, PointMatcher
from src.llm.providers.base import VLMProvider
from src.llm.types import InspectionReport, VLMImage

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent / "prompts" / "figure.md"
_PROMPT = _PROMPT_PATH.read_text()


def _df_to_records(df: pd.DataFrame) -> list[dict]:
    """Coerce a pipeline DataFrame into the inspector schema.

    The pipeline emits a few different column conventions across figure
    kinds; this normalizer reduces them to ``{series, x, y, value?}``.
    """
    if df.empty:
        return []

    # Resolve aliases produced by the figure pipeline.
    aliases = {
        "X": "x",
        "Y": "y",
        "Y_left": "y",
        "Y_Left": "y",
        "Y_right": "value",
        "Y_Right": "value",
        "Data_Value": "value",
        "series_label": "series",
        "Series": "series",
    }
    df = df.rename(columns={k: v for k, v in aliases.items() if k in df.columns})

    cols = [c for c in ("series", "x", "y", "value") if c in df.columns]
    if not {"x", "y"}.issubset(cols):
        return []  # no usable points
    return df[cols].to_dict(orient="records")


class FigureInspector:
    """Inspector for a single figure.

    Stateless per-call; one instance can fan out across many figures of a
    paper.  The provider and matcher are injected so ``main.py`` controls
    *which* VLM is consulted and *how* points are matched.
    """

    def __init__(
        self,
        vlm: VLMProvider,
        cfg: VLMConfig,
        matcher: PointMatcher | None = None,
        policy: FusionPolicy | None = None,
    ):
        self.vlm = vlm
        self.cfg = cfg
        self.matcher = matcher or NearestPointMatcher(tol=0.05)
        self.policy = policy   # if None, the report's fused_records stays empty

    def inspect(
        self,
        image_path: Path,
        pipeline_df: pd.DataFrame,
        source_id: str,
    ) -> InspectionReport:
        pipeline_records = _df_to_records(pipeline_df)
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
                kind="figure",
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

        vlm_records = _extract_points(payload)
        matches, unmatched_p, unmatched_v = self.matcher.match(pipeline_records, vlm_records)

        denom = max(len(pipeline_records), len(vlm_records), 1)
        match_rate = len(matches) / denom
        review_needed = match_rate < self.cfg.match_threshold

        logger.info(
            "figure inspect source=%s vlm=%s pipe=%d vlm_rec=%d matched=%d rate=%.2f review=%s",
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
            kind="figure",
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


def _extract_points(payload: dict) -> list[dict]:
    """Pull a flat list of points out of the VLM JSON schema."""
    pts = payload.get("points") or []
    if not isinstance(pts, list):
        return []
    out: list[dict] = []
    for p in pts:
        if not isinstance(p, dict):
            continue
        if "x" not in p or "y" not in p:
            continue
        out.append({k: p.get(k) for k in ("series", "x", "y", "value") if k in p})
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
