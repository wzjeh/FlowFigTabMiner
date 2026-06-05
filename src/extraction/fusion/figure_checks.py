"""Concrete consistency checks for the figure extraction track.

Currently shipping
------------------
- ``PointCountConsistency`` — guards against the most common figure error
  mode: YOLO ``data_point`` detection count diverges from PaddleOCR
  ``value_label`` count. When the gap is large, the chart probably has
  missed points (YOLO undercount) or hallucinated points (YOLO overcount).

Planned follow-ups (separate PRs, not in this scaffold)
-------------------------------------------------------
- ``RansacInlierConsistency``  — flag ``axis_fit_inlier_ratio < 0.7``.
- ``LegendColorMargin``        — flag legend colors with pairwise CIE Lab
                                  distance < threshold (likely series swap).
- ``OcrParseRate``             — flag pages where < 80 % of tick labels
                                  parse as numeric.
"""

from __future__ import annotations

import logging
from typing import Any

from src.extraction.fusion.base import (
    ConsistencyCheck,
    ConsistencyResult,
    Issue,
    Severity,
)

logger = logging.getLogger(__name__)


class PointCountConsistency(ConsistencyCheck):
    """Cross-check YOLO data-point count vs OCR value-label count.

    When YOLO finds 20 data points but OCR only reads 12 value labels (or
    vice versa), the chart probably has an extraction error somewhere.
    Stricter mismatches incur larger confidence penalties.

    Required keys in ``stage_outputs``:
      - ``yolo_point_count``  (int) — YOLOv11m-DataDet ``data_point`` detections
      - ``ocr_value_count``   (int) — PaddleOCR-detected numeric value labels

    Optional keys:
      - ``expected_min`` (int) — caller's minimum-plausible count.
    """

    id = "point_count_consistency"
    required_keys = ("yolo_point_count", "ocr_value_count")

    def __init__(
        self,
        *,
        warn_ratio: float = 0.5,   # 50 % gap → warning
        error_ratio: float = 0.25,  # 75 % gap → error
    ):
        if not 0.0 < error_ratio < warn_ratio < 1.0:
            raise ValueError("must satisfy 0 < error_ratio < warn_ratio < 1")
        self.warn_ratio = warn_ratio
        self.error_ratio = error_ratio

    def run(self, stage_outputs: dict[str, Any]) -> ConsistencyResult:
        missing = self._missing_keys(stage_outputs)
        if missing is not None:
            return ConsistencyResult(issues=[missing])

        yolo_n = int(stage_outputs["yolo_point_count"])
        ocr_n = int(stage_outputs["ocr_value_count"])

        if yolo_n == 0 and ocr_n == 0:
            logger.debug("point_count_consistency yolo=0 ocr=0 → skip (empty chart)")
            return ConsistencyResult()  # nothing to check; downstream handles

        ratio = min(yolo_n, ocr_n) / max(yolo_n, ocr_n, 1)
        logger.debug(
            "point_count_consistency yolo=%d ocr=%d ratio=%.3f warn=%.2f err=%.2f",
            yolo_n,
            ocr_n,
            ratio,
            self.warn_ratio,
            self.error_ratio,
        )

        if ratio >= self.warn_ratio:
            # gap is small enough that label count vs point count drift is
            # plausibly just unlabelled-by-design points (common in dense
            # scatter plots); no action.
            return ConsistencyResult()

        if ratio >= self.error_ratio:
            return ConsistencyResult(
                issues=[
                    Issue(
                        check=self.id,
                        severity=Severity.WARNING,
                        message=f"YOLO {yolo_n} pts vs OCR {ocr_n} value labels (ratio {ratio:.2f})",
                        detail={"yolo": yolo_n, "ocr": ocr_n, "ratio": ratio},
                    )
                ],
                confidence_delta=-0.1,
            )

        return ConsistencyResult(
            issues=[
                Issue(
                    check=self.id,
                    severity=Severity.ERROR,
                    message=(
                        f"large gap: YOLO {yolo_n} pts vs OCR {ocr_n} labels (ratio {ratio:.2f}). "
                        "One side almost certainly missed half the chart."
                    ),
                    detail={"yolo": yolo_n, "ocr": ocr_n, "ratio": ratio},
                )
            ],
            confidence_delta=-0.3,
        )
