"""Pluggable matchers for pairing pipeline records with VLM records.

Each matcher implements a small protocol::

    match(pipeline_records, vlm_records) -> tuple[
        list[MatchPair],     # mutually matched pairs
        list[dict],          # pipeline records left unmatched
        list[dict],          # VLM records left unmatched
    ]

The default ``NearestPointMatcher`` does greedy 1-1 assignment based on a
normalised distance with a tolerance cutoff; this is enough for inspection
(an OCR miss shows up as ``unmatched_pipeline``) without forcing the
caller to think about optimal transport.
"""

from __future__ import annotations

import abc
import math
from typing import Iterable

from src.llm.types import MatchPair


class PointMatcher(abc.ABC):
    """Match figure data points (series + numeric x/y) between pipeline & VLM."""

    @abc.abstractmethod
    def match(
        self,
        pipeline_records: list[dict],
        vlm_records: list[dict],
    ) -> tuple[list[MatchPair], list[dict], list[dict]]: ...


class CellMatcher(abc.ABC):
    """Match table cells (row, column, value) between pipeline & VLM."""

    @abc.abstractmethod
    def match(
        self,
        pipeline_records: list[dict],
        vlm_records: list[dict],
    ) -> tuple[list[MatchPair], list[dict], list[dict]]: ...


# ─────────────────────────────────────────────────────────────────── default impls


def _rel_distance(p_val: float, v_val: float, scale: float) -> float:
    if scale <= 0 or any(math.isnan(x) for x in (p_val, v_val)):
        return math.inf
    return abs(p_val - v_val) / scale


class NearestPointMatcher(PointMatcher):
    """Greedy nearest-neighbour matching on relative coordinate distance.

    Records are expected to have at least ``x`` and ``y`` keys; ``series``
    is used as a soft hint (only points sharing a series are paired
    preferentially).  Anything pair with ``max(|Δx|/range_x, |Δy|/range_y)
    > tol`` is left unmatched and shows up in the report's ``unmatched_*``
    arrays — exactly where OCR errors hide.
    """

    def __init__(self, tol: float = 0.05):
        if not 0.0 < tol < 1.0:
            raise ValueError("tol must be in (0, 1)")
        self.tol = tol

    def match(
        self,
        pipeline_records: list[dict],
        vlm_records: list[dict],
    ) -> tuple[list[MatchPair], list[dict], list[dict]]:
        if not pipeline_records or not vlm_records:
            return [], list(pipeline_records), list(vlm_records)

        # Coordinate ranges (use combined extent so x/y are weighted equally).
        def _xy(records: Iterable[dict]) -> tuple[list[float], list[float]]:
            xs, ys = [], []
            for r in records:
                if "x" in r and "y" in r:
                    try:
                        xs.append(float(r["x"]))
                        ys.append(float(r["y"]))
                    except (TypeError, ValueError):
                        continue
            return xs, ys

        all_x, all_y = _xy([*pipeline_records, *vlm_records])
        range_x = (max(all_x) - min(all_x)) if all_x else 1.0
        range_y = (max(all_y) - min(all_y)) if all_y else 1.0

        used_vlm: set[int] = set()
        matches: list[MatchPair] = []
        unmatched_pipe: list[dict] = []

        for p_rec in pipeline_records:
            try:
                px, py = float(p_rec["x"]), float(p_rec["y"])
            except (KeyError, TypeError, ValueError):
                unmatched_pipe.append(p_rec)
                continue

            best_j, best_d = -1, math.inf
            for j, v_rec in enumerate(vlm_records):
                if j in used_vlm:
                    continue
                if "x" not in v_rec or "y" not in v_rec:
                    continue
                try:
                    vx, vy = float(v_rec["x"]), float(v_rec["y"])
                except (TypeError, ValueError):
                    continue

                # Soft series penalty: same series gets the raw distance,
                # different series gets a small additive cost so it only
                # wins if no same-series candidate is close.
                d = max(_rel_distance(px, vx, range_x), _rel_distance(py, vy, range_y))
                if p_rec.get("series") and v_rec.get("series"):
                    if p_rec["series"] != v_rec["series"]:
                        d += 0.5  # diverges past tol almost certainly

                if d < best_d:
                    best_d, best_j = d, j

            if best_j >= 0 and best_d <= self.tol:
                matches.append(
                    MatchPair(
                        pipeline_record=p_rec,
                        vlm_record=vlm_records[best_j],
                        distance=best_d,
                    )
                )
                used_vlm.add(best_j)
            else:
                unmatched_pipe.append(p_rec)

        unmatched_vlm = [v for j, v in enumerate(vlm_records) if j not in used_vlm]
        return matches, unmatched_pipe, unmatched_vlm


def _canonical_text(value: object) -> str:
    """Conservative normalization for exact cell comparison."""
    if value is None:
        return ""
    text = str(value).strip()
    return " ".join(text.lower().split())


class ExactCellMatcher(CellMatcher):
    """Exact text match on cells, keyed on ``(row_idx, col_idx)`` when
    both sides supply indices; otherwise on header-aligned dict keys.

    Pipeline produces cells with ``{row, col, value}``; VLM is asked for
    the same.  Indices are integer rows / columns starting at 0.
    """

    def match(
        self,
        pipeline_records: list[dict],
        vlm_records: list[dict],
    ) -> tuple[list[MatchPair], list[dict], list[dict]]:
        # Index VLM cells by (row, col).
        by_rc: dict[tuple[int, int], int] = {}
        for j, v in enumerate(vlm_records):
            try:
                rc = (int(v["row"]), int(v["col"]))
            except (KeyError, TypeError, ValueError):
                continue
            by_rc[rc] = j

        used: set[int] = set()
        matches: list[MatchPair] = []
        unmatched_pipe: list[dict] = []
        for p in pipeline_records:
            try:
                rc = (int(p["row"]), int(p["col"]))
            except (KeyError, TypeError, ValueError):
                unmatched_pipe.append(p)
                continue
            if rc not in by_rc:
                unmatched_pipe.append(p)
                continue
            j = by_rc[rc]
            v = vlm_records[j]
            if _canonical_text(p.get("value")) == _canonical_text(v.get("value")):
                matches.append(MatchPair(pipeline_record=p, vlm_record=v, distance=0.0))
                used.add(j)
            else:
                unmatched_pipe.append(p)

        unmatched_vlm = [v for j, v in enumerate(vlm_records) if j not in used]
        return matches, unmatched_pipe, unmatched_vlm
