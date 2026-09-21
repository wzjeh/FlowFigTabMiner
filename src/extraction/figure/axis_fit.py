"""Robust axis calibration helpers (pure functions, unit-tested).

Why: the coordinate mapper fitted pixel→value lines with a RANSAC whose
residual threshold was a fixed 10.0 *value units*.  On a log axis the
targets are exponents spanning ~2 units, so a single mis-read tick
("10-1" → "101", "10-1.5" → "10-1.") was never rejected and dragged the
whole axis by factors of 3–30.  On linear axes "-10 0" → -100 did the same.

Fixes, all deterministic:
* residual threshold in the axis's own units (0.15 decades on log axes,
  4 % of the value span on linear axes, floor 1.0);
* monotonic-subsequence filter for X ticks (value must increase with x);
* fit quality reported (inliers / total, max residual) so callers can
  refuse a calibration built on fewer than 2 consistent ticks.
"""

from __future__ import annotations

import math
import re
import unicodedata
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LinearRegression


class LineModel:
    """Minimal sklearn-like predictor for one axis: value = a * pixel + b."""

    def __init__(self, slope: float, intercept: float, inliers: int, total: int, max_residual: float):
        self.slope, self.intercept = float(slope), float(intercept)
        self.inliers, self.total, self.max_residual = int(inliers), int(total), float(max_residual)

    def predict(self, X):
        arr = np.asarray(X, dtype=float).reshape(-1)
        return self.slope * arr + self.intercept

    @property
    def quality(self) -> Dict[str, Any]:
        return {"inliers": self.inliers, "total": self.total, "max_residual": round(self.max_residual, 4)}


def monotonic_subsequence(cands: Sequence[Sequence[Any]], pixel_idx: int, value_idx: int = 1,
                          direction: str = "increasing") -> List[Sequence[Any]]:
    """Longest strictly monotonic (in value) subsequence when sorted by pixel.
    Drops ticks whose value breaks the axis order (a lost minus sign, a
    stray number from the plot area)."""
    if not cands:
        return []
    ordered = sorted(cands, key=lambda c: c[pixel_idx])
    vals = [c[value_idx] for c in ordered]
    n = len(vals)
    dp, parent = [1] * n, [-1] * n
    for i in range(n):
        for j in range(i):
            ok = vals[i] > vals[j] if direction == "increasing" else vals[i] < vals[j]
            if ok and dp[j] + 1 > dp[i]:
                dp[i], parent[i] = dp[j] + 1, j
    end = max(range(n), key=lambda i: dp[i])
    keep = []
    while end != -1:
        keep.append(end); end = parent[end]
    return [ordered[i] for i in reversed(keep)]


def best_monotonic_subsequence(cands: Sequence[Sequence[Any]], pixel_idx: int, value_idx: int = 1,
                               prefer: str = "decreasing") -> Tuple[List[Sequence[Any]], str]:
    """Longest monotonic subsequence in EITHER direction (ties → ``prefer``).
    A Y axis that increases downwards (e.g. a temperature axis drawn 50 → 75
    top to bottom) used to be cut to a single tick by a decreasing-only filter."""
    dec = monotonic_subsequence(cands, pixel_idx, value_idx, "decreasing")
    inc = monotonic_subsequence(cands, pixel_idx, value_idx, "increasing")
    if len(inc) > len(dec) or (len(inc) == len(dec) and prefer == "increasing"):
        return inc, "increasing"
    return dec, "decreasing"


def decide_dual_axis(n_left_parsed: int, n_right_parsed: int, min_right: int = 3) -> bool:
    """A right Y axis needs at least ``min_right`` parseable tick candidates.
    Two or three stray numbers on the right (legend entries "-78 / -48 / 24",
    annotations "4", "Cl") are not an axis and must not feed Y_Right."""
    return n_left_parsed >= 1 and n_right_parsed >= min_right


def residual_threshold(values: Sequence[float], is_log: bool) -> float:
    if is_log:
        return 0.15                                  # decades
    span = (max(values) - min(values)) if len(values) > 1 else 0.0
    return max(1.0, 0.04 * span)


def robust_fit(pairs: Sequence[Tuple[float, float]], is_log: bool, min_inliers: int = 2) -> Optional[LineModel]:
    """Fit value = a·pixel + b, rejecting ticks that disagree with the consensus line.

    Deterministic pairwise consensus (tick counts are tiny): every 2-tick
    line is scored by how many ticks lie within the unit-aware threshold;
    the line with most inliers wins, ties go to the SMALLER |slope| — OCR
    errors (lost minus sign, fused digits, "-10 0" → -100) produce extreme
    values, hence steeper lines.  The winner is refit on its inliers.
    Returns None when fewer than ``min_inliers`` ticks agree.
    """
    pts = [(float(p), float(v)) for p, v in pairs if p is not None and v is not None]
    if len(pts) < 2:
        return None
    X = np.array([[p] for p, _ in pts]); y = np.array([v for _, v in pts])
    thr = residual_threshold(y, is_log)
    if len(pts) == 2:
        lr = LinearRegression().fit(X, y)
        return LineModel(lr.coef_[0], lr.intercept_, 2, 2, 0.0)
    best_mask, best_key = None, None
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            (p1, v1), (p2, v2) = pts[i], pts[j]
            if p1 == p2:
                continue
            a = (v2 - v1) / (p2 - p1); b = v1 - a * p1
            m = np.array([abs(a * p + b - v) <= thr for p, v in pts])
            key = (int(m.sum()), -abs(a))
            if best_key is None or key > best_key:
                best_key, best_mask = key, m
    if best_mask is None or best_mask.sum() < min_inliers:
        return None
    lr = LinearRegression().fit(X[best_mask], y[best_mask])
    resid = np.abs(lr.predict(X[best_mask]) - y[best_mask])
    return LineModel(lr.coef_[0], lr.intercept_, int(best_mask.sum()), len(pts), float(resid.max()) if len(resid) else 0.0)


_INTERNAL_SPACE = re.compile(r"^\s*[+\-]?\d+\s+\d+\s*$")


def tick_text_is_ambiguous(txt: str) -> bool:
    """OCR strings that must not become tick values: two digit groups
    separated by a space ("-10 0" would fuse to -100), or an exponent with
    its decimals cut off ("10-1." could be 10^-1 or 10^-1.5)."""
    t = (txt or "").strip()
    if _INTERNAL_SPACE.match(t):
        return True
    if t.startswith("10") and t.endswith(".") and len(t) > 3:
        return True
    return False


# ── chart geometry ──────────────────────────────────────────────────────────
def _levels(values: Sequence[float], rel_gap: float = 0.08, log: bool = False) -> int:
    vals = sorted(float(v) for v in values if v is not None and not (isinstance(v, float) and np.isnan(v)))
    if log:
        vals = [np.log10(v) for v in vals if v > 0]
    if not vals:
        return 0
    span = vals[-1] - vals[0]
    if span <= 0:
        return 1
    n = 1
    for a, b in zip(vals, vals[1:]):
        if b - a > rel_gap * span:
            n += 1
    return n


def is_grid_like(raw_data: Sequence[Dict[str, Any]], x_log: bool = True) -> Dict[str, Any]:
    """Heatmaps put their points on a lattice: a few X levels × a few Y levels
    covering most of the lattice.  Scatter plots with value labels do not.
    Returns {"grid": bool, "x_levels": int, "y_levels": int, "n": int}."""
    xs = [p.get("X") for p in raw_data if isinstance(p, dict)]
    ys = [p.get("Y_Left") for p in raw_data if isinstance(p, dict)]
    n = sum(1 for x, y in zip(xs, ys) if x is not None and y is not None)
    nx, ny = _levels([x for x in xs if x is not None], log=x_log), _levels([y for y in ys if y is not None])
    grid = bool(n >= 4 and 2 <= nx <= 15 and 2 <= ny <= 12 and nx * ny <= 1.6 * n and n <= 1.2 * nx * ny)
    return {"grid": grid, "x_levels": nx, "y_levels": ny, "n": n}


# ── tick text parsing (lifted from CoordinateMapper.map_coordinates) ────────
def is_scientific_text(txt: str) -> bool:
    """OCR / VLM string that unambiguously denotes 10^x: "10-2.0", "10 -1.5",
    "100.5", "10^-2", "10^0".  Bare "100" / "101" are NOT treated as
    exponents: on a linear 0-100 axis "100" is one hundred, and the VLM
    reader writes log ticks in the caret form, so a fused OCR "100" against a
    VLM "10^0" becomes a conflict that the geometric fit settles."""
    t = (txt or "").strip()
    return bool(re.match(r"^10(\s+[+\-]?\d|\^[+\-]?\d|[+\-]\d|\d*\.\d)", t))


def parse_tick_text(txt: str) -> Optional[float]:
    """Parse an OCR / VLM tick string into a number.

    Log-axis strings return the EXPONENT (``"10-1.5"`` → -1.5, ``"10^0.5"`` →
    0.5, ``"100.5"`` → 0.5, ``"101"`` → 1.0); plain numbers return the value
    (``"-40"`` → -40.0).  Ambiguous strings (``"-10 0"``, ``"10-1."``) and
    strings that look like body text return None.  Behaviour is identical to
    the former nested ``parse_val`` in the coordinate mapper; the VLM writes
    exponents as ``10^-1.5`` which the caret branch handles.
    """
    txt = unicodedata.normalize("NFKC", (txt or "")).strip()     # "０" → "0"
    if not txt:
        return None
    # A tick mark glued to the number ("100-", "-20-", "80_") is not a sign.
    m = re.match(r"^([+\-]?\d+(?:\.\d+)?)\s*[-\u2013\u2014_|']+$", txt)
    if m:
        txt = m.group(1)
    if tick_text_is_ambiguous(txt):
        return None
    # A number with a unit glued on ("0.5mm", "1.59 mm", "20 °C", "5%"): the
    # value is the number; the unit lives in the axis title / label text.
    m = re.match(r"^([+\-]?\d+(?:\.\d+)?)\s*(?:[a-zA-Zµμ°%]{1,6}|°C)$", txt)
    if m and not txt.startswith("10"):
        try:
            return float(m.group(1))
        except ValueError:
            pass
    digits_in = sum(1 for c in txt if c.isdigit())
    alpha_in = sum(1 for c in txt if c.isalpha())
    if alpha_in > 0 and not txt.lstrip("-").startswith("10"):
        return None
    if alpha_in > digits_in:
        return None
    m = re.match(r"^10\s+([+\-]?\d+\.?\d*)$", txt)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    m = re.match(r"^10\^([+\-]?\d+\.?\d*)$", txt)          # explicit caret (VLM form)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    if txt.startswith("10") and len(txt) > 2 and is_scientific_text(txt):
        rest_clean = re.sub(r"[^\d\.\-+]", "", txt[2:])
        if rest_clean:
            try:
                exp = float(rest_clean)
                if -10 <= exp <= 10:
                    return exp
            except ValueError:
                pass
    if txt.startswith("10") and not any(c.isdigit() for c in txt[2:]):
        return None
    clean = re.sub(r"[^\d\.\-eE+]", "", txt)
    try:
        return float(clean)
    except ValueError:
        return None


def parse_value_text(txt: Optional[str]) -> Optional[float]:
    """Parse a heatmap cell / point label: a plain number (``"43"``, ``"10"``,
    ``"_69"`` → 69).  Unlike ``parse_tick_text`` there is no 10^x rule, so
    ``"10"`` is the value ten, not an ambiguous log tick."""
    t = (txt or "").strip()
    if not t:
        return None
    if sum(c.isalpha() for c in t) > sum(c.isdigit() for c in t):
        return None
    m = re.search(r"[+\-]?\d+(?:\.\d+)?", t)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


# ── OCR × VLM fusion (deterministic) ────────────────────────────────────────
def fuse_tick_readings(ocr_text: Optional[str], vlm_text: Optional[str]) -> List[Tuple[float, str]]:
    """Candidate values for ONE tick box from two readers.

    agree → one candidate (``agree``); one reader only → that reader; both
    parse but disagree → BOTH candidates (``conflict:ocr`` / ``conflict:vlm``)
    at the same pixel, so the geometric consensus fit decides — never an
    average, never a preference.
    """
    o = parse_tick_text(ocr_text) if ocr_text else None
    v = parse_tick_text(vlm_text) if vlm_text else None
    if o is not None and v is not None:
        if abs(o - v) <= 1e-9:
            return [(o, "agree")]
        return [(o, "conflict:ocr"), (v, "conflict:vlm")]
    if o is not None:
        return [(o, "ocr")]
    if v is not None:
        return [(v, "vlm")]
    return []


def _valid_value(v: Optional[float]) -> bool:
    return isinstance(v, (int, float)) and not (isinstance(v, float) and np.isnan(v)) and 0 <= v <= 100


def fuse_value_readings(ocr_val: Optional[float], vlm_val: Optional[float],
                        policy: str = "vlm") -> Tuple[Optional[float], str]:
    """Value for ONE heatmap cell label from two readers (no geometric arbiter).

    equal → ``agree``; one reader missing / out of [0, 100] → the other;
    both valid but different → ``policy`` decides (``vlm`` | ``ocr`` |
    ``null``), and the source records the conflict either way.
    """
    o_ok, v_ok = _valid_value(ocr_val), _valid_value(vlm_val)
    if o_ok and v_ok:
        if abs(float(ocr_val) - float(vlm_val)) <= 1e-9:
            return float(ocr_val), "agree"
        if policy == "ocr":
            return float(ocr_val), "conflict:ocr"
        if policy == "null":
            return None, "conflict:null"
        return float(vlm_val), "conflict:vlm"
    if v_ok:
        return float(vlm_val), "vlm"
    if o_ok:
        return float(ocr_val), "ocr"
    return None, "none"


def _greedy_pairs(point_centers, box_centers, radius, shift=(0.0, 0.0)):
    """Greedy nearest-pair 1:1 assignment on offset-corrected distances.
    Returns ({point_idx: box_idx}, total_distance)."""
    pairs = []
    for i, (px, py) in enumerate(point_centers):
        for j, (bx, by) in enumerate(box_centers):
            d = math.hypot(bx - shift[0] - px, by - shift[1] - py)
            if d < radius:
                pairs.append((d, i, j))
    pairs.sort()
    used_p, used_b, out, total = set(), set(), {}, 0.0
    for d, i, j in pairs:
        if i in used_p or j in used_b:
            continue
        out[i] = j; used_p.add(i); used_b.add(j); total += d
    return out, total


def match_labels_to_points(point_centers, box_centers, radius: float = 120.0) -> dict:
    """One value box labels at most ONE point (and vice versa).

    The figure's label convention ("value printed below-right of the
    marker") is an offset vector.  Candidate offsets are the most frequent
    (box - point) vectors over each point's two nearest boxes (10 px grid);
    each candidate is scored by the size of the 1:1 assignment it yields on
    offset-corrected distances, then by total distance.  So a label lying
    between two marker rows goes to the marker it belongs to, not to
    whichever is a few pixels nearer.  Returns {point_idx: box_idx}.

    Without the 1:1 rule a lone stray box (a legend number such as "6" on an
    xy plot) is copied onto every point within the search radius, and on
    heatmaps points whose own label YOLO missed inherit a neighbour's value.
    """
    if not point_centers or not box_centers:
        return {}
    best, total = _greedy_pairs(point_centers, box_centers, radius)
    if len(best) < 3:
        return best
    best_key = (len(best), -total)
    votes: Dict[Tuple[int, int], int] = {}
    for px, py in point_centers:
        near = sorted(((math.hypot(bx - px, by - py), bx - px, by - py) for bx, by in box_centers))[:2]
        for d, ox, oy in near:
            if d < radius:
                key = (int(round(ox / 10.0)), int(round(oy / 10.0)))
                votes[key] = votes.get(key, 0) + 1
    for (gx, gy), _ in sorted(votes.items(), key=lambda kv: -kv[1])[:3]:
        out, t = _greedy_pairs(point_centers, box_centers, radius, (gx * 10.0, gy * 10.0))
        if (len(out), -t) > best_key:
            best, best_key = out, (len(out), -t)
    return best
