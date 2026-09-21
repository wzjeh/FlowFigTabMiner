"""Filled bars in a bar chart (OpenCV, no model).

The micro detector is a scatter-plot model: a bar chart yields zero data
points.  When the mapper has a Y-axis model and at least two X tick labels
but no points, the filled rectangles standing on the baseline are the data:
each bar's top edge is its value, its centre its category.  Pure image
geometry, gated by the caller; hatched or outlined bars are not detected
(fill ratio), which errs on the side of no data rather than wrong data.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np


def detect_bars(img: np.ndarray, x_range: Sequence[float], y_range: Sequence[float],
                min_w: int = 8, min_h: int = 8, fill_ratio: float = 0.85,
                ink_threshold: int = 235, baseline: float = None) -> List[Dict[str, Any]]:
    """Solid bars inside the plot area ``x_range`` × ``y_range`` (pixels).

    Returns ``[{cx, top, bottom, x1, x2, w, h}]`` sorted left to right.  A
    component counts as a bar when it is at least ``min_w`` × ``min_h``, no
    wider than half the plot, filled (area / bbox area ≥ ``fill_ratio``) and
    taller than it is thin (axis lines and text are rejected by size / fill).
    With ``baseline`` (pixel y of the lowest Y tick) a component must stand on
    it — its bottom no higher than 6 % of the plot height above it — which
    rejects legend swatches and in-plot boxes.
    """
    if img is None or img.size == 0:
        return []
    h_img, w_img = img.shape[:2]
    x0, x1 = int(max(0, min(x_range))), int(min(w_img, max(x_range)))
    y0, y1 = int(max(0, min(y_range))), int(min(h_img, max(y_range)))
    if x1 - x0 < 2 * min_w or y1 - y0 < min_h:
        return []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    roi = gray[y0:y1, x0:x1]
    ink = (roi < ink_threshold).astype(np.uint8)
    # drop hairlines first (axis / grid lines up to 4 px, which would otherwise
    # glue every bar standing on the axis into one wide component), then close
    # anti-aliased edges
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    plot_w = x1 - x0
    bars = []
    for i in range(1, n):
        bx, by, bw, bh, area = (int(v) for v in stats[i])
        if bw < min_w or bh < min_h or bw > 0.5 * plot_w:
            continue
        if area < fill_ratio * bw * bh:
            continue                       # outline, hatch, text, marker cluster
        if baseline is not None and (y0 + by + bh) < baseline - 0.06 * (y1 - y0):
            continue                       # floating box (legend swatch), not a bar on the axis
        bars.append({"cx": x0 + bx + bw / 2.0, "top": float(y0 + by), "bottom": float(y0 + by + bh),
                     "x1": float(x0 + bx), "x2": float(x0 + bx + bw), "w": bw, "h": bh})
    bars.sort(key=lambda b: b["cx"])
    return bars
