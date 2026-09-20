"""Numbered contact sheet of label crops for constrained VLM reading.

Every YOLO box the coordinate mapper is about to OCR (axis ticks, heatmap
cell labels) is cropped with the mapper's own padding / upscale rules and
tiled into one image, each cell headed by its index.  A single VLM call then
transcribes "cell #i" → text; the reader never has to locate anything, which
is what keeps hallucination out of the loop.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np

HEADER_H = 22          # px reserved above each crop for the index label
GAP = 8                # px between cells


def crop_box(img: np.ndarray, box: Sequence[float], pad: int = 15, min_side: int = 60,
             max_side: int = 320) -> Tuple[np.ndarray, List[int]]:
    """Crop ``box`` from ``img`` with padding; upscale tiny crops (cubic) so the
    text is legible; cap the longest side.  Returns (crop, [x1, y1, x2, y2])."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return np.full((min_side, min_side, 3), 255, dtype=np.uint8), [x1, y1, x2, y2]
    ch, cw = crop.shape[:2]
    scale = 1.0
    if min(ch, cw) < min_side:
        scale = min_side / max(1, min(ch, cw))
    if max(ch, cw) * scale > max_side:
        scale = max_side / max(ch, cw)
    if abs(scale - 1.0) > 1e-3:
        crop = cv2.resize(crop, (max(1, int(cw * scale)), max(1, int(ch * scale))),
                          interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
    return crop, [x1, y1, x2, y2]


def build_contact_sheet(img: np.ndarray, dets: Sequence[Dict[str, Any]], *, pad: int = 15,
                        min_side: int = 60, cols: int = 4, cell_max: int = 320
                        ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Tile the crops of ``dets`` (dicts with ``box``) into a numbered grid.

    Returns ``(sheet_bgr, cells)`` where ``cells[i] == {"index": i, "det":
    dets[i], "crop_box": [...]}`` — the index printed above cell i is exactly
    the position of the detection in ``dets``.
    """
    cells: List[Dict[str, Any]] = []
    crops: List[np.ndarray] = []
    for i, d in enumerate(dets):
        crop, cb = crop_box(img, d["box"], pad=pad, min_side=min_side, max_side=cell_max)
        crops.append(crop)
        cells.append({"index": i, "det": d, "crop_box": cb})
    if not crops:
        return np.full((min_side, min_side, 3), 255, dtype=np.uint8), cells
    cols = max(1, min(cols, len(crops)))
    rows = (len(crops) + cols - 1) // cols
    col_w = [0] * cols
    row_h = [0] * rows
    for i, c in enumerate(crops):
        r, k = divmod(i, cols)
        col_w[k] = max(col_w[k], c.shape[1])
        row_h[r] = max(row_h[r], c.shape[0] + HEADER_H)
    width = sum(col_w) + GAP * (cols + 1)
    height = sum(row_h) + GAP * (rows + 1)
    sheet = np.full((height, width, 3), 255, dtype=np.uint8)
    y = GAP
    for r in range(rows):
        x = GAP
        for k in range(cols):
            i = r * cols + k
            if i < len(crops):
                c = crops[i]
                label = f"#{i}"
                cv2.putText(sheet, label, (x + 2, y + HEADER_H - 6), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (200, 0, 0), 2, cv2.LINE_AA)
                cy = y + HEADER_H
                sheet[cy:cy + c.shape[0], x:x + c.shape[1]] = c
                cv2.rectangle(sheet, (x - 2, y - 2), (x + col_w[k] + 1, y + row_h[r] + 1), (180, 180, 180), 1)
            x += col_w[k] + GAP
        y += row_h[r] + GAP
    return sheet, cells
