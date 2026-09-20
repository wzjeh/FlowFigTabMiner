"""Series recovery for xy plots whose legend the macro YOLO did not crop.

When ``elements['legend']`` is empty, ``LegendMatcher.parse_legend_crops``
returns no prototypes and every point ends up ``Series=Default`` even though
the VLM metadata call read 2-5 series names.  This module rebuilds the same
``{name: {"hsv": np.array([H, S, V]), "count": 1}}`` prototype dict from two
deterministic sources so the unchanged ``LegendMatcher.match_points`` can
assign points:

1. **PDF text layer** (vector figures): the legend strings sit in the text
   layer with their bbox; the marker swatch printed next to the text is
   sampled from the TF-ID crop.
2. **VLM marker colours**: the metadata call also names the printed colour of
   each legend entry (``legend_markers``); colour words map to fixed HSV
   centres.

Both sources are all-or-nothing: unless *every* series gets a distinct
prototype, no prototypes are returned (a partial set would force the
unmatched series' points onto a wrong series because ``match_points`` is a
threshold-free argmin).  Positions are never taken from the VLM.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np

# OpenCV HSV scale: H 0-180, S/V 0-255.  Centres of printed marker colours.
NAMED_COLORS_HSV: Dict[str, np.ndarray] = {
    "red":     np.array([0,   220, 200]),
    "orange":  np.array([12,  230, 230]),
    "yellow":  np.array([28,  220, 230]),
    "green":   np.array([60,  200, 160]),
    "cyan":    np.array([90,  220, 220]),
    "blue":    np.array([115, 220, 200]),
    "purple":  np.array([140, 180, 160]),
    "magenta": np.array([155, 200, 200]),
    "pink":    np.array([165, 120, 230]),
    "brown":   np.array([10,  180, 120]),
    "black":   np.array([0,   0,   30]),
    "gray":    np.array([0,   0,   128]),
    "grey":    np.array([0,   0,   128]),
    # Monochrome legends: the VLM may only say open/filled → both print black.
    "open":    np.array([0,   0,   30]),
    "filled":  np.array([0,   0,   30]),
}
_COLOR_ALIASES = {"dark blue": "blue", "light blue": "cyan", "navy": "blue", "violet": "purple",
                  "dark green": "green", "light green": "green", "olive": "green", "teal": "cyan",
                  "dark red": "red", "crimson": "red", "maroon": "brown", "gold": "yellow",
                  "lime": "green", "sky blue": "cyan", "dark gray": "gray", "dark grey": "gray"}
# Two prototypes closer than this (weighted HSV distance, same formula as
# LegendMatcher.weighted_hsv_dist) cannot be told apart by colour.
MIN_PROTOTYPE_SEPARATION = 0.35
_MIN_FG_PIXELS = 8
_MAX_LEGEND_LINE_PX = 60          # 15 pt at 4 px/pt; taller = vertical text


@dataclass
class SeriesPrototypes:
    prototypes: Dict[str, Dict[str, Any]]
    source: str                      # "text_layer" | "vlm_colors"
    notes: List[str]


def _hsv_dist(c1: np.ndarray, c2: np.ndarray) -> float:
    c1 = c1.astype(float); c2 = c2.astype(float)
    dh = min(abs(c1[0] - c2[0]), 180 - abs(c1[0] - c2[0])) / 180.0
    ds = abs(c1[1] - c2[1]) / 255.0
    dv = abs(c1[2] - c2[2]) / 255.0
    return 4.0 * dh + 1.5 * ds + 0.5 * dv


def _well_separated(protos: Dict[str, Dict[str, Any]]) -> bool:
    keys = list(protos)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            if _hsv_dist(protos[keys[i]]["hsv"], protos[keys[j]]["hsv"]) < MIN_PROTOTYPE_SEPARATION:
                return False
    return True


def _norm_color_word(word: Optional[str]) -> Optional[str]:
    if not word:
        return None
    w = re.sub(r"[^a-z ]", " ", word.lower()).strip()
    w = re.sub(r"\s+", " ", w)
    if w in NAMED_COLORS_HSV:
        return w
    if w in _COLOR_ALIASES:
        return _COLOR_ALIASES[w]
    for tok in w.split():                      # "red circles" → red
        if tok in NAMED_COLORS_HSV:
            return tok
    return None


def prototypes_from_markers(markers: Sequence[Dict[str, Any]]) -> Optional[SeriesPrototypes]:
    """Build prototypes from VLM ``legend_markers`` (name + printed colour word).

    Returns None unless ≥ 2 series all carry a recognised, mutually distinct
    colour word.
    """
    protos: Dict[str, Dict[str, Any]] = {}
    notes: List[str] = []
    for m in markers or []:
        name = (m.get("name") or "").strip()
        if not name:
            continue
        col = _norm_color_word(m.get("color"))
        if col is None:
            notes.append(f"no colour word for '{name}' ({m.get('color')!r})")
            return None
        protos[name] = {"hsv": NAMED_COLORS_HSV[col].copy(), "count": 1, "color_word": col}
    if len(protos) < 2:
        return None
    if not _well_separated(protos):
        return None
    return SeriesPrototypes(protos, "vlm_colors", notes)


def _norm_name(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    s = s.replace("−", "-").replace("–", "-").replace("—", "-").replace("°", "")
    return re.sub(r"\s+", "", s).casefold()


def _foreground_hsv(crop: np.ndarray) -> Optional[np.ndarray]:
    """Mean colour of the non-white pixels (same background rule as
    ``LegendMatcher.get_dominant_color``); None when the window is blank."""
    if crop is None or crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask_fg = ~((hsv[:, :, 1] < 40) & (hsv[:, :, 2] > 200))
    if int(np.sum(mask_fg)) < _MIN_FG_PIXELS:
        return None
    mean_bgr = np.mean(crop[mask_fg], axis=0).astype(np.uint8)
    return cv2.cvtColor(np.array([[mean_bgr]]), cv2.COLOR_BGR2HSV)[0][0]


def _find_line(name: str, inner_lines: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    key = _norm_name(name)
    if len(key) < 1:
        return None
    exact = [ln for ln in inner_lines if _norm_name(ln.get("text", "")) == key]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        return None                                    # ambiguous → don't guess
    partial = [ln for ln in inner_lines
               if key in _norm_name(ln.get("text", "")) and len(_norm_name(ln.get("text", ""))) <= 2 * len(key) + 4]
    return partial[0] if len(partial) == 1 else None


def prototypes_from_text_layer(figure_img: np.ndarray, crop_bbox_pt: Sequence[float],
                               inner_lines: Sequence[Dict[str, Any]], names: Sequence[str],
                               scale: float = 4.0) -> Optional[SeriesPrototypes]:
    """Sample the marker swatch printed beside each legend string.

    ``figure_img`` is the TF-ID crop (rendered at ``scale`` px/pt) whose page
    bbox is ``crop_bbox_pt``; ``inner_lines`` are the text-layer lines inside
    that bbox with their page-space ``bbox_pt``.  For each series name the
    window one line-height tall and two line-heights wide immediately LEFT of
    the text (then RIGHT) is scanned for foreground pixels.  Returns None
    unless every name is found and every swatch yields a distinct colour.
    """
    if figure_img is None or figure_img.size == 0 or len(names) < 2:
        return None
    H, W = figure_img.shape[:2]
    ox, oy = float(crop_bbox_pt[0]), float(crop_bbox_pt[1])
    protos: Dict[str, Dict[str, Any]] = {}
    notes: List[str] = []
    all_boxes = [ln.get("bbox_pt") for ln in inner_lines if ln.get("bbox_pt")]
    for name in names:
        ln = _find_line(name, inner_lines)
        if ln is None:
            notes.append(f"'{name}' not in text layer")
            return None
        bx0, by0, bx1, by1 = [float(v) for v in ln["bbox_pt"]]
        x0, y0, x1, y1 = (bx0 - ox) * scale, (by0 - oy) * scale, (bx1 - ox) * scale, (by1 - oy) * scale
        h = max(4.0, y1 - y0)
        if h > _MAX_LEGEND_LINE_PX:                    # rotated axis title, not a legend entry
            notes.append(f"'{name}' line too tall ({h:.0f}px)")
            return None
        found = None
        for side in ("left", "right"):
            if side == "left":
                wx0, wx1 = x0 - 2.0 * h, x0 - 0.15 * h
            else:
                wx0, wx1 = x1 + 0.15 * h, x1 + 2.0 * h
            # Never look into a neighbouring text line (horizontal legends).
            for ob in all_boxes:
                if ob is ln.get("bbox_pt"):
                    continue
                ox0, oy0, ox1, oy1 = [(float(v) - (ox if k % 2 == 0 else oy)) * scale for k, v in enumerate(ob)]
                if oy1 < y0 or oy0 > y1:
                    continue
                if side == "left" and ox1 > wx0 and ox1 < x0:
                    wx0 = max(wx0, ox1 + 0.15 * h)
                if side == "right" and ox0 < wx1 and ox0 > x1:
                    wx1 = min(wx1, ox0 - 0.15 * h)
            ix0, ix1 = int(max(0, round(wx0))), int(min(W, round(wx1)))
            iy0, iy1 = int(max(0, round(y0))), int(min(H, round(y1)))
            if ix1 - ix0 < 2 or iy1 - iy0 < 2:
                continue
            hsv = _foreground_hsv(figure_img[iy0:iy1, ix0:ix1])
            if hsv is not None:
                found = hsv
                break
        if found is None:
            notes.append(f"no swatch beside '{name}'")
            return None
        protos[name] = {"hsv": np.array(found), "count": 1}
    if len(protos) < 2 or not _well_separated(protos):
        return None
    return SeriesPrototypes(protos, "text_layer", notes)


def recover_series_prototypes(series_names: Sequence[str], legend_markers: Sequence[Dict[str, Any]],
                              figure_img_path: Optional[str], context: Optional[Dict[str, Any]],
                              log=print) -> Optional[SeriesPrototypes]:
    """Text layer first (measured colours), VLM colour words second."""
    names = [str(n).strip() for n in (series_names or []) if n and str(n).strip()]
    if len(names) < 2:
        return None
    ctx = context or {}
    inner_lines = ctx.get("inner_lines") or []
    if figure_img_path and inner_lines and ctx.get("bbox_pt"):
        img = cv2.imread(figure_img_path)
        res = prototypes_from_text_layer(img, ctx["bbox_pt"], inner_lines, names)
        if res:
            log(f"      [Series] recovered {len(res.prototypes)} series from PDF text layer swatches")
            return res
    res = prototypes_from_markers(legend_markers or [])
    if res:
        log(f"      [Series] recovered {len(res.prototypes)} series from VLM colour words "
            f"{ {k: v.get('color_word') for k, v in res.prototypes.items()} }")
    return res
