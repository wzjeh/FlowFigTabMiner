"""VLM second reader for chart labels (constrained reading).

The coordinate mapper hands over the YOLO boxes it is about to OCR (axis
tick labels, heatmap cell labels).  This module tiles them into a numbered
contact sheet, asks the VLM to transcribe cell #i → text, and returns a
per-box reading.  The VLM never locates anything and never emits data
points: it only reads printed characters, which is where PaddleOCR fails
("10⁻¹" → "101", "-10 0" → -100) and vision models excel.

Fusion with the OCR reading happens in ``axis_fit.fuse_tick_readings`` /
``fuse_value_readings``; geometry (``robust_fit``) arbitrates tick conflicts.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence

import cv2
import numpy as np
from pydantic import BaseModel, Field

from src.extraction.figure.contact_sheet import build_contact_sheet
from src.llm.config import VLMConfig
from src.llm.providers.base import VLMProvider
from src.llm.types import VLMImage

_PROMPT_PATH = Path(__file__).parent / "prompts" / "label_reader.md"
_PROMPT = _PROMPT_PATH.read_text(encoding="utf-8")


class LabelReading(BaseModel):
    index: int
    text: str = ""
    kind: Literal["tick", "value", "empty", "other"] = "tick"


class LabelReadingsResponse(BaseModel):
    """Schema enforced on the VLM response."""
    readings: List[LabelReading] = Field(default_factory=list)


class LabelReadResult(BaseModel):
    """Per-figure outcome: index → text plus call metadata (for facts / forensics)."""
    texts: Dict[int, str] = Field(default_factory=dict)
    kinds: Dict[int, str] = Field(default_factory=dict)
    model: Optional[str] = None
    latency_ms: Optional[float] = None
    tokens_in: Optional[int] = None
    tokens_out: Optional[int] = None
    cache_hit: bool = False
    sheet_path: Optional[str] = None
    n_boxes: int = 0
    notes: Optional[str] = None
    ok: bool = False


class VLMLabelReader:
    """One VLM call per figure over a numbered contact sheet of label crops."""

    def __init__(self, vlm: VLMProvider, cfg: VLMConfig, *, max_boxes: int = 80):
        self.vlm = vlm
        self.cfg = cfg
        self.max_boxes = max_boxes

    def read(self, img_bgr: np.ndarray, dets: Sequence[Dict[str, Any]], figure_id: str,
             out_dir: Optional[str]) -> LabelReadResult:
        result = LabelReadResult(n_boxes=len(dets))
        if not len(dets):
            result.notes = "no boxes"
            return result
        dets = list(dets)[: self.max_boxes]
        result.n_boxes = len(dets)
        sheet, cells = build_contact_sheet(img_bgr, dets)
        sheet_dir = out_dir or os.getcwd()
        os.makedirs(sheet_dir, exist_ok=True)
        sheet_path = os.path.join(sheet_dir, f"{figure_id}_labels_sheet.png")
        try:
            cv2.imwrite(sheet_path, sheet)
        except Exception as exc:  # pragma: no cover
            result.notes = f"sheet write failed: {exc}"
            return result
        result.sheet_path = sheet_path
        t0 = time.perf_counter()
        try:
            meta, parsed = self.vlm.inspect(
                image=VLMImage(path=Path(sheet_path), mime_type="image/png"),
                system_prompt="",
                user_prompt=_PROMPT,
                cfg=self.cfg,
                response_schema=LabelReadingsResponse,
            )
            resp = LabelReadingsResponse.model_validate(parsed)
        except Exception as exc:
            result.notes = f"vlm call failed: {exc}"
            result.latency_ms = (time.perf_counter() - t0) * 1000.0
            return result
        for r in resp.readings:
            if 0 <= r.index < len(dets):
                result.texts[r.index] = (r.text or "").strip()
                result.kinds[r.index] = r.kind
        result.model = getattr(meta, "model", None)
        result.latency_ms = getattr(meta, "latency_ms", None)
        result.tokens_in = getattr(meta, "tokens_in", None)
        result.tokens_out = getattr(meta, "tokens_out", None)
        result.cache_hit = bool(getattr(meta, "cache_hit", False))
        result.ok = True
        return result


def write_forensics(out_dir: Optional[str], figure_id: str, rows: List[Dict[str, Any]],
                    summary: Dict[str, Any]) -> None:
    """``{figure_id}_labels_vlm.json``: per-box ocr / vlm / used / source + summary."""
    if not out_dir:
        return
    try:
        with open(os.path.join(out_dir, f"{figure_id}_labels_vlm.json"), "w") as f:
            json.dump({"summary": summary, "boxes": rows}, f, indent=2, ensure_ascii=False)
    except Exception:
        pass
