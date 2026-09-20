"""Figure metadata extractor — Gemini metadata-only VLM call.

This module owns the figure text fields that PaddleOCR cannot reliably
read: axis labels, axis units, legend chemistry names, title, and
footnote.  Data points are NOT extracted here; they belong to the YOLO +
RANSAC pipeline (per the canonical owner table).

Structured outputs (Gemini ``response_schema``) guarantee the SDK
returns JSON conforming to ``FigureMetadataResponse`` — no regex parsing
needed downstream.
"""

from __future__ import annotations

import logging
import re
import time
import unicodedata
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel

from src.evidence.types import FieldSource, FieldValue
from src.llm.config import VLMConfig
from src.llm.providers.base import VLMProvider
from src.llm.types import VLMImage

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent / "prompts" / "metadata.md"
_PROMPT = _PROMPT_PATH.read_text()


# Hygiene for VLM string fields.  A structured-output call can still return
# junk (observed: a 2 kB "series name" made of backspace characters, and
# strings echoing the schema field names such as "legend_series_names_null");
# nothing downstream can use those, so they are dropped deterministically.
_MAX_TEXT_CHARS = {"default": 300, "short": 120}
_SCHEMA_ECHO = re.compile(r"(x_axis_label|y_axis_label|x_axis_unit|y_axis_unit|legend_series_names|"
                          r"legend_markers|footnote_null|_null_null)", re.I)


def clean_vlm_text(v: Any, max_chars: int = 120) -> Optional[str]:
    """Strip control characters and reject over-long or schema-echo strings."""
    if v is None:
        return None
    if not isinstance(v, str):
        v = str(v)
    v = "".join(ch for ch in v if unicodedata.category(ch)[0] != "C" or ch in "\n\t")
    v = re.sub(r"[ \t]+", " ", v).strip()
    if not v or len(v) > max_chars or _SCHEMA_ECHO.search(v):
        return None
    return v


class LegendMarker(BaseModel):
    """How one legend entry is printed (used only to recover series when the
    macro detector found no legend box; never for positions)."""

    name: str
    color: Optional[str] = None      # colour word as printed: red / blue / black / open / filled …
    marker: Optional[str] = None     # shape word: circle / square / triangle / line …


class FigureMetadataResponse(BaseModel):
    """Schema enforced on the VLM response."""

    title: Optional[str] = None
    x_axis_label: Optional[str] = None
    x_axis_unit: Optional[str] = None
    y_axis_label: Optional[str] = None
    y_axis_unit: Optional[str] = None
    legend_series_names: list[Optional[str]] = []
    legend_markers: list[LegendMarker] = []
    footnote: Optional[str] = None


class FigureMetadata(BaseModel):
    """Convenience grouping of the seven FieldValues emitted per figure."""

    title: FieldValue
    x_axis_label: FieldValue
    y_axis_label: FieldValue
    x_axis_unit: FieldValue
    y_axis_unit: FieldValue
    legend_series_names: FieldValue
    legend_markers: FieldValue
    footnote: FieldValue


def _wrap_missing(notes: str | None = None) -> FigureMetadata:
    """Return an all-MISSING FigureMetadata bundle (graceful failure)."""
    miss = lambda: FieldValue(value=None, source=FieldSource.MISSING, notes=notes)
    return FigureMetadata(
        title=miss(),
        x_axis_label=miss(),
        y_axis_label=miss(),
        x_axis_unit=miss(),
        y_axis_unit=miss(),
        legend_series_names=FieldValue(value=[], source=FieldSource.MISSING, notes=notes),
        legend_markers=FieldValue(value=[], source=FieldSource.MISSING, notes=notes),
        footnote=miss(),
    )


class FigureMetadataExtractor:
    """Single-call Gemini metadata extractor for one figure.

    ``vlm`` and ``cfg`` are injected from ``main.py`` so this module
    never imports the SDK directly.  Construction is cheap (no I/O); the
    actual call happens in ``extract()``.
    """

    def __init__(self, vlm: VLMProvider, cfg: VLMConfig):
        self.vlm = vlm
        self.cfg = cfg

    def extract(self, image: Path) -> FigureMetadata:
        try:
            t0 = time.perf_counter()
            meta, parsed = self.vlm.inspect(
                image=VLMImage(path=image, mime_type=_mime_for(image)),
                system_prompt="",
                user_prompt=_PROMPT,
                cfg=self.cfg,
                response_schema=FigureMetadataResponse,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
        except Exception as exc:
            logger.warning(
                "figure_metadata_vlm failed image=%s exc=%s — emitting MISSING",
                image.name, exc,
            )
            return _wrap_missing(notes=f"vlm call failed: {exc}")

        # Validate against schema; the SDK already constrains it, but we
        # double-check so a buggy provider wrapper surfaces here, not in
        # the assembler.
        resp = FigureMetadataResponse.model_validate(parsed)

        model_id = meta.model
        latency_ms = meta.latency_ms

        def fv(v, fallback_source: FieldSource = FieldSource.VLM_METADATA) -> FieldValue:
            v = clean_vlm_text(v)
            return FieldValue(
                value=v,
                source=FieldSource.MISSING if v is None or v == "" else fallback_source,
                model_id=model_id,
                latency_ms=latency_ms,
            )

        legend_names = [n for n in (clean_vlm_text(s) for s in resp.legend_series_names) if n]
        legend_markers = []
        for m in resp.legend_markers:
            name = clean_vlm_text(m.name)
            if name:
                legend_markers.append({"name": name, "color": clean_vlm_text(m.color), "marker": clean_vlm_text(m.marker)})

        return FigureMetadata(
            title=fv(resp.title),
            x_axis_label=fv(resp.x_axis_label),
            y_axis_label=fv(resp.y_axis_label),
            x_axis_unit=fv(resp.x_axis_unit),
            y_axis_unit=fv(resp.y_axis_unit),
            legend_series_names=FieldValue(
                value=legend_names,
                source=FieldSource.VLM_METADATA if legend_names else FieldSource.MISSING,
                model_id=model_id,
                latency_ms=latency_ms,
            ),
            legend_markers=FieldValue(
                value=legend_markers,
                source=FieldSource.VLM_METADATA if legend_markers else FieldSource.MISSING,
                model_id=model_id,
                latency_ms=latency_ms,
            ),
            footnote=fv(resp.footnote),
        )


def _mime_for(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".")
    return {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "tif": "image/tiff",
        "tiff": "image/tiff",
        "webp": "image/webp",
        "bmp": "image/bmp",
    }.get(suffix, "image/png")
