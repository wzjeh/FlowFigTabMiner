"""VLM table transcriber — one call per table crop.

Design (mirrors ``src/extraction/figure/metadata_vlm.py``): the provider is
injected, the prompt lives in ``prompts/table_transcribe.md``, the response
is constrained by a pydantic schema, and every failure comes back as a
result with ``ok=False`` instead of an exception.  The VLM defines the grid
itself (header rows + data rows); drawn molecules are the token
``[STRUCTURE]`` and receive SMILES later from MolNexTR
(``grid_ops.align_structures``).  Positions and SMILES never come from the
VLM.
"""
from __future__ import annotations

import logging
import re
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple

from pydantic import BaseModel

from src.extraction.figure.metadata_vlm import clean_vlm_text
from src.llm.config import VLMConfig
from src.llm.providers.base import VLMProvider
from src.llm.types import VLMImage

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent / "prompts" / "table_transcribe.md"
_PROMPT = _PROMPT_PATH.read_text(encoding="utf-8")
_MAX_CELL_CHARS = 400

_FREE_MODE_SUFFIX = (
    "\n\nReturn ONLY one JSON object (no markdown fences, no commentary) with exactly these keys: "
    "caption (string or null), scheme_conditions (string or null), header_rows (array of arrays of strings), "
    "data_rows (array of arrays of strings), footnotes (string or null).\n"
)


class TableTranscriptionResponse(BaseModel):
    """Schema enforced on the VLM response."""

    caption: Optional[str] = None
    scheme_conditions: Optional[str] = None
    header_rows: List[List[str]] = []
    data_rows: List[List[str]] = []
    footnotes: Optional[str] = None


class TableTranscription(BaseModel):
    """Result of one transcription call, with telemetry."""

    ok: bool = False
    header_rows: List[List[str]] = []
    data_rows: List[List[str]] = []
    n_cols: int = 0
    caption: Optional[str] = None
    scheme_conditions: Optional[str] = None
    footnotes: Optional[str] = None
    model: Optional[str] = None
    latency_ms: Optional[float] = None
    tokens_in: Optional[int] = None
    tokens_out: Optional[int] = None
    cache_hit: bool = False
    notes: Optional[str] = None


def _clean_cell(v) -> str:
    if v is None:
        return ""
    s = "".join(ch for ch in str(v) if unicodedata.category(ch)[0] != "C")
    s = re.sub(r"\s+", " ", s).strip()
    return s[:_MAX_CELL_CHARS]


def _one_line(v: Optional[str]) -> Optional[str]:
    """Caption / footnote / scheme text: sanitised and whitespace-collapsed
    (the model sometimes emits line breaks for glyphs it cannot read)."""
    v = clean_vlm_text(v, max_chars=4000)
    return re.sub(r"\s+", " ", v).strip() if v else v


def normalize_grid(header_rows: List[List[str]], data_rows: List[List[str]]
                   ) -> Tuple[List[List[str]], List[List[str]], int, List[str]]:
    """Clean cells and force a rectangular grid: the modal data-row width
    wins (header width as fallback); short rows are padded with ``""``,
    long rows trimmed (non-empty trimmed cells are reported in notes)."""
    notes: List[str] = []
    hdr = [[_clean_cell(c) for c in row] for row in (header_rows or []) if row is not None]
    dat = [[_clean_cell(c) for c in row] for row in (data_rows or []) if row is not None]
    dat = [r for r in dat if any(r)]
    widths = Counter(len(r) for r in dat) or Counter(len(r) for r in hdr)
    if not widths:
        return hdr, dat, 0, ["empty grid"]
    best = max(widths.values())
    cands = sorted(w for w, n in widths.items() if n == best)
    hdr_w = len(hdr[0]) if hdr else None
    n_cols = hdr_w if hdr_w in cands else cands[-1]     # tie → trust the header width, else the widest
    padded = trimmed = 0

    def fix(rows):
        nonlocal padded, trimmed
        out = []
        for r in rows:
            if len(r) < n_cols:
                padded += 1; r = r + [""] * (n_cols - len(r))
            elif len(r) > n_cols:
                if any(r[n_cols:]):
                    trimmed += 1
                r = r[:n_cols]
            out.append(r)
        return out

    hdr, dat = fix(hdr), fix(dat)
    if padded:
        notes.append(f"{padded} rows padded to {n_cols} cols")
    if trimmed:
        notes.append(f"{trimmed} rows had non-empty cells beyond col {n_cols} (trimmed)")
    return hdr, dat, n_cols, notes


def _mime_for(path: Path) -> str:
    return {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
            "tif": "image/tiff", "tiff": "image/tiff"}.get(path.suffix.lower().lstrip("."), "image/png")


class TableTranscriber:
    """One structured VLM call per table image (any ``VLMProvider``)."""

    def __init__(self, vlm: VLMProvider, cfg: VLMConfig):
        self.vlm = vlm
        self.cfg = cfg

    def transcribe(self, image: Path) -> TableTranscription:
        image = Path(image)
        last_exc = None
        first = None          # a degenerate grid from an earlier attempt, kept if no later attempt is better
        # Attempt 1: constrained (schema) decoding at temperature 0.
        # Attempt 2: free-text JSON (schema pasted into the prompt) at 0 — the
        #   constrained decoder falls into repetition loops on symbols such as
        #   ± or a superscript minus ("50.207 ±" → thousands of "\n" → prose);
        #   free decoding writes the same cell verbatim.
        # Attempt 3: free-text JSON at 0.4 — a last re-roll for loops that
        #   greedy decoding would reproduce, and for a grid returned as one
        #   cell per row (n_cols == 1 for a table that plainly has columns).
        hot = self.cfg.model_copy(update={"temperature": max(0.4, self.cfg.temperature)})
        attempts = ((self.cfg, TableTranscriptionResponse), (self.cfg, None), (hot, None))
        for attempt, (cfg, schema) in enumerate(attempts, 1):
            prompt = _PROMPT if schema is not None else _PROMPT + _FREE_MODE_SUFFIX
            try:
                t0 = time.perf_counter()
                meta, parsed = self.vlm.inspect(
                    image=VLMImage(path=image, mime_type=_mime_for(image)),
                    system_prompt="",
                    user_prompt=prompt,
                    cfg=cfg,
                    response_schema=schema,
                )
                elapsed = (time.perf_counter() - t0) * 1000.0
                resp = TableTranscriptionResponse.model_validate(parsed)
                last_exc = None
                hdr, dat, n_cols, notes = normalize_grid(resp.header_rows, resp.data_rows)
                if attempt > 1:
                    notes.append(f"attempt {attempt}: {'free-text' if schema is None else 'structured'} JSON at T={cfg.temperature}")
                if attempt < len(attempts) and n_cols == 1 and len(hdr) + len(dat) >= 3:
                    logger.warning("table_vlm single-column grid image=%s (%d rows) on attempt %d; re-rolling", image.name, len(hdr) + len(dat), attempt)
                    if first is None:
                        first = (meta, resp, hdr, dat, n_cols, notes + [f"single-column grid on attempt {attempt}"], elapsed)
                    continue
                break
            except Exception as exc:
                last_exc = exc
                logger.warning("table_vlm attempt %d failed image=%s exc=%s", attempt, image.name, str(exc)[:200])
        if last_exc is not None or (first is not None and n_cols == 1):
            if first is None:
                return TableTranscription(ok=False, notes=f"vlm call failed: {last_exc}")
            meta, resp, hdr, dat, n_cols, notes, elapsed = first
        fin = str(getattr(meta, "finish_reason", None) or "").split(".")[-1].upper()
        if fin and fin not in ("STOP", "END_TURN"):
            notes.append(f"finish_reason={fin}")
        out = TableTranscription(
            ok=bool(dat) and n_cols > 0,
            header_rows=hdr, data_rows=dat, n_cols=n_cols,
            caption=_one_line(resp.caption),
            scheme_conditions=_one_line(resp.scheme_conditions),
            footnotes=_one_line(resp.footnotes),
            model=meta.model, latency_ms=getattr(meta, "latency_ms", None) or elapsed,
            tokens_in=meta.tokens_in, tokens_out=meta.tokens_out,
            cache_hit=bool(getattr(meta, "cache_hit", False)),
            notes="; ".join(notes) or None,
        )
        if not out.ok and not out.notes:
            out.notes = "empty grid"
        logger.info("table_vlm image=%s ok=%s grid=%dx%d cache_hit=%s tokens_in=%s tokens_out=%s notes=%s",
                    image.name, out.ok, len(hdr) + len(dat), n_cols, out.cache_hit, out.tokens_in, out.tokens_out, out.notes)
        return out
