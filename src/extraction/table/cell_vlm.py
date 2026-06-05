"""Table cell extractor — Gemini full-cell VLM call with R1 circuit breaker.

This module owns table cell text.  TATR has already given us the grid
dimensions ``(m, n)``; the VLM transcribes every cell into a dense 2-D
array that we validate against ``(m, n)``.  If the grid alignment fails
on the first call, we retry once with a stricter prompt.  If it still
fails, the assembler downstream marks the table as ``alignment_failed``
and refuses to emit cells — better than silently shifting later cells
into wrong columns.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from src.evidence.types import FieldSource, TableCell
from src.llm.config import VLMConfig
from src.llm.providers.base import VLMProvider
from src.llm.types import VLMImage

logger = logging.getLogger(__name__)

_PROMPT_TEMPLATE_PATH = Path(__file__).parent / "prompts" / "table_cells.md"
_PROMPT_TEMPLATE = _PROMPT_TEMPLATE_PATH.read_text()


class TableCellsResponse(BaseModel):
    """Schema enforced on the VLM response.

    Pydantic's per-list size validation can't constrain "exactly {m}
    rows of exactly {n} cells" generically, so we accept any ``rows``
    here and validate the shape post-call.
    """

    rows: list[list[str]] = Field(default_factory=list)


class CellAlignmentError(Exception):
    """Raised when the VLM-returned grid shape disagrees with TATR."""

    def __init__(self, expected: tuple[int, int], actual_rows: int, actual_cols: list[int]):
        self.expected = expected
        self.actual_rows = actual_rows
        self.actual_cols = actual_cols
        super().__init__(
            f"shape mismatch: expected {expected[0]}×{expected[1]}, "
            f"got {actual_rows} rows with column counts {actual_cols}"
        )


class TableCellExtractionResult(BaseModel):
    """Wrapper carrying the cells plus status, so the assembler can
    write ``EvidenceStatus.ALIGNMENT_FAILED`` when appropriate without
    inspecting a list length downstream.
    """

    model_config = {"arbitrary_types_allowed": True}

    cells: list[TableCell] = Field(default_factory=list)
    aligned: bool
    source: FieldSource
    model_id: Optional[str] = None
    notes: Optional[str] = None


class TableCellExtractor:
    """Two-attempt cell extractor with R1 circuit breaker."""

    def __init__(self, vlm: VLMProvider, cfg: VLMConfig):
        self.vlm = vlm
        self.cfg = cfg

    def extract(self, image: Path, m: int, n: int) -> TableCellExtractionResult:
        """Run up to two Gemini calls; emit ``aligned=False`` if both fail.

        Parameters
        ----------
        image: path to the table crop (post-molecule-masking is fine)
        m: expected row count from TATR
        n: expected column count from TATR
        """
        if m <= 0 or n <= 0:
            logger.warning("table_cell_vlm aborting: TATR gave empty grid (%dx%d)", m, n)
            return TableCellExtractionResult(
                cells=[], aligned=False,
                source=FieldSource.ALIGNMENT_FAILED,
                notes=f"tatr emitted empty grid m={m} n={n}",
            )

        # Attempt 1 — base prompt.
        try:
            cells, model_id = self._one_attempt(image, m, n, retry_note=None)
            return TableCellExtractionResult(
                cells=cells, aligned=True,
                source=FieldSource.VLM_CELL, model_id=model_id,
            )
        except CellAlignmentError as exc:
            logger.warning(
                "table_cell_vlm attempt 1 misaligned: %s — retrying with stricter prompt",
                exc,
            )
            note = (
                f"Your previous output had {exc.actual_rows} rows "
                f"with column counts {exc.actual_cols}; expected exactly "
                f"{exc.expected[0]} rows × {exc.expected[1]} columns. "
                "Output the corrected grid, never skipping a column."
            )

        # Attempt 2 — stricter prompt with feedback on the first mismatch.
        try:
            cells, model_id = self._one_attempt(image, m, n, retry_note=note)
            return TableCellExtractionResult(
                cells=cells, aligned=True,
                source=FieldSource.VLM_CELL, model_id=model_id,
                notes="aligned after one retry",
            )
        except CellAlignmentError as exc:
            logger.error("table_cell_vlm attempt 2 still misaligned: %s — circuit breaker fires", exc)
            return TableCellExtractionResult(
                cells=[], aligned=False,
                source=FieldSource.ALIGNMENT_FAILED,
                notes=f"alignment failed twice: {exc}",
            )
        except Exception as exc:
            logger.exception("table_cell_vlm attempt 2 raised %s", exc)
            return TableCellExtractionResult(
                cells=[], aligned=False,
                source=FieldSource.ALIGNMENT_FAILED,
                notes=f"vlm call failed: {exc}",
            )

    def _one_attempt(
        self,
        image: Path,
        m: int,
        n: int,
        retry_note: Optional[str],
    ) -> tuple[list[TableCell], str]:
        prompt = _PROMPT_TEMPLATE.format(m=m, n=n)
        if retry_note:
            prompt = f"{prompt}\n\n## RETRY NOTE\n{retry_note}"

        meta, parsed = self.vlm.inspect(
            image=VLMImage(path=image, mime_type=_mime_for(image)),
            system_prompt="",
            user_prompt=prompt,
            cfg=self.cfg,
            response_schema=TableCellsResponse,
        )

        resp = TableCellsResponse.model_validate(parsed)

        # Shape check: exactly m rows, each of exactly n cells.
        col_lens = [len(r) for r in resp.rows]
        if len(resp.rows) != m or any(c != n for c in col_lens):
            raise CellAlignmentError(expected=(m, n), actual_rows=len(resp.rows), actual_cols=col_lens)

        cells = [
            TableCell(row=ri, col=ci, text=text)
            for ri, row in enumerate(resp.rows)
            for ci, text in enumerate(row)
        ]
        logger.info(
            "table_cell_vlm aligned %dx%d cells=%d image=%s",
            m, n, len(cells), image.name,
        )
        return cells, meta.model


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
