"""
VLM-assisted table header correction.

Uses TATR's `table column header` bounding boxes (preserved in structure.py)
to cheaply estimate the number of header rows; when TATR output looks
unreliable, calls a ``VLMProvider`` (injected from ``main.py``) to re-read
the header strip.  The provider abstraction means this module does not
know whether it's talking to Gemini, Qwen-VL, or any other backend.
"""

import json
import logging
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from PIL import Image

from src.llm.config import VLMConfig
from src.llm.providers.base import VLMProvider
from src.llm.types import VLMImage

logger = logging.getLogger(__name__)


class HeaderCorrector:
    def __init__(
        self,
        config: dict,
        vlm: Optional[VLMProvider] = None,
        vlm_cfg: Optional[VLMConfig] = None,
    ):
        """
        Args:
            config:  ``tables.header_correction`` block (enabled / trigger_threshold /
                     send_header_crop_only).
            vlm:     injected VLM provider; when ``None`` the corrector still
                     reports whether correction would have been attempted but
                     skips the actual API call.
            vlm_cfg: typed config for the VLM call (model id, temperature, …).
                     Required iff ``vlm`` is given.
        """
        self.enabled = config.get("enabled", False)
        self.trigger_threshold = float(config.get("trigger_threshold", 0.5))
        self.send_header_crop_only = config.get("send_header_crop_only", True)
        self.vlm = vlm
        self.vlm_cfg = vlm_cfg
        if self.enabled and self.vlm is not None and self.vlm_cfg is None:
            raise ValueError("HeaderCorrector: vlm provided without vlm_cfg")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def needs_correction(self, grid: list, structure_data: dict) -> bool:
        """
        Heuristic (no API cost): returns True if VLM correction should be attempted.

        Conservative design — only triggers in clear failure cases:
        1. row-0 non-empty ratio < trigger_threshold  (spanning header → many cells empty)
        2. row-0 AND row-1 both heavily text-dominant (≥70%) AND neither row contains any
           numeric cell (otherwise row-1 is likely data, not a second header row)

        Short-circuit: if TATR found high-confidence header_regions AND row-0 is reasonably
        filled (≥trigger_threshold), trust TATR and skip VLM.
        """
        if not grid:
            return False

        row0 = grid[0]
        if not row0:
            return False

        col_count = len(row0)
        non_empty = sum(1 for c in row0 if str(c).strip())
        non_empty_ratio = non_empty / col_count

        # Short-circuit: TATR header_regions look reliable — row-0 is sufficiently filled
        header_regions = structure_data.get("header_regions", [])
        if header_regions and non_empty_ratio >= self.trigger_threshold:
            avg_conf = sum(r.get("score", 0.0) for r in header_regions) / len(header_regions)
            if avg_conf >= 0.8:
                logger.debug(
                    f"HeaderCorrector skip: TATR header_regions conf={avg_conf:.2f}, "
                    f"row-0 fill={non_empty_ratio:.2f} — trusting TATR"
                )
                return False

        # Condition 1: too many empty cells in row-0 (likely a spanning/merged header)
        if non_empty_ratio < self.trigger_threshold:
            logger.debug(
                f"HeaderCorrector trigger: row-0 non-empty ratio {non_empty_ratio:.2f} < {self.trigger_threshold}"
            )
            return True

        # Condition 2: double header row — both row-0 and row-1 are HEAVILY text-dominant
        # (≥70% text) AND row-1 has zero numeric cells (i.e., it really is a header, not data)
        if len(grid) >= 2:
            row1 = grid[1]
            r0_text = self._text_ratio(row0)
            r1_text = self._text_ratio(row1)
            r1_has_numbers = any(
                re.match(r'^[\d\.\-\+]+\s*[%°]?$', str(c).strip())
                for c in row1 if str(c).strip()
            )
            if r0_text >= 0.7 and r1_text >= 0.7 and not r1_has_numbers:
                logger.debug(
                    f"HeaderCorrector trigger: row-0 text={r0_text:.2f}, row-1 text={r1_text:.2f}, "
                    f"no numerics in row-1 → likely double header"
                )
                return True

        return False

    def correct(self, image_path: str, grid: list, structure_data: dict) -> list:
        """
        Main entry point.  Returns corrected grid or original grid on any failure.
        """
        if not self.enabled or not grid:
            return grid

        col_count = max(len(row) for row in grid) if grid else 0
        if col_count == 0:
            return grid

        try:
            header_row_count = self._estimate_header_rows(structure_data, grid)
            logger.info(
                f"   HeaderCorrector: estimated {header_row_count} header row(s) from TATR"
            )

            send_path = image_path
            tmp_crop_path = None
            if self.send_header_crop_only:
                tmp_crop_path = self._crop_header_image(image_path, structure_data)
                if tmp_crop_path:
                    send_path = tmp_crop_path

            parsed = self._call_vlm(send_path, col_count, header_row_count)
            if tmp_crop_path and os.path.exists(tmp_crop_path):
                os.remove(tmp_crop_path)

            if parsed is None:
                logger.warning("   HeaderCorrector: VLM call failed, keeping original grid")
                return grid

            columns = parsed.get("columns", []) if isinstance(parsed, dict) else []
            if len(columns) != col_count:
                logger.warning(
                    "   HeaderCorrector: VLM returned %d columns, expected %d — keeping original",
                    len(columns),
                    col_count,
                )
                return grid

            # Build corrected header row from VLM output
            vl_header_row = [""] * col_count
            for col_info in parsed["columns"]:
                idx = col_info.get("index", -1)
                if 0 <= idx < col_count:
                    vl_header_row[idx] = col_info.get("header", "")

            # Use VLM's header_row_count if available (prefer TATR estimate as fallback)
            skip_rows = parsed.get("header_row_count", header_row_count)
            skip_rows = max(1, skip_rows)  # always skip at least 1 row

            new_grid = [vl_header_row] + grid[skip_rows:]
            logger.info(
                f"   HeaderCorrector: replaced {skip_rows} TATR header row(s) with VLM headers"
            )
            return new_grid

        except Exception as e:
            logger.error(f"   HeaderCorrector: unexpected error — {e}", exc_info=True)
            return grid

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _text_ratio(self, row: list) -> float:
        """Fraction of non-empty cells whose content is non-numeric."""
        non_empty = [str(c).strip() for c in row if str(c).strip()]
        if not non_empty:
            return 0.0
        text_cells = sum(1 for c in non_empty if not re.match(r'^[\d\.\-\+\s%°]+$', c))
        return text_cells / len(non_empty)

    def _estimate_header_rows(self, structure_data: dict, grid: list) -> int:
        """
        Use TATR `header_regions` Y-coordinates to count how many rows fall inside.
        Falls back to 1 if no header_regions detected.
        """
        header_regions = structure_data.get("header_regions", [])
        rows = structure_data.get("rows", [])

        if not header_regions or not rows:
            return 1

        # Use the union of all header region boxes
        header_y_max = max(r["box"][3] for r in header_regions)
        header_y_min = min(r["box"][1] for r in header_regions)

        count = 0
        for row in rows:
            ry1, ry2 = row["box"][1], row["box"][3]
            row_center = (ry1 + ry2) / 2
            if header_y_min <= row_center <= header_y_max:
                count += 1

        return max(1, count)

    def _crop_header_image(self, image_path: str, structure_data: dict, buffer_px: int = 10) -> Optional[str]:
        """
        Crop the header region from the image and save to a temp file.
        Returns the temp file path, or None if cropping fails.
        """
        header_regions = structure_data.get("header_regions", [])
        if not header_regions:
            return None

        try:
            img = Image.open(image_path).convert("RGB")
            w, h = img.size

            x1 = max(0, min(r["box"][0] for r in header_regions) - buffer_px)
            y1 = max(0, min(r["box"][1] for r in header_regions) - buffer_px)
            x2 = min(w, max(r["box"][2] for r in header_regions) + buffer_px)
            y2 = min(h, max(r["box"][3] for r in header_regions) + buffer_px)

            crop = img.crop((x1, y1, x2, y2))
            fd, tmp_path = tempfile.mkstemp(suffix=".png")
            os.close(fd)
            crop.save(tmp_path)
            return tmp_path

        except Exception as e:
            logger.warning(f"   HeaderCorrector: crop failed — {e}")
            return None

    def _call_vlm(self, image_path: str, col_count: int, header_row_count: int) -> Optional[dict]:
        """Ask the injected VLM provider to read the header strip.

        Returns the parsed ``{"header_row_count": int, "columns": [...]}``
        dict, or ``None`` on any failure (logged).  The provider returns
        already-decoded JSON, so we skip the historical regex parsing.
        """
        if self.vlm is None or self.vlm_cfg is None:
            logger.warning("   HeaderCorrector: no VLM provider injected; skip")
            return None

        indices = ", ".join(str(i) for i in range(col_count))
        prompt = (
            f"This is a chemistry paper table. Count the columns carefully by looking at the data rows — "
            f"there are exactly {col_count} columns (indices {indices}).\n\n"
            f"Task: identify the header text for each column.\n\n"
            f"Rules:\n"
            f"- header_row_count: how many top rows are headers (not data), usually 1 or 2.\n"
            f"- You MUST output exactly {col_count} entries, one per column, left-to-right.\n"
            f"- NEVER skip a column. If a column has no visible header text (e.g. a numbering column "
            f"like '4a', '4b'), still include it with header \"\".\n"
            f"- For merged/spanning headers (e.g. 'Yield' over 'GC%' and 'Isolated%'), "
            f"write each sub-column as 'Yield / GC%' and 'Yield / Isolated%'.\n\n"
            f"Return ONLY valid JSON, no explanation:\n"
            f'{{\"header_row_count\": <int>, \"columns\": [{{\"index\": 0, \"header\": \"...\"}}, ...]}}'
        )

        try:
            _meta, parsed = self.vlm.inspect(
                image=VLMImage(path=Path(image_path)),
                system_prompt="",
                user_prompt=prompt,
                cfg=self.vlm_cfg,
            )
            return parsed
        except Exception as exc:
            logger.error(f"   HeaderCorrector: VLM provider error — {exc}")
            return None

