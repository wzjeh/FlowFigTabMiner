"""VLM second reader for a structure crop MolNexTR read implausibly.

Same division of labour as the tick-label reader: the reading is taken only
when it is a plausible molecule (RDKit-valid, organic elements, no shards).
On 21 disagreements judged by eye MolNexTR was right 12 times and the VLM 5,
so the VLM never replaces a plausible first reading.
"""
from __future__ import annotations

import hashlib
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

import cv2
from pydantic import BaseModel

from src.adjudication.entity_pool import canonical_smiles
from src.extraction.common.structure_plausibility import suspicious_smiles
from src.llm.types import VLMImage

logger = logging.getLogger(__name__)

_PROMPT = (Path(__file__).parent / "prompts" / "structure_smiles.md").read_text(encoding="utf-8")


class StructureReading(BaseModel):
    smiles: str = ""


class VLMStructureReader:
    def __init__(self, vlm, cfg, scratch_dir: Optional[str] = None):
        self.vlm = vlm
        self.cfg = cfg
        self.scratch_dir = scratch_dir or os.path.join(tempfile.gettempdir(), "fftm_structure_reads")
        self.n_calls = 0
        self.n_accepted = 0

    def read(self, crop_bgr) -> Optional[str]:
        """Canonical SMILES the VLM reads off ``crop_bgr``, or None when the
        reading is empty, invalid or itself suspicious."""
        os.makedirs(self.scratch_dir, exist_ok=True)
        ok, buf = cv2.imencode(".png", crop_bgr)
        if not ok:
            return None
        path = os.path.join(self.scratch_dir, f"structure_{hashlib.md5(buf.tobytes()).hexdigest()[:16]}.png")
        with open(path, "wb") as f:
            f.write(buf.tobytes())
        self.n_calls += 1
        try:
            _meta, parsed = self.vlm.inspect(image=VLMImage(path=Path(path), mime_type="image/png"), system_prompt="",
                                             user_prompt=_PROMPT, cfg=self.cfg, response_schema=StructureReading)
            smi = (StructureReading.model_validate(parsed).smiles or "").strip()
        except Exception as exc:
            logger.warning("structure_second_reader failed: %s", exc)
            return None
        if not smi or suspicious_smiles(smi):
            return None
        out = canonical_smiles(smi)
        if out:
            self.n_accepted += 1
        return out
