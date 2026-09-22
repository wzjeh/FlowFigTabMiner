"""Harvest ``label -> SMILES`` from reaction schemes that TF-ID filed as figures.

A scheme drawn as a figure (Scheme 3: "1 -> 2 -> 3") is the only place many
papers show the structure behind a label such as "3" or "c-9".  Macro YOLO
rejects these crops ("no target_image"), so nothing downstream ever saw them
and the entity pool could not resolve the label that every heatmap record of
that paper carries.

Division of labour (same as the tick-label reader): molecule YOLO finds the
structures, MolNexTR reads them, and the VLM only copies the label printed
next to each numbered structure on a contact sheet.  An R-group family
("c-8 (R = Me) / c-10 (R = Ph)" under one drawing) is expanded with RDKit
from the wildcard core MolNexTR returns.
"""
from __future__ import annotations

import glob
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
from pydantic import BaseModel

from src.adjudication.entity_pool import canonical_smiles
from src.extraction.common.structure_second_reader import VLMStructureReader
from src.extraction.figure.contact_sheet import build_contact_sheet
from src.llm.providers.base import VLMImage

logger = logging.getLogger(__name__)

_PROMPT = (Path(__file__).parent / "prompts" / "compound_labels.md").read_text(encoding="utf-8")


class RVariant(BaseModel):
    label: str
    r_group: str


class CompoundReading(BaseModel):
    index: int
    label: Optional[str] = None
    variants: List[RVariant] = []


class CompoundLabelsResponse(BaseModel):
    readings: List[CompoundReading] = []


# Substituent as a SMILES fragment whose first atom takes the wildcard's bond.
R_GROUPS = {
    "h": "[H]", "me": "C", "methyl": "C", "et": "CC", "ethyl": "CC",
    "pr": "CCC", "n-pr": "CCC", "npr": "CCC", "propyl": "CCC",
    "i-pr": "C(C)C", "ipr": "C(C)C", "isopropyl": "C(C)C",
    "bu": "CCCC", "n-bu": "CCCC", "nbu": "CCCC", "butyl": "CCCC",
    "s-bu": "C(C)CC", "sbu": "C(C)CC", "t-bu": "C(C)(C)C", "tbu": "C(C)(C)C", "tert-butyl": "C(C)(C)C",
    "ph": "c1ccccc1", "phenyl": "c1ccccc1", "bn": "Cc1ccccc1", "benzyl": "Cc1ccccc1",
    "ome": "OC", "oet": "OCC", "tms": "[Si](C)(C)C", "sime3": "[Si](C)(C)C", "me3si": "[Si](C)(C)C",
    "cf3": "C(F)(F)F", "f": "F", "cl": "Cl", "br": "Br", "cn": "C#N", "no2": "[N+](=O)[O-]",
}


def substitute_r(core: str, r_name: str) -> Optional[str]:
    """``*C1OC1(C)c1ccccc1`` + "Me" -> ``CC1OC1(C)c1ccccc1``.  None when the core
    has other than one wildcard or the group is not in the table."""
    frag = R_GROUPS.get(re.sub(r"\s+", "", (r_name or "")).lower())
    if not frag or core.count("*") != 1:
        return None
    try:
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog("rdApp.*")
        mol = Chem.MolFromSmiles(core)
        if mol is None:
            return None
        if frag == "[H]":
            out = Chem.DeleteSubstructs(mol, Chem.MolFromSmarts("[#0]"))
        else:
            out = Chem.ReplaceSubstructs(mol, Chem.MolFromSmarts("[#0]"), Chem.MolFromSmiles(frag), replaceAll=True)[0]
        return canonical_smiles(Chem.MolToSmiles(out))
    except Exception:
        return None


def filtered_scheme_crops(intermediate_dir: str) -> List[str]:
    """Figure crops macro YOLO rejected as non-charts, in page order."""
    out = []
    for sp in sorted(glob.glob(os.path.join(glob.escape(intermediate_dir), "status", "*.json"))):
        try:
            st = json.load(open(sp))
        except Exception:
            continue
        if st.get("stage") == "macro_clean" and st.get("outcome") == "filtered":
            png = os.path.join(intermediate_dir, "figures", st["source_id"] + ".png")
            if os.path.exists(png):
                out.append(png)
    return out


def label_cells(mols: List[Dict[str, Any]], img_shape) -> List[Dict[str, Any]]:
    """The contact-sheet cell for each structure: the box widened a little and
    extended downwards, where the label is printed."""
    h, w = img_shape[:2]
    cells = []
    for m in mols:
        x1, y1, x2, y2 = m["box"]
        bw, bh = x2 - x1, y2 - y1
        cells.append({"box": [max(0, x1 - 0.25 * bw), max(0, y1 - 0.15 * bh), min(w, x2 + 0.25 * bw), min(h, y2 + 0.9 * bh)]})
    return cells


def readings_to_pool(mols: List[Dict[str, Any]], readings: List[CompoundReading]) -> Dict[str, str]:
    """Pair each reading with its structure; expand R-group families."""
    pool: Dict[str, str] = {}
    for r in readings:
        if not (0 <= r.index < len(mols)):
            continue
        smi = (mols[r.index].get("smiles") or "").strip()
        if not smi or smi.lower() == "<invalid>":
            continue
        if r.variants:
            for v in r.variants:
                s = substitute_r(smi, v.r_group)
                if s and v.label and v.label.strip():
                    pool.setdefault(v.label.strip(), s)
            continue
        c = canonical_smiles(smi)
        if c and "*" not in c and "." not in c and r.label and r.label.strip():
            pool.setdefault(r.label.strip(), c)
    return pool


class FigureSchemeHarvester:
    def __init__(self, molecule_processor, content_recognizer, vlm, cfg, *, max_boxes: int = 24):
        self.mp = molecule_processor
        self.cr = content_recognizer
        self.vlm = vlm
        self.cfg = cfg
        self.max_boxes = max_boxes
        self.second_reader = VLMStructureReader(vlm, cfg)

    def harvest(self, intermediate_dir: str) -> Dict[str, str]:
        """label -> canonical SMILES over every filtered figure crop of one paper.
        Forensics: ``schemes/{id}_compounds_sheet.png`` + ``{id}_compounds.json``."""
        out_dir = os.path.join(intermediate_dir, "schemes")
        pool: Dict[str, str] = {}
        for png in filtered_scheme_crops(intermediate_dir):
            sid = os.path.splitext(os.path.basename(png))[0]
            try:
                pool_i = self._one(png, sid, out_dir)
            except Exception as exc:
                logger.warning("figure_scheme %s failed: %s", sid, exc)
                continue
            for k, v in pool_i.items():
                pool.setdefault(k, v)
        return pool

    def _one(self, png: str, sid: str, out_dir: str) -> Dict[str, str]:
        os.makedirs(out_dir, exist_ok=True)
        _, mols = self.mp.process_image(png, self.cr, mask_only=True, output_path=os.path.join(out_dir, f"{sid}_molecules.png"),
                                        second_reader=self.second_reader)
        mols = [m for m in (mols or []) if m.get("smiles")][: self.max_boxes]
        if not mols:
            return {}
        img = cv2.imread(png)
        sheet, _cells = build_contact_sheet(img, label_cells(mols, img.shape), pad=0, cell_max=420)
        sheet_path = os.path.join(out_dir, f"{sid}_compounds_sheet.png")
        cv2.imwrite(sheet_path, sheet)
        t0 = time.perf_counter()
        meta, parsed = self.vlm.inspect(image=VLMImage(path=Path(sheet_path), mime_type="image/png"), system_prompt="",
                                        user_prompt=_PROMPT, cfg=self.cfg, response_schema=CompoundLabelsResponse)
        resp = CompoundLabelsResponse.model_validate(parsed)
        pool = readings_to_pool(mols, resp.readings)
        json.dump({"source_id": sid, "n_structures": len(mols), "latency_ms": (time.perf_counter() - t0) * 1000.0,
                   "cache_hit": bool(getattr(meta, "cache_hit", False)),
                   "structures": [{"index": i, "box": [int(v) for v in m["box"]], "smiles": m.get("smiles"), "reader": m.get("reader")} for i, m in enumerate(mols)],
                   "readings": [r.model_dump() for r in resp.readings], "pool": pool},
                  open(os.path.join(out_dir, f"{sid}_compounds.json"), "w"), indent=1)
        logger.info("figure_scheme %s structures=%d labels=%d pool=%d", sid, len(mols), sum(bool(r.label or r.variants) for r in resp.readings), len(pool))
        return pool
