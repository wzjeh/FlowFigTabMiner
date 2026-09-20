"""Table extraction: YOLO segmentation → MolNexTR SMILES → VLM transcription.

Per table crop (``tables/{b}.png`` from TF-ID):

1. ``TableFilter`` (YOLO tab-seg) rejects non-tables and cuts the table body
   (input for the molecule detector) and any ``table_scheme`` drawing (input
   for Step 3.5).
2. ``MoleculeProcessor`` (YOLO tab-mol + MolNexTR) → ``[{box, smiles}]`` in
   body-crop pixels.  MolNexTR is the only SMILES source.
3. ``TableTranscriber`` — ONE VLM call on the full, unmasked crop returns
   caption / scheme conditions / header rows / data rows / footnotes; drawn
   molecules are the token ``[STRUCTURE]``.
4. ``grid_ops.align_structures`` puts the MolNexTR SMILES into the token
   cells by geometry; ``grid_ops.text_agreement`` cross-checks the numbers
   against the PDF text layer (``context/{b}_context.json``).
5. Dense grid → headerless CSV (header rows first) + ``{b}_evidence.json``
   (always written; ``parse_status`` = ok | unverified | failed:<reason>) +
   per-source status record.

Caption / footnote precedence is decided downstream
(``source_discovery.apply_context_to_evidence``): the PDF text layer wins
when present, the VLM transcription is the fallback.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import pandas as pd
import yaml

from src.extraction.common.content_recognizer import ContentRecognizer
from src.extraction.common.molecule_processor import MoleculeProcessor
from src.extraction.table.grid_ops import align_structures, text_agreement, text_layer_anchors
from src.extraction.table.table_vlm import TableTranscriber
from src.parsing.caption_locator import load_context
from src.parsing.table_filter import TableFilter
from src.pipeline.status import write_status

logger = logging.getLogger(__name__)

_DEFAULT_KEYWORDS = ['yield', 'conversion', 'selectivity', 'product', 'composition', 'conditions', 'reaction']


def _load_keywords() -> List[str]:
    for kw_path in ("keywords.yaml", os.path.join(os.path.dirname(__file__), "..", "..", "..", "keywords.yaml")):
        if os.path.exists(kw_path):
            try:
                with open(kw_path) as f:
                    return (yaml.safe_load(f) or {}).get('figure_table_keywords', _DEFAULT_KEYWORDS)
            except Exception:
                pass
    return _DEFAULT_KEYWORDS


def _smiles_is_valid(smiles: str) -> bool:
    """MolNexTR / OCR-fallback output that RDKit cannot parse must not enter a cell."""
    if not smiles or smiles.startswith("<"):
        return False
    try:
        from rdkit import Chem
        from rdkit import RDLogger
        RDLogger.DisableLog("rdApp.*")
        return Chem.MolFromSmiles(smiles) is not None
    except Exception:
        return True


def _crop_scale(intermediate_dir: str) -> float:
    """Render scale of the TF-ID crops (``layout.json`` ``crop_scale``; 4 px/pt)."""
    try:
        with open(os.path.join(intermediate_dir, "layout.json")) as f:
            return float(json.load(f).get("crop_scale", 4))
    except Exception:
        return 4.0


def _drop_scheme_boxes(mol_meta: List[Dict[str, Any]], scheme_boxes: List[List[float]], y_offset: float) -> List[Dict[str, Any]]:
    """Remove molecule boxes lying (≥50 % of their area) inside a YOLO
    ``table_scheme`` box — a reaction scheme drawn inside the body cut is not
    a table cell.  Scheme boxes are in crop pixels, molecules in body pixels."""
    if not scheme_boxes:
        return mol_meta
    keep = []
    for m in mol_meta:
        x1, y1, x2, y2 = m["box"]
        area = max(1.0, (x2 - x1) * (y2 - y1))
        inside = False
        for sx1, sy1, sx2, sy2 in scheme_boxes:
            sy1, sy2 = sy1 - y_offset, sy2 - y_offset
            ix = max(0.0, min(x2, sx2) - max(x1, sx1)); iy = max(0.0, min(y2, sy2) - max(y1, sy1))
            if ix * iy / area >= 0.5:
                inside = True; break
        if not inside:
            keep.append(m)
    return keep


class TablePipeline:
    def __init__(self, transcriber: TableTranscriber, content_recognizer: Optional[ContentRecognizer] = None,
                 min_text_agreement: float = 0.6, table_filter=None, molecule_processor=None):
        """
        Args:
            transcriber: VLM table transcriber (any ``VLMProvider`` behind it).
            content_recognizer: shared MolNexTR/OCR engine (constructed if None).
            min_text_agreement: grid-vs-text-layer numeric recall below which
                the table is marked ``parse_status="unverified"``.
            table_filter / molecule_processor: optional pre-built stage models
                (tests); otherwise loaded once from config.yaml.
        """
        from src.utils.config import load_config
        self.cfg = load_config()
        self.tables_cfg = self.cfg.get("tables", {})
        self.transcriber = transcriber
        self.min_text_agreement = float(min_text_agreement)
        self.keywords = _load_keywords()
        self.filter = table_filter or TableFilter(model_path=self.tables_cfg.get("segmentation", {}).get("model_path"))
        mol_cfg = self.tables_cfg.get("molecule_detection", {})
        self.molecule_processor = molecule_processor or MoleculeProcessor(
            model_path=mol_cfg.get("model_path"), conf_threshold=mol_cfg.get("confidence_threshold", 0.25))
        self.recognizer = content_recognizer or ContentRecognizer()

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _status(image_path, output_dir, stage, outcome, reason="", **extra):
        """Per-source outcome record (``{intermediate}/status/{table}.json``);
        ``output_dir`` is ``{intermediate}/tables``."""
        source_id = os.path.splitext(os.path.basename(image_path))[0]
        write_status(os.path.dirname(os.path.abspath(output_dir)), source_id, stage, outcome, reason, **extra)

    def _write_evidence(self, table_output_dir: str, table_basename: str, data: Dict[str, Any]) -> str:
        json_path = os.path.join(table_output_dir, f"{table_basename}_evidence.json")
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"   -> Saved Evidence JSON to: {json_path}")
        return json_path

    # ------------------------------------------------------------------ main
    def process_table(self, image_path: str, output_dir: str) -> Dict[str, Any]:
        """Process one TF-ID table crop; ``output_dir`` is ``{intermediate}/tables``."""
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Processing Table: {image_path}")
        table_basename = os.path.splitext(os.path.basename(image_path))[0]
        intermediate_dir = os.path.dirname(os.path.abspath(output_dir))

        # 1. YOLO filter + segmentation
        filter_res = self.filter.filter_tables([image_path], conf_threshold=0.4)[0]
        if not filter_res['is_table']:
            logger.info(f"   -> Rejected by Table Filter (Conf: {filter_res.get('conf')})")
            self._status(image_path, output_dir, "table_filter", "filtered",
                         f"rejected by TabSeg YOLO (conf={filter_res.get('conf')})")
            return {'is_valid': False, 'reason': 'Filtered by YOLO'}

        table_output_dir = os.path.join(output_dir, table_basename)
        os.makedirs(table_output_dir, exist_ok=True)
        components = filter_res.get('components', {}) or {}
        scheme_boxes = []
        for i, item in enumerate(components.get('table_scheme', [])):
            cv2.imwrite(os.path.join(table_output_dir, f"{table_basename}_table_scheme_{i}.png"), item['crop'])
            if item.get('box') is not None:
                scheme_boxes.append([float(v) for v in item['box']])
        body_crop = filter_res.get('table_body_crop')
        body_path = os.path.join(table_output_dir, f"{table_basename}_body_main.png")
        if body_crop is None:
            body_crop = cv2.imread(image_path)
        cv2.imwrite(body_path, body_crop)
        crop_coords = filter_res.get('crop_coords') or [0, 0, 0, 0]

        # 2. Molecules → SMILES (MolNexTR), boxes in body pixels
        mol_meta: List[Dict[str, Any]] = []
        try:
            _, mol_meta = self.molecule_processor.process_image(
                body_path, self.recognizer, mask_only=True,
                output_path=os.path.join(table_output_dir, f"{table_basename}_body_main_debug_yolo.png"))
        except Exception as exc:
            logger.warning(f"   -> molecule detection failed: {exc}")
        mol_meta = _drop_scheme_boxes(mol_meta or [], scheme_boxes, float(crop_coords[1] or 0))
        for m in mol_meta:
            if not _smiles_is_valid(str(m.get("smiles") or "")):
                m["smiles"] = ""
        logger.info(f"   -> {len(mol_meta)} molecule boxes, {sum(1 for m in mol_meta if m['smiles'])} with valid SMILES")

        # 3. VLM transcription of the full crop
        tr = self.transcriber.transcribe(Path(image_path))
        context = load_context(intermediate_dir, table_basename) or {}
        transcriber_meta = {"model": tr.model, "latency_ms": tr.latency_ms, "tokens_in": tr.tokens_in,
                            "tokens_out": tr.tokens_out, "cache_hit": tr.cache_hit, "notes": tr.notes}
        if not tr.ok:
            reason = f"failed:vlm:{tr.notes or 'unknown'}"
            json_path = self._write_evidence(table_output_dir, table_basename, {
                "csv_path": None, "num_extracted": 0, "n_rows": 0, "n_cols": 0, "header_row_count": 0,
                "caption_text": "", "table_note_text": "", "scheme_conditions": None,
                "is_relevant": False, "parse_status": reason, "grid_text_agreement": None,
                "structure_alignment": None, "n_molecules": len(mol_meta), "transcriber": transcriber_meta,
            })
            self._status(image_path, output_dir, "table_vlm", "failed", reason)
            return {'is_valid': False, 'reason': 'vlm_failed', 'json_path': json_path}

        # 4. Structures by geometry (text-layer anchors when available) + cross-check
        full_grid = tr.header_rows + tr.data_rows
        row_c = col_c = None
        if context.get("inner_lines") and context.get("bbox_pt"):
            scale = _crop_scale(intermediate_dir)
            img_w = float(cv2.imread(image_path).shape[1]) if os.path.exists(image_path) else 0.0
            row_c, col_c = text_layer_anchors(full_grid, len(tr.header_rows), context["inner_lines"], context["bbox_pt"],
                                              scale, (float(crop_coords[0] or 0), float(crop_coords[1] or 0)), img_w)
        grid, align = align_structures(full_grid, mol_meta, row_centres=row_c, col_centres=col_c)
        agreement = text_agreement(tr.data_rows, context.get("inner_text"))
        parse_status = "unverified" if (agreement is not None and agreement < self.min_text_agreement) else "ok"
        logger.info(f"   -> grid {len(tr.header_rows)}+{len(tr.data_rows)}x{tr.n_cols}, structures {align['status']} "
                    f"({align['assigned']}/{align['n_tokens']}), text agreement {agreement}, status {parse_status}")

        # 5. CSV (dense, headerless, header rows first)
        df = pd.DataFrame(grid)
        csv_path = os.path.join(output_dir, f"{table_basename}_extracted.csv")
        df.to_csv(csv_path, index=False, header=False)

        # 6. Relevance (soft): PDF caption when located, else the VLM's
        caption_pdf = context.get("caption", "") if context.get("caption_source") == "pdf_text" else ""
        check_text = " ".join(x for x in (caption_pdf, tr.caption or "", tr.footnotes or "",
                                          df.head(3).to_string(index=False, header=False)) if x).lower()
        is_relevant = any(kw in check_text for kw in self.keywords)
        if not is_relevant:
            logger.info("   -> Table marked not relevant by keyword filter (soft reject, CSV still written)")

        # 7. Evidence + status
        evidence = {
            "csv_path": csv_path,
            "num_extracted": int(df.shape[0] * df.shape[1]),
            "n_rows": int(len(tr.data_rows)), "n_cols": int(tr.n_cols),
            "header_row_count": int(len(tr.header_rows)),
            "caption_text": tr.caption or "",
            "table_note_text": tr.footnotes or "",
            "scheme_conditions": tr.scheme_conditions,
            "is_relevant": bool(is_relevant),
            "parse_status": parse_status,
            "grid_text_agreement": agreement,
            "structure_alignment": align,
            "n_molecules": len(mol_meta),
            "molecules": [{"box": [round(float(v), 1) for v in m["box"]], "smiles": m.get("smiles", "")} for m in mol_meta],
            "transcriber": transcriber_meta,
        }
        json_path = self._write_evidence(table_output_dir, table_basename, evidence)
        self._status(image_path, output_dir, "evidence", "ok", "",
                     rows=int(len(tr.data_rows)), is_relevant=bool(is_relevant), parse_status=parse_status)
        return {
            'is_valid': True, 'is_relevant': is_relevant, 'csv_path': csv_path, 'dataframe': df,
            'json_path': json_path, 'header_row_count': len(tr.header_rows), 'parse_status': parse_status,
            'caption_text': tr.caption or "", 'table_note_text': tr.footnotes or "",
        }
