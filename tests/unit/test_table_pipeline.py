"""TablePipeline: YOLO filter → MolNexTR → VLM transcription → aligned CSV + evidence."""
import json
import os
from unittest.mock import Mock, patch

import cv2
import numpy as np
import pandas as pd
import pytest

from src.extraction.table.pipeline import TablePipeline
from src.extraction.table.table_vlm import TableTranscription


def _filter(is_table=True, scheme=None):
    body = np.full((200, 300, 3), 255, dtype=np.uint8)
    res = {"is_table": is_table, "conf": 0.9, "components": {"table_body": [{"crop": body, "box": [0, 40, 300, 240], "conf": 0.9}]},
           "table_body_crop": body, "crop_coords": [0, 40, 300, 240]}
    if scheme:
        res["components"]["table_scheme"] = scheme
    f = Mock(); f.filter_tables.return_value = [res]
    return f


def _mols(meta):
    m = Mock(); m.process_image.return_value = (None, meta)
    return m


def _transcriber(tr):
    t = Mock(); t.transcribe.return_value = tr
    return t


@pytest.fixture
def paper(tmp_path):
    """{intermediate}/tables/page_1_table_0.png + a text-layer context."""
    inter = tmp_path / "paper"
    tables = inter / "tables"; tables.mkdir(parents=True)
    img = tables / "page_1_table_0.png"
    cv2.imwrite(str(img), np.full((240, 300, 3), 255, dtype=np.uint8))
    (inter / "context").mkdir()
    ctx = {"source_id": "page_1_table_0", "kind": "table", "caption": "Table 1. Scope.", "caption_source": "pdf_text",
           "footnote": "", "inner_text": "Entry | Product | Yield\n1 | 93\n2 | 20\n3 | 71"}
    (inter / "context" / "page_1_table_0_context.json").write_text(json.dumps(ctx))
    return inter, str(img), str(tables)


def _pipe(filter_, mols, tr, min_agreement=0.6):
    with patch("src.utils.config.load_config", return_value={"tables": {}}):
        return TablePipeline(transcriber=_transcriber(tr), content_recognizer=Mock(), min_text_agreement=min_agreement,
                             table_filter=filter_, molecule_processor=mols)


_TR = TableTranscription(ok=True, header_rows=[["Entry", "Product", "Yield [%]"]],
                         data_rows=[["1", "[STRUCTURE]", "93"], ["2", "[STRUCTURE]", "20"], ["3", "[STRUCTURE]", "71"]],
                         n_cols=3, caption="Table 1. Scope.", footnotes="[a] GC yield.", model="stub", tokens_in=1, tokens_out=1)


def test_filtered_table_writes_status_only(paper):
    inter, img, tables = paper
    res = _pipe(_filter(is_table=False), _mols([]), _TR).process_table(img, tables)
    assert res["is_valid"] is False and res["reason"] == "Filtered by YOLO"
    st = json.load(open(inter / "status" / "page_1_table_0.json"))
    assert st["stage"] == "table_filter" and st["outcome"] == "filtered"
    assert not (inter / "tables" / "page_1_table_0" / "page_1_table_0_evidence.json").exists()


def test_vlm_failure_writes_failed_evidence_and_no_csv(paper):
    inter, img, tables = paper
    res = _pipe(_filter(), _mols([]), TableTranscription(ok=False, notes="vlm call failed: boom")).process_table(img, tables)
    assert res["is_valid"] is False
    ev = json.load(open(inter / "tables" / "page_1_table_0" / "page_1_table_0_evidence.json"))
    assert ev["parse_status"].startswith("failed:vlm") and ev["csv_path"] is None and ev["is_relevant"] is False
    assert not (inter / "tables" / "page_1_table_0_extracted.csv").exists()


def test_happy_path_aligns_smiles_and_cross_checks(paper):
    inter, img, tables = paper
    meta = [{"box": [100, 10, 140, 40], "smiles": "CCO", "conf": 0.9},
            {"box": [100, 70, 140, 100], "smiles": "not a smiles", "conf": 0.9},
            {"box": [100, 130, 140, 160], "smiles": "c1ccccc1", "conf": 0.9}]
    res = _pipe(_filter(), _mols(meta), _TR).process_table(img, tables)
    assert res["is_valid"] and res["is_relevant"] and res["header_row_count"] == 1 and res["parse_status"] == "ok"
    df = pd.read_csv(res["csv_path"], header=None, dtype=str)
    assert list(df.iloc[0]) == ["Entry", "Product", "Yield [%]"]
    assert list(df.iloc[1]) == ["1", "CCO", "93"] and df.iloc[2, 1] == "[STRUCTURE]" and df.iloc[3, 1] == "c1ccccc1"
    ev = json.load(open(res["json_path"]))
    assert ev["structure_alignment"]["status"] in ("ok", "anchored", "anchored_partial") and ev["structure_alignment"]["assigned"] == 2
    assert ev["structure_alignment"]["unresolved"] == 1                  # RDKit-invalid SMILES stays a token
    assert ev["grid_text_agreement"] == 1.0 and ev["header_row_count"] == 1 and ev["n_rows"] == 3 and ev["ditto_filled"] == 0
    assert ev["caption_text"] == "Table 1. Scope." and ev["table_note_text"] == "[a] GC yield."
    st = json.load(open(inter / "status" / "page_1_table_0.json"))
    assert st["stage"] == "evidence" and st["outcome"] == "ok"
    assert not (inter / "tables" / "cells").exists() and not (inter / "tables" / "page_1_table_0" / "page_1_table_0_body_main_masked.png").exists()


def test_low_text_agreement_marks_unverified(paper):
    inter, img, tables = paper
    tr = _TR.model_copy(update={"data_rows": [["1", "[STRUCTURE]", "11"], ["2", "[STRUCTURE]", "22"], ["3", "[STRUCTURE]", "33"]]})
    res = _pipe(_filter(), _mols([]), tr).process_table(img, tables)
    ev = json.load(open(res["json_path"]))
    assert res["parse_status"] == "unverified" and ev["grid_text_agreement"] < 0.6 and ev["structure_alignment"]["status"] == "none"


def test_scheme_boxes_are_dropped_before_alignment(paper):
    inter, img, tables = paper
    scheme = [{"crop": np.zeros((30, 300, 3), dtype=np.uint8), "box": [0, 40, 300, 80], "conf": 0.8}]   # crop coords; body starts at y=40
    meta = [{"box": [10, 5, 50, 30], "smiles": "CCO", "conf": 0.9},          # inside the scheme (body y 0..40)
            {"box": [100, 60, 140, 90], "smiles": "CCC", "conf": 0.9},
            {"box": [100, 120, 140, 150], "smiles": "CCCC", "conf": 0.9},
            {"box": [100, 170, 140, 199], "smiles": "CCCCC", "conf": 0.9}]
    res = _pipe(_filter(scheme=scheme), _mols(meta), _TR).process_table(img, tables)
    ev = json.load(open(res["json_path"]))
    assert ev["n_molecules"] == 3 and ev["structure_alignment"]["status"] in ("ok", "anchored") and ev["structure_alignment"]["assigned"] == 3
    assert os.path.exists(inter / "tables" / "page_1_table_0" / "page_1_table_0_table_scheme_0.png")
