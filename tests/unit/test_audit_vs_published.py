"""scripts/audit_vs_published.py: the published dataset is the answer key."""
import importlib.util
import json
import os

import pandas as pd

_spec = importlib.util.spec_from_file_location("audit_vs_published", os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "audit_vs_published.py"))
audit_mod = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(audit_mod)


def _truth(rows):
    return pd.DataFrame(rows, columns=["paper_id", "paper_doi", "product", "product_smiles_canonical", "tR1_s", "T1_C", "tR2_s", "yield_pct"])


def test_match_accepts_either_residence_time_step_and_checks_structure(tmp_path):
    recs = [  # a figure varied the SECOND step: the pipeline files it under residence_time_2_s
        {"yield_pct": 61.0, "conditions": {"temperature_C": -47.0, "residence_time_s": None, "residence_time_2_s": 0.0123}, "product_smiles": "CC(C)(C)OC(=O)c1ccccc1", "__source_id": "page_7_figure_0_t0"},
        {"yield_pct": 30.0, "conditions": {"temperature_C": -78.0, "residence_time_s": 1.5}, "product_smiles": "*OC(=O)c1ccccc1", "__source_id": "page_3_figure_0_t1"},
        {"yield_pct": 30.5, "conditions": {"temperature_C": -78.0, "residence_time_s": 9.0}, "product_smiles": None, "__source_id": "page_3_figure_0_t2"},
    ]
    (tmp_path / "P_normalized.json").write_text(json.dumps(recs))
    truth = _truth([
        ("pid", "doi", "3a", "CC(C)(C)OC(=O)C1=CC=CC=C1", 1.6, -48.0, 0.011, 61.0),   # step 2 varied -> matched, SMILES agrees
        ("pid", "doi", "3a", "CC(C)(C)OC(=O)C1=CC=CC=C1", 1.5, -78.0, None, 30.0),   # matched on tR1, generic '*' SMILES disagrees
        ("pid", "doi", "3a", "CC(C)(C)OC(=O)C1=CC=CC=C1", 90.0, -78.0, None, 30.0),  # yield present but tR off -> no cond match
    ])
    summary, pairs = audit_mod.audit(truth, {"pid": "x/P.pdf"}, str(tmp_path))
    s = summary.iloc[0]
    assert (s.pub_rows, s.yield_match, s.cond_match, s.smiles_present, s.smiles_agree) == (3, 3, 2, 2, 1)
    assert pairs.pipe_src.tolist() == ["page_7_figure_0_t0", "page_3_figure_0_t1"]


def test_paper_without_output_counts_zero(tmp_path):
    truth = _truth([("pid", "doi", "3a", "CCO", 1.0, 20.0, None, 50.0)])
    summary, pairs = audit_mod.audit(truth, {"pid": "x/missing.pdf"}, str(tmp_path))
    assert summary.iloc[0].cond_match == 0 and pairs.empty
