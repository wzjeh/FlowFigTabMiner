"""Unit tests for post_processor data-hygiene helpers.

Covers:
  - _safe_sheet_name: openpyxl-safe, length-capped, de-duplicated sheet titles
  - _clean_metric_value: coerce yield/conversion/selectivity/ee to a [0,100]
    float, preserving the raw via the caller (note signals a non-trivial change)
"""
from __future__ import annotations

from src.adjudication.post_processor import _safe_sheet_name, _clean_metric_value


# ── _safe_sheet_name (B1) ───────────────────────────────────────────────────
def test_sheet_name_strips_forbidden_chars():
    used = set()
    # ':' is the bug that crashed openpyxl ("Figure 3: yield vs tR")
    assert ":" not in _safe_sheet_name("Figure 3: yield vs tR", used)
    out = _safe_sheet_name("a/b\\c?d*e[f]g", set())
    assert not (set(out) & set(r':\/?*[]'))


def test_sheet_name_caps_length():
    assert len(_safe_sheet_name("x" * 50, set())) <= 31


def test_sheet_name_dedupes():
    used = {"All Records"}
    a = _safe_sheet_name("Table 1", used)
    b = _safe_sheet_name("Table 1", used)
    c = _safe_sheet_name("Table 1", used)
    assert a == "Table 1" and b != a and c not in (a, b)
    assert len({a, b, c}) == 3


def test_sheet_name_empty_fallback():
    assert _safe_sheet_name(":::", set())  # not empty after stripping


# ── _clean_metric_value (B2) ─────────────────────────────────────────────────
def test_metric_clean_number_unchanged():
    assert _clean_metric_value(85) == (85.0, None)
    assert _clean_metric_value(99.5) == (99.5, None)
    assert _clean_metric_value("85") == (85.0, None)


def test_metric_range_to_midpoint():
    v, note = _clean_metric_value("60-80%")
    assert v == 70.0 and note
    v2, note2 = _clean_metric_value("60–80")  # en dash
    assert v2 == 70.0 and note2


def test_metric_bounded_value():
    v, note = _clean_metric_value(">99")
    assert v == 99.0 and note
    v2, note2 = _clean_metric_value("<5")
    assert v2 == 5.0 and note2


def test_metric_out_of_range_dropped():
    assert _clean_metric_value(105)[0] is None
    assert _clean_metric_value("120%")[0] is None


def test_metric_empty_and_bool():
    for empty in (None, "", "null", "nan"):
        assert _clean_metric_value(empty) == (None, None)
    assert _clean_metric_value(True) == (None, None)  # bool is not a metric


# ── promote_smiles_in_names ─────────────────────────────────────────────────
def test_smiles_filed_as_a_name_moves_to_the_smiles_field():
    from src.adjudication.post_processor import promote_smiles_in_names
    r = {"reactant1_name": "Brc1cccnc1Br", "reactant1_smiles": None,
         "reactant2_name": "MeI", "reactant2_smiles": None,
         "product_name": "Cc1cccnc1Br", "product_smiles": "CCO"}       # product SMILES already set → name kept
    promote_smiles_in_names(r)
    assert r["reactant1_smiles"] == "Brc1cccnc1Br" and r["reactant1_name"] is None
    assert r["reactant2_name"] == "MeI" and r["reactant2_smiles"] is None   # a word, not a SMILES
    assert r["product_name"] == "Cc1cccnc1Br" and r["product_smiles"] == "CCO"


def test_excel_safe_frame_strips_control_characters():
    import pandas as pd
    from src.adjudication.post_processor import _excel_safe_frame
    df = pd.DataFrame({"a": ["Figure 9 \x02 Eﬀect", "ok"], "b": [1, 2]})
    out = _excel_safe_frame(df)
    assert out["a"].tolist() == ["Figure 9  Eﬀect", "ok"] and out["b"].tolist() == [1, 2]


def test_invalid_smiles_fields_are_cleared():
    from src.adjudication.post_processor import drop_invalid_smiles
    r = {"product_smiles": "[STRUCTURE]", "reactant1_smiles": "CCCCN1CC(c2ccccc2)C(O)c3ccco31", "reactant2_smiles": "CCO"}
    drop_invalid_smiles(r)
    assert r["product_smiles"] is None and r["reactant1_smiles"] is None and r["reactant2_smiles"] == "CCO"
    assert r["__smiles_dropped"] == "reactant1=CCCCN1CC(c2ccccc2)C(O)c3ccco31; product=[STRUCTURE]"


def test_generic_product_smiles_is_parked_so_name_lookups_can_fill_the_slot():
    from src.adjudication.post_processor import park_generic_smiles
    r = park_generic_smiles({"product_smiles": "*OC(=O)c1ccccc1", "product_name": "tert-butyl benzoate"})
    assert r["product_smiles"] is None and r["product_core_smiles"] == "*OC(=O)c1ccccc1" and "generic" in r["__smiles_dropped"]
    r = park_generic_smiles({"product_smiles": "C=C.C=C.C[Si](C)(C)c1ccccc1-c1ccc(Cl)cc1"})
    assert r["product_smiles"] is None and r["product_core_smiles"].startswith("C=C.")
    for keep in ("CC(C)(C)OC(=O)c1ccccc1", "[Na+].[Cl-]", "CCO.CC(=O)O"):      # one compound; salts / two real fragments stay
        r = park_generic_smiles({"product_smiles": keep})
        assert r["product_smiles"] == keep and "product_core_smiles" not in r
