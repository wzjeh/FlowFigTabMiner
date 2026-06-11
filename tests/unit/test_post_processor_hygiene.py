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
