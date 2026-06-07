"""Unit tests for the hollow-record flag in post_processor.

A hollow record has no chemical identity AND no reaction-outcome metric —
the signature of a non-reaction figure (e.g. a temperature-vs-time process
trace) mis-read as reaction data.
"""
from __future__ import annotations

from src.adjudication.post_processor import _is_hollow_record


def test_hollow_when_only_conditions():
    # temperature / solvent / time only — a process-monitoring trace point
    assert _is_hollow_record({"conditions": {"temperature_C": 5},
                              "other_metrics": {"time_min": 0.4}}) is True


def test_not_hollow_with_conversion_only():
    # conversion is a real reaction outcome (process papers report it)
    assert _is_hollow_record({"conversion_pct": 85}) is False


def test_not_hollow_with_yield():
    assert _is_hollow_record({"yield_pct": 73}) is False


def test_not_hollow_with_product_identity():
    assert _is_hollow_record({"product_label": "2b"}) is False
    assert _is_hollow_record({"product_name": "aniline"}) is False
    assert _is_hollow_record({"product_smiles": "c1ccccc1N"}) is False


def test_not_hollow_with_reactant_identity():
    assert _is_hollow_record({"reactant1_name": "n-BuLi"}) is False


def test_not_hollow_with_selectivity_or_ee():
    assert _is_hollow_record({"selectivity_pct": 90}) is False
    assert _is_hollow_record({"ee_pct": 99}) is False
