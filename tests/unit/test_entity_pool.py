"""Unit tests for the paper-level compound identity pool.

Focus is the PRECISION guards (the earlier naive version produced wrong
backfills): RDKit validation, canonicalisation, name normalisation, the
relaxed label regex, the no-multi-fragment / no-wildcard SMILES gate, the
adjacent-only CSV association, and the refusal to pool bare row indices.
"""
from __future__ import annotations

import csv
import os
import tempfile

import pytest

from src.adjudication.entity_pool import (
    EntityPool,
    build_global_entity_pool,
    canonical_smiles,
    is_record_label,
    normalize_name,
    resolve_record_smiles,
)


# ── canonical_smiles ───────────────────────────────────────────────────
def test_canonical_smiles_valid_and_invalid():
    assert canonical_smiles("CCN(CC)c1ccc(OC)cc1") is not None
    # a compound label is not a molecule
    assert canonical_smiles("2b") is None
    assert canonical_smiles("") is None
    assert canonical_smiles(None) is None


def test_canonical_smiles_is_canonical():
    # two equivalent SMILES collapse to the same canonical form
    a = canonical_smiles("OC")
    b = canonical_smiles("CO")
    assert a == b


# ── normalize_name ─────────────────────────────────────────────────────
def test_normalize_name_strips_surface_variation():
    assert normalize_name("4-Methoxy-N,N-diethylaniline") == "4methoxynndiethylaniline"
    # same molecule, different surface form → same key
    assert normalize_name("4-methoxy-N,N-diethylaniline ") == normalize_name(
        "4 Methoxy N,N diethylaniline"
    )


# ── label regex ────────────────────────────────────────────────────────
@pytest.mark.parametrize("lab", ["2b", "4a'", "S1", "12c", "1"])
def test_is_record_label_accepts(lab):
    assert is_record_label(lab)


@pytest.mark.parametrize("lab", ["crisaborole intermediate", "", "abc", "tetrahydrofuran"])
def test_is_record_label_rejects(lab):
    assert not is_record_label(lab)


# ── pool gate: no multi-fragment / wildcard SMILES ─────────────────────
def test_pool_rejects_garbage_smiles():
    p = EntityPool()
    p.add_label("2a", "C#CC#C.C#CC#CC#CC#C")   # multi-fragment MolNexTR junk
    p.add_label("1a", "*.O=Cc1ccc(Cl)cc1")     # wildcard / attachment
    p.add_label("ok", "c1ccccc1")              # clean
    p.finalize()
    assert "2a" not in p.label_to_smiles
    assert "1a" not in p.label_to_smiles
    assert p.label_to_smiles.get("ok") == canonical_smiles("c1ccccc1")


# ── conflict resolution by majority vote ───────────────────────────────
def test_label_conflict_majority_vote():
    p = EntityPool()
    p.add_label("3a", "c1ccccc1")
    p.add_label("3a", "c1ccccc1")
    p.add_label("3a", "CCO")
    p.finalize()
    assert p.label_to_smiles["3a"] == canonical_smiles("c1ccccc1")


# ── CSV harvest: adjacent association, reject scrambled + row indices ───
def _write_csv(path, rows):
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(rows)


def test_csv_clean_adjacent_label_is_harvested():
    with tempfile.TemporaryDirectory() as d:
        tdir = os.path.join(d, "tables", "t1")
        os.makedirs(tdir)
        # label "2b" sits right next to its product SMILES (distance 1)
        _write_csv(os.path.join(tdir, "x_extracted.csv"), [
            ["Entry", "Substrate", "Product_SMILES", "Label", "Yield"],
            ["1", "COc1ccc(Br)cc1", "CCN(CC)c1ccc(OC)cc1", "2b", "93"],
        ])
        pool = build_global_entity_pool(d, {}, [])
        assert pool.label_to_smiles.get("2b") == canonical_smiles("CCN(CC)c1ccc(OC)cc1")


def test_csv_scrambled_far_label_not_harvested():
    with tempfile.TemporaryDirectory() as d:
        tdir = os.path.join(d, "tables", "t1")
        os.makedirs(tdir)
        # label "1b" (col 0) is 3 columns from any SMILES (col 3) → scrambled
        _write_csv(os.path.join(tdir, "x_extracted.csv"), [
            ["Label", "t1", "t2", "Product"],
            ["1b", "18", "10", "C=CC=C(O[Si](C)(C)C)[Si](C)(C)C"],
        ])
        pool = build_global_entity_pool(d, {}, [])
        assert "1b" not in pool.label_to_smiles


def test_csv_pure_row_index_not_harvested():
    with tempfile.TemporaryDirectory() as d:
        tdir = os.path.join(d, "tables", "t1")
        os.makedirs(tdir)
        _write_csv(os.path.join(tdir, "x_extracted.csv"), [
            ["Entry", "Product"],
            ["1", "c1ccccc1"],   # "1" is a row index, must NOT become a label
        ])
        pool = build_global_entity_pool(d, {}, [])
        assert "1" not in pool.label_to_smiles


# ── cross-record propagation + entry-index collision guard ─────────────
def test_cross_record_propagation_by_letter_entry():
    records = [
        {"entry_number": "2b", "product_smiles": "CCN(CC)c1ccc(OC)cc1"},
        {"entry_number": "2b", "product_smiles": None, "product_label": None},
    ]
    pool = build_global_entity_pool(tempfile.mkdtemp(), {}, records)
    rec = dict(records[1])
    resolve_record_smiles(rec, pool)
    assert rec["product_smiles"] == canonical_smiles("CCN(CC)c1ccc(OC)cc1")


def test_bare_entry_index_does_not_cross_fill():
    # entry "1" of table A must NOT leak into entry "1" of table B
    records = [
        {"entry_number": "1", "product_smiles": "CCN(CC)c1ccc(OC)cc1"},
        {"entry_number": "1", "product_smiles": None, "product_label": None,
         "product_name": None},
    ]
    pool = build_global_entity_pool(tempfile.mkdtemp(), {}, records)
    rec = dict(records[1])
    resolve_record_smiles(rec, pool)
    assert rec.get("product_smiles") is None  # bare index never pooled


# ── resolve only fills empties, never overwrites ───────────────────────
def test_resolve_does_not_overwrite_existing():
    pool = EntityPool()
    pool.add_label("2b", "c1ccccc1")
    pool.finalize()
    rec = {"product_label": "2b", "product_smiles": "CCO"}
    resolve_record_smiles(rec, pool)
    assert rec["product_smiles"] == "CCO"  # untouched
