"""C1 — SMILES-gap forensic (READ-ONLY, no LLM, deterministic).

For every non-hollow record that HAS a product identity (label or name) but NO
product_smiles, classify WHERE in the chain the SMILES was lost, into four
buckets:

  POOL_HAS_IT            entity_pool resolves this record's label/name to a
                         SMILES, yet the record is still empty → resolve/assembly
                         bug or stale output (should be ~0 on fresh runs).
  STRUCT_EXISTS_UNLINKED record has a compound LABEL, no pool hit, but the paper
                         DOES contain recognised structures (MolNexTR SMILES in
                         some table CSV) → structure exists, label never linked
                         (scheme→pool dead / label-norm mismatch / wrong column).
  NO_STRUCT_SOURCE       record has a label, no pool hit, and the paper has NO
                         recognised structures at all → MolNexTR found nothing /
                         no structures drawn.
  NAME_ONLY              record carries only a name (no label-shaped id) → can
                         only be resolved by name→structure lookup (PubChem).

Reuses src/adjudication/entity_pool to rebuild each paper's pool exactly as
PostProcessor does, so the verdict reflects the real pipeline state.

Usage:
  flowfigtabminer/bin/python scripts/forensic_smiles_gap.py \
      ["data/input/Clean organolithium"]   # default = that dir
Writes eval/forensic_smiles_gap.{json,md}.
"""
from __future__ import annotations

import csv
import glob
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.adjudication.entity_pool import (
    build_global_entity_pool,
    canonical_smiles,
    is_record_label,
    _is_compound_label,
)

FINAL_DIR = "data/final_output"
INTER_DIR = "data/intermediate"
DEFAULT_PDF_DIR = "data/input/Clean organolithium"

BUCKETS = ["POOL_HAS_IT", "STRUCT_EXISTS_UNLINKED", "NO_STRUCT_SOURCE",
           "NAME_ONLY_STRUCT", "NAME_ONLY_NO_STRUCT"]


def _load_scheme_pools(intermediate_dir: str) -> dict:
    """Mirror PostProcessor's compound_pool.json loading."""
    pool_path = os.path.join(intermediate_dir, "compound_pool.json")
    if not os.path.exists(pool_path):
        return {}
    try:
        pd = json.load(open(pool_path))
    except Exception:
        return {}
    return pd if isinstance(pd.get("reactant_pool"), dict) else {"compound_pool": pd}


def _count_struct_smiles(intermediate_dir: str) -> int:
    """How many table cells across this paper parse as a real SMILES (i.e.
    MolNexTR recognised a structure).  Counts cells REJECTED by the pool's
    poolable gate too (multi-fragment '.', wildcard '*'), since those still mean
    'a structure was recognised, just not cleanly linked'."""
    n = 0
    tables_dir = os.path.join(intermediate_dir, "tables")
    if not os.path.isdir(tables_dir):
        return 0
    for csv_path in glob.glob(os.path.join(tables_dir, "**", "*_extracted.csv"), recursive=True):
        try:
            with open(csv_path, newline="") as f:
                for row in csv.reader(f):
                    for cell in row:
                        cell = (cell or "").strip()
                        # cheap pre-filter: SMILES have no spaces and >=2 chars
                        if len(cell) < 2 or " " in cell:
                            continue
                        if canonical_smiles(cell):
                            n += 1
        except Exception:
            continue
    return n


def _has_label(rec: dict) -> bool:
    return (is_record_label(rec.get("product_label"))
            or is_record_label(rec.get("product_name"))
            or _is_compound_label(rec.get("entry_number")))


_TABLE_CELLS_CACHE: dict[str, list] = {}


def _table_cells(intermediate_dir: str) -> list:
    if intermediate_dir in _TABLE_CELLS_CACHE:
        return _TABLE_CELLS_CACHE[intermediate_dir]
    rows = []
    for cp in glob.glob(os.path.join(intermediate_dir, "tables", "**", "*_extracted.csv"), recursive=True):
        try:
            with open(cp, newline="") as f:
                for row in csv.reader(f):
                    rows.append([(c or "").strip() for c in row])
        except Exception:
            continue
    _TABLE_CELLS_CACHE[intermediate_dir] = rows
    return rows


def _table_label_has_struct(intermediate_dir: str, label) -> bool:
    """A table row contains a cell == label AND a sibling cell that parses as a
    SMILES (wildcard '*' allowed) → harvest could link them."""
    if not label:
        return False
    lab = str(label).strip().lower()
    for row in _table_cells(intermediate_dir):
        if lab in [c.lower() for c in row]:
            for c in row:
                if len(c) >= 2 and " " not in c and canonical_smiles(c):
                    return True
    return False


def classify_fix(rec: dict, intermediate_dir: str) -> str:
    """For a LINKING-gap record (structure exists but unlinked), what fix could
    resolve it DETERMINISTICALLY? HARD = needs scheme-pool revival or semantic
    name→structure linking (no deterministic path)."""
    label = rec.get("product_label")
    name = rec.get("product_name")
    if label and canonical_smiles(str(label)):
        return "FIX_label_is_smiles"      # label field holds a (often '*') SMILES
    if name and canonical_smiles(str(name)):
        return "FIX_name_is_smiles"
    if _table_label_has_struct(intermediate_dir, label):
        return "FIX_table_harvest"        # label sits in a table row with a structure
    return "HARD_no_anchor"               # figure fixed-product / descriptive name


DETERMINISTIC_FIXES = {"FIX_label_is_smiles", "FIX_name_is_smiles", "FIX_table_harvest"}
LINKING_BUCKETS = {"STRUCT_EXISTS_UNLINKED", "NAME_ONLY_STRUCT"}


def classify_paper(basename: str) -> list[dict]:
    norm_path = os.path.join(FINAL_DIR, f"{basename}_normalized.json")
    if not os.path.exists(norm_path):
        return []
    try:
        records = json.load(open(norm_path))
    except Exception:
        return []
    if not isinstance(records, list):
        return []

    intermediate_dir = os.path.join(INTER_DIR, basename)
    scheme_pools = _load_scheme_pools(intermediate_dir)
    pool = build_global_entity_pool(intermediate_dir, scheme_pools, records)
    struct_count = _count_struct_smiles(intermediate_dir)

    out = []
    for rec in records:
        if rec.get("is_hollow") is True:
            continue
        if rec.get("product_smiles"):
            continue
        label = rec.get("product_label")
        name = rec.get("product_name")
        entry = rec.get("entry_number")
        if not (label or name):
            continue  # no product identity at all — not a SMILES gap

        hit = pool.lookup(label=label, name=name)
        if not hit and _is_compound_label(entry):
            hit = pool.lookup(label=entry)

        if hit:
            bucket = "POOL_HAS_IT"
        elif _has_label(rec):
            bucket = "STRUCT_EXISTS_UNLINKED" if struct_count > 0 else "NO_STRUCT_SOURCE"
        else:
            # name only, no label-shaped id: split by whether the paper has any
            # recognised structure (a name-link miss that COULD be fixed by
            # name normalisation) vs no structure at all (PubChem territory).
            bucket = "NAME_ONLY_STRUCT" if struct_count > 0 else "NAME_ONLY_NO_STRUCT"

        fix = classify_fix(rec, intermediate_dir) if bucket in LINKING_BUCKETS else None
        out.append({
            "paper": basename,
            "source": rec.get("source_table_or_figure", ""),
            "product_label": label,
            "product_name": name,
            "entry_number": entry,
            "paper_struct_smiles": struct_count,
            "pool_size": pool.size,
            "bucket": bucket,
            "fix": fix,
        })
    return out


def _resolve_basenames(spec: str) -> list[str]:
    """spec = 'ALL' (every normalized.json), '@file' (one basename per line),
    a PDF directory, or a single basename."""
    if spec == "ALL":
        return sorted(os.path.basename(f)[: -len("_normalized.json")]
                      for f in glob.glob(os.path.join(FINAL_DIR, "*_normalized.json")))
    if spec.startswith("@"):
        with open(spec[1:]) as f:
            return [ln.strip() for ln in f if ln.strip()]
    if os.path.isdir(spec):
        return sorted(os.path.splitext(os.path.basename(p))[0]
                      for p in glob.glob(os.path.join(spec, "*.pdf")))
    return [spec]


def main(spec: str, label: str = "gap", exclude_dirs: list[str] | None = None):
    basenames = _resolve_basenames(spec)
    # exclude basenames belonging to other named sets (for lateral splits)
    excluded = set()
    for d in (exclude_dirs or []):
        excluded |= set(_resolve_basenames(d))
    basenames = [b for b in basenames if b not in excluded]

    all_gaps = []
    per_paper = {}
    for b in basenames:
        rows = classify_paper(b)
        all_gaps.extend(rows)
        per_paper[b] = len(rows)

    total = len(all_gaps)
    counts = {bk: sum(1 for r in all_gaps if r["bucket"] == bk) for bk in BUCKETS}

    # ── Deterministic-fixability of LINKING gaps (the C2 question) ──
    link_gaps = [r for r in all_gaps if r["bucket"] in LINKING_BUCKETS]
    fix_rec = Counter(r["fix"] for r in link_gaps)
    fix_uniq_keys = {}
    for r in link_gaps:
        fix_uniq_keys.setdefault((r["paper"], str(r["product_label"]), str(r["product_name"])), r["fix"])
    fix_uniq = Counter(fix_uniq_keys.values())
    det_rec = sum(v for k, v in fix_rec.items() if k in DETERMINISTIC_FIXES)
    det_uniq = sum(v for k, v in fix_uniq.items() if k in DETERMINISTIC_FIXES)
    hard_rec = fix_rec.get("HARD_no_anchor", 0)
    hard_uniq = fix_uniq.get("HARD_no_anchor", 0)
    link_rec = len(link_gaps)
    link_uniq = len(fix_uniq_keys)

    # Unique-compound view: figures emit one record per data point for the SAME
    # fixed product, inflating record counts. De-dupe by (paper, label, name) so
    # the bucket mix reflects distinct compounds, not plot density.
    uniq_keys = {}
    for r in all_gaps:
        uniq_keys.setdefault((r["paper"], str(r["product_label"]), str(r["product_name"])), r["bucket"])
    uniq_total = len(uniq_keys)
    uniq_counts = {bk: sum(1 for b in uniq_keys.values() if b == bk) for bk in BUCKETS}

    # ── report ──
    lines = [f"# SMILES-gap forensic — {label}", "",
             f"Corpus: `{spec}` ({len(basenames)} papers"
             + (f", excl {len(excluded)}" if excluded else "") + ")",
             f"Total gap records (non-hollow, has identity, no product_smiles): **{total}**", "",
             "| Bucket | Count | % | Meaning |", "|---|---|---|---|"]
    meaning = {
        "POOL_HAS_IT": "pool resolves it but record empty → resolve/stale bug",
        "STRUCT_EXISTS_UNLINKED": "labelled, paper HAS structures, label not linked",
        "NO_STRUCT_SOURCE": "labelled, paper has NO structures (MolNexTR/none drawn)",
        "NAME_ONLY_STRUCT": "name only, paper HAS structures → name-link miss (fixable)",
        "NAME_ONLY_NO_STRUCT": "name only, no structures → PubChem territory",
    }
    for bk in BUCKETS:
        pct = round(100.0 * counts[bk] / total, 1) if total else 0.0
        lines.append(f"| {bk} | {counts[bk]} | {pct}% | {meaning[bk]} |")

    lines += ["",
              f"### Unique-compound view ({uniq_total} distinct (paper,label,name); "
              "de-dupes figure multi-point inflation)", "",
              "| Bucket | Unique | % |", "|---|---|---|"]
    for bk in BUCKETS:
        upct = round(100.0 * uniq_counts[bk] / uniq_total, 1) if uniq_total else 0.0
        lines.append(f"| {bk} | {uniq_counts[bk]} | {upct}% |")

    # ── linking-gap deterministic fixability (the lateral-validation headline) ──
    dr = lambda x: round(100.0 * x / link_rec, 1) if link_rec else 0.0
    du = lambda x: round(100.0 * x / link_uniq, 1) if link_uniq else 0.0
    lines += ["", "## Linking-gap deterministic fixability", "",
              f"Of the {link_rec} linking-gap records ({link_uniq} unique compounds), how many "
              "could a DETERMINISTIC fix (wildcard-`*` / label-is-SMILES / table-harvest) resolve "
              "vs HARD (figure fixed-product / descriptive name → needs scheme-pool or semantic linking):", "",
              "| Fixability | Records | % | Unique | % |", "|---|---|---|---|---|",
              f"| **Deterministic** | {det_rec} | {dr(det_rec)}% | {det_uniq} | {du(det_uniq)}% |",
              f"| **HARD_no_anchor** | {hard_rec} | {dr(hard_rec)}% | {hard_uniq} | {du(hard_uniq)}% |", "",
              "Fix breakdown (records / unique):"]
    for fx in ("FIX_label_is_smiles", "FIX_name_is_smiles", "FIX_table_harvest", "HARD_no_anchor"):
        lines.append(f"- {fx}: {fix_rec.get(fx, 0)} / {fix_uniq.get(fx, 0)}")

    # samples per bucket
    lines += ["", "## Samples (up to 3 per bucket)", ""]
    for bk in BUCKETS:
        ex, seen = [], set()
        for r in all_gaps:
            if r["bucket"] != bk:
                continue
            k = (r["paper"], str(r["product_label"]), str(r["product_name"]))
            if k in seen:
                continue
            seen.add(k)
            ex.append(r)
            if len(ex) >= 3:
                break
        lines.append(f"**{bk}** ({counts[bk]})")
        if not ex:
            lines.append("- (none)")
        for r in ex:
            lines.append(f"- `{r['paper'][:30]}` {r['source']} | label={r['product_label']!r} "
                         f"name={str(r['product_name'])[:28]!r} struct_in_paper={r['paper_struct_smiles']}")
        lines.append("")

    # per-paper gap counts
    lines += ["## Gap records per paper", ""]
    for b in sorted(per_paper, key=lambda k: -per_paper[k]):
        if per_paper[b]:
            lines.append(f"- {per_paper[b]:>4}  {b[:60]}")

    md = "\n".join(lines)
    print(md)
    os.makedirs("eval", exist_ok=True)
    stem = f"eval/forensic_smiles_gap_{label}"
    with open(stem + ".json", "w") as f:
        json.dump({"corpus": spec, "label": label, "papers": len(basenames),
                   "total": total, "counts": counts,
                   "uniq_total": uniq_total, "uniq_counts": uniq_counts,
                   "linking_fixability": {
                       "records": {"deterministic": det_rec, "hard": hard_rec, "total": link_rec},
                       "unique": {"deterministic": det_uniq, "hard": hard_uniq, "total": link_uniq},
                       "fix_records": dict(fix_rec), "fix_unique": dict(fix_uniq)},
                   "per_paper": per_paper, "gaps": all_gaps}, f, indent=2, ensure_ascii=False)
    with open(stem + ".md", "w") as f:
        f.write(md)
    print(f"\n[forensic] wrote {stem}.{{json,md}}")
    return {"label": label, "papers": len(basenames), "total": total,
            "link_rec": link_rec, "det_rec": det_rec, "hard_rec": hard_rec,
            "link_uniq": link_uniq, "det_uniq": det_uniq, "hard_uniq": hard_uniq}


if __name__ == "__main__":
    spec = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PDF_DIR
    label = sys.argv[2] if len(sys.argv) > 2 else "gap"
    main(spec, label)
