#!/usr/bin/env python3
"""
check_pipeline_smiles_correctness.py
-------------------------------------
Compare Pipeline (MolNexTR) SMILES against Gemini and Claude VLM outputs.

When Gemini and Claude agree on a canonical SMILES, that forms the
"G∩C consensus" ground truth.  We then check how many Pipeline SMILES
match that consensus.

Sources:
  - Pipeline: layer1_table_annotation.csv (cell_type == "smiles")
  - Gemini:   images/tables/T{1-5}.json
  - Claude:   images/tables/T{1-5}_claude.json
"""

import csv
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

try:
    from rdkit import Chem
    HAS_RDKIT = True
except ImportError:
    print("WARNING: RDKit not available. Cannot canonicalize SMILES.")
    HAS_RDKIT = False


# ── JSON extraction (handles trailing content after JSON object) ──

def extract_json(text):
    depth = 0
    start = None
    for i, c in enumerate(text):
        if c in '{[':
            if depth == 0:
                start = i
            depth += 1
        elif c in '}]':
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i+1])
    return json.loads(text)


# ── SMILES detection (same logic as check_gc_row_alignment.py) ──

def _looks_like_smiles(s):
    if not s or len(s) < 4:
        return False
    # Exclude pure numbers with optional footnotes like "34[b]", "70 (82)[c]"
    if re.match(r'^[\d\s.,\-\u2013()\[\]a-d%]+$', s):
        return False
    # Exclude short abbreviation-like strings (MeLi, LDA, etc.)
    if re.match(r'^[A-Z][a-z]*[A-Z]?[a-z]*\d?$', s) and len(s) < 6:
        return False
    # Must contain typical SMILES bond/branch characters
    smiles_chars = set("CNOSFPIBrcnos=#()[]@+\\/-12345678.%")
    ratio = sum(1 for c in s if c in smiles_chars) / len(s)
    return ratio > 0.7 and any(c in s for c in "=#()[]") and any(c in s for c in "CNOSFPIBrcnos")


def canonical(smi):
    """Return canonical SMILES string, or None if invalid."""
    if not HAS_RDKIT:
        return smi
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


# ── Extract SMILES from VLM JSON (Gemini / Claude) ──

def extract_smiles_from_vlm_json(data):
    """
    Returns list of (row_idx, col_idx, col_name, raw_smiles) tuples.
    Uses _looks_like_smiles to detect SMILES cells.
    """
    columns = data.get("columns", [])
    rows = data.get("rows", data.get("data", []))
    results = []
    for row_idx, row in enumerate(rows):
        if isinstance(row, list):
            for col_idx, val in enumerate(row):
                val_str = str(val).strip() if val is not None else ""
                if _looks_like_smiles(val_str):
                    col_name = columns[col_idx] if col_idx < len(columns) else f"col_{col_idx}"
                    results.append((row_idx, col_idx, col_name, val_str))
    return results


# ── Extract Pipeline SMILES from CSV ──

def load_pipeline_smiles_from_csv(csv_path):
    """
    Returns dict: table_id -> list of (data_row, col_idx, col_name, raw_smiles)
    Only includes rows where cell_type == "smiles".
    """
    result = defaultdict(list)
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        # Handle Windows line endings
        content = f.read().replace('\r\n', '\n').replace('\r', '\n')
        reader = csv.DictReader(content.splitlines())
        for row in reader:
            if row.get('cell_type') == 'smiles':
                table_id = row['label']
                data_row = int(row['data_row'])
                col_idx = int(row['col_idx'])
                col_name = row.get('col_name', '')
                val = row.get('predicted_value', '')
                result[table_id].append((data_row, col_idx, col_name, val))
    return result


# ── Filter Pipeline SMILES: only keep actual molecular SMILES ──

def is_actual_smiles(raw_smi):
    """
    Determine if a pipeline 'smiles'-typed cell actually contains a SMILES string
    (as opposed to numbers with footnotes that were misclassified).
    We use _looks_like_smiles as primary filter, but also try RDKit parse.
    """
    if _looks_like_smiles(raw_smi):
        return True
    # Also accept if RDKit can parse it
    if HAS_RDKIT:
        mol = Chem.MolFromSmiles(raw_smi)
        if mol is not None and mol.GetNumAtoms() >= 3:
            return True
    return False


# ── Main analysis ──

def analyze_table(table_id, pipeline_entries, gemini_path, claude_path):
    """Analyze Pipeline vs G∩C consensus for one table."""

    # Load VLM data
    with open(gemini_path, 'r') as f:
        g_data = extract_json(f.read())
    with open(claude_path, 'r') as f:
        c_data = extract_json(f.read())

    g_entries = extract_smiles_from_vlm_json(g_data)
    c_entries = extract_smiles_from_vlm_json(c_data)

    # ── Gemini SMILES ──
    g_raw = [e[3] for e in g_entries]
    g_canon_list = [(s, canonical(s)) for s in g_raw]
    g_valid_set = set(c for _, c in g_canon_list if c is not None)

    # ── Claude SMILES ──
    c_raw = [e[3] for e in c_entries]
    c_canon_list = [(s, canonical(s)) for s in c_raw]
    c_valid_set = set(c for _, c in c_canon_list if c is not None)

    # ── G∩C consensus ──
    gc_consensus = g_valid_set & c_valid_set

    # ── Pipeline SMILES ──
    # Filter to actual SMILES (not misclassified numbers/text)
    p_all_entries = pipeline_entries  # (data_row, col_idx, col_name, raw)
    p_smiles_entries = []
    p_non_smiles_entries = []
    for entry in p_all_entries:
        raw = entry[3]
        if is_actual_smiles(raw):
            p_smiles_entries.append(entry)
        else:
            p_non_smiles_entries.append(entry)

    p_raw = [e[3] for e in p_smiles_entries]
    p_canon_list = [(raw, canonical(raw)) for raw in p_raw]
    p_valid = [(raw, c) for raw, c in p_canon_list if c is not None]
    p_invalid = [(raw, c) for raw, c in p_canon_list if c is None]
    p_valid_set = set(c for _, c in p_valid)

    # ── Classification ──
    confirmed = p_valid_set & gc_consensus
    match_g_only = (p_valid_set & g_valid_set) - gc_consensus
    match_c_only = (p_valid_set & c_valid_set) - gc_consensus
    partially_confirmed = match_g_only | match_c_only
    unconfirmed = p_valid_set - g_valid_set - c_valid_set
    pipeline_missed = gc_consensus - p_valid_set

    # ── Detailed mismatch info ──
    # For each unconfirmed pipeline SMILES, find closest in G∩C
    unconfirmed_details = []
    for p_canon in sorted(unconfirmed):
        # Find the raw string
        raw_for_canon = [raw for raw, c in p_valid if c == p_canon]
        unconfirmed_details.append({
            "pipeline_raw": raw_for_canon,
            "pipeline_canon": p_canon,
        })

    # For each pipeline-missed G∩C consensus molecule
    missed_details = []
    for gc_canon in sorted(pipeline_missed):
        # Find raw strings in G and C
        g_raws = [raw for raw, c in g_canon_list if c == gc_canon]
        c_raws = [raw for raw, c in c_canon_list if c == gc_canon]
        missed_details.append({
            "gc_canon": gc_canon,
            "gemini_raw": g_raws,
            "claude_raw": c_raws,
        })

    return {
        "table_id": table_id,
        # Pipeline stats
        "p_total_smiles_typed": len(p_all_entries),
        "p_actual_smiles": len(p_smiles_entries),
        "p_non_smiles_filtered": len(p_non_smiles_entries),
        "p_valid_rdkit": len(p_valid),
        "p_invalid_rdkit": len(p_invalid),
        "p_unique_molecules": len(p_valid_set),
        # VLM stats
        "g_smiles_count": len(g_raw),
        "g_valid": len(g_valid_set),
        "c_smiles_count": len(c_raw),
        "c_valid": len(c_valid_set),
        "gc_consensus_count": len(gc_consensus),
        # Classification
        "confirmed_correct": len(confirmed),
        "confirmed_set": sorted(confirmed),
        "match_g_only": len(match_g_only),
        "match_g_only_set": sorted(match_g_only),
        "match_c_only": len(match_c_only),
        "match_c_only_set": sorted(match_c_only),
        "partially_confirmed": len(partially_confirmed),
        "unconfirmed": len(unconfirmed),
        "unconfirmed_details": unconfirmed_details,
        "pipeline_missed": len(pipeline_missed),
        "missed_details": missed_details,
        # Raw lists for debugging
        "p_valid_list": p_valid,
        "p_invalid_list": p_invalid,
        "p_non_smiles_list": p_non_smiles_entries,
        "gc_consensus_set": sorted(gc_consensus),
    }


def print_report(result):
    tid = result["table_id"]
    print(f"\n{'='*90}")
    print(f"  {tid}")
    print(f"{'='*90}")

    # Pipeline stats
    print(f"\n  Pipeline SMILES-typed cells:  {result['p_total_smiles_typed']}")
    print(f"  Actual SMILES (after filter): {result['p_actual_smiles']}")
    if result['p_non_smiles_filtered'] > 0:
        print(f"  Filtered out (not SMILES):    {result['p_non_smiles_filtered']}")
        for entry in result['p_non_smiles_list']:
            print(f"    row={entry[0]} col={entry[1]} ({entry[2]}): \"{entry[3]}\"")
    print(f"  RDKit valid:                  {result['p_valid_rdkit']}")
    if result['p_invalid_rdkit'] > 0:
        print(f"  RDKit invalid:                {result['p_invalid_rdkit']}")
        for raw, _ in result['p_invalid_list']:
            print(f"    \"{raw}\"")
    print(f"  Unique molecules:             {result['p_unique_molecules']}")

    # VLM stats
    print(f"\n  Gemini SMILES: {result['g_smiles_count']} total, {result['g_valid']} unique valid")
    print(f"  Claude SMILES: {result['c_smiles_count']} total, {result['c_valid']} unique valid")
    print(f"  G∩C consensus: {result['gc_consensus_count']} molecules")

    # Classification
    print(f"\n  --- Pipeline vs G∩C Consensus ---")
    p_unique = result['p_unique_molecules']
    confirmed = result['confirmed_correct']
    print(f"  Confirmed correct (in G∩C):     {confirmed}/{p_unique}"
          f"  ({confirmed/p_unique:.0%})" if p_unique > 0 else f"  Confirmed correct: {confirmed}")
    if result['confirmed_set']:
        for s in result['confirmed_set']:
            print(f"    OK  {s}")

    if result['match_g_only'] > 0:
        print(f"  Match Gemini only (not Claude): {result['match_g_only']}")
        for s in result['match_g_only_set']:
            print(f"    ~G  {s}")

    if result['match_c_only'] > 0:
        print(f"  Match Claude only (not Gemini): {result['match_c_only']}")
        for s in result['match_c_only_set']:
            print(f"    ~C  {s}")

    if result['unconfirmed'] > 0:
        print(f"  Unconfirmed (match neither):    {result['unconfirmed']}")
        for d in result['unconfirmed_details']:
            print(f"    ??  {d['pipeline_canon']}  (raw: {d['pipeline_raw']})")

    if result['pipeline_missed'] > 0:
        print(f"  Pipeline missed from G∩C:       {result['pipeline_missed']}")
        for d in result['missed_details']:
            print(f"    MISS {d['gc_canon']}")
            print(f"         G raw: {d['gemini_raw']}")
            print(f"         C raw: {d['claude_raw']}")

    # Positional comparison (row-aligned) where possible
    # This is only meaningful for T1 and T3 where row structure is clear
    print(f"\n  --- G∩C Consensus Set ---")
    if result['gc_consensus_set']:
        for i, s in enumerate(result['gc_consensus_set']):
            in_p = "IN_PIPELINE" if s in set(c for _, c in result['p_valid_list']) else "MISSED"
            print(f"    [{i+1}] {s}  -> {in_p}")
    else:
        print(f"    (no consensus - G and C disagree on all molecules)")


def main():
    eval_dir = Path(__file__).parent
    base_dir = eval_dir / "images" / "tables"
    csv_path = eval_dir / "layer1_table_annotation.csv"

    # Load pipeline data
    pipeline_data = load_pipeline_smiles_from_csv(csv_path)

    tables = ["T1", "T2", "T3", "T4", "T5"]
    all_results = []

    for tid in tables:
        gemini_path = base_dir / f"{tid}.json"
        claude_path = base_dir / f"{tid}_claude.json"

        if not gemini_path.exists() or not claude_path.exists():
            print(f"Skipping {tid}: missing VLM files")
            continue

        p_entries = pipeline_data.get(tid, [])
        result = analyze_table(tid, p_entries, gemini_path, claude_path)
        all_results.append(result)
        print_report(result)

    # ── Overall Summary ──
    print(f"\n{'='*90}")
    print(f"  OVERALL SUMMARY")
    print(f"{'='*90}")

    total_p_smiles_typed = sum(r['p_total_smiles_typed'] for r in all_results)
    total_p_actual = sum(r['p_actual_smiles'] for r in all_results)
    total_p_valid = sum(r['p_valid_rdkit'] for r in all_results)
    total_p_unique = sum(r['p_unique_molecules'] for r in all_results)
    total_gc = sum(r['gc_consensus_count'] for r in all_results)
    total_confirmed = sum(r['confirmed_correct'] for r in all_results)
    total_g_only = sum(r['match_g_only'] for r in all_results)
    total_c_only = sum(r['match_c_only'] for r in all_results)
    total_partial = sum(r['partially_confirmed'] for r in all_results)
    total_unconfirmed = sum(r['unconfirmed'] for r in all_results)
    total_missed = sum(r['pipeline_missed'] for r in all_results)

    print(f"\n  Pipeline:")
    print(f"    SMILES-typed cells in CSV:   {total_p_smiles_typed}")
    print(f"    Actual molecular SMILES:     {total_p_actual}")
    print(f"    RDKit valid:                 {total_p_valid}")
    print(f"    Unique molecules:            {total_p_unique}")

    print(f"\n  G∩C Consensus (ground truth):")
    print(f"    Total consensus molecules:   {total_gc}")

    print(f"\n  Pipeline Correctness:")
    pct = total_confirmed / total_p_unique * 100 if total_p_unique > 0 else 0
    print(f"    Confirmed correct (in G∩C):  {total_confirmed}/{total_p_unique} ({pct:.1f}%)")
    pct2 = total_partial / total_p_unique * 100 if total_p_unique > 0 else 0
    print(f"    Partially confirmed:         {total_partial}/{total_p_unique} ({pct2:.1f}%)")
    print(f"      - match Gemini only:       {total_g_only}")
    print(f"      - match Claude only:       {total_c_only}")
    pct3 = total_unconfirmed / total_p_unique * 100 if total_p_unique > 0 else 0
    print(f"    Unconfirmed (neither):       {total_unconfirmed}/{total_p_unique} ({pct3:.1f}%)")
    pct4 = total_missed / total_gc * 100 if total_gc > 0 else 0
    print(f"    Pipeline missed from G∩C:    {total_missed}/{total_gc} ({pct4:.1f}%)")

    print(f"\n  Per-table summary:")
    print(f"    {'Table':<6} {'P_unique':<10} {'G∩C':<6} {'Confirmed':<12} {'Partial':<10} {'Unconf':<10} {'Missed':<10}")
    print(f"    {'-----':<6} {'--------':<10} {'---':<6} {'---------':<12} {'-------':<10} {'------':<10} {'------':<10}")
    for r in all_results:
        pu = r['p_unique_molecules']
        gc = r['gc_consensus_count']
        conf = r['confirmed_correct']
        part = r['partially_confirmed']
        unc = r['unconfirmed']
        miss = r['pipeline_missed']
        conf_pct = f"{conf}/{pu} ({conf/pu:.0%})" if pu > 0 else f"{conf}/0"
        print(f"    {r['table_id']:<6} {pu:<10} {gc:<6} {conf_pct:<12} {part:<10} {unc:<10} {miss:<10}")

    print(f"\n  CONCLUSION:")
    print(f"    {total_confirmed}/{total_p_unique} Pipeline unique molecules confirmed correct by G∩C consensus ({pct:.1f}%)")
    print(f"    {total_partial} additionally match one VLM (partially confirmed)")
    print(f"    {total_unconfirmed} match neither Gemini nor Claude (potentially wrong)")
    print(f"    {total_missed}/{total_gc} G∩C consensus molecules were missed by Pipeline")


if __name__ == "__main__":
    main()
