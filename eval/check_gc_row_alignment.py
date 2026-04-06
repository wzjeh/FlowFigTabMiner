#!/usr/bin/env python3
"""
check_gc_row_alignment.py
-------------------------
Compare Gemini vs Claude table SMILES outputs to determine whether low
positional (row-by-row) match rates are caused by row reordering or
genuine molecular differences.

For each table T1-T5:
  1. Extract all SMILES from both Gemini and Claude JSON outputs
  2. Canonicalize with RDKit
  3. Positional comparison (same row index)
  4. Set-based comparison (molecule-level, ignoring position)
"""

import json, os, re, sys
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


# ── SMILES detection (same logic as compute_vlm_comparison.py) ──

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
        return smi  # fallback: raw string comparison
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


# ── Extract ordered SMILES list from a table JSON ──

def extract_smiles_ordered(data):
    """
    Returns list of (row_idx, col_idx, col_name, raw_smiles) tuples,
    preserving original row order.
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


# ── Main analysis ──

def analyze_table(table_id, gemini_path, claude_path):
    """Analyze one table pair. Returns a dict of results."""
    # Load
    with open(gemini_path, 'r') as f:
        g_data = extract_json(f.read())
    with open(claude_path, 'r') as f:
        c_data = extract_json(f.read())

    g_entries = extract_smiles_ordered(g_data)
    c_entries = extract_smiles_ordered(c_data)

    g_raw = [e[3] for e in g_entries]
    c_raw = [e[3] for e in c_entries]

    # Canonicalize
    g_canon = [canonical(s) for s in g_raw]
    c_canon = [canonical(s) for s in c_raw]

    # ── Positional (row-by-row) comparison ──
    min_len = min(len(g_canon), len(c_canon))
    positional_matches = 0
    positional_mismatches = []
    for i in range(min_len):
        gc = g_canon[i]
        cc = c_canon[i]
        if gc is not None and cc is not None and gc == cc:
            positional_matches += 1
        else:
            positional_mismatches.append({
                "idx": i,
                "gemini_raw": g_raw[i],
                "gemini_canon": gc,
                "claude_raw": c_raw[i],
                "claude_canon": cc,
                "gemini_pos": f"row{g_entries[i][0]}/col{g_entries[i][1]}({g_entries[i][2]})",
                "claude_pos": f"row{c_entries[i][0]}/col{c_entries[i][1]}({c_entries[i][2]})",
            })

    positional_rate = positional_matches / min_len if min_len > 0 else 0.0

    # ── Set-based (molecule-level) comparison ──
    g_valid_set = set(s for s in g_canon if s is not None)
    c_valid_set = set(s for s in c_canon if s is not None)

    intersection = g_valid_set & c_valid_set
    union = g_valid_set | c_valid_set

    set_intersection_count = len(intersection)
    set_union_count = len(union)
    jaccard = set_intersection_count / set_union_count if set_union_count > 0 else 0.0
    pct_of_gemini = set_intersection_count / len(g_valid_set) if g_valid_set else 0.0
    pct_of_claude = set_intersection_count / len(c_valid_set) if c_valid_set else 0.0

    only_gemini = g_valid_set - c_valid_set
    only_claude = c_valid_set - g_valid_set

    # Count invalid
    g_invalid = sum(1 for s in g_canon if s is None)
    c_invalid = sum(1 for s in c_canon if s is None)

    return {
        "table_id": table_id,
        "gemini_smiles_count": len(g_raw),
        "claude_smiles_count": len(c_raw),
        "gemini_valid": len(g_raw) - g_invalid,
        "claude_valid": len(c_raw) - c_invalid,
        "gemini_invalid": g_invalid,
        "claude_invalid": c_invalid,
        "positional_compared": min_len,
        "positional_matches": positional_matches,
        "positional_rate": positional_rate,
        "positional_mismatches": positional_mismatches,
        "set_gemini_unique_molecules": len(g_valid_set),
        "set_claude_unique_molecules": len(c_valid_set),
        "set_intersection": set_intersection_count,
        "set_union": set_union_count,
        "set_jaccard": jaccard,
        "set_pct_of_gemini": pct_of_gemini,
        "set_pct_of_claude": pct_of_claude,
        "only_in_gemini": sorted(only_gemini),
        "only_in_claude": sorted(only_claude),
    }


def print_report(result):
    tid = result["table_id"]
    print(f"\n{'='*80}")
    print(f"  {tid}")
    print(f"{'='*80}")

    print(f"\n  SMILES extracted:  Gemini={result['gemini_smiles_count']}  Claude={result['claude_smiles_count']}")
    print(f"  RDKit valid:       Gemini={result['gemini_valid']}  Claude={result['claude_valid']}")
    print(f"  RDKit invalid:     Gemini={result['gemini_invalid']}  Claude={result['claude_invalid']}")

    # Positional
    print(f"\n  --- Positional (row-by-row) comparison ---")
    print(f"  Compared (min length): {result['positional_compared']}")
    print(f"  Canonical matches:     {result['positional_matches']}")
    print(f"  Match rate:            {result['positional_rate']:.1%}")
    if result['positional_mismatches']:
        print(f"\n  Positional mismatches ({len(result['positional_mismatches'])}):")
        for m in result['positional_mismatches']:
            print(f"    [{m['idx']}] G: {m['gemini_raw'][:60]}")
            print(f"         -> canon: {m['gemini_canon']}")
            print(f"         C: {m['claude_raw'][:60]}")
            print(f"         -> canon: {m['claude_canon']}")

    # Set-based
    print(f"\n  --- Set-based (molecule-level) comparison ---")
    print(f"  Unique molecules:  Gemini={result['set_gemini_unique_molecules']}  Claude={result['set_claude_unique_molecules']}")
    print(f"  Intersection:      {result['set_intersection']}  (Jaccard={result['set_jaccard']:.1%})")
    print(f"  % of Gemini set:   {result['set_pct_of_gemini']:.1%}")
    print(f"  % of Claude set:   {result['set_pct_of_claude']:.1%}")

    if result['only_in_gemini']:
        print(f"\n  Only in Gemini ({len(result['only_in_gemini'])}):")
        for s in result['only_in_gemini']:
            print(f"    {s}")
    if result['only_in_claude']:
        print(f"\n  Only in Claude ({len(result['only_in_claude'])}):")
        for s in result['only_in_claude']:
            print(f"    {s}")

    # Verdict
    print(f"\n  >>> VERDICT:")
    pos_rate = result['positional_rate']
    set_rate = result['set_jaccard']
    pct_g = result['set_pct_of_gemini']
    pct_c = result['set_pct_of_claude']

    if result['gemini_smiles_count'] == 0 and result['claude_smiles_count'] == 0:
        print(f"      No SMILES in either output. N/A.")
    elif result['gemini_smiles_count'] == 0 or result['claude_smiles_count'] == 0:
        print(f"      One side has no SMILES. Cannot compare.")
    elif pos_rate >= 0.8:
        print(f"      HIGH positional match ({pos_rate:.1%}). Rows are well-aligned; minimal reordering.")
    elif set_rate >= 0.8 or (pct_g >= 0.8 and pct_c >= 0.8):
        gap = set_rate - pos_rate
        print(f"      REORDERING detected: positional={pos_rate:.1%} but set={set_rate:.1%} (gap={gap:.1%}).")
        print(f"      The low positional match is largely due to row reordering, NOT molecular differences.")
    else:
        print(f"      GENUINE molecular differences: positional={pos_rate:.1%}, set Jaccard={set_rate:.1%}.")
        print(f"      Both alignment and molecule identity differ between Gemini and Claude.")


def main():
    base_dir = Path(__file__).parent / "images" / "tables"

    tables = ["T1", "T2", "T3", "T4", "T5"]
    all_results = []

    for tid in tables:
        gemini_path = base_dir / f"{tid}.json"
        claude_path = base_dir / f"{tid}_claude.json"

        if not gemini_path.exists() or not claude_path.exists():
            print(f"Skipping {tid}: missing files")
            continue

        result = analyze_table(tid, gemini_path, claude_path)
        all_results.append(result)
        print_report(result)

    # ── Overall Summary ──
    print(f"\n{'='*80}")
    print(f"  OVERALL SUMMARY")
    print(f"{'='*80}")

    total_pos_compared = sum(r['positional_compared'] for r in all_results)
    total_pos_match = sum(r['positional_matches'] for r in all_results)
    overall_pos_rate = total_pos_match / total_pos_compared if total_pos_compared > 0 else 0.0

    # For set-based overall, aggregate across tables
    total_set_inter = sum(r['set_intersection'] for r in all_results)
    total_set_union = sum(r['set_union'] for r in all_results)
    overall_jaccard = total_set_inter / total_set_union if total_set_union > 0 else 0.0

    print(f"\n  Across all tables:")
    print(f"    Total SMILES compared positionally: {total_pos_compared}")
    print(f"    Total positional matches:           {total_pos_match}")
    print(f"    Overall positional match rate:       {overall_pos_rate:.1%}")
    print(f"    Total set intersections:             {total_set_inter}")
    print(f"    Total set unions:                    {total_set_union}")
    print(f"    Overall set Jaccard:                 {overall_jaccard:.1%}")

    print(f"\n  Per-table summary:")
    print(f"    {'Table':<6} {'G#':<5} {'C#':<5} {'Pos%':<8} {'Set Jacc%':<10} {'Verdict'}")
    print(f"    {'-----':<6} {'---':<5} {'---':<5} {'------':<8} {'---------':<10} {'-------'}")
    for r in all_results:
        pos_pct = f"{r['positional_rate']:.0%}"
        set_pct = f"{r['set_jaccard']:.0%}"
        if r['gemini_smiles_count'] == 0 or r['claude_smiles_count'] == 0:
            verdict = "No SMILES"
        elif r['positional_rate'] >= 0.8:
            verdict = "Aligned"
        elif r['set_jaccard'] >= 0.8 or (r['set_pct_of_gemini'] >= 0.8 and r['set_pct_of_claude'] >= 0.8):
            verdict = "Reordering"
        else:
            verdict = "Mol. diff"
        print(f"    {r['table_id']:<6} {r['gemini_smiles_count']:<5} {r['claude_smiles_count']:<5} {pos_pct:<8} {set_pct:<10} {verdict}")

    gap = overall_jaccard - overall_pos_rate
    print(f"\n  CONCLUSION:")
    print(f"    Overall positional match rate = {overall_pos_rate:.1%}")
    print(f"    Overall set-based Jaccard     = {overall_jaccard:.1%}")
    print(f"    Gap (set - positional)        = {gap:.1%}")
    if gap > 0.15:
        print(f"    => Row reordering is a MAJOR factor in the observed SMILES mismatch.")
    elif gap > 0.05:
        print(f"    => Row reordering is a MODERATE factor; some genuine molecular differences also exist.")
    else:
        print(f"    => Row reordering is NOT the main cause; differences are primarily molecular.")


if __name__ == "__main__":
    main()
