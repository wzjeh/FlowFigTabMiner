"""
Parse VLM output JSON files for the tR sub-dataset.

Reads VLM responses from:
  data/vlm_input_tr/vlm_output/heatmaps/*.json
  data/vlm_input_tr/vlm_output/tables/*.json

Outputs:
  data/vlm_input_tr/parsed/heatmap_points.csv   ← all heatmap data points
  data/vlm_input_tr/parsed/tables/*.csv          ← per-table extracted data
  data/vlm_input_tr/parsed/tr_dataset_vlm.csv    ← unified tR dataset

Usage:
  cd /Users/zhaowenyuan/Projects/FlowFigTabMiner
  python scripts/parse_vlm_tr_output.py
"""

import os
import sys
import csv
import json
import re
import glob

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.path.join(PROJECT_ROOT, "data/vlm_input_tr")

# ──────────────────────────────────────────────────────────────
# Heatmap parsing
# ──────────────────────────────────────────────────────────────

def parse_heatmap_json(json_path, image_name, manifest_info):
    """
    Parse a single VLM heatmap JSON → list of flat data point dicts.

    VLM output format (from heatmap_prompt.txt):
    {
      "x_axis": {"title": "tR1/s", "scale": "log", "tick_values": [...]},
      "y_axis": {"title": "T/°C", "scale": "linear", "tick_values": [...]},
      "data_points": [{"tR_s": 0.01, "T_C": -78, "yield_pct": 24}, ...],
      "caption": "...",
      "context": {
        "sub_figure_label": "a",
        "reaction_description": "...",
        "compound_ids": ["2"],
        "electrophile_or_substrate": "...",
        "molecule_in_figure": "..."
      }
    }
    """
    with open(json_path, "r") as f:
        raw = f.read()

    # Strip markdown code fences if present
    raw = re.sub(r"^```(?:json)?\s*\n?", "", raw.strip())
    raw = re.sub(r"\n?```\s*$", "", raw.strip())

    data = json.loads(raw)

    x_axis = data.get("x_axis", {})
    y_axis = data.get("y_axis", {})
    caption = data.get("caption", "")
    context = data.get("context", {})
    points = data.get("data_points", [])

    rows = []
    for pt in points:
        rows.append({
            "source_type": "heatmap",
            "image_file": image_name,
            "paper": manifest_info.get("paper", ""),
            "v10_index": manifest_info.get("v10_index", ""),
            "sub_figure_label": context.get("sub_figure_label", ""),
            "tR_s": pt.get("tR_s", ""),
            "T_C": pt.get("T_C", ""),
            "yield_pct": pt.get("yield_pct", ""),
            "x_axis_title": x_axis.get("title", ""),
            "x_axis_scale": x_axis.get("scale", ""),
            "y_axis_title": y_axis.get("title", ""),
            "y_axis_scale": y_axis.get("scale", ""),
            "reaction_description": context.get("reaction_description", ""),
            "compound_ids": ";".join(context.get("compound_ids", [])) if isinstance(context.get("compound_ids"), list) else str(context.get("compound_ids", "")),
            "electrophile_or_substrate": context.get("electrophile_or_substrate", ""),
            "molecule_in_figure": context.get("molecule_in_figure", ""),
            "caption": caption,
        })

    return rows


def parse_all_heatmaps(manifest_lookup):
    """Parse all heatmap VLM outputs."""
    hm_output_dir = os.path.join(BASE_DIR, "vlm_output", "heatmaps")
    if not os.path.isdir(hm_output_dir):
        print(f"[Heatmap] VLM output dir not found: {hm_output_dir}")
        print(f"  → Place VLM JSON responses in this directory")
        return []

    json_files = sorted(glob.glob(os.path.join(hm_output_dir, "*.json")))
    if not json_files:
        print(f"[Heatmap] No JSON files in {hm_output_dir}")
        return []

    all_rows = []
    errors = []

    for jf in json_files:
        name = os.path.splitext(os.path.basename(jf))[0]
        image_key = f"heatmaps/{name}.png"
        manifest_info = manifest_lookup.get(image_key, {})

        try:
            rows = parse_heatmap_json(jf, image_key, manifest_info)
            all_rows.extend(rows)
            print(f"  {name}: {len(rows)} data points")
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            errors.append((name, str(e)))
            print(f"  {name}: ERROR - {e}")

    print(f"[Heatmap] Parsed {len(json_files)} files → {len(all_rows)} total points")
    if errors:
        print(f"  Errors ({len(errors)}):")
        for name, err in errors:
            print(f"    - {name}: {err}")

    return all_rows


# ──────────────────────────────────────────────────────────────
# Table parsing
# ──────────────────────────────────────────────────────────────

def _is_structure_cell(val):
    """Check if cell is a molecule structure placeholder or VLM-attempted SMILES."""
    s = str(val).strip()
    if not s:
        return False
    if s.lower() in ("[structure]", "[mol]", "[化学结构]", "[分子结构]"):
        return True
    if len(s) > 10:
        smiles_chars = set("CNOSPFClBr=#()[]/@+-.0123456789cnos")
        ratio = sum(1 for c in s if c in smiles_chars) / len(s)
        if ratio > 0.7 and any(c in s for c in "=#()[]"):
            return True
    return False


def _rdkit_valid(smi):
    """Check SMILES validity via RDKit (lazy import)."""
    if not smi or not isinstance(smi, str):
        return False
    try:
        from rdkit import Chem
        return Chem.MolFromSmiles(smi) is not None
    except Exception:
        return False


def merge_smiles_simple(columns, rows, mol_meta):
    """
    Lightweight SMILES merge for paper 80 tables.
    Replaces [structure] cells with MolNexTR SMILES from mol_meta.

    Uses spatial Y-clustering to match mol_meta entries to table rows,
    then X-ordering within each row.
    """
    if not mol_meta:
        return rows, {"replaced": 0, "kept_vlm": 0, "no_match": 0}

    n_rows = len(rows)

    # Find structure cells per row
    structure_rows = []  # [(row_idx, [col_indices])]
    for ri, row in enumerate(rows):
        cols_with_struct = [ci for ci, cell in enumerate(row) if _is_structure_cell(cell)]
        if cols_with_struct:
            structure_rows.append((ri, cols_with_struct))

    if not structure_rows:
        return rows, {"replaced": 0, "kept_vlm": 0, "no_match": 0}

    # Cluster mol_meta by Y coordinate
    sorted_mols = sorted(mol_meta, key=lambda m: (m["box"][1] + m["box"][3]) / 2)
    gap_threshold = 30  # pixels
    groups = []
    current = [sorted_mols[0]]
    for i in range(1, len(sorted_mols)):
        cy_prev = (current[-1]["box"][1] + current[-1]["box"][3]) / 2
        cy_curr = (sorted_mols[i]["box"][1] + sorted_mols[i]["box"][3]) / 2
        if cy_curr - cy_prev > gap_threshold:
            groups.append(current)
            current = [sorted_mols[i]]
        else:
            current.append(sorted_mols[i])
    groups.append(current)

    # Sort each group by X
    for g in groups:
        g.sort(key=lambda m: (m["box"][0] + m["box"][2]) / 2)

    # Match groups to structure_rows
    replaced = 0
    kept_vlm = 0
    no_match = 0

    for gi, (ri, col_indices) in enumerate(structure_rows):
        row_mols = groups[gi] if gi < len(groups) else []
        for ci_idx, ci in enumerate(col_indices):
            vlm_val = str(rows[ri][ci]).strip()
            vlm_valid = _rdkit_valid(vlm_val)
            if ci_idx < len(row_mols):
                mol_smi = row_mols[ci_idx].get("smiles", "")
                mol_valid = _rdkit_valid(mol_smi)
                if vlm_valid:
                    kept_vlm += 1
                elif mol_valid:
                    rows[ri][ci] = mol_smi
                    replaced += 1
                else:
                    kept_vlm += 1
            else:
                no_match += 1

    return rows, {"replaced": replaced, "kept_vlm": kept_vlm, "no_match": no_match}


def parse_table_json(json_path, image_name, manifest_info):
    """
    Parse a single VLM table JSON → (columns, rows) tuple.

    VLM output format:
    {
      "columns": ["col1", "col2", ...],
      "rows": [["cell1", "cell2", ...], ...]
    }
    """
    with open(json_path, "r") as f:
        raw = f.read()

    raw = re.sub(r"^```(?:json)?\s*\n?", "", raw.strip())
    raw = re.sub(r"\n?```\s*$", "", raw.strip())

    data = json.loads(raw)
    columns = data.get("columns", [])
    rows = data.get("rows", [])

    if not rows:
        return columns, []

    # Normalize row lengths
    n_cols = len(columns) if columns else max(len(r) for r in rows)
    for i, row in enumerate(rows):
        if len(row) < n_cols:
            rows[i] = row + [""] * (n_cols - len(row))
        elif len(row) > n_cols:
            rows[i] = row[:n_cols]

    # Merge mol_meta SMILES if available
    mol_meta_name = image_name.replace("tables/", "").replace(".png", "")
    mol_meta_path = os.path.join(BASE_DIR, "tables", f"{mol_meta_name}_mol_meta.json")
    if os.path.exists(mol_meta_path):
        with open(mol_meta_path, "r") as f:
            mol_meta = json.load(f)
        if mol_meta:
            rows, stats = merge_smiles_simple(columns, rows, mol_meta)
            print(f"    SMILES merge: {stats}")

    return columns, rows


def parse_all_tables(manifest_lookup):
    """Parse all table VLM outputs."""
    tbl_output_dir = os.path.join(BASE_DIR, "vlm_output", "tables")
    if not os.path.isdir(tbl_output_dir):
        print(f"[Table] VLM output dir not found: {tbl_output_dir}")
        print(f"  → Place VLM JSON responses in this directory")
        return []

    json_files = sorted(glob.glob(os.path.join(tbl_output_dir, "*.json")))
    if not json_files:
        print(f"[Table] No JSON files in {tbl_output_dir}")
        return []

    parsed_dir = os.path.join(BASE_DIR, "parsed", "tables")
    os.makedirs(parsed_dir, exist_ok=True)

    all_table_info = []
    errors = []

    for jf in json_files:
        name = os.path.splitext(os.path.basename(jf))[0]
        image_key = f"tables/{name}.png"
        manifest_info = manifest_lookup.get(image_key, {})

        try:
            columns, rows = parse_table_json(jf, image_key, manifest_info)
            if not rows:
                print(f"  {name}: empty table, skipped")
                continue

            # Save per-table CSV
            csv_path = os.path.join(parsed_dir, f"{name}.csv")
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                if columns:
                    writer.writerow(columns)
                writer.writerows(rows)

            print(f"  {name}: {len(rows)} rows × {len(columns)} cols → {csv_path}")

            all_table_info.append({
                "name": name,
                "image_key": image_key,
                "paper": manifest_info.get("paper", ""),
                "columns": columns,
                "rows": rows,
            })

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            errors.append((name, str(e)))
            print(f"  {name}: ERROR - {e}")

    print(f"[Table] Parsed {len(json_files)} files → {len(all_table_info)} tables")
    if errors:
        print(f"  Errors ({len(errors)}):")
        for name, err in errors:
            print(f"    - {name}: {err}")

    return all_table_info


# ──────────────────────────────────────────────────────────────
# Table → tR data points extraction
# ──────────────────────────────────────────────────────────────

# Column name patterns for identifying tR columns
TR_COL_PATTERNS = [
    r"\bt\^?[Rr]\d*",         # tR, tR1, t^R1, t^R2 (word boundary avoids "Entry")
    r"[Rr]esidence\s*[Tt]ime",  # residence time, Residence Time
    r"\bt\d+\s*[/\(]",        # t1 (min), t2/min (require digit after t)
]

T_COL_PATTERNS = [
    r"[Tt]emp",
    r"\bT\d*\s*[/\(°]",     # T (°C), T2(°C)
    r"\bT\s*$",
]

YIELD_COL_PATTERNS = [
    r"[Yy]ield",
    r"产率",
]


def _match_col(col_name, patterns):
    """Check if column name matches any pattern."""
    for p in patterns:
        if re.search(p, col_name):
            return True
    return False


def _extract_numeric(val):
    """Extract first numeric value from a cell (handles footnotes like '70 (82)[c]')."""
    s = str(val).strip()
    if not s or s == "–" or s == "-" or s == "—":
        return None
    # Match first number (int or float, possibly negative)
    m = re.search(r"(-?\d+\.?\d*)", s)
    return float(m.group(1)) if m else None


def extract_tr_from_table(table_info):
    """
    Extract tR-related data points from a parsed table.

    Returns list of dicts with: source_type, paper, tR_s, T_C, yield_pct, ...
    """
    columns = table_info["columns"]
    rows = table_info["rows"]
    paper = table_info["paper"]
    name = table_info["name"]

    if not columns or not rows:
        return []

    # Identify relevant columns
    tr_cols = [(i, c) for i, c in enumerate(columns) if _match_col(c, TR_COL_PATTERNS)]
    t_cols = [(i, c) for i, c in enumerate(columns) if _match_col(c, T_COL_PATTERNS)]
    yield_cols = [(i, c) for i, c in enumerate(columns) if _match_col(c, YIELD_COL_PATTERNS)]

    if not tr_cols:
        print(f"    {name}: no tR column found in {columns}")
        return []

    # Determine tR unit from column name
    def _get_tr_unit(col_name):
        col_lower = col_name.lower()
        if "min" in col_lower:
            return "min"
        return "s"  # default seconds

    points = []
    for row in rows:
        for tr_idx, tr_col_name in tr_cols:
            tr_val = _extract_numeric(row[tr_idx]) if tr_idx < len(row) else None
            if tr_val is None:
                continue

            # Convert to seconds if needed
            tr_unit = _get_tr_unit(tr_col_name)
            tr_s = tr_val * 60.0 if tr_unit == "min" else tr_val

            # Get temperature (use first T column if available)
            t_c = None
            if t_cols:
                ti, _ = t_cols[0]
                t_c = _extract_numeric(row[ti]) if ti < len(row) else None

            # Get yield (use first yield column)
            yield_pct = None
            if yield_cols:
                yi, _ = yield_cols[0]
                yield_pct = _extract_numeric(row[yi]) if yi < len(row) else None

            # Collect all other columns as context
            other_data = {}
            for ci, col in enumerate(columns):
                if ci < len(row) and ci not in [x[0] for x in tr_cols + t_cols + yield_cols]:
                    other_data[col] = row[ci]

            points.append({
                "source_type": "table",
                "image_file": table_info["image_key"],
                "paper": paper,
                "v10_index": "",
                "sub_figure_label": "",
                "tR_s": tr_s,
                "T_C": t_c if t_c is not None else "",
                "yield_pct": yield_pct if yield_pct is not None else "",
                "x_axis_title": tr_col_name,
                "x_axis_scale": "",
                "y_axis_title": t_cols[0][1] if t_cols else "",
                "y_axis_scale": "",
                "reaction_description": "",
                "compound_ids": "",
                "electrophile_or_substrate": "",
                "molecule_in_figure": "",
                "caption": "",
                "table_context": json.dumps(other_data, ensure_ascii=False) if other_data else "",
            })

    return points


# ──────────────────────────────────────────────────────────────
# Manifest loader
# ──────────────────────────────────────────────────────────────

def load_manifest():
    """Load manifest.csv into a lookup dict keyed by file path."""
    manifest_path = os.path.join(BASE_DIR, "manifest.csv")
    lookup = {}
    if not os.path.exists(manifest_path):
        print(f"[WARN] manifest.csv not found: {manifest_path}")
        return lookup

    with open(manifest_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lookup[row["file"]] = row

    return lookup


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

UNIFIED_COLUMNS = [
    "source_type", "image_file", "paper", "v10_index",
    "sub_figure_label", "tR_s", "T_C", "yield_pct",
    "x_axis_title", "x_axis_scale", "y_axis_title", "y_axis_scale",
    "reaction_description", "compound_ids",
    "electrophile_or_substrate", "molecule_in_figure",
    "caption", "table_context",
]


def main():
    print(f"Base directory: {BASE_DIR}")

    parsed_dir = os.path.join(BASE_DIR, "parsed")
    os.makedirs(parsed_dir, exist_ok=True)

    manifest = load_manifest()
    print(f"Manifest: {len(manifest)} entries\n")

    # ── Heatmaps ──
    print("=" * 60)
    print("Parsing heatmap VLM outputs...")
    print("=" * 60)
    hm_rows = parse_all_heatmaps(manifest)

    if hm_rows:
        hm_csv = os.path.join(parsed_dir, "heatmap_points.csv")
        with open(hm_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=UNIFIED_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(hm_rows)
        print(f"\n→ Saved {len(hm_rows)} heatmap points to {hm_csv}")

    # ── Tables ──
    print("\n" + "=" * 60)
    print("Parsing table VLM outputs...")
    print("=" * 60)
    table_infos = parse_all_tables(manifest)

    # Extract tR data points from tables
    tbl_rows = []
    for ti in table_infos:
        pts = extract_tr_from_table(ti)
        if pts:
            tbl_rows.extend(pts)
            print(f"    → {ti['name']}: {len(pts)} tR points extracted")
        else:
            print(f"    → {ti['name']}: no tR points (may lack tR column)")

    # ── Unified dataset ──
    print("\n" + "=" * 60)
    print("Unified tR dataset")
    print("=" * 60)

    all_rows = []
    for r in hm_rows:
        row = {k: r.get(k, "") for k in UNIFIED_COLUMNS}
        all_rows.append(row)
    for r in tbl_rows:
        row = {k: r.get(k, "") for k in UNIFIED_COLUMNS}
        all_rows.append(row)

    if all_rows:
        unified_csv = os.path.join(parsed_dir, "tr_dataset_vlm.csv")
        with open(unified_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=UNIFIED_COLUMNS)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"→ Saved {len(all_rows)} total points to {unified_csv}")
    else:
        print("→ No data points parsed (no VLM output files found?)")

    # Summary
    print(f"\nHeatmap points: {len(hm_rows)}")
    print(f"Table points:   {len(tbl_rows)}")
    print(f"Total:          {len(all_rows)}")

    # Check for missing VLM outputs
    hm_output_dir = os.path.join(BASE_DIR, "vlm_output", "heatmaps")
    tbl_output_dir = os.path.join(BASE_DIR, "vlm_output", "tables")

    hm_expected = [k for k, v in manifest.items() if v.get("type") == "heatmap"]
    tbl_expected = [k for k, v in manifest.items() if v.get("type") == "table"]

    hm_done = set()
    if os.path.isdir(hm_output_dir):
        hm_done = {f"heatmaps/{os.path.splitext(f)[0]}.png"
                    for f in os.listdir(hm_output_dir) if f.endswith(".json")}

    tbl_done = set()
    if os.path.isdir(tbl_output_dir):
        tbl_done = {f"tables/{os.path.splitext(f)[0]}.png"
                    for f in os.listdir(tbl_output_dir) if f.endswith(".json")}

    hm_missing = [k for k in hm_expected if k not in hm_done]
    tbl_missing = [k for k in tbl_expected if k not in tbl_done]

    if hm_missing or tbl_missing:
        print(f"\nMissing VLM outputs:")
        if hm_missing:
            print(f"  Heatmaps ({len(hm_missing)}/{len(hm_expected)}):")
            for m in hm_missing[:5]:
                print(f"    - {m}")
            if len(hm_missing) > 5:
                print(f"    ... and {len(hm_missing) - 5} more")
        if tbl_missing:
            print(f"  Tables ({len(tbl_missing)}/{len(tbl_expected)}):")
            for m in tbl_missing:
                print(f"    - {m}")


if __name__ == "__main__":
    main()
