"""
Build the final VLM-based tR sub-dataset.

Combines:
  1. VLM-extracted heatmap + table data (coordinates: tR, T, yield)
  2. v10 context metadata (reaction_class, reactants, products, solvent, etc.)

Context transfer: v10 review_id ↔ VLM v10_index (same figure)

Output:
  data/final_output/organolithium_tr_subdataset_vlm.csv

Usage:
  cd /Users/zhaowenyuan/Projects/FlowFigTabMiner
  python scripts/build_tr_dataset_vlm.py
"""

import os
import csv
import json
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VLM_PARSED = os.path.join(PROJECT_ROOT, "data/vlm_input_tr/parsed/tr_dataset_vlm.csv")
V10_HEATMAP = os.path.join(PROJECT_ROOT, "data/final_output/dataset_comparison/organolithium_tr_heatmap_points_v10.csv")
OUTPUT_CSV = os.path.join(PROJECT_ROOT, "data/final_output/organolithium_tr_subdataset_vlm.csv")

# Context fields to transfer from v10
V10_CONTEXT_FIELDS = [
    "reaction_class",
    "organolithium_role_v5",
    "reagent_family_v2",
    "solvent",
    "reactor_type",
    "reactant1_name",
    "reactant2_name",
    "product_name",
    "paper_doi",
    "paper_year",
]

# Output columns
OUTPUT_COLUMNS = [
    # Identity
    "point_id",
    "source_type",        # heatmap / table
    "image_file",
    "paper",
    "v10_index",
    # Coordinates (VLM-extracted)
    "tR_s",
    "T_C",
    "yield_pct",
    # Axis metadata
    "x_axis_title",
    "x_axis_scale",
    "y_axis_title",
    "y_axis_scale",
    # VLM context (from figure)
    "sub_figure_label",
    "reaction_description",
    "compound_ids",
    "electrophile_or_substrate",
    "caption",
    # v10 context (transferred by v10_index)
    "reaction_class",
    "organolithium_role",
    "reagent_family",
    "solvent",
    "reactor_type",
    "reactant1_name",
    "reactant2_name",
    "product_name",
    "paper_doi",
    "paper_year",
    # Table-specific
    "table_context",
    # Provenance
    "coordinate_source",   # "vlm" always
    "context_source",      # "v10_transfer" or "vlm_only"
]


def load_v10_context():
    """Load v10 context, grouped by review_id (= v10_index)."""
    ctx = {}
    if not os.path.exists(V10_HEATMAP):
        print(f"[WARN] v10 file not found: {V10_HEATMAP}")
        return ctx

    with open(V10_HEATMAP) as f:
        for row in csv.DictReader(f):
            rid = row.get("review_id_v10", "")
            # Skip zz_ test entries
            paper_dir = row.get("paper_dir", "")
            if "zz_" in paper_dir:
                continue
            if rid and rid not in ctx:
                ctx[rid] = {
                    "reaction_class": row.get("reaction_class", ""),
                    "organolithium_role_v5": row.get("organolithium_role_v5", ""),
                    "reagent_family_v2": row.get("reagent_family_v2", ""),
                    "solvent": row.get("solvent", ""),
                    "reactor_type": row.get("reactor_type", ""),
                    "reactant1_name": row.get("reactant1_name", ""),
                    "reactant2_name": row.get("reactant2_name", ""),
                    "product_name": row.get("product_name", ""),
                    "paper_doi": row.get("paper_doi", ""),
                    "paper_year": row.get("paper_year", ""),
                }
    print(f"[v10] Loaded context for {len(ctx)} figure groups (excluding zz_)")
    return ctx


def build_dataset():
    """Build the final VLM-based tR sub-dataset."""
    # Load VLM data
    with open(VLM_PARSED) as f:
        vlm_rows = list(csv.DictReader(f))
    print(f"[VLM] {len(vlm_rows)} points loaded")

    # Load v10 context
    v10_ctx = load_v10_context()

    # Build output rows
    output = []
    ctx_matched = 0
    ctx_missing = 0

    for i, r in enumerate(vlm_rows):
        vid = r.get("v10_index", "")
        v10 = v10_ctx.get(vid, {})

        has_v10 = bool(v10)
        if has_v10:
            ctx_matched += 1
        else:
            ctx_missing += 1

        row = {
            "point_id": i + 1,
            "source_type": r.get("source_type", ""),
            "image_file": r.get("image_file", ""),
            "paper": r.get("paper", ""),
            "v10_index": vid,
            "tR_s": r.get("tR_s", ""),
            "T_C": r.get("T_C", ""),
            "yield_pct": r.get("yield_pct", ""),
            "x_axis_title": r.get("x_axis_title", ""),
            "x_axis_scale": r.get("x_axis_scale", ""),
            "y_axis_title": r.get("y_axis_title", ""),
            "y_axis_scale": r.get("y_axis_scale", ""),
            "sub_figure_label": r.get("sub_figure_label", ""),
            "reaction_description": r.get("reaction_description", ""),
            "compound_ids": r.get("compound_ids", ""),
            "electrophile_or_substrate": r.get("electrophile_or_substrate", ""),
            "caption": r.get("caption", ""),
            # v10 context transfer
            "reaction_class": v10.get("reaction_class", ""),
            "organolithium_role": v10.get("organolithium_role_v5", ""),
            "reagent_family": v10.get("reagent_family_v2", ""),
            "solvent": v10.get("solvent", ""),
            "reactor_type": v10.get("reactor_type", ""),
            "reactant1_name": v10.get("reactant1_name", ""),
            "reactant2_name": v10.get("reactant2_name", ""),
            "product_name": v10.get("product_name", ""),
            "paper_doi": v10.get("paper_doi", ""),
            "paper_year": v10.get("paper_year", ""),
            # Table-specific
            "table_context": r.get("table_context", ""),
            # Provenance
            "coordinate_source": "vlm",
            "context_source": "v10_transfer" if has_v10 else "vlm_only",
        }
        output.append(row)

    # Write output
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(output)

    # Summary
    hm = [r for r in output if r["source_type"] == "heatmap"]
    tb = [r for r in output if r["source_type"] == "table"]

    print(f"\n{'=' * 60}")
    print(f"FINAL DATASET: {OUTPUT_CSV}")
    print(f"{'=' * 60}")
    print(f"Total points:     {len(output)}")
    print(f"  Heatmap:        {len(hm)}")
    print(f"  Table:          {len(tb)}")
    print(f"Context transfer: {ctx_matched} matched / {ctx_missing} vlm_only")

    # Context coverage
    print(f"\nContext field coverage:")
    for field in ["reaction_class", "organolithium_role", "reagent_family",
                   "solvent", "reactor_type", "reactant1_name", "reactant2_name", "product_name"]:
        filled = sum(1 for r in output if r.get(field) and r[field] not in ("", "missing_or_ambiguous", "Unclassified"))
        pct = filled / len(output) * 100
        print(f"  {field:25s}: {filled:>5}/{len(output)} ({pct:5.1f}%)")

    # Value ranges
    trs = [float(r["tR_s"]) for r in output if r["tR_s"]]
    ts = [float(r["T_C"]) for r in output if r["T_C"]]
    ys = [float(r["yield_pct"]) for r in output if r["yield_pct"]]
    print(f"\nValue ranges:")
    print(f"  tR_s:     {min(trs):.4f} ~ {max(trs):.1f} s")
    print(f"  T_C:      {min(ts):.0f} ~ {max(ts):.0f} °C")
    print(f"  yield:    {min(ys):.0f} ~ {max(ys):.0f} %")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    build_dataset()
