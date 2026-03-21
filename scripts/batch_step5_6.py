"""
Batch runner: re-run Step 5 (GlobalAssembly) + Step 6 (PostProcessor) for all organolithium PDFs,
then combine all _normalized.json into one Excel for submission.

Usage:
  python scripts/batch_step5_6.py [--force-assembly] [--smiles-lookup]

  --force-assembly : Force LLM re-run even if _final.json already exists
  --smiles-lookup  : Query PubChem to fill missing SMILES (slow)
  --combine-only   : Skip Step 5+6, only rebuild the combined Excel from existing _normalized.json
"""

import os
import sys
import json
import argparse
import traceback
from datetime import datetime

sys.path.insert(0, os.getcwd())

from src.adjudication.global_assembly import GlobalAssembly
from src.adjudication.post_processor import PostProcessor

PDF_DIR = "data/input/organolithium"
FINAL_DIR = "data/final_output"


def get_all_pdfs():
    pdfs = sorted([
        os.path.join(PDF_DIR, f)
        for f in os.listdir(PDF_DIR)
        if f.endswith(".pdf")
    ])
    return pdfs


def run_step5_6(pdf_path, assembler, post, force_assembly, smiles_lookup):
    basename = os.path.splitext(os.path.basename(pdf_path))[0]
    intermediate_dir = os.path.join("data/intermediate", basename)
    print(f"\n{'='*60}")
    print(f"Processing: {basename}")
    print(f"{'='*60}")
    try:
        assembler.run(pdf_path, intermediate_dir, force=force_assembly)
        norm = post.run(pdf_path, intermediate_dir, smiles_lookup=smiles_lookup)
        return norm
    except Exception as e:
        print(f"[ERROR] {basename}: {e}")
        traceback.print_exc()
        return None


def combine_all(output_path):
    """
    Combine all _normalized.json into one Excel.
    Sheet structure:
      - "All Records"     : every record from every paper, flat
      - "By Paper"        : pivot-style summary (one row per paper)
      Per-source sheets are omitted in the combined file to keep it manageable.
    """
    import pandas as pd

    all_rows = []
    norm_files = sorted([
        os.path.join(FINAL_DIR, f)
        for f in os.listdir(FINAL_DIR)
        if f.endswith("_normalized.json")
    ])

    print(f"\n[Combine] Found {len(norm_files)} _normalized.json files")

    for nf in norm_files:
        basename = os.path.basename(nf).replace("_normalized.json", "")
        try:
            with open(nf) as f:
                records = json.load(f)
            if not isinstance(records, list):
                continue
            for r in records:
                row = {}
                # Paper-level fields first
                row["paper_basename"] = basename
                row["paper_doi"] = r.get("paper_doi")
                row["paper_year"] = r.get("paper_year")
                row["reaction_class"] = r.get("reaction_class")

                # Compound fields
                row["reactant1_name"] = r.get("reactant1_name")
                row["reactant1_smiles"] = r.get("reactant1_smiles")
                row["reactant2_name"] = r.get("reactant2_name")
                row["reactant2_smiles"] = r.get("reactant2_smiles")
                row["product_name"] = r.get("product_name")
                row["product_smiles"] = r.get("product_smiles")
                row["product_label"] = r.get("product_label")
                row["reaction_smiles"] = r.get("reaction_smiles")

                # Result fields
                row["yield_pct"] = r.get("yield_pct")
                row["yield_type"] = r.get("yield_type")
                row["batch_yield_pct"] = r.get("batch_yield_pct")
                row["conversion_pct"] = r.get("conversion_pct")
                row["selectivity_pct"] = r.get("selectivity_pct")
                row["diastereomeric_ratio"] = r.get("diastereomeric_ratio")
                row["ee_pct"] = r.get("ee_pct")
                row["stoichiometry"] = r.get("stoichiometry")

                # Conditions
                conds = r.get("conditions") or {}
                row["temperature_C"] = conds.get("temperature_C")
                row["residence_time_s"] = conds.get("residence_time_s")
                row["flow_rate_mL_min"] = conds.get("flow_rate_mL_min")
                row["flow_rate_stream1_mL_min"] = conds.get("flow_rate_stream1_mL_min")
                row["flow_rate_stream2_mL_min"] = conds.get("flow_rate_stream2_mL_min")
                row["solvent"] = conds.get("solvent")
                row["solvent_list"] = ", ".join(conds["solvent_list"]) if conds.get("solvent_list") else None
                row["catalyst"] = conds.get("catalyst")
                row["catalyst_metal"] = conds.get("catalyst_metal")
                row["catalyst_loading_pct"] = conds.get("catalyst_loading_pct")
                row["ligand"] = conds.get("ligand")
                row["ligand_loading_pct"] = conds.get("ligand_loading_pct")
                row["additive"] = conds.get("additive")
                row["pressure_bar"] = conds.get("pressure_bar")
                row["reactor_type"] = conds.get("reactor_type")

                # Other metrics: flatten into columns prefixed "metric_"
                other = r.get("other_metrics") or {}
                for mk, mv in other.items():
                    row[f"metric_{mk}"] = mv

                # Source / notes
                row["source_table_or_figure"] = r.get("source_table_or_figure")
                row["notes"] = r.get("notes")

                all_rows.append(row)

        except Exception as e:
            print(f"[Combine] Failed to load {nf}: {e}")

    if not all_rows:
        print("[Combine] No records found!")
        return

    df = pd.DataFrame(all_rows)

    # --- By Paper summary sheet ---
    paper_summary = []
    for basename, grp in df.groupby("paper_basename", sort=True):
        summary = {
            "paper_basename": basename,
            "paper_doi": grp["paper_doi"].dropna().iloc[0] if grp["paper_doi"].notna().any() else None,
            "paper_year": grp["paper_year"].dropna().iloc[0] if grp["paper_year"].notna().any() else None,
            "reaction_class": grp["reaction_class"].dropna().iloc[0] if grp["reaction_class"].notna().any() else None,
            "total_records": len(grp),
            "records_with_yield": grp["yield_pct"].notna().sum(),
            "records_with_ee": grp["ee_pct"].notna().sum(),
            "records_with_conversion": grp["conversion_pct"].notna().sum(),
            "unique_solvents": ", ".join(sorted(grp["solvent"].dropna().unique())),
            "unique_catalysts": ", ".join(sorted(grp["catalyst"].dropna().unique())),
            "temp_range_C": f"{grp['temperature_C'].min()} ~ {grp['temperature_C'].max()}" if grp["temperature_C"].notna().any() else None,
        }
        paper_summary.append(summary)

    df_summary = pd.DataFrame(paper_summary)

    # Write Excel
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="All Records", index=False)
        df_summary.to_excel(writer, sheet_name="By Paper", index=False)

    print(f"\n[Combine] Total records: {len(df)}")
    print(f"[Combine] Papers covered: {df['paper_basename'].nunique()}")
    print(f"[Combine] Records with yield_pct: {df['yield_pct'].notna().sum()}")
    print(f"[Combine] Records with ee_pct: {df['ee_pct'].notna().sum()}")
    print(f"[Combine] Saved -> {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-assembly", action="store_true",
                        help="Force LLM re-run even if _final.json exists")
    parser.add_argument("--smiles-lookup", action="store_true",
                        help="PubChem SMILES lookup (slow)")
    parser.add_argument("--combine-only", action="store_true",
                        help="Skip Step 5+6, only rebuild combined Excel")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    combined_path = os.path.join(FINAL_DIR, f"ALL_organolithium_normalized_{timestamp}.xlsx")

    if not args.combine_only:
        pdfs = get_all_pdfs()
        print(f"Found {len(pdfs)} PDFs to process")

        assembler = GlobalAssembly()
        post = PostProcessor()
        failed = []

        for i, pdf_path in enumerate(pdfs, 1):
            print(f"\n[{i}/{len(pdfs)}] {os.path.basename(pdf_path)}")
            result = run_step5_6(pdf_path, assembler, post, args.force_assembly, args.smiles_lookup)
            if result is None:
                failed.append(os.path.basename(pdf_path))

        if failed:
            print(f"\n[WARN] Failed PDFs ({len(failed)}):")
            for f in failed:
                print(f"  - {f}")

    combine_all(combined_path)
    print(f"\nDone. Combined file: {combined_path}")


if __name__ == "__main__":
    main()
