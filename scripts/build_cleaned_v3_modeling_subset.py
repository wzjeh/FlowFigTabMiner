#!/usr/bin/env python3
"""Build a modeling-oriented subset from cleaned v3 by excluding artifact candidates."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path("/Users/zhaowenyuan/Projects/FlowFigTabMiner")
COMPARE_DIR = ROOT / "data" / "final_output" / "dataset_comparison"
V3_XLSX = COMPARE_DIR / "ALL_organolithium_normalized_20260327_1022_cleaned_v3.xlsx"
V3_MODELING_XLSX = COMPARE_DIR / "ALL_organolithium_normalized_20260327_1022_cleaned_v3_modeling.xlsx"
V3_MODELING_JSON = COMPARE_DIR / "my_dataset_cleaning_v3_modeling_summary.json"


def main() -> None:
    df = pd.read_excel(V3_XLSX)
    if "exclude_candidate_v3" not in df.columns:
        raise SystemExit("cleaned_v3 is missing exclude_candidate_v3")

    modeling_df = df[~df["exclude_candidate_v3"].fillna(False)].copy()

    summary = {
        "input_file": str(V3_XLSX),
        "output_file": str(V3_MODELING_XLSX),
        "input_rows": int(len(df)),
        "excluded_artifact_rows": int(df["exclude_candidate_v3"].fillna(False).sum()),
        "output_rows": int(len(modeling_df)),
        "remaining_unclassified": int((modeling_df["reagent_family"] == "Unclassified").sum()),
    }

    with pd.ExcelWriter(V3_MODELING_XLSX, engine="openpyxl") as writer:
        modeling_df.to_excel(writer, sheet_name="All Records", index=False)
    V3_MODELING_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved modeling subset: {V3_MODELING_XLSX}")
    print(f"Saved summary: {V3_MODELING_JSON}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
