#!/usr/bin/env python3
"""Build cleaned v6 dataset with rescued tR(s) column for heatmap use."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path("/Users/zhaowenyuan/Projects/FlowFigTabMiner")
COMPARE_DIR = ROOT / "data" / "final_output" / "dataset_comparison"

V5_XLSX = COMPARE_DIR / "ALL_organolithium_normalized_20260327_1022_cleaned_v5.xlsx"
V6_XLSX = COMPARE_DIR / "ALL_organolithium_normalized_20260327_1022_cleaned_v6.xlsx"
V6_MODELING_XLSX = COMPARE_DIR / "ALL_organolithium_normalized_20260327_1022_cleaned_v6_modeling.xlsx"
V6_SUMMARY_JSON = COMPARE_DIR / "my_dataset_cleaning_v6_summary.json"


TR_PATTERNS = [
    ("note_tR1_sec", re.compile(r"\btR1\s*=\s*([0-9]*\.?[0-9]+)\s*s", re.I), "R1"),
    ("note_tR_sec", re.compile(r"\btR\s*=\s*([0-9]*\.?[0-9]+)\s*s", re.I), "generic"),
    ("note_residence_R1_sec", re.compile(r"residence time in R1\s*=\s*([0-9]*\.?[0-9]+)\s*s", re.I), "R1"),
    ("note_residence_sec", re.compile(r"residence time\s*\(?([0-9]*\.?[0-9]+)\s*s\)?", re.I), "generic"),
]


def parse_note_tr(notes: str) -> tuple[float | None, str | None, str | None]:
    text = str(notes or "")
    for source, pattern, stage in TR_PATTERNS:
        m = pattern.search(text)
        if m:
            return float(m.group(1)), source, stage
    return None, None, None


def choose_tr(row: pd.Series) -> tuple[float | None, str | None, str | None, bool]:
    for col, stage in [
        ("metric_tR1_sec", "R1"),
        ("metric_tR1_s", "R1"),
        ("metric_tR_sec", "generic"),
        ("metric_tR_s", "generic"),
        ("metric_tR2_sec", "R2"),
        ("metric_tR2_s", "R2"),
    ]:
        if col in row.index and pd.notna(row[col]):
            return float(row[col]), col, stage, True

    note_val, note_source, note_stage = parse_note_tr(row.get("notes"))
    if note_val is not None:
        return note_val, note_source, note_stage, True

    for col in ["cleaned_residence_time_s", "residence_time_s"]:
        if col in row.index and pd.notna(row[col]):
            return float(row[col]), f"{col}_fallback", "generic", False

    return None, None, None, False


def main() -> None:
    df = pd.read_excel(V5_XLSX)

    values = []
    sources = []
    stages = []
    explicits = []
    for _, row in df.iterrows():
        value, source, stage, explicit = choose_tr(row)
        values.append(value)
        sources.append(source)
        stages.append(stage)
        explicits.append(explicit)

    df["tr_s_v6"] = values
    df["tr_source_v6"] = sources
    df["tr_stage_v6"] = stages
    df["tr_is_explicit_v6"] = explicits

    with pd.ExcelWriter(V6_XLSX, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="All Records", index=False)

    modeling_df = df.loc[~df["exclude_candidate_v3"].fillna(False)].copy()
    with pd.ExcelWriter(V6_MODELING_XLSX, engine="openpyxl") as writer:
        modeling_df.to_excel(writer, sheet_name="All Records", index=False)

    summary = {
        "input_file": str(V5_XLSX),
        "output_file": str(V6_XLSX),
        "modeling_output_file": str(V6_MODELING_XLSX),
        "rows": int(len(df)),
        "rows_modeling": int(len(modeling_df)),
        "tr_nonnull_all": int(df["tr_s_v6"].notna().sum()),
        "tr_nonnull_modeling": int(modeling_df["tr_s_v6"].notna().sum()),
        "tr_explicit_all": int(df["tr_is_explicit_v6"].sum()),
        "tr_explicit_modeling": int(modeling_df["tr_is_explicit_v6"].sum()),
        "tr_source_counts_all": df["tr_source_v6"].fillna("missing").value_counts().to_dict(),
        "tr_source_counts_modeling": modeling_df["tr_source_v6"].fillna("missing").value_counts().to_dict(),
    }
    V6_SUMMARY_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved cleaned v6 workbook: {V6_XLSX}")
    print(f"Saved cleaned v6 modeling workbook: {V6_MODELING_XLSX}")
    print(f"Saved v6 summary: {V6_SUMMARY_JSON}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
