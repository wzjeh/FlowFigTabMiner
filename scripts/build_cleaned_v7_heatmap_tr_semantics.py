#!/usr/bin/env python3
"""Build cleaned v7 dataset with heatmap figure groups using tR(s) semantics."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path("/Users/zhaowenyuan/Projects/FlowFigTabMiner")
COMPARE_DIR = ROOT / "data" / "final_output" / "dataset_comparison"

V6_XLSX = COMPARE_DIR / "ALL_organolithium_normalized_20260327_1022_cleaned_v6.xlsx"
V7_XLSX = COMPARE_DIR / "ALL_organolithium_normalized_20260327_1022_cleaned_v7.xlsx"
V7_MODELING_XLSX = COMPARE_DIR / "ALL_organolithium_normalized_20260327_1022_cleaned_v7_modeling.xlsx"
V7_HEATMAP_CSV = COMPARE_DIR / "heatmap_figure_groups_v7.csv"
V7_SUMMARY_JSON = COMPARE_DIR / "my_dataset_cleaning_v7_summary.json"


def identify_heatmap_groups(df: pd.DataFrame) -> pd.DataFrame:
    figs = df[df["source_table_or_figure"].fillna("").str.contains("Figure", case=False, regex=False)].copy()
    summary = (
        figs.groupby(["paper_basename", "source_table_or_figure"], dropna=False)
        .agg(
            rows=("paper_basename", "size"),
            temp_nonnull=("temperature_C", lambda s: int(s.notna().sum())),
            tr_nonnull=("tr_s_v6", lambda s: int(s.notna().sum())),
            unique_temp=("temperature_C", lambda s: int(s.dropna().nunique())),
            unique_tr=("tr_s_v6", lambda s: int(s.dropna().nunique())),
            yield_nonnull=("yield_pct", lambda s: int(s.notna().sum())),
            conv_nonnull=("conversion_pct", lambda s: int(s.notna().sum())),
            sel_nonnull=("selectivity_pct", lambda s: int(s.notna().sum())),
        )
        .reset_index()
    )
    summary["is_heatmap_v7"] = (
        (summary["unique_temp"] > 1)
        & (summary["unique_tr"] > 1)
        & (summary["rows"] >= 4)
    )
    return summary.sort_values(
        ["is_heatmap_v7", "unique_temp", "unique_tr", "rows"],
        ascending=[False, False, False, False],
    )


def main() -> None:
    df = pd.read_excel(V6_XLSX)
    heatmap_groups = identify_heatmap_groups(df)
    heatmap_groups.to_csv(V7_HEATMAP_CSV, index=False)

    heatmap_key = set(
        zip(
            heatmap_groups.loc[heatmap_groups["is_heatmap_v7"], "paper_basename"],
            heatmap_groups.loc[heatmap_groups["is_heatmap_v7"], "source_table_or_figure"],
        )
    )

    pair_series = list(zip(df["paper_basename"], df["source_table_or_figure"]))
    df["is_heatmap_figure_v7"] = [pair in heatmap_key for pair in pair_series]
    df["time_axis_value_s_v7"] = pd.NA
    df["time_axis_label_v7"] = pd.NA
    df["time_axis_source_v7"] = pd.NA

    mask = df["is_heatmap_figure_v7"]
    df.loc[mask, "time_axis_value_s_v7"] = df.loc[mask, "tr_s_v6"]
    df.loc[mask, "time_axis_label_v7"] = "tR (s)"
    df.loc[mask, "time_axis_source_v7"] = df.loc[mask, "tr_source_v6"]

    with pd.ExcelWriter(V7_XLSX, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="All Records", index=False)

    modeling_df = df.loc[~df["exclude_candidate_v3"].fillna(False)].copy()
    with pd.ExcelWriter(V7_MODELING_XLSX, engine="openpyxl") as writer:
        modeling_df.to_excel(writer, sheet_name="All Records", index=False)

    summary = {
        "input_file": str(V6_XLSX),
        "output_file": str(V7_XLSX),
        "modeling_output_file": str(V7_MODELING_XLSX),
        "heatmap_group_csv": str(V7_HEATMAP_CSV),
        "heatmap_groups_total": int(heatmap_groups["is_heatmap_v7"].sum()),
        "heatmap_rows_total": int(df["is_heatmap_figure_v7"].sum()),
        "heatmap_rows_with_tr_total": int((df["is_heatmap_figure_v7"] & df["time_axis_value_s_v7"].notna()).sum()),
        "heatmap_groups_preview": heatmap_groups.loc[heatmap_groups["is_heatmap_v7"]].head(20).to_dict(orient="records"),
    }
    V7_SUMMARY_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved cleaned v7 workbook: {V7_XLSX}")
    print(f"Saved cleaned v7 modeling workbook: {V7_MODELING_XLSX}")
    print(f"Saved heatmap group CSV: {V7_HEATMAP_CSV}")
    print(f"Saved v7 summary: {V7_SUMMARY_JSON}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
