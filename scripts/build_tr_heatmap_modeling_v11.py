#!/usr/bin/env python3
"""Build a modeling-oriented v11 subset from tR heatmap points."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pandas as pd


ROOT = Path("/Users/zhaowenyuan/Projects/FlowFigTabMiner")
COMPARE_DIR = ROOT / "data" / "final_output" / "dataset_comparison"

V10_POINTS = COMPARE_DIR / "organolithium_tr_heatmap_points_v10.csv"
OUT_CSV = COMPARE_DIR / "organolithium_tr_heatmap_points_v11_modeling.csv"
OUT_XLSX = COMPARE_DIR / "organolithium_tr_heatmap_points_v11_modeling.xlsx"
OUT_SUMMARY = COMPARE_DIR / "organolithium_tr_heatmap_points_v11_modeling_summary.json"


ARTIFACT_PATTERNS = [
    re.compile(r"^example\d*$", re.I),
    re.compile(r"^zz_codex", re.I),
    re.compile(r"^\d+$"),
]


def infer_tr_stage(x_title: str) -> str:
    text = str(x_title or "").lower()
    if "r1" in text:
        return "R1"
    if "r2" in text:
        return "R2"
    return "generic"


def is_artifact_like(paper_dir: str, paper_doi: object, match_score: object) -> bool:
    name = str(paper_dir or "")
    if any(p.search(name) for p in ARTIFACT_PATTERNS):
        return True
    if pd.isna(paper_doi) and (pd.isna(match_score) or float(match_score) < 0.5):
        return True
    return False


def choose_context_confidence(row: pd.Series) -> str:
    if row.get("context_match_level_v10") == "exact_figure":
        return "high"
    if row.get("context_match_level_v10") == "paper_only":
        return "medium"
    return "low"


def main() -> None:
    df = pd.read_csv(V10_POINTS)

    out = pd.DataFrame()
    out["point_id_v11"] = range(1, len(df) + 1)
    out["review_id_v10"] = df["review_id_v10"]
    out["figure_key_v11"] = (
        df["paper_dir"].astype(str)
        + " :: "
        + df["page_tag"].astype(str)
        + " :: "
        + df["evidence_file"].astype(str)
    )
    out["paper_dir"] = df["paper_dir"]
    out["paper_doi"] = df["paper_doi"]
    out["paper_year"] = df["paper_year"]
    out["page_tag"] = df["page_tag"]
    out["figure_num_1based"] = df["figure_num_1based"]
    out["series_label"] = df["series_label"]

    out["tr_s"] = pd.to_numeric(df["tr_s"], errors="coerce")
    out["log10_tr_s"] = out["tr_s"].map(lambda x: math.log10(x) if pd.notna(x) and x > 0 else pd.NA)
    out["tr_stage_v11"] = df["x_title_ocr_raw"].map(infer_tr_stage)

    out["temperature_C"] = pd.to_numeric(df["temperature_C_from_figure"], errors="coerce")
    out["yield_pct"] = pd.to_numeric(df["yield_pct_from_figure"], errors="coerce")

    out["reaction_class"] = df["reaction_class"]
    out["organolithium_role_v5"] = df["organolithium_role_v5"]
    out["reagent_family_v2"] = df["reagent_family_v2"]
    out["solvent"] = df["solvent"]
    out["reactor_type"] = df["reactor_type"]
    out["reactant1_name"] = df["reactant1_name"]
    out["reactant2_name"] = df["reactant2_name"]
    out["product_name"] = df["product_name"]

    out["context_match_level_v10"] = df["context_match_level_v10"]
    out["context_confidence_v11"] = df.apply(choose_context_confidence, axis=1)
    out["paper_match_score_v10"] = pd.to_numeric(df["paper_match_score_v10"], errors="coerce")

    out["artifact_candidate_v11"] = df.apply(
        lambda row: is_artifact_like(row.get("paper_dir"), row.get("paper_doi"), row.get("paper_match_score_v10")),
        axis=1,
    )
    out["missing_doi_v11"] = out["paper_doi"].isna()
    out["missing_role_v11"] = out["organolithium_role_v5"].isna()
    out["missing_reagent_family_v11"] = out["reagent_family_v2"].isna() | (out["reagent_family_v2"] == "Unclassified")
    out["missing_product_v11"] = out["product_name"].isna()
    out["paper_only_context_v11"] = out["context_match_level_v10"].eq("paper_only")

    out["yield_out_of_range_v11"] = ~(out["yield_pct"].between(0, 100, inclusive="both"))
    out["temperature_out_of_range_v11"] = ~(out["temperature_C"].between(-120, 150, inclusive="both"))
    out["tr_out_of_range_v11"] = ~((out["tr_s"] >= 1e-5) & (out["tr_s"] <= 86400))

    out["modeling_priority_v11"] = "B"
    high_mask = (
        ~out["artifact_candidate_v11"]
        & ~out["yield_out_of_range_v11"]
        & ~out["temperature_out_of_range_v11"]
        & ~out["tr_out_of_range_v11"]
        & ~out["missing_doi_v11"]
        & ~out["missing_role_v11"]
    )
    out.loc[high_mask, "modeling_priority_v11"] = "A"
    low_mask = out["artifact_candidate_v11"] | out["yield_out_of_range_v11"] | out["temperature_out_of_range_v11"] | out["tr_out_of_range_v11"]
    out.loc[low_mask, "modeling_priority_v11"] = "C"

    out["recommended_keep_for_modeling_v11"] = out["modeling_priority_v11"].isin(["A", "B"]) & ~out["artifact_candidate_v11"]

    out.to_csv(OUT_CSV, index=False)
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        out.to_excel(writer, sheet_name="tr_heatmap_modeling_v11", index=False)

    summary = {
        "input_points_csv": str(V10_POINTS),
        "output_csv": str(OUT_CSV),
        "output_xlsx": str(OUT_XLSX),
        "rows": int(len(out)),
        "unique_figures": int(out["figure_key_v11"].nunique()),
        "unique_papers": int(out["paper_dir"].nunique()),
        "artifact_candidate_rows": int(out["artifact_candidate_v11"].sum()),
        "recommended_keep_rows": int(out["recommended_keep_for_modeling_v11"].sum()),
        "priority_counts": out["modeling_priority_v11"].value_counts().to_dict(),
        "tr_stage_counts": out["tr_stage_v11"].value_counts().to_dict(),
        "context_confidence_counts": out["context_confidence_v11"].value_counts().to_dict(),
        "coverage": {
            "paper_doi": int(out["paper_doi"].notna().sum()),
            "reaction_class": int(out["reaction_class"].notna().sum()),
            "organolithium_role_v5": int(out["organolithium_role_v5"].notna().sum()),
            "reagent_family_v2": int(out["reagent_family_v2"].notna().sum()),
            "solvent": int(out["solvent"].notna().sum()),
            "reactor_type": int(out["reactor_type"].notna().sum()),
            "reactant1_name": int(out["reactant1_name"].notna().sum()),
            "reactant2_name": int(out["reactant2_name"].notna().sum()),
            "product_name": int(out["product_name"].notna().sum()),
        },
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
