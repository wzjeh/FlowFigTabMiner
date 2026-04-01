#!/usr/bin/env python3
"""Build a raw-image-only audit pack and editable Excel for tR/T/Yield point review."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd


ROOT = Path("/Users/zhaowenyuan/Projects/FlowFigTabMiner")
COMPARE_DIR = ROOT / "data" / "final_output" / "dataset_comparison"

POINTS_CSV = COMPARE_DIR / "organolithium_tr_heatmap_points_v10.csv"
SOURCE_PACK = COMPARE_DIR / "organolithium_tr_points_source_figures_v10_review_pack"

RAW_PACK = COMPARE_DIR / "organolithium_tr_points_raw_figures_only_v10"
AUDIT_XLSX = COMPARE_DIR / "organolithium_tr_points_audit_v10.xlsx"
AUDIT_CSV = COMPARE_DIR / "organolithium_tr_points_audit_v10.csv"


def main() -> None:
    points = pd.read_csv(POINTS_CSV).copy()

    RAW_PACK.mkdir(parents=True, exist_ok=True)

    # Copy only raw figure images for each source folder.
    source_rows = []
    for folder in sorted([p for p in SOURCE_PACK.iterdir() if p.is_dir()]):
        dest_folder = RAW_PACK / folder.name
        dest_folder.mkdir(parents=True, exist_ok=True)

        raw_files = sorted(folder.rglob("*_raw.png"))
        for src in raw_files:
            rel = src.relative_to(folder)
            dst = dest_folder / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            source_rows.append(
                {
                    "source_folder_name": folder.name,
                    "raw_image_path": str(dst),
                }
            )

    # Build editable audit sheet with only the three key values and identifiers.
    review = points.loc[:, [
        "review_id_v10",
        "paper_dir",
        "page_tag",
        "figure_num_1based",
        "evidence_file",
        "series_label",
        "tr_s",
        "temperature_C_from_figure",
        "yield_pct_from_figure",
    ]].copy()
    review = review.sort_values(
        ["review_id_v10", "page_tag", "evidence_file", "tr_s", "temperature_C_from_figure", "yield_pct_from_figure"]
    ).reset_index(drop=True)

    review["audit_point_id_v10"] = range(1, len(review) + 1)
    review["raw_image_path"] = review.apply(
        lambda row: str(
            RAW_PACK
            / f"{int(row['review_id_v10']):02d}_{row['paper_dir']}"
            / str(row["page_tag"])
            / str(row["evidence_file"]).replace("_evidence.json", "_raw.png")
        ),
        axis=1,
    )

    review["tr_s_corrected"] = review["tr_s"]
    review["temperature_C_corrected"] = review["temperature_C_from_figure"]
    review["yield_pct_corrected"] = review["yield_pct_from_figure"]
    review["audit_status"] = ""
    review["audit_notes"] = ""

    review.to_csv(AUDIT_CSV, index=False)
    with pd.ExcelWriter(AUDIT_XLSX, engine="openpyxl") as writer:
        review.to_excel(writer, sheet_name="point_audit", index=False)
        pd.DataFrame(source_rows).to_excel(writer, sheet_name="raw_images", index=False)

    print(
        {
            "raw_pack": str(RAW_PACK),
            "audit_xlsx": str(AUDIT_XLSX),
            "audit_csv": str(AUDIT_CSV),
            "figures": int(review[["review_id_v10", "paper_dir", "page_tag"]].drop_duplicates().shape[0]),
            "points": int(len(review)),
        }
    )


if __name__ == "__main__":
    main()
