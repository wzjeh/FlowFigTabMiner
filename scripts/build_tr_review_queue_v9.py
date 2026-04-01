#!/usr/bin/env python3
"""Build a manual review queue for tR-like intermediate heatmap figures."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import pandas as pd


ROOT = Path("/Users/zhaowenyuan/Projects/FlowFigTabMiner")
OUT_DIR = ROOT / "data" / "final_output" / "dataset_comparison"

INPUT_CSV = OUT_DIR / "intermediate_tr_heatmap_candidates_ge8.csv"
OUT_REVIEW_CSV = OUT_DIR / "intermediate_tr_review_queue_v9.csv"
OUT_REVIEW_XLSX = OUT_DIR / "intermediate_tr_review_queue_v9.xlsx"
OUT_SUBSET_CSV = OUT_DIR / "intermediate_tr_candidate_subset_v9.csv"
OUT_SUMMARY_JSON = OUT_DIR / "intermediate_tr_review_queue_v9_summary.json"


def normalize_text(text: str) -> str:
    text = str(text or "")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def is_tr_like(text: str) -> bool:
    norm = normalize_text(text)
    return bool(norm) and (
        "residencetime" in norm
        or ("tr" in norm and "s" in norm)
        or "r1s" in norm
        or "r2s" in norm
    )


def is_possible_tr(text: str) -> bool:
    if is_tr_like(text):
        return True
    norm = normalize_text(text)
    if not norm:
        return False
    # OCR-corrupted but still plausible tR-like labels.
    if "trz" in norm:
        return True
    if norm in {"trmin", "minrr", "tr2", "tr2s", "tr1s"}:
        return True
    if "tr2" in norm:
        return True
    if "minrr" in norm:
        return True
    if "time" in norm or "residence" in norm:
        return True
    if re.search(r"r[12].*s", norm):
        return True
    return False


def infer_time_unit(text: str) -> tuple[str, str]:
    raw = str(text or "")
    norm = normalize_text(raw)
    if "min" in raw.lower() or "min" in norm:
        return "min", "explicit_min"
    if "(s" in raw.lower() or "/s" in raw.lower() or "seconds" in raw.lower():
        return "s", "explicit_s"
    if "residencetime" in norm:
        return "s", "residence_time_default_s"
    if any(token in norm for token in ["trs", "tr1s", "tr2s", "r1s", "r2s", "trz"]):
        return "s", "ocr_variant_inferred_s"
    if "tr" in norm:
        return "unknown", "tr_like_unit_unclear"
    return "unknown", "unit_unclear"


def classify_row(row: pd.Series) -> str:
    x_titles = str(row.get("x_titles") or "")
    if is_tr_like(x_titles):
        return "high_confidence_tR"
    if is_possible_tr(x_titles):
        return "possible_tR"
    return "non_tR"


def main() -> None:
    df = pd.read_csv(INPUT_CSV)
    df["tr_class_v9"] = df.apply(classify_row, axis=1)
    df["time_unit_v9"], df["time_unit_basis_v9"] = zip(*df["x_titles"].map(infer_time_unit))

    review_df = df.loc[df["tr_class_v9"] != "non_tR"].copy()
    review_df["needs_manual_review_v9"] = True
    review_df["review_decision_v9"] = ""
    review_df["review_notes_v9"] = ""
    review_df["proposed_keep_in_tr_subset_v9"] = review_df["tr_class_v9"] == "high_confidence_tR"

    review_df = review_df.sort_values(
        ["tr_class_v9", "time_unit_v9", "n_data_values", "paper_dir", "page_num", "figure_num_1based"],
        ascending=[True, True, False, True, True, True],
    )

    subset_df = review_df.loc[:, [
        "paper_dir",
        "page_num",
        "figure_num_1based",
        "n_data_values",
        "min_data_value",
        "median_data_value",
        "max_data_value",
        "pct_in_0_100",
        "x_titles",
        "y_titles",
        "tr_class_v9",
        "time_unit_v9",
        "time_unit_basis_v9",
        "data_value_yield_like",
        "evidence_file_count",
        "proposed_keep_in_tr_subset_v9",
        "needs_manual_review_v9",
        "review_decision_v9",
        "review_notes_v9",
    ]].copy()

    review_df.to_csv(OUT_REVIEW_CSV, index=False)
    subset_df.to_csv(OUT_SUBSET_CSV, index=False)
    with pd.ExcelWriter(OUT_REVIEW_XLSX, engine="openpyxl") as writer:
        review_df.to_excel(writer, sheet_name="tR Review Queue", index=False)
        subset_df.to_excel(writer, sheet_name="Candidate Subset", index=False)

    summary = {
        "input_csv": str(INPUT_CSV),
        "review_queue_csv": str(OUT_REVIEW_CSV),
        "review_queue_xlsx": str(OUT_REVIEW_XLSX),
        "candidate_subset_csv": str(OUT_SUBSET_CSV),
        "total_candidates_ge8": int(len(df)),
        "high_confidence_tR": int((df["tr_class_v9"] == "high_confidence_tR").sum()),
        "possible_tR": int((df["tr_class_v9"] == "possible_tR").sum()),
        "excluded_non_tR": int((df["tr_class_v9"] == "non_tR").sum()),
        "unit_counts": review_df["time_unit_v9"].value_counts(dropna=False).to_dict(),
        "preview": review_df.head(30).to_dict(orient="records"),
    }
    OUT_SUMMARY_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
