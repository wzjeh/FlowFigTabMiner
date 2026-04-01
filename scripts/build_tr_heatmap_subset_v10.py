#!/usr/bin/env python3
"""Extract tR-T-yield points from reviewed heatmap figures and safely supplement context."""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd


ROOT = Path("/Users/zhaowenyuan/Projects/FlowFigTabMiner")
INTERMEDIATE_DIR = ROOT / "data" / "intermediate"
COMPARE_DIR = ROOT / "data" / "final_output" / "dataset_comparison"
REVIEW_DIR = COMPARE_DIR / "intermediate_heatmap_candidates_ge8_review_pack_v1"

KEEP_CSV = REVIEW_DIR / "candidate_manifest_keep_only_v10.csv"
V8_XLSX = COMPARE_DIR / "ALL_organolithium_normalized_20260327_1022_cleaned_v8.xlsx"

OUT_POINTS_CSV = COMPARE_DIR / "organolithium_tr_heatmap_points_v10.csv"
OUT_POINTS_XLSX = COMPARE_DIR / "organolithium_tr_heatmap_points_v10.xlsx"
OUT_FIGURE_AUDIT_CSV = COMPARE_DIR / "organolithium_tr_heatmap_figure_audit_v10.csv"
OUT_SUMMARY_JSON = COMPARE_DIR / "organolithium_tr_heatmap_subset_v10_summary.json"


def normalize_text(text: str) -> str:
    text = str(text or "").strip()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = " ".join(text.split())
    return text.lower()


def tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", normalize_text(text)))


def safe_float(value: object) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def best_paper_match(paper_dir: str, paper_index: list[tuple[str, set[str], str]]) -> tuple[str | None, float]:
    query_tokens = tokenize(paper_dir)
    if not query_tokens:
        return None, 0.0

    best_name = None
    best_score = 0.0
    for _, target_tokens, target_name in paper_index:
        union = len(query_tokens | target_tokens) or 1
        score = len(query_tokens & target_tokens) / union
        if score > best_score:
            best_score = score
            best_name = target_name
    return best_name, best_score


def choose_safe_value(exact_rows: pd.DataFrame, paper_rows: pd.DataFrame, col: str) -> tuple[object | None, str]:
    for source_name, frame in [("exact_figure_unique", exact_rows), ("paper_unique", paper_rows)]:
        if col not in frame.columns:
            continue
        vals = [v for v in frame[col].dropna().tolist() if str(v).strip()]
        unique_vals = []
        seen = set()
        for v in vals:
            key = str(v).strip()
            if key not in seen:
                seen.add(key)
                unique_vals.append(v)
        if len(unique_vals) == 1:
            return unique_vals[0], source_name
    return None, "missing_or_ambiguous"


def dominant_value(frame: pd.DataFrame, col: str) -> tuple[object | None, float]:
    if col not in frame.columns:
        return None, 0.0
    vals = [str(v).strip() for v in frame[col].dropna().tolist() if str(v).strip()]
    if not vals:
        return None, 0.0
    counter = Counter(vals)
    value, count = counter.most_common(1)[0]
    return value, count / len(vals)


def build_context(v8_df: pd.DataFrame, paper_dir: str, figure_num: int) -> tuple[dict[str, object], dict[str, object]]:
    exact_paper, score = best_paper_match(
        paper_dir,
        [(name, tokenize(name), name) for name in sorted(v8_df["paper_basename"].dropna().astype(str).unique())],
    )
    if exact_paper is None:
        return {
            "paper_match_name_v10": None,
            "paper_match_score_v10": 0.0,
            "context_match_level_v10": "none",
        }, {
            "paper_match_name_v10": None,
            "paper_match_score_v10": 0.0,
            "paper_rows_v10": 0,
            "exact_figure_rows_v10": 0,
        }

    paper_rows = v8_df.loc[v8_df["paper_basename"] == exact_paper].copy()
    exact_rows = paper_rows.loc[paper_rows["figure_num_v10"] == figure_num].copy()

    context: dict[str, object] = {
        "paper_match_name_v10": exact_paper,
        "paper_match_score_v10": round(score, 4),
        "context_match_level_v10": "exact_figure" if len(exact_rows) else "paper_only",
    }
    audit: dict[str, object] = {
        "paper_match_name_v10": exact_paper,
        "paper_match_score_v10": round(score, 4),
        "paper_rows_v10": int(len(paper_rows)),
        "exact_figure_rows_v10": int(len(exact_rows)),
    }

    for col in [
        "paper_doi",
        "paper_year",
        "reaction_class",
        "organolithium_role_v5",
        "reagent_family_v2",
        "solvent",
        "reactor_type",
        "reactant1_name",
        "reactant2_name",
        "product_name",
    ]:
        value, source = choose_safe_value(exact_rows, paper_rows, col)
        context[col] = value
        context[f"{col}_source_v10"] = source
        top_value, top_share = dominant_value(exact_rows if len(exact_rows) else paper_rows, col)
        audit[f"{col}_top_value_v10"] = top_value
        audit[f"{col}_top_share_v10"] = round(top_share, 4)

    return context, audit


def main() -> None:
    keep_df = pd.read_csv(KEEP_CSV)
    v8_df = pd.read_excel(V8_XLSX)
    v8_df["figure_num_v10"] = pd.to_numeric(
        v8_df["source_table_or_figure"].fillna("").astype(str).str.extract(r"Figure\s*(\d+)")[0],
        errors="coerce",
    )

    point_rows: list[dict[str, object]] = []
    figure_rows: list[dict[str, object]] = []

    for _, keep in keep_df.iterrows():
        paper_dir = str(keep["paper_dir"])
        page_tag = str(keep["page_tag"])
        figure_num = int(keep["figure_num_1based"])
        context, audit = build_context(v8_df, paper_dir, figure_num)

        extracted_points = 0
        evidence_files = sorted((INTERMEDIATE_DIR / paper_dir / "macro_cleaned").glob(f"{page_tag}_*_evidence.json"))
        for evidence_path in evidence_files:
            try:
                obj = json.loads(evidence_path.read_text(encoding="utf-8", errors="ignore"))
            except Exception:
                continue

            text_evidence = obj.get("text_evidence") or {}
            x_title = " | ".join(
                item.get("text", "")
                for item in (text_evidence.get("x_axis_title") or [])
                if isinstance(item, dict)
            ).strip()
            y_title = " | ".join(
                item.get("text", "")
                for item in (text_evidence.get("y_axis_title") or [])
                if isinstance(item, dict)
            ).strip()

            for row in obj.get("raw_data") or []:
                tr_s = safe_float(row.get("X"))
                temperature_c = safe_float(row.get("Y_Left"))
                yield_pct = safe_float(row.get("Y_Right/Data_Value"))
                if tr_s is None or temperature_c is None or yield_pct is None:
                    continue

                extracted_points += 1
                out = {
                    "review_id_v10": int(keep["review_id"]),
                    "paper_dir": paper_dir,
                    "page_tag": page_tag,
                    "figure_num_1based": figure_num,
                    "evidence_file": evidence_path.name,
                    "series_label": row.get("Series"),
                    "tr_s": tr_s,
                    "temperature_C_from_figure": temperature_c,
                    "yield_pct_from_figure": yield_pct,
                    "x_axis_label_v10": "tR (s)",
                    "y_axis_label_v10": "T (°C)",
                    "data_value_label_v10": "Yield (%)",
                    "x_title_ocr_raw": x_title,
                    "y_title_ocr_raw": y_title,
                    "review_source_v10": "user_reviewed_heatmap_keep",
                }
                out.update(context)
                point_rows.append(out)

        fig_row = {
            "review_id_v10": int(keep["review_id"]),
            "paper_dir": paper_dir,
            "page_tag": page_tag,
            "figure_num_1based": figure_num,
            "kept_by_user_v10": True,
            "x_axis_final_v10": keep.get("x_axis_final_v10"),
            "y_axis_final_v10": keep.get("y_axis_final_v10"),
            "time_unit_final_v10": keep.get("time_unit_final_v10"),
            "extracted_points_v10": extracted_points,
            "evidence_files_found_v10": len(evidence_files),
        }
        fig_row.update(context)
        fig_row.update(audit)
        figure_rows.append(fig_row)

    points_df = pd.DataFrame(point_rows).sort_values(
        ["paper_dir", "figure_num_1based", "evidence_file", "tr_s", "temperature_C_from_figure", "yield_pct_from_figure"]
    )
    figure_df = pd.DataFrame(figure_rows).sort_values(["paper_dir", "figure_num_1based"])

    points_df.to_csv(OUT_POINTS_CSV, index=False)
    with pd.ExcelWriter(OUT_POINTS_XLSX, engine="openpyxl") as writer:
        points_df.to_excel(writer, sheet_name="tR_heatmap_points", index=False)
        figure_df.to_excel(writer, sheet_name="figure_audit", index=False)
    figure_df.to_csv(OUT_FIGURE_AUDIT_CSV, index=False)

    summary = {
        "input_keep_csv": str(KEEP_CSV),
        "input_v8_xlsx": str(V8_XLSX),
        "output_points_csv": str(OUT_POINTS_CSV),
        "output_points_xlsx": str(OUT_POINTS_XLSX),
        "output_figure_audit_csv": str(OUT_FIGURE_AUDIT_CSV),
        "kept_figures": int(len(figure_df)),
        "extracted_points": int(len(points_df)),
        "figures_with_points": int((figure_df["extracted_points_v10"] > 0).sum()),
        "figures_with_exact_context": int((figure_df["context_match_level_v10"] == "exact_figure").sum()),
        "figures_with_paper_context_only": int((figure_df["context_match_level_v10"] == "paper_only").sum()),
        "paper_doi_covered": int(points_df["paper_doi"].notna().sum()) if "paper_doi" in points_df.columns else 0,
        "reaction_class_covered": int(points_df["reaction_class"].notna().sum()) if "reaction_class" in points_df.columns else 0,
        "organolithium_role_covered": int(points_df["organolithium_role_v5"].notna().sum()) if "organolithium_role_v5" in points_df.columns else 0,
        "reagent_family_covered": int(points_df["reagent_family_v2"].notna().sum()) if "reagent_family_v2" in points_df.columns else 0,
        "solvent_covered": int(points_df["solvent"].notna().sum()) if "solvent" in points_df.columns else 0,
        "reactor_type_covered": int(points_df["reactor_type"].notna().sum()) if "reactor_type" in points_df.columns else 0,
        "paper_list_preview": figure_df[[
            "paper_dir",
            "page_tag",
            "extracted_points_v10",
            "paper_doi",
            "paper_year",
            "reaction_class",
            "organolithium_role_v5",
            "reagent_family_v2",
            "context_match_level_v10",
        ]].head(30).to_dict(orient="records"),
    }
    OUT_SUMMARY_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
