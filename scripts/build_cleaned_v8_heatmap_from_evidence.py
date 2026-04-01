#!/usr/bin/env python3
"""Build cleaned v8 dataset with OCR-variant tR rescue and evidence-driven heatmap tagging."""

from __future__ import annotations

import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import pandas as pd


ROOT = Path("/Users/zhaowenyuan/Projects/FlowFigTabMiner")
COMPARE_DIR = ROOT / "data" / "final_output" / "dataset_comparison"
INTERMEDIATE_DIR = ROOT / "data" / "intermediate"

V7_XLSX = COMPARE_DIR / "ALL_organolithium_normalized_20260327_1022_cleaned_v7.xlsx"
V8_XLSX = COMPARE_DIR / "ALL_organolithium_normalized_20260327_1022_cleaned_v8.xlsx"
V8_MODELING_XLSX = COMPARE_DIR / "ALL_organolithium_normalized_20260327_1022_cleaned_v8_modeling.xlsx"
V8_HEATMAP_CSV = COMPARE_DIR / "heatmap_figure_groups_v8.csv"
V8_SUMMARY_JSON = COMPARE_DIR / "my_dataset_cleaning_v8_summary.json"

DATA_VALUE_THRESHOLD = 14

TR_PATTERNS = [
    ("note_tR1_sec", re.compile(r"\b(?:t\s*r|tr|ts|rs)\s*1\s*=\s*([0-9]*\.?[0-9]+)\s*s\b", re.I), "R1"),
    ("note_tR2_sec", re.compile(r"\b(?:t\s*r|tr|ts|rs)\s*2\s*=\s*([0-9]*\.?[0-9]+)\s*s\b", re.I), "R2"),
    ("note_tR_sec", re.compile(r"\b(?:t\s*r|tr|ts|rs)\s*=\s*([0-9]*\.?[0-9]+)\s*s\b", re.I), "generic"),
    (
        "note_residence_R1_sec",
        re.compile(r"(?:residence\s*time\s*in\s*R1|R1\s*residence\s*time)\s*=\s*([0-9]*\.?[0-9]+)\s*s\b", re.I),
        "R1",
    ),
    (
        "note_residence_R2_sec",
        re.compile(r"(?:residence\s*time\s*in\s*R2|R2\s*residence\s*time)\s*=\s*([0-9]*\.?[0-9]+)\s*s\b", re.I),
        "R2",
    ),
    ("note_residence_sec", re.compile(r"\bresidence\s*time\s*=\s*([0-9]*\.?[0-9]+)\s*s\b", re.I), "generic"),
]


def normalize_text(text: str) -> str:
    text = str(text or "").strip()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = " ".join(text.split())
    return text.lower()


def tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", normalize_text(text)))


def parse_note_tr(notes: str) -> tuple[float | None, str | None, str | None]:
    text = str(notes or "")
    for source, pattern, stage in TR_PATTERNS:
        match = pattern.search(text)
        if match:
            return float(match.group(1)), source, stage
    return None, None, None


def choose_tr_v8(row: pd.Series) -> tuple[float | None, str | None, str | None, bool]:
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


def build_intermediate_dir_map() -> tuple[dict[str, str], list[tuple[str, set[str], str]]]:
    exact: dict[str, str] = {}
    fuzzy: list[tuple[str, set[str], str]] = []
    for paper_dir in sorted(p for p in INTERMEDIATE_DIR.iterdir() if p.is_dir()):
        norm = normalize_text(paper_dir.name)
        exact[norm] = paper_dir.name
        fuzzy.append((norm, tokenize(paper_dir.name), paper_dir.name))
    return exact, fuzzy


def match_intermediate_dir(paper_name: str, exact_map: dict[str, str], fuzzy_rows: list[tuple[str, set[str], str]]) -> str | None:
    norm = normalize_text(paper_name)
    if norm in exact_map:
        return exact_map[norm]

    tokens = tokenize(paper_name)
    if not tokens:
        return None

    best_dir = None
    best_score = 0.0
    for _, dir_tokens, dir_name in fuzzy_rows:
        union = len(tokens | dir_tokens) or 1
        score = len(tokens & dir_tokens) / union
        if score > best_score:
            best_score = score
            best_dir = dir_name

    return best_dir if best_score >= 0.55 else None


def build_evidence_heatmap_index() -> pd.DataFrame:
    records: dict[tuple[str, int], dict[str, object]] = defaultdict(
        lambda: {
            "max_data_values": 0,
            "x_titles": set(),
            "page_figs": set(),
            "evidence_files": set(),
        }
    )

    for path in INTERMEDIATE_DIR.glob("*/macro_cleaned/*figure*_evidence.json"):
        match = re.search(r"page_(\d+)_figure_(\d+)", path.name)
        if not match:
            continue

        page_num = int(match.group(1))
        figure_num = int(match.group(2)) + 1
        paper_dir = path.parent.parent.name

        try:
            obj = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            continue

        raw_data = obj.get("raw_data") or []
        data_value_count = 0
        for row in raw_data:
            for key, value in row.items():
                if "Data_Value" in key and value not in (None, "", "nan"):
                    data_value_count += 1

        text_evidence = obj.get("text_evidence") or {}
        x_axis_title = " ".join(
            item.get("text", "")
            for item in (text_evidence.get("x_axis_title") or [])
            if isinstance(item, dict)
        ).strip()

        rec = records[(normalize_text(paper_dir), figure_num)]
        rec["max_data_values"] = max(int(rec["max_data_values"]), data_value_count)
        if x_axis_title:
            rec["x_titles"].add(x_axis_title)
        rec["page_figs"].add(f"page_{page_num}_figure_{figure_num - 1}")
        rec["evidence_files"].add(path.name)

    rows = []
    for (paper_key_norm, figure_num), rec in records.items():
        rows.append(
            {
                "paper_key_norm_v8": paper_key_norm,
                "figure_num_v8": figure_num,
                "heatmap_evidence_data_values_v8": int(rec["max_data_values"]),
                "heatmap_evidence_x_title_v8": " | ".join(sorted(rec["x_titles"])),
                "heatmap_evidence_page_figs_v8": " | ".join(sorted(rec["page_figs"])),
                "heatmap_evidence_file_count_v8": len(rec["evidence_files"]),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    df = pd.read_excel(V7_XLSX)

    tr_values = []
    tr_sources = []
    tr_stages = []
    tr_explicit = []
    for _, row in df.iterrows():
        value, source, stage, explicit = choose_tr_v8(row)
        tr_values.append(value)
        tr_sources.append(source)
        tr_stages.append(stage)
        tr_explicit.append(explicit)

    df["tr_s_v8"] = tr_values
    df["tr_source_v8"] = tr_sources
    df["tr_stage_v8"] = tr_stages
    df["tr_is_explicit_v8"] = tr_explicit

    exact_map, fuzzy_rows = build_intermediate_dir_map()
    paper_to_dir = {
        paper: match_intermediate_dir(paper, exact_map, fuzzy_rows)
        for paper in df["paper_basename"].fillna("").astype(str).unique()
    }
    df["matched_intermediate_dir_v8"] = df["paper_basename"].map(paper_to_dir)
    df["paper_key_norm_v8"] = df["matched_intermediate_dir_v8"].map(normalize_text)
    df["figure_num_v8"] = pd.to_numeric(
        df["source_table_or_figure"].fillna("").astype(str).str.extract(r"Figure\s*(\d+)")[0],
        errors="coerce",
    )

    evidence_df = build_evidence_heatmap_index()
    df = df.merge(evidence_df, on=["paper_key_norm_v8", "figure_num_v8"], how="left")

    df["is_heatmap_figure_v8"] = (
        df["source_table_or_figure"].fillna("").str.contains("Figure", case=False, regex=False)
        & (df["heatmap_evidence_data_values_v8"].fillna(0) >= DATA_VALUE_THRESHOLD)
    )
    df["time_axis_value_s_v8"] = pd.NA
    df["time_axis_label_v8"] = pd.NA
    df["time_axis_source_v8"] = pd.NA
    mask = df["is_heatmap_figure_v8"]
    df.loc[mask, "time_axis_value_s_v8"] = df.loc[mask, "tr_s_v8"]
    df.loc[mask, "time_axis_label_v8"] = "tR (s)"
    df.loc[mask, "time_axis_source_v8"] = df.loc[mask, "tr_source_v8"]

    heatmap_groups = (
        df.loc[mask, [
            "paper_basename",
            "paper_doi",
            "paper_year",
            "source_table_or_figure",
            "matched_intermediate_dir_v8",
            "figure_num_v8",
            "heatmap_evidence_data_values_v8",
            "heatmap_evidence_x_title_v8",
            "heatmap_evidence_page_figs_v8",
            "time_axis_label_v8",
        ]]
        .drop_duplicates()
        .sort_values(["heatmap_evidence_data_values_v8", "paper_basename"], ascending=[False, True])
    )
    heatmap_groups.to_csv(V8_HEATMAP_CSV, index=False)

    with pd.ExcelWriter(V8_XLSX, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="All Records", index=False)

    modeling_df = df.loc[~df["exclude_candidate_v3"].fillna(False)].copy()
    with pd.ExcelWriter(V8_MODELING_XLSX, engine="openpyxl") as writer:
        modeling_df.to_excel(writer, sheet_name="All Records", index=False)

    summary = {
        "input_file": str(V7_XLSX),
        "output_file": str(V8_XLSX),
        "modeling_output_file": str(V8_MODELING_XLSX),
        "heatmap_group_csv": str(V8_HEATMAP_CSV),
        "data_value_threshold_v8": DATA_VALUE_THRESHOLD,
        "rows": int(len(df)),
        "rows_modeling": int(len(modeling_df)),
        "tr_nonnull_all_v8": int(df["tr_s_v8"].notna().sum()),
        "tr_nonnull_modeling_v8": int(modeling_df["tr_s_v8"].notna().sum()),
        "tr_explicit_all_v8": int(df["tr_is_explicit_v8"].sum()),
        "tr_explicit_modeling_v8": int(modeling_df["tr_is_explicit_v8"].sum()),
        "tr_explicit_gain_vs_v6": int(df["tr_is_explicit_v8"].sum() - df["tr_is_explicit_v6"].fillna(False).sum()),
        "heatmap_groups_total_v8": int(len(heatmap_groups)),
        "heatmap_rows_total_v8": int(mask.sum()),
        "heatmap_rows_with_tr_total_v8": int((mask & df["time_axis_value_s_v8"].notna()).sum()),
        "heatmap_groups_modeling_v8": int(
            modeling_df.loc[modeling_df["is_heatmap_figure_v8"], ["paper_basename", "source_table_or_figure"]].drop_duplicates().shape[0]
        ),
        "heatmap_rows_modeling_v8": int(modeling_df["is_heatmap_figure_v8"].sum()),
        "heatmap_groups_preview_v8": heatmap_groups.head(30).to_dict(orient="records"),
        "tr_source_counts_all_v8": df["tr_source_v8"].fillna("missing").value_counts().to_dict(),
        "tr_source_counts_modeling_v8": modeling_df["tr_source_v8"].fillna("missing").value_counts().to_dict(),
    }
    V8_SUMMARY_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved cleaned v8 workbook: {V8_XLSX}")
    print(f"Saved cleaned v8 modeling workbook: {V8_MODELING_XLSX}")
    print(f"Saved heatmap group CSV: {V8_HEATMAP_CSV}")
    print(f"Saved v8 summary: {V8_SUMMARY_JSON}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
