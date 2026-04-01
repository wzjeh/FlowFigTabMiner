#!/usr/bin/env python3
"""Analyze intermediate figure evidence for tR-like heatmap candidates."""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import pandas as pd


ROOT = Path("/Users/zhaowenyuan/Projects/FlowFigTabMiner")
INTERMEDIATE_DIR = ROOT / "data" / "intermediate"
OUT_DIR = ROOT / "data" / "final_output" / "dataset_comparison"

OUT_ALL = OUT_DIR / "intermediate_tr_heatmap_candidates_ge8.csv"
OUT_TR = OUT_DIR / "intermediate_tr_heatmap_candidates_ge8_tr_like.csv"
OUT_SUMMARY = OUT_DIR / "intermediate_tr_heatmap_candidates_ge8_summary.json"

MIN_DATA_VALUES = 8


def normalize_text(text: str) -> str:
    text = str(text or "")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def is_tr_like_x_axis(text: str) -> bool:
    norm = normalize_text(text)
    if not norm:
        return False
    return (
        "residencetime" in norm
        or ("tr" in norm and "s" in norm)
        or "r1s" in norm
        or "r2s" in norm
    )


def safe_float(value: object) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def is_yield_like(values: list[float]) -> bool:
    if not values:
        return False
    within = [v for v in values if 0 <= v <= 100]
    pct = len(within) / len(values)
    vmax = max(values)
    # Allow some OCR noise, but keep the overall shape close to a percentage axis.
    return pct >= 0.7 and vmax <= 150


def main() -> None:
    grouped: dict[tuple[str, int, int], dict[str, object]] = defaultdict(
        lambda: {
            "values": [],
            "x_titles": set(),
            "y_titles": set(),
            "files": set(),
        }
    )

    for path in INTERMEDIATE_DIR.glob("*/macro_cleaned/*figure*_evidence.json"):
        match = re.search(r"page_(\d+)_figure_(\d+)", path.name)
        if not match:
            continue
        page_num = int(match.group(1))
        fig_idx = int(match.group(2))
        key = (path.parent.parent.name, page_num, fig_idx + 1)

        try:
            obj = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
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

        rec = grouped[key]
        if x_title:
            rec["x_titles"].add(x_title)
        if y_title:
            rec["y_titles"].add(y_title)
        rec["files"].add(path.name)

        for row in obj.get("raw_data") or []:
            value = safe_float(row.get("Y_Right/Data_Value"))
            if value is not None:
                rec["values"].append(value)

    rows: list[dict[str, object]] = []
    for (paper_dir, page_num, fig_num), rec in grouped.items():
        values = list(rec["values"])
        if len(values) < MIN_DATA_VALUES:
            continue
        within_0_100 = [v for v in values if 0 <= v <= 100]
        x_titles = " || ".join(sorted(rec["x_titles"]))
        y_titles = " || ".join(sorted(rec["y_titles"]))
        rows.append(
            {
                "paper_dir": paper_dir,
                "page_num": page_num,
                "figure_num_1based": fig_num,
                "n_data_values": len(values),
                "min_data_value": min(values),
                "median_data_value": float(pd.Series(values).median()),
                "max_data_value": max(values),
                "pct_in_0_100": round(100 * len(within_0_100) / len(values), 2),
                "x_titles": x_titles,
                "y_titles": y_titles,
                "x_axis_tr_like": is_tr_like_x_axis(x_titles),
                "data_value_yield_like": is_yield_like(values),
                "evidence_file_count": len(rec["files"]),
            }
        )

    all_df = pd.DataFrame(rows).sort_values(
        ["x_axis_tr_like", "data_value_yield_like", "n_data_values", "paper_dir"],
        ascending=[False, False, False, True],
    )
    tr_df = all_df.loc[all_df["x_axis_tr_like"]].copy()

    all_df.to_csv(OUT_ALL, index=False)
    tr_df.to_csv(OUT_TR, index=False)

    summary = {
        "candidate_groups_ge8": int(len(all_df)),
        "tr_like_groups_ge8": int(len(tr_df)),
        "tr_like_and_yield_like_groups_ge8": int(
            len(all_df.loc[all_df["x_axis_tr_like"] & all_df["data_value_yield_like"]])
        ),
        "output_all_csv": str(OUT_ALL),
        "output_tr_csv": str(OUT_TR),
        "top_tr_like_groups": tr_df.head(30).to_dict(orient="records"),
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
