#!/usr/bin/env python3
"""Create manuscript-style source diversity comparison plots for my dataset and ORD."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import PercentFormatter


ROOT = Path("/Users/zhaowenyuan/Projects/FlowFigTabMiner")
COMPARE_DIR = ROOT / "data" / "final_output" / "dataset_comparison"
MY_CLEANED_XLSX = COMPARE_DIR / "ALL_organolithium_normalized_20260327_1022_cleaned_v3.xlsx"
ORD_RICH_CSV = ROOT / "other dataset" / "ord" / "organolithium" / "ord_organolithium_reactions_rich.csv"

PLOT_PATH = COMPARE_DIR / "organolithium_source_diversity_comparison_v3.png"
SUMMARY_CSV = COMPARE_DIR / "organolithium_source_diversity_summary_v3.csv"


def gini(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or np.all(arr == 0):
        return 0.0
    arr = np.sort(arr)
    n = arr.size
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * arr) / (n * np.sum(arr))) - (n + 1) / n)


def cumulative_curve(counts: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    ordered = counts.sort_values(ascending=False).to_numpy(dtype=float)
    n = len(ordered)
    x = np.arange(1, n + 1, dtype=float) / n * 100.0
    y = np.cumsum(ordered) / ordered.sum() * 100.0
    return x, y


def concentration_summary(name: str, counts: pd.Series) -> dict:
    ordered = counts.sort_values(ascending=False)
    return {
        "dataset": name,
        "records": int(ordered.sum()),
        "unique_sources": int(len(ordered)),
        "top1_share_pct": float(ordered.iloc[0] / ordered.sum() * 100),
        "top5_share_pct": float(ordered.head(5).sum() / ordered.sum() * 100),
        "median_records_per_source": float(ordered.median()),
        "gini_source_concentration": gini(ordered.to_numpy(dtype=float)),
    }


def main() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    COMPARE_DIR.mkdir(parents=True, exist_ok=True)

    my_df = pd.read_excel(MY_CLEANED_XLSX)
    ord_df = pd.read_csv(ORD_RICH_CSV, low_memory=False)

    my_df["source_id"] = my_df["paper_doi"].fillna("").astype(str).str.strip()
    missing = my_df["source_id"].eq("") | my_df["source_id"].eq("nan")
    my_df.loc[missing, "source_id"] = my_df.loc[missing, "paper_basename"].astype(str).str.strip()
    my_source_counts = my_df["source_id"].value_counts()
    ord_source_counts = ord_df["doi"].fillna("unknown_source").astype(str).str.strip().value_counts()

    summary_df = pd.DataFrame(
        [
            concentration_summary("My dataset", my_source_counts),
            concentration_summary("ORD organolithium", ord_source_counts),
        ]
    )
    summary_df.to_csv(SUMMARY_CSV, index=False)

    year_df = my_df.loc[:, ["source_id", "paper_year"]].drop_duplicates()
    year_df["paper_year"] = pd.to_numeric(year_df["paper_year"], errors="coerce")
    year_df = year_df[year_df["paper_year"].between(1900, 2026, inclusive="both")]
    year_counts = year_df["paper_year"].astype(int).value_counts().sort_index()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    ordered_my = my_source_counts.sort_values(ascending=False)
    ordered_ord = ord_source_counts.sort_values(ascending=False)
    my_top5 = ordered_my.head(5)

    my_pie_labels = list(ordered_my.index.astype(str))
    my_pie_values = [float(v) for v in ordered_my.to_numpy(dtype=float)]
    ord_pie_labels = list(ordered_ord.index.astype(str))
    ord_pie_values = [float(v) for v in ordered_ord.to_numpy(dtype=float)]

    pie_colors_my = sns.blend_palette(
        ["#0b3c5d", "#1f77b4", "#6baed6", "#b3d7ef", "#e8f4fb"],
        n_colors=len(my_pie_values),
    )
    pie_colors_ord = sns.color_palette("Greens", n_colors=len(ord_pie_values))

    ax = axes[0, 0]
    wedges, _, autotexts = ax.pie(
        my_pie_values,
        labels=None,
        autopct=lambda pct: f"{pct:.1f}%" if pct >= 4 else "",
        startangle=90,
        counterclock=False,
        colors=pie_colors_my,
        wedgeprops={"linewidth": 0.3, "edgecolor": "white"},
        textprops={"fontsize": 10},
    )
    ax.set_title("My Dataset: Source Composition")
    ax.legend(
        [wedges[i] for i in range(min(5, len(wedges)))],
        [f"{label}" for label in my_top5.index.astype(str)],
        title="Top 5 DOI sources",
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        fontsize=9,
        title_fontsize=10,
        frameon=True,
    )
    ax.text(
        0.0,
        -1.25,
        f"3,195 records from 114 source documents\nTop-1 share = 5.1%; Top-5 share = 17.5%",
        ha="center",
        va="center",
        fontsize=10,
    )

    ax = axes[0, 1]
    wedges, _, autotexts = ax.pie(
        ord_pie_values,
        labels=None,
        autopct=lambda pct: f"{pct:.1f}%",
        startangle=90,
        counterclock=False,
        colors=pie_colors_ord,
        wedgeprops={"linewidth": 1, "edgecolor": "white"},
        textprops={"fontsize": 10},
    )
    ax.set_title("ORD: Source Composition")
    ax.legend(
        wedges,
        [str(label) for label in ord_pie_labels],
        title="DOI-level sources",
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        fontsize=9,
        title_fontsize=10,
        frameon=True,
    )
    ax.text(
        0.0,
        -1.25,
        "43,004 records from 4 sources\n"
        "Top-1 share = 79.8%; Top-5 share = 100.0%\n"
        "Dominant source: USPTO patents (1976-Sep 2016)",
        ha="center",
        va="center",
        fontsize=10,
    )

    ax = axes[1, 0]
    my_year_pct = year_counts / year_counts.sum() * 100.0
    ax.bar(my_year_pct.index.to_numpy(), my_year_pct.to_numpy(), color="#1f77b4", width=0.8)
    ax.set_title("My Dataset: Unique Sources by Year")
    ax.set_xlabel("Publication year")
    ax.set_ylabel("Source document share (%)")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    if len(my_year_pct) > 0:
        ax.set_xlim(my_year_pct.index.min() - 0.7, my_year_pct.index.max() + 0.7)
    ax.tick_params(axis="x", rotation=45)

    ord_year_pct = pd.Series(
        {
            "1976-2016": float(ordered_ord.get("10.6084/m9.figshare.5104873.v1", 0.0)),
            "2016": float(ordered_ord.get("10.1002/chem.201603436", 0.0)),
            "2018": float(ordered_ord.get("10.1039/C8SC04228D", 0.0)),
            "2023": float(ordered_ord.get("10.1038/s41557", 0.0)),
        }
    )
    ord_year_pct = ord_year_pct / ord_year_pct.sum() * 100.0
    ax = axes[1, 1]
    ax.bar(ord_year_pct.index.to_list(), ord_year_pct.to_numpy(), color="#2ca02c", width=0.8)
    ax.set_title("ORD: Source Year Distribution")
    ax.set_xlabel("Source year / source span")
    ax.set_ylabel("Record share (%)")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.tick_params(axis="x", rotation=25)

    axes[1, 1].text(
        0.98,
        0.95,
        "USPTO-derived source is treated as a\nsingle 1976-2016 bucket.",
        transform=axes[1, 1].transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.9, "edgecolor": "#cccccc"},
    )

    plt.savefig(PLOT_PATH, dpi=260, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot: {PLOT_PATH}")
    print(f"Saved summary: {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
