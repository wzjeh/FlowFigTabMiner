#!/usr/bin/env python3
"""Plot v5 organolithium role analysis for the cleaned dataset."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from matplotlib.ticker import PercentFormatter


ROOT = Path("/Users/zhaowenyuan/Projects/FlowFigTabMiner")
COMPARE_DIR = ROOT / "data" / "final_output" / "dataset_comparison"
V5_XLSX = COMPARE_DIR / "ALL_organolithium_normalized_20260327_1022_cleaned_v5.xlsx"
V5_MODELING_XLSX = COMPARE_DIR / "ALL_organolithium_normalized_20260327_1022_cleaned_v5_modeling.xlsx"

ROLE_HEATMAP = COMPARE_DIR / "my_dataset_reagent_role_heatmap_v5.png"
ROLE_FOCUS_PLOT = COMPARE_DIR / "my_dataset_role_focus_LDA_LiNp_v5.png"
ROLE_SANKEY_HTML = COMPARE_DIR / "my_dataset_reagent_role_mode_sankey_v5.html"
ROLE_SANKEY_PNG = COMPARE_DIR / "my_dataset_reagent_role_mode_sankey_v5.png"
ROLE_SUMMARY_CSV = COMPARE_DIR / "organolithium_role_v5_summary.csv"

ROLE_ORDER = [
    "Base-mediated deprotonation",
    "Halogen-lithium exchange",
    "Nucleophilic organolithium",
    "C-C bond-forming organolithium",
    "SET reductant / reductive lithiation",
    "Anionic polymerization initiator",
    "Ambiguous / other",
]

REAGENT_ORDER = [
    "n-BuLi",
    "s-BuLi",
    "LDA",
    "PhLi",
    "MesLi",
    "Aryl/heteroarylLi",
    "Lithium naphthalenide",
    "MeLi",
    "HexylLi",
    "LiHMDS",
]


def classify_mode(df: pd.DataFrame) -> pd.Series:
    if "flow_batch_label_v1" in df.columns:
        label = df["flow_batch_label_v1"].fillna("Flow").astype(str)
        return label.where(label.eq("Batch"), "Flow")
    text = (
        df["paper_basename"].fillna("").astype(str)
        + " "
        + df["notes"].fillna("").astype(str)
        + " "
        + df["reactor_type"].fillna("").astype(str)
    ).str.lower()
    batch = text.str.contains(r"\bbatch\b|\bflask\b|\bround-bottom\b", regex=True)
    return batch.map({True: "Batch", False: "Flow"})


def build_heatmap(df: pd.DataFrame) -> None:
    plot_df = df.loc[
        ~df["reagent_family"].isin(["Unclassified", "Other organolithium"])
        & df["reagent_family"].isin(REAGENT_ORDER)
        & df["organolithium_role_v5"].isin(ROLE_ORDER)
    ].copy()
    counts = pd.crosstab(plot_df["reagent_family"], plot_df["organolithium_role_v5"])
    counts = counts.reindex(index=[r for r in REAGENT_ORDER if r in counts.index], columns=ROLE_ORDER, fill_value=0)
    perc = counts.div(counts.sum(axis=1), axis=0).fillna(0) * 100.0

    plt.figure(figsize=(14, 7.5))
    ax = sns.heatmap(
        perc,
        cmap="YlGnBu",
        annot=True,
        fmt=".0f",
        linewidths=0.5,
        cbar_kws={"label": "Within-family percentage (%)"},
    )
    ax.set_title("My Dataset: Organolithium Reagent Roles (v5, modeling subset)", fontsize=16, pad=14)
    ax.set_xlabel("Organolithium role")
    ax.set_ylabel("Reagent family")
    ax.tick_params(axis="x", labelrotation=35, labelsize=10)
    ax.tick_params(axis="y", labelsize=10)
    plt.tight_layout()
    plt.savefig(ROLE_HEATMAP, dpi=300, bbox_inches="tight")
    plt.close()


def build_focus_plot(df: pd.DataFrame) -> None:
    focus = df.loc[df["reagent_family"].isin(["LDA", "Lithium naphthalenide"])].copy()
    summary = (
        focus.groupby(["reagent_family", "organolithium_role_v5"])
        .size()
        .reset_index(name="rows")
    )
    totals = summary.groupby("reagent_family")["rows"].transform("sum")
    summary["pct"] = summary["rows"] / totals * 100.0

    plt.figure(figsize=(10, 5.5))
    ax = sns.barplot(
        data=summary,
        x="organolithium_role_v5",
        y="pct",
        hue="reagent_family",
        order=ROLE_ORDER,
        palette=["#0B5FA5", "#C97B2C"],
    )
    ax.set_title("Focused Role Distribution for LDA and Lithium Naphthalenide (v5)", fontsize=15, pad=12)
    ax.set_xlabel("Organolithium role")
    ax.set_ylabel("Percentage (%)")
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.tick_params(axis="x", labelrotation=35)
    plt.tight_layout()
    plt.savefig(ROLE_FOCUS_PLOT, dpi=300, bbox_inches="tight")
    plt.close()


def build_sankey(df: pd.DataFrame) -> None:
    plot_df = df.loc[
        ~df["reagent_family"].isin(["Unclassified", "Other organolithium"])
        & df["reagent_family"].isin(REAGENT_ORDER)
    ].copy()
    plot_df["mode_v5"] = classify_mode(plot_df)

    left = plot_df.groupby(["reagent_family", "organolithium_role_v5"]).size().reset_index(name="value")
    right = plot_df.groupby(["organolithium_role_v5", "mode_v5"]).size().reset_index(name="value")

    reagent_nodes = list(dict.fromkeys(left["reagent_family"].tolist()))
    role_nodes = [r for r in ROLE_ORDER if r in set(plot_df["organolithium_role_v5"])]
    mode_nodes = ["Flow", "Batch"]
    labels = reagent_nodes + role_nodes + mode_nodes
    idx = {label: i for i, label in enumerate(labels)}

    sources = []
    targets = []
    values = []
    for _, row in left.iterrows():
        sources.append(idx[row["reagent_family"]])
        targets.append(idx[row["organolithium_role_v5"]])
        values.append(int(row["value"]))
    for _, row in right.iterrows():
        sources.append(idx[row["organolithium_role_v5"]])
        targets.append(idx[row["mode_v5"]])
        values.append(int(row["value"]))

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(label=labels, pad=18, thickness=18, color="#9ecae1"),
                link=dict(source=sources, target=targets, value=values),
            )
        ]
    )
    fig.update_layout(title_text="My Dataset: Reagent Family -> Organolithium Role -> Mode (v5)", font_size=12)
    fig.write_html(str(ROLE_SANKEY_HTML))
    try:
        fig.write_image(str(ROLE_SANKEY_PNG), width=1500, height=800, scale=2)
    except Exception:
        pass


def main() -> None:
    sns.set_theme(style="whitegrid", context="talk")

    full_df = pd.read_excel(V5_XLSX)
    modeling_df = pd.read_excel(V5_MODELING_XLSX)

    build_heatmap(modeling_df)
    build_focus_plot(modeling_df)
    build_sankey(modeling_df)

    summary_rows = []
    for name, df in [("all", full_df), ("modeling", modeling_df)]:
        for role, count in df["organolithium_role_v5"].value_counts().items():
            summary_rows.append({"subset": name, "role": role, "rows": int(count)})
    pd.DataFrame(summary_rows).to_csv(ROLE_SUMMARY_CSV, index=False)

    print(f"Saved heatmap: {ROLE_HEATMAP}")
    print(f"Saved focus plot: {ROLE_FOCUS_PLOT}")
    print(f"Saved sankey html: {ROLE_SANKEY_HTML}")
    print(f"Saved sankey png: {ROLE_SANKEY_PNG}")
    print(f"Saved summary: {ROLE_SUMMARY_CSV}")


if __name__ == "__main__":
    main()
