#!/usr/bin/env python3
"""Plot flow/batch and reagent comparisons, plus richer my-dataset-only structure views."""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from matplotlib.ticker import PercentFormatter


ROOT = Path("/Users/zhaowenyuan/Projects/FlowFigTabMiner")
COMPARE_DIR = ROOT / "data" / "final_output" / "dataset_comparison"
MY_CLEANED_XLSX = COMPARE_DIR / "ALL_organolithium_normalized_20260327_1022_cleaned_v3.xlsx"
ORD_RICH_CSV = ROOT / "other dataset" / "ord" / "organolithium" / "ord_organolithium_reactions_rich.csv"

MODE_PLOT = COMPARE_DIR / "organolithium_flow_batch_comparison_v3.png"
REAGENT_PLOT = COMPARE_DIR / "organolithium_reagent_composition_comparison_v3.png"
SANKEY_HTML = COMPARE_DIR / "my_dataset_reagent_reaction_mode_sankey_v3.html"
SANKEY_PNG = COMPARE_DIR / "my_dataset_reagent_reaction_mode_sankey_v3.png"
HEATMAP_PNG = COMPARE_DIR / "my_dataset_reagent_reaction_heatmap_v3.png"
SUMMARY_CSV = COMPARE_DIR / "organolithium_flow_batch_reagent_structure_summary_v3.csv"


def classify_reagent_family_from_text(text: str) -> str | None:
    if not text:
        return None
    t = str(text).lower().strip()
    if not t or t == "nan":
        return None

    patterns = [
        ("n-BuLi", [r"\bn-?butyl ?lithium\b", r"\bnbuli\b", r"\bn-buli\b"]),
        ("s-BuLi", [r"\bsec-?butyl ?lithium\b", r"\bs-?butyllithium\b", r"\bsbuli\b"]),
        ("t-BuLi", [r"\btert-?butyl ?lithium\b", r"\bt-?butyllithium\b", r"\btbuli\b", r"\btertbutyllithium\b"]),
        ("MeLi", [r"\bmethyl ?lithium\b", r"\bmeli\b"]),
        ("PhLi", [r"\bphenyl ?lithium\b", r"\bphli\b"]),
        ("LDA", [r"\blithium diisopropylamide\b", r"\blda\b"]),
        ("LiHMDS", [r"\blihmds\b", r"lithium hexamethyldisilazide"]),
        ("Lithium naphthalenide", [r"lithium naphthalenide", r"\bnaphthalenide\b"]),
        ("HexylLi", [r"\bhexyl ?lithium\b", r"\bn-?hexyllithium\b"]),
        ("EthylLi", [r"\bethyl ?lithium\b"]),
        ("MesLi", [r"\bmesli\b", r"\bmesityllithium\b"]),
        ("Vinyl/alkynylLi", [r"vinyllithium", r"lithio.*yne", r"lithiophenylacetylene", r"ynolate"]),
        ("Aryl/heteroarylLi", [r"aryllithium", r"lithiothiophene", r"lithiopyrid", r"lithiofuran", r"lithiotoluene", r"phenyllithium"]),
        ("Other organolithium", [r"organolithium", r"alkyllithium", r"dilithio", r"\blithio\b"]),
    ]
    for label, pats in patterns:
        if any(re.search(p, t) for p in pats):
            return label
    return None


def classify_reagent_family_from_smiles(text: str) -> str | None:
    if not text:
        return None
    s = str(text)
    checks = [
        ("n-BuLi", ["C(CCC)[Li]", "[Li]CCCC", "[Li+].CCC[CH2-]"]),
        ("s-BuLi", ["[Li]C(C)CC", "C(C)(CC)[Li]"]),
        ("t-BuLi", ["C(C)(C)(C)[Li]", "[Li]C(C)(C)C"]),
        ("MeLi", ["C[Li]", "[Li]C"]),
        ("PhLi", ["C1(=CC=CC=C1)[Li]", "[Li]c1ccccc1"]),
        ("HexylLi", ["C(CCCCC)[Li]"]),
        ("EthylLi", ["[Li]CC", "C(C)[Li]"]),
    ]
    for label, pats in checks:
        if any(p in s for p in pats):
            return label
    if "[Li]" in s or "[Li+]" in s:
        return "Other organolithium"
    return None


def classify_my_reagent_family(row: pd.Series) -> str:
    candidates = [
        row.get("reactant2_name"),
        row.get("reactant1_name"),
        row.get("reactant2_smiles_enriched"),
        row.get("reactant1_smiles_enriched"),
        row.get("reactant2_smiles"),
        row.get("reactant1_smiles"),
    ]
    for c in candidates:
        fam = classify_reagent_family_from_text(c if isinstance(c, str) else "")
        if fam:
            return fam
        fam = classify_reagent_family_from_smiles(c if isinstance(c, str) else "")
        if fam:
            return fam
    return "Unclassified"


def rescue_my_reagent_family(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["reagent_family_raw"] = out.apply(classify_my_reagent_family, axis=1)
    out["reagent_family"] = out["reagent_family_raw"]
    out["reagent_family_assignment"] = "direct"

    mask = out["reagent_family"].eq("Unclassified")
    notes_family = out.loc[mask, "notes"].map(classify_reagent_family_from_text)
    note_mask = mask & notes_family.notna()
    out.loc[note_mask, "reagent_family"] = notes_family.loc[note_mask]
    out.loc[note_mask, "reagent_family_assignment"] = "notes"

    def apply_unique_group(keys: list[str], label: str) -> None:
        classified = out[out["reagent_family"] != "Unclassified"].copy()
        if classified.empty:
            return
        group_nunique = classified.groupby(keys)["reagent_family"].nunique()
        valid_keys = group_nunique[group_nunique == 1].index
        if len(valid_keys) == 0:
            return
        mapping = classified.groupby(keys)["reagent_family"].first().to_dict()
        unresolved = out["reagent_family"].eq("Unclassified")
        for idx, row in out.loc[unresolved, keys].iterrows():
            key = tuple(row[k] for k in keys) if len(keys) > 1 else row[keys[0]]
            if key in mapping:
                out.at[idx, "reagent_family"] = mapping[key]
                out.at[idx, "reagent_family_assignment"] = label

    apply_unique_group(["paper_doi", "source_table_or_figure"], "same_doi_same_figure")
    apply_unique_group(["paper_doi"], "same_doi_unique")
    apply_unique_group(["paper_basename", "source_table_or_figure"], "same_paper_same_figure")
    return out


def classify_ord_reagent_family(matched_components_json: str) -> str:
    try:
        items = json.loads(matched_components_json) if isinstance(matched_components_json, str) else []
    except Exception:
        items = []
    for item in items:
        val = str(item.get("matched_value", ""))
        if item.get("identifier_type") == "NAME":
            fam = classify_reagent_family_from_text(val)
            if fam:
                return fam
    for item in items:
        val = str(item.get("matched_value", ""))
        if item.get("identifier_type") == "SMILES":
            fam = classify_reagent_family_from_smiles(val)
            if fam:
                return fam
    return "Unclassified"


def my_mode(series: pd.Series) -> pd.Series:
    mode = pd.Series("Flow", index=series.index, dtype=object)
    mode[series.astype(str).str.lower().eq("batch")] = "Batch"
    return mode


def ord_mode(df: pd.DataFrame) -> pd.Series:
    combined = (
        df["conditions_json"].fillna("").astype(str)
        + " "
        + df["setup_json"].fillna("").astype(str)
        + " "
        + df["notes_json"].fillna("").astype(str)
    ).str.lower()
    mode = pd.Series("Batch", index=df.index, dtype=object)
    mode[combined.str.contains("flow")] = "Flow"
    return mode


def pct(series: pd.Series, order: list[str] | None = None) -> pd.Series:
    out = series.value_counts(dropna=False) / len(series) * 100.0
    if order is not None:
        out = out.reindex(order, fill_value=0.0)
    return out


def save_mode_plot(my_mode_series: pd.Series, ord_mode_series: pd.Series) -> None:
    order = ["Flow", "Batch"]
    comp = pd.DataFrame({"My dataset": pct(my_mode_series, order), "ORD": pct(ord_mode_series, order)})
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    comp.T.plot(kind="bar", stacked=True, ax=ax, color=["#1f77b4", "#d95f02"], width=0.58)
    ax.set_title("Reaction Mode Comparison")
    ax.set_ylabel("Record share (%)")
    ax.set_xlabel("")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.set_ylim(0, 100)
    ax.legend(title="", loc="upper right", frameon=True)
    for idx, row in enumerate(comp.T.itertuples(index=False)):
        cumulative = 0.0
        for value in row:
            if value >= 5:
                ax.text(idx, cumulative + value / 2, f"{value:.0f}%", ha="center", va="center", color="white", fontsize=11)
            cumulative += value
    plt.tight_layout()
    plt.savefig(MODE_PLOT, dpi=260)
    plt.close(fig)


def save_reagent_plot(my_reagents: pd.Series, ord_reagents: pd.Series) -> None:
    order = [
        "n-BuLi",
        "s-BuLi",
        "t-BuLi",
        "MeLi",
        "PhLi",
        "LDA",
        "LiHMDS",
        "Lithium naphthalenide",
        "MesLi",
        "HexylLi",
        "EthylLi",
        "Vinyl/alkynylLi",
        "Aryl/heteroarylLi",
        "Other organolithium",
        "Unclassified",
    ]
    comp = pd.DataFrame({"My dataset": pct(my_reagents, order), "ORD": pct(ord_reagents, order)})
    comp = comp.loc[comp.max(axis=1) >= 1.0].sort_values("My dataset", ascending=False)
    fig, ax = plt.subplots(figsize=(12.2, 6.8))
    x = range(len(comp))
    width = 0.38
    ax.bar([i - width / 2 for i in x], comp["My dataset"], width=width, color="#1f77b4", label="My dataset")
    ax.bar([i + width / 2 for i in x], comp["ORD"], width=width, color="#2ca02c", label="ORD")
    ax.set_xticks(list(x))
    ax.set_xticklabels(comp.index, rotation=35, ha="right")
    ax.set_ylabel("Record share (%)")
    ax.set_xlabel("")
    ax.set_title("Organolithium Reagent Composition")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.set_ylim(0, 100)
    ax.legend(frameon=True, loc="upper right")
    plt.tight_layout()
    plt.savefig(REAGENT_PLOT, dpi=260)
    plt.close(fig)


def save_heatmap(my_df: pd.DataFrame) -> None:
    filtered_df = my_df[~my_df["reagent_family"].isin(["Unclassified", "Other organolithium"])].copy()
    top_reagents = filtered_df["reagent_family"].value_counts().head(8).index
    top_reactions = my_df["reaction_class_clean"].value_counts().head(8).index
    heat = (
        filtered_df[filtered_df["reagent_family"].isin(top_reagents) & filtered_df["reaction_class_clean"].isin(top_reactions)]
        .groupby(["reagent_family", "reaction_class_clean"])
        .size()
        .unstack(fill_value=0)
    )
    heat = heat.div(heat.sum(axis=1), axis=0) * 100.0
    fig, ax = plt.subplots(figsize=(13.6, 8.6))
    sns.heatmap(
        heat,
        cmap="Blues",
        annot=True,
        fmt=".0f",
        annot_kws={"size": 8},
        cbar_kws={"label": "Within-reagent share (%)", "shrink": 0.9},
        ax=ax,
    )
    ax.set_title("My Dataset: Reaction Types within Major Reagent Families")
    ax.set_xlabel("Reaction type")
    ax.set_ylabel("Organolithium reagent family")
    ax.tick_params(axis="x", labelrotation=30, labelsize=10)
    ax.tick_params(axis="y", labelsize=10)
    plt.tight_layout()
    plt.savefig(HEATMAP_PNG, dpi=260)
    plt.close(fig)


def save_sankey(my_df: pd.DataFrame) -> None:
    sankey_df = my_df[my_df["reagent_family"] != "Unclassified"].copy()
    sankey_df["mode_clean"] = my_mode(sankey_df["flow_batch_label_v1"])
    top_reagents = sankey_df["reagent_family"].value_counts().head(7).index.tolist()
    top_reactions = sankey_df["reaction_class_clean"].value_counts().head(7).index.tolist()
    sankey_df["reagent_plot"] = sankey_df["reagent_family"].where(sankey_df["reagent_family"].isin(top_reagents), "Other reagents")
    sankey_df["reaction_plot"] = sankey_df["reaction_class_clean"].where(sankey_df["reaction_class_clean"].isin(top_reactions), "Other reaction types")

    reagent_nodes = [f"Reagent: {x}" for x in pd.Index(top_reagents + ["Other reagents"]).unique()]
    reaction_nodes = [f"Reaction: {x}" for x in pd.Index(top_reactions + ["Other reaction types"]).unique()]
    mode_nodes = ["Mode: Flow", "Mode: Batch"]
    labels = reagent_nodes + reaction_nodes + mode_nodes
    node_index = {label: i for i, label in enumerate(labels)}

    src = []
    tgt = []
    val = []
    for (reagent, reaction), count in sankey_df.groupby(["reagent_plot", "reaction_plot"]).size().items():
        src.append(node_index[f"Reagent: {reagent}"])
        tgt.append(node_index[f"Reaction: {reaction}"])
        val.append(int(count))
    for (reaction, mode), count in sankey_df.groupby(["reaction_plot", "mode_clean"]).size().items():
        src.append(node_index[f"Reaction: {reaction}"])
        tgt.append(node_index[f"Mode: {mode}"])
        val.append(int(count))

    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                node=dict(
                    pad=16,
                    thickness=18,
                    line=dict(color="rgba(80,80,80,0.3)", width=0.5),
                    label=labels,
                    color=["#9ecae1"] * len(reagent_nodes) + ["#6baed6"] * len(reaction_nodes) + ["#3182bd", "#e6550d"],
                ),
                link=dict(source=src, target=tgt, value=val, color="rgba(100,149,237,0.25)"),
            )
        ]
    )
    fig.update_layout(
        title="My Dataset: Reagent Family -> Reaction Type -> Mode",
        font=dict(size=12),
        width=1200,
        height=700,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    fig.write_html(SANKEY_HTML, include_plotlyjs="cdn")
    fig.write_image(SANKEY_PNG, scale=2)


def main() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    COMPARE_DIR.mkdir(parents=True, exist_ok=True)

    my_df = pd.read_excel(MY_CLEANED_XLSX)
    ord_df = pd.read_csv(ORD_RICH_CSV, low_memory=False)

    if "reagent_family" not in my_df.columns:
        my_df = rescue_my_reagent_family(my_df)
    ord_df["reagent_family"] = ord_df["matched_components_json"].map(classify_ord_reagent_family)
    my_df["reaction_class_clean"] = my_df["reaction_class"].fillna("Unspecified")

    my_mode_series = my_mode(my_df["flow_batch_label_v1"])
    ord_mode_series = ord_mode(ord_df)

    save_mode_plot(my_mode_series, ord_mode_series)
    save_reagent_plot(my_df["reagent_family"], ord_df["reagent_family"])
    save_heatmap(my_df)
    save_sankey(my_df)

    summary = pd.DataFrame(
        [
            {"metric": "my_flow_pct", "value": float((my_mode_series == "Flow").mean() * 100)},
            {"metric": "my_batch_pct", "value": float((my_mode_series == "Batch").mean() * 100)},
            {"metric": "ord_flow_pct", "value": float((ord_mode_series == "Flow").mean() * 100)},
            {"metric": "ord_batch_pct", "value": float((ord_mode_series == "Batch").mean() * 100)},
            {"metric": "my_unique_reagent_families", "value": int(my_df["reagent_family"].nunique())},
            {"metric": "ord_unique_reagent_families", "value": int(ord_df["reagent_family"].nunique())},
            {"metric": "my_unclassified_after_v3", "value": int((my_df["reagent_family"] == "Unclassified").sum())},
        ]
    )
    summary.to_csv(SUMMARY_CSV, index=False)

    print(f"Saved mode plot: {MODE_PLOT}")
    print(f"Saved reagent plot: {REAGENT_PLOT}")
    print(f"Saved heatmap: {HEATMAP_PNG}")
    print(f"Saved sankey html: {SANKEY_HTML}")
    print(f"Saved sankey png: {SANKEY_PNG}")
    print(f"Used cleaned workbook: {MY_CLEANED_XLSX}")
    print(f"Saved summary: {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
