#!/usr/bin/env python3
"""Plot flow/batch mode, reaction type, and organolithium reagent composition comparisons."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import PercentFormatter


ROOT = Path("/Users/zhaowenyuan/Projects/FlowFigTabMiner")
COMPARE_DIR = ROOT / "data" / "final_output" / "dataset_comparison"
MY_CLEANED_XLSX = COMPARE_DIR / "ALL_organolithium_normalized_20260327_1022_cleaned_v1.xlsx"
ORD_RICH_CSV = ROOT / "other dataset" / "ord" / "organolithium" / "ord_organolithium_reactions_rich.csv"

MODE_PLOT = COMPARE_DIR / "organolithium_mode_comparison.png"
REAGENT_PLOT = COMPARE_DIR / "organolithium_reagent_composition_comparison.png"
REACTIONTYPE_PLOT = COMPARE_DIR / "my_dataset_reaction_type_composition.png"
SUMMARY_CSV = COMPARE_DIR / "organolithium_mode_reactiontype_reagent_summary.csv"


def classify_reagent_family_from_text(text: str) -> str | None:
    if not text:
        return None
    t = str(text).lower().strip()
    if not t or t == "nan":
        return None

    patterns = [
        ("n-BuLi", [r"\bn-?butyl ?lithium\b", r"\bnbuli\b", r"\bn-buli\b", r"\bn-butyllithium\b"]),
        ("s-BuLi", [r"\bsec-?butyl ?lithium\b", r"\bs-?butyllithium\b", r"\bsbuli\b"]),
        ("t-BuLi", [r"\btert-?butyl ?lithium\b", r"\bt-?butyllithium\b", r"\btbuli\b", r"\btertbutyllithium\b"]),
        ("MeLi", [r"\bmethyl ?lithium\b", r"\bmeli\b"]),
        ("PhLi", [r"\bphenyl ?lithium\b", r"\bphli\b"]),
        ("LDA", [r"\blithium diisopropylamide\b", r"\blda\b"]),
        ("LiHMDS", [r"\blihmds\b", r"lithium hexamethyldisilazide"]),
        ("Lithium naphthalenide", [r"lithium naphthalenide", r"naphthalenide"]),
        ("HexylLi", [r"\bhexyl ?lithium\b", r"\bn-?hexyllithium\b"]),
        ("EthylLi", [r"\bethyl ?lithium\b", r"\bethyllithium\b"]),
        ("MesLi", [r"\bmesli\b", r"\bmesityllithium\b"]),
        ("Vinyl/alkynylLi", [r"vinyllithium", r"lithio.*yne", r"lithiophenylacetylene", r"ynolate"]),
        ("Aryl/heteroarylLi", [r"aryllithium", r"lithio[a-z]", r"phenyllithium", r"lithiothiophene", r"lithiopyrid", r"lithiofuran", r"lithiotoluene"]),
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


def classify_ord_reagent_family(matched_components_json: str) -> str:
    try:
        items = json.loads(matched_components_json) if isinstance(matched_components_json, str) else []
    except Exception:
        items = []

    names = []
    smiles = []
    for item in items:
        value = str(item.get("matched_value", ""))
        if item.get("identifier_type") == "NAME":
            names.append(value)
        elif item.get("identifier_type") == "SMILES":
            smiles.append(value)

    for name in names:
        fam = classify_reagent_family_from_text(name)
        if fam:
            return fam
    for smi in smiles:
        fam = classify_reagent_family_from_smiles(smi)
        if fam:
            return fam
    return "Unclassified"


def normalize_mode_my(series: pd.Series) -> pd.Series:
    return series.replace({"unknown": "Not explicit"})


def infer_ord_mode(df: pd.DataFrame) -> pd.Series:
    combined = (
        df["conditions_json"].fillna("").astype(str)
        + " "
        + df["setup_json"].fillna("").astype(str)
        + " "
        + df["notes_json"].fillna("").astype(str)
    ).str.lower()
    has_flow = combined.str.contains("flow")
    has_batch = combined.str.contains("batch")
    mode = pd.Series("Not explicit", index=df.index, dtype=object)
    mode[has_batch] = "Batch"
    mode[has_flow] = "Flow"
    return mode


def composition_percent(series: pd.Series, order: list[str] | None = None) -> pd.Series:
    counts = series.value_counts(dropna=False)
    pct = counts / counts.sum() * 100.0
    if order is not None:
        pct = pct.reindex(order, fill_value=0.0)
    return pct


def main() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    COMPARE_DIR.mkdir(parents=True, exist_ok=True)

    my_df = pd.read_excel(MY_CLEANED_XLSX)
    ord_df = pd.read_csv(ORD_RICH_CSV, low_memory=False)

    my_mode = normalize_mode_my(my_df["flow_batch_label_v1"]).replace({"flow": "Flow", "batch": "Batch"})
    ord_mode = infer_ord_mode(ord_df)
    mode_order = ["Flow", "Batch", "Not explicit"]

    my_df["reagent_family"] = my_df.apply(classify_my_reagent_family, axis=1)
    ord_df["reagent_family"] = ord_df["matched_components_json"].map(classify_ord_reagent_family)

    reagent_order = [
        "n-BuLi",
        "s-BuLi",
        "t-BuLi",
        "MeLi",
        "PhLi",
        "LDA",
        "LiHMDS",
        "Lithium naphthalenide",
        "HexylLi",
        "EthylLi",
        "MesLi",
        "Vinyl/alkynylLi",
        "Aryl/heteroarylLi",
        "Other organolithium",
        "Unclassified",
    ]

    summary_rows = []
    for dataset_name, series in [
        ("My dataset mode", my_mode),
        ("ORD mode", ord_mode),
        ("My dataset reagent", my_df["reagent_family"]),
        ("ORD reagent", ord_df["reagent_family"]),
        ("My dataset reaction_class", my_df["reaction_class"].fillna("Unspecified")),
    ]:
        for label, value in composition_percent(series).items():
            summary_rows.append({"dataset_metric": dataset_name, "label": label, "percent": float(value)})
    pd.DataFrame(summary_rows).to_csv(SUMMARY_CSV, index=False)

    mode_comp = pd.DataFrame(
        {
            "My dataset": composition_percent(my_mode, mode_order),
            "ORD": composition_percent(ord_mode, mode_order),
        }
    )
    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    mode_comp.T.plot(kind="bar", stacked=True, ax=ax, color=["#1f77b4", "#ff7f0e", "#bdbdbd"], width=0.6)
    ax.set_ylabel("Record share (%)")
    ax.set_xlabel("")
    ax.set_title("Reaction Mode Annotation")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.legend(title="", loc="upper right", frameon=True)
    ax.set_ylim(0, 100)
    for idx, row in enumerate(mode_comp.T.itertuples(index=False)):
        total = 0
        for value in row:
            if value >= 6:
                ax.text(idx, total + value / 2, f"{value:.0f}%", ha="center", va="center", fontsize=10, color="white" if value > 20 else "black")
            total += value
    plt.tight_layout()
    plt.savefig(MODE_PLOT, dpi=260)
    plt.close(fig)

    reagent_comp = pd.DataFrame(
        {
            "My dataset": composition_percent(my_df["reagent_family"], reagent_order),
            "ORD": composition_percent(ord_df["reagent_family"], reagent_order),
        }
    ).loc[lambda x: x.max(axis=1) >= 1.0]
    reagent_comp = reagent_comp.sort_values("My dataset", ascending=True)
    fig, ax = plt.subplots(figsize=(10.5, 7.8))
    y = range(len(reagent_comp))
    ax.barh([i - 0.18 for i in y], reagent_comp["My dataset"], height=0.34, color="#1f77b4", label="My dataset")
    ax.barh([i + 0.18 for i in y], reagent_comp["ORD"], height=0.34, color="#2ca02c", label="ORD")
    ax.set_yticks(list(y))
    ax.set_yticklabels(reagent_comp.index)
    ax.set_xlabel("Record share (%)")
    ax.set_title("Organolithium Reagent Composition")
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.legend(frameon=True, loc="lower right")
    plt.tight_layout()
    plt.savefig(REAGENT_PLOT, dpi=260)
    plt.close(fig)

    reaction_counts = composition_percent(my_df["reaction_class"].fillna("Unspecified"))
    reaction_counts = reaction_counts[reaction_counts >= 1.0].sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    ax.barh(reaction_counts.index, reaction_counts.values, color="#1f77b4")
    ax.set_xlabel("Record share (%)")
    ax.set_title("My Dataset: Reaction Type Composition")
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.text(
        0.98,
        0.05,
        "ORD comparison is not shown here because\n"
        "the extracted ORD subset does not provide\n"
        "standardized reaction-type labels comparable\n"
        "to the curated reaction_class field.",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.9, "edgecolor": "#cccccc"},
    )
    plt.tight_layout()
    plt.savefig(REACTIONTYPE_PLOT, dpi=260)
    plt.close(fig)

    print(f"Saved mode plot: {MODE_PLOT}")
    print(f"Saved reagent plot: {REAGENT_PLOT}")
    print(f"Saved reaction-type plot: {REACTIONTYPE_PLOT}")
    print(f"Saved summary: {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
