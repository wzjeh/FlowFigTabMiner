"""
OrgLi Dataset Diversity Comparison: Flow vs ORD vs USPTO.

3-panel figure showing:
  (a) OrgLi reagent type distribution
  (b) Intermediate structural class diversity (Flow-unique)
  (c) Experimental condition coverage (tR × T grid vs single-point)

Output: data/final_output/dataset_comparison/orgli_diversity_comparison.png

Usage:
  cd /Users/zhaowenyuan/Projects/FlowFigTabMiner
  python scripts/ml_lifetime/plot_orgli_diversity_comparison.py
"""

import os, sys, json, csv
import numpy as np
import pandas as pd
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

FLOW_CSV = os.path.join(PROJECT_ROOT, "data/final_output/organolithium_tr_subdataset_vlm_enriched.csv")
USPTO_CSV = os.path.join(PROJECT_ROOT, "data/input/uspto_organolithium_extracted_rich.csv")
ORD_CSV = os.path.join(PROJECT_ROOT, "other dataset/ord/organolithium/ord_organolithium_reactions_rich.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data/final_output/dataset_comparison")
OUTPUT_PNG = os.path.join(OUTPUT_DIR, "orgli_diversity_comparison.png")

COLORS = {"Flow (this work)": "#2171B5", "ORD": "#41AB5D", "USPTO": "#F16913"}


# ── Data loading ──

def load_flow_reagents():
    df = pd.read_csv(FLOW_CSV)
    mapping = {
        "s-BuLi": "s-BuLi", "n-BuLi": "n-BuLi", "PhLi": "PhLi",
        "LDA": "LDA", "lithium naphthalenide (LiNp)": "LiNp",
    }
    return Counter(mapping.get(r, "other") for r in df["organolithium_reagent"].dropna())


def load_uspto_reagents():
    df = pd.read_csv(USPTO_CSV)
    def classify(frag):
        f = str(frag).replace(" ", "")
        if f in ("C(CCC)[Li]", "[Li]CCCC", "CCCC[Li]"): return "n-BuLi"
        if f in ("C(C)(CC)[Li]", "[Li]C(C)CC", "CCC([Li])C", "CC([Li])CC"): return "s-BuLi"
        if f in ("C(C)(C)(C)[Li]", "[Li]C(C)(C)C", "CC(C)(C)[Li]"): return "t-BuLi"
        if f in ("C[Li]", "[Li]C"): return "MeLi"
        if "C1" in f and "[Li]" in f and len(f) < 25: return "PhLi"
        return "other"
    return Counter(classify(f) for f in df["matched_fragment"].dropna())


def load_ord_reagents():
    df = pd.read_csv(ORD_CSV, low_memory=False)
    counts = Counter()
    for mc in df["matched_components_json"].dropna():
        try:
            for comp in json.loads(mc):
                val = comp.get("matched_value", "").lower().strip()
                if "n-butyllithium" in val: counts["n-BuLi"] += 1
                elif "sec-butyllithium" in val: counts["s-BuLi"] += 1
                elif "tert-butyllithium" in val or "t-butyllithium" in val: counts["t-BuLi"] += 1
                elif "methyllithium" in val: counts["MeLi"] += 1
                elif "phenyllithium" in val: counts["PhLi"] += 1
                elif "lithium" in val: counts["other"] += 1
                break
        except: pass
    return counts


def load_flow_intermediates():
    """Classify flow intermediates into structural types."""
    df = pd.read_csv(FLOW_CSV)
    classes = Counter()
    for _, row in df.iterrows():
        inter = str(row.get("intermediate", ""))
        smi = str(row.get("intermediate_smiles", ""))
        if "benzonitrile" in inter or "C#N" in smi:
            classes["ArLi-EWG\n(CN)"] += 1
        elif "benzoate" in inter or "OC(=O)" in smi:
            classes["ArLi-EWG\n(ester)"] += 1
        elif "Br" in inter and "Li" in inter and "benzene" in inter:
            classes["ArLi-halo\n(Br)"] += 1
        elif "I" in inter and "Li" in inter and "benzene" in inter:
            classes["ArLi-halo\n(I)"] += 1
        elif "oxiranyl" in inter or "C1CO1" in smi:
            classes["Epoxy-Li"] += 1
        elif "carbenoid" in inter.lower() or ("CHLi" in inter and ("F" in inter or "Cl" in inter)):
            classes["Carbenoid\n(sp3)"] += 1
        elif "fluoromethyl" in inter.lower():
            classes["Carbenoid\n(sp3)"] += 1
        elif "benzyl" in inter.lower():
            classes["BenzylLi"] += 1
        elif "methoxyphenyl" in inter or "phenyllithium" in inter.lower() or inter == "aryllithium":
            classes["ArLi\n(plain)"] += 1
        elif "glyco" in inter.lower():
            classes["ArLi\n(glycosyl)"] += 1
        elif "alkyllithium" in inter.lower():
            classes["AlkylLi\n(functional)"] += 1
        else:
            classes["Other\nArLi"] += 1
    return classes


def load_flow_coverage():
    """Get (n_temperatures, n_tR_levels) per intermediate."""
    df = pd.read_csv(FLOW_CSV)
    by_inter = defaultdict(lambda: {"temps": set(), "trs": set()})
    for _, row in df.iterrows():
        inter = row.get("intermediate", "")
        t = row.get("T1_C", "") or row.get("T2_C", "")
        tr = row.get("tR1_s", "") or row.get("tR2_s", "")
        if inter and t and tr:
            by_inter[inter]["temps"].add(str(t))
            by_inter[inter]["trs"].add(str(tr))
    return {k: (len(v["temps"]), len(v["trs"])) for k, v in by_inter.items()}


# ── Plotting ──

def plot_panel_a(ax, flow, ord_, uspto):
    """Panel (a): Reagent type distribution grouped bar chart."""
    reagents = ["n-BuLi", "s-BuLi", "t-BuLi", "PhLi", "MeLi", "LDA", "LiNp", "other"]

    def to_pct(counts):
        total = sum(counts.values())
        return [counts.get(r, 0) / total * 100 for r in reagents]

    flow_pct = to_pct(flow)
    ord_pct = to_pct(ord_)
    uspto_pct = to_pct(uspto)

    x = np.arange(len(reagents))
    w = 0.25
    bars1 = ax.bar(x - w, flow_pct, w, label="Flow (this work)", color=COLORS["Flow (this work)"], edgecolor="white", zorder=3)
    bars2 = ax.bar(x, ord_pct, w, label="ORD", color=COLORS["ORD"], edgecolor="white", zorder=3)
    bars3 = ax.bar(x + w, uspto_pct, w, label="USPTO", color=COLORS["USPTO"], edgecolor="white", zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(reagents, fontsize=9, rotation=30, ha="right")
    ax.set_ylabel("Fraction (%)", fontsize=11)
    ax.set_title("(a) OrgLi Reagent Distribution", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(0, 85)
    ax.grid(axis="y", alpha=0.3, zorder=0)

    # Annotate Flow-unique reagents
    for i, r in enumerate(reagents):
        if r in ("LDA", "LiNp") and flow_pct[i] > 0:
            ax.annotate("Flow\nonly", xy=(x[i] - w, flow_pct[i]),
                        xytext=(0, 8), textcoords="offset points",
                        fontsize=7, color=COLORS["Flow (this work)"],
                        ha="center", fontweight="bold")

    # Add dataset size annotations
    ax.text(0.98, 0.85, f"Flow: {sum(flow.values()):,} rxns\nORD: {sum(ord_.values()):,} rxns\nUSPTO: {sum(uspto.values()):,} rxns",
            transform=ax.transAxes, fontsize=8, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))


def plot_panel_b(ax, inter_classes):
    """Panel (b): Intermediate structural diversity."""
    # Sort by count
    sorted_classes = sorted(inter_classes.items(), key=lambda x: -x[1])
    names = [k for k, v in sorted_classes]
    counts = [v for k, v in sorted_classes]

    # Color by category
    color_map = {
        "ArLi-EWG\n(CN)": "#2171B5", "ArLi-EWG\n(ester)": "#4292C6",
        "ArLi-halo\n(Br)": "#E6550D", "ArLi-halo\n(I)": "#FD8D3C",
        "ArLi\n(plain)": "#969696", "Other\nArLi": "#BDBDBD",
        "Epoxy-Li": "#6A51A3", "Carbenoid\n(sp3)": "#9E9AC8",
        "BenzylLi": "#31A354", "ArLi\n(glycosyl)": "#74C476",
        "AlkylLi\n(functional)": "#756BB1",
    }
    colors = [color_map.get(n, "#CCCCCC") for n in names]

    bars = ax.barh(range(len(names)), counts, color=colors, edgecolor="white")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Data Points", fontsize=11)
    ax.set_title("(b) Intermediate Structural Classes\n(Flow dataset — tracked for first time)", fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    # Add count labels
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                str(cnt), va="center", fontsize=8)

    # Add annotation for ORD/USPTO
    ax.text(0.95, 0.95,
            "ORD / USPTO:\nOrgLi recorded as\nreagent only (n-BuLi)\n— intermediate not tracked",
            transform=ax.transAxes, fontsize=8, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFE0E0", alpha=0.9),
            style="italic")


def plot_panel_c(ax, coverage):
    """Panel (c): Condition coverage (tR × T grid)."""
    names = []
    n_temps = []
    n_trs = []
    for inter, (nt, ntr) in sorted(coverage.items(), key=lambda x: -(x[1][0] * x[1][1])):
        if nt >= 2 and ntr >= 2:
            short = inter[:25]
            names.append(short)
            n_temps.append(nt)
            n_trs.append(ntr)

    scatter = ax.scatter(n_temps, n_trs, s=120, c=COLORS["Flow (this work)"],
                          edgecolors="navy", alpha=0.7, zorder=5)

    # Annotate some points
    for i in range(min(8, len(names))):
        ax.annotate(names[i], (n_temps[i], n_trs[i]),
                     xytext=(5, 5), textcoords="offset points", fontsize=6, alpha=0.8)

    # Add batch data region
    ax.scatter([1], [1], s=200, c=COLORS["ORD"], marker="s", edgecolors="darkgreen",
               alpha=0.7, zorder=5, label="Batch (ORD/USPTO)")
    ax.annotate("Batch reactions\n(1 temp, 1 time point)", (1, 1),
                 xytext=(1.5, 2), fontsize=8, color="darkgreen",
                 arrowprops=dict(arrowstyle="->", color="darkgreen"))

    ax.set_xlabel("Temperature Levels per Intermediate", fontsize=11)
    ax.set_ylabel("Residence Time Levels per Intermediate", fontsize=11)
    ax.set_title("(c) Experimental Condition Coverage", fontsize=12, fontweight="bold")
    ax.set_xlim(0, max(n_temps) + 1)
    ax.set_ylim(0, max(n_trs) + 1)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Shade the flow region
    ax.fill_between([1.5, max(n_temps) + 0.5], 1.5, max(n_trs) + 0.5,
                     alpha=0.08, color=COLORS["Flow (this work)"], zorder=0)
    ax.text(max(n_temps) * 0.6, max(n_trs) * 0.85, "Flow: 2D grid\n(enables kinetic fitting)",
            fontsize=9, color=COLORS["Flow (this work)"], fontweight="bold", ha="center")


def main():
    print("Loading data...")
    flow_reagents = load_flow_reagents()
    uspto_reagents = load_uspto_reagents()
    ord_reagents = load_ord_reagents()
    inter_classes = load_flow_intermediates()
    coverage = load_flow_coverage()

    print(f"  Flow reagents: {dict(flow_reagents)}")
    print(f"  USPTO reagents: {dict(uspto_reagents)}")
    print(f"  ORD reagents: {dict(ord_reagents)}")
    print(f"  Intermediate classes: {dict(inter_classes)}")
    print(f"  Coverage: {len(coverage)} intermediates with 2D data")

    # Create figure
    fig = plt.figure(figsize=(18, 6.5), constrained_layout=True)
    gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1.1, 1.0, 1.0])

    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])
    ax_c = fig.add_subplot(gs[2])

    plot_panel_a(ax_a, flow_reagents, ord_reagents, uspto_reagents)
    plot_panel_b(ax_b, inter_classes)
    plot_panel_c(ax_c, coverage)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_PNG.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved: {OUTPUT_PNG}")
    print(f"Saved: {OUTPUT_PNG.replace('.png', '.pdf')}")


if __name__ == "__main__":
    main()
