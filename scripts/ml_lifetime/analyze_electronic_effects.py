"""
Electronic Effect Analysis for Organolithium Intermediate Stability.

Generates:
  1. Hammett sigma vs Ea plot (ArLi subset)
  2. Decomposition mechanism classification
  3. sp2 vs sp3 comparison
  4. Position effect (o/m/p-CN)
  5. Stability ranking with reactor zone annotations
  6. Ea vs ln_A compensation effect plot

Output: data/ml_lifetime/analysis_figures/*.png
        data/ml_lifetime/electronic_analysis.csv

Usage:
  cd /Users/zhaowenyuan/Projects/FlowFigTabMiner
  python scripts/ml_lifetime/analyze_electronic_effects.py
"""

import os, csv
import numpy as np
from scipy.stats import pearsonr, linregress

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARRHENIUS_CSV = os.path.join(PROJECT_ROOT, "data/ml_lifetime/phase_b_arrhenius.csv")
HALFLIFE_CSV = os.path.join(PROJECT_ROOT, "data/ml_lifetime/phase_a_halflives.csv")
PLOT_DIR = os.path.join(PROJECT_ROOT, "data/ml_lifetime/analysis_figures")
OUTPUT_CSV = os.path.join(PROJECT_ROOT, "data/ml_lifetime/electronic_analysis.csv")

os.makedirs(PLOT_DIR, exist_ok=True)

# ── Hammett sigma + mechanism assignments ──

INTERMEDIATE_PROPERTIES = {
    "p-lithiobenzonitrile (from 1c)": {
        "sigma": 0.66, "category": "ArLi-EWG", "ewg": "CN (para)",
        "mechanism": "conjugation-stabilized", "hybridization": "sp2",
        "short": "p-CN-ArLi",
    },
    "m-lithiobenzonitrile (from 1b)": {
        "sigma": 0.56, "category": "ArLi-EWG", "ewg": "CN (meta)",
        "mechanism": "conjugation-stabilized", "hybridization": "sp2",
        "short": "m-CN-ArLi",
    },
    "tert-butyl 4-(lithio)benzoate (aryllithium)": {
        "sigma": 0.45, "category": "ArLi-EWG", "ewg": "COOtBu (para)",
        "mechanism": "conjugation-stabilized", "hybridization": "sp2",
        "short": "p-ester-ArLi",
    },
    "tert-butyl o-(lithio)benzoate (aryllithium)": {
        "sigma": 0.45, "category": "ArLi-EWG", "ewg": "COOtBu (ortho)",
        "mechanism": "conjugation-stabilized", "hybridization": "sp2",
        "short": "o-ester-ArLi",
    },
    "aryllithium (then borylated to arylboronate)": {
        "sigma": 0.0, "category": "ArLi-plain", "ewg": "none",
        "mechanism": "protonation/polymerization", "hybridization": "sp2",
        "short": "PhLi (plain)",
    },
    "(Br, Li substituents on benzene ring)": {
        "sigma": 0.39, "category": "ArLi-halo", "ewg": "Br (ortho)",
        "mechanism": "benzyne elimination", "hybridization": "sp2",
        "short": "o-Br-ArLi",
    },
    "(I, Li substituents on benzene ring)": {
        "sigma": 0.35, "category": "ArLi-halo", "ewg": "I (ortho)",
        "mechanism": "benzyne elimination", "hybridization": "sp2",
        "short": "o-I-ArLi",
    },
    "oxiranyllithium": {
        "sigma": None, "category": "Epoxy-Li", "ewg": "epoxide",
        "mechanism": "ring-opening", "hybridization": "sp3",
        "short": "oxiranyl-Li",
    },
    "Li-CH2-F (fluoromethyllithium), lifetime 13 ms at -60 \u00b0C": {
        "sigma": None, "category": "Carbenoid", "ewg": "F",
        "mechanism": "alpha-elimination", "hybridization": "sp3",
        "short": "LiCH\u2082F",
    },
    "CHLi(I)(F) (iodofluoromethyllithium), lifetime 82 ms at -40 \u00b0C": {
        "sigma": None, "category": "Carbenoid", "ewg": "I + F",
        "mechanism": "alpha-elimination", "hybridization": "sp3",
        "short": "CHLi(I)(F)",
    },
    "chloroiodomethyllithium (CHLi(I)(Cl))": {
        "sigma": None, "category": "Carbenoid", "ewg": "I + Cl",
        "mechanism": "alpha-elimination", "hybridization": "sp3",
        "short": "CHLi(I)(Cl)",
    },
}


def load_arrhenius():
    rows = []
    with open(ARRHENIUS_CSV) as f:
        for r in csv.DictReader(f):
            rows.append({
                "intermediate": r["intermediate"],
                "Ea": float(r["Ea_decomp_kJ_mol"]),
                "ln_A": float(r["ln_A"]),
                "t_half_m40": float(r["t_half_m40C_s"]),
                "r2": float(r["arrhenius_r2"]),
                "n_T": int(r["n_temperatures"]),
            })
    return rows


def load_halflives():
    rows = []
    with open(HALFLIFE_CSV) as f:
        for r in csv.DictReader(f):
            if r["model"] == "competing" and r["t_half_s"]:
                rows.append({
                    "intermediate": r["intermediate"],
                    "T_C": float(r["T_C"]),
                    "t_half": float(r["t_half_s"]),
                })
    return rows


# ── Plot 1: Hammett sigma vs Ea ──

def plot_hammett(data):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Panel (a): sigma vs Ea for ArLi
    arli = [d for d in data if d["props"]["sigma"] is not None]

    sigmas = [d["props"]["sigma"] for d in arli]
    eas = [d["Ea"] for d in arli]
    names = [d["props"]["short"] for d in arli]
    cats = [d["props"]["category"] for d in arli]

    cat_colors = {
        "ArLi-EWG": "#2171B5", "ArLi-plain": "#969696",
        "ArLi-halo": "#E6550D",
    }
    colors = [cat_colors.get(c, "#999") for c in cats]

    ax1.scatter(sigmas, eas, s=120, c=colors, edgecolors="black", zorder=5)

    for i, name in enumerate(names):
        offset = (8, 8) if "Br" not in name else (8, -12)
        ax1.annotate(name, (sigmas[i], eas[i]), xytext=offset,
                     textcoords="offset points", fontsize=8, fontweight="bold")

    # Fit line excluding benzyne outliers
    non_benzyne = [(s, e) for d, s, e in zip(arli, sigmas, eas)
                   if d["props"]["mechanism"] != "benzyne elimination"]
    if len(non_benzyne) >= 3:
        s_fit, e_fit = zip(*non_benzyne)
        slope, intercept, r, p, se = linregress(s_fit, e_fit)
        x_line = np.linspace(-0.1, 0.75, 50)
        y_line = slope * x_line + intercept
        ax1.plot(x_line, y_line, "b--", alpha=0.5, linewidth=1.5,
                 label=f"EWG trend: Ea = {slope:.1f}\u03c3 + {intercept:.1f}\n(r={r:.2f}, excl. benzyne)")

    # Mark benzyne zone
    ax1.axhspan(70, 90, alpha=0.1, color="red")
    ax1.text(0.05, 84, "benzyne\nelimination\nzone", fontsize=8, color="red", style="italic")

    ax1.set_xlabel("Hammett \u03c3", fontsize=12)
    ax1.set_ylabel("Ea (kJ/mol)", fontsize=12)
    ax1.set_title("(a) Hammett \u03c3 vs Decomposition Ea\n(ArLi intermediates only)", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    # Panel (b): sp2 vs sp3 comparison
    sp2 = [d for d in data if d["props"]["hybridization"] == "sp2"]
    sp3 = [d for d in data if d["props"]["hybridization"] == "sp3"]

    sp2_eas = [d["Ea"] for d in sp2]
    sp3_eas = [d["Ea"] for d in sp3]
    sp2_names = [d["props"]["short"] for d in sp2]
    sp3_names = [d["props"]["short"] for d in sp3]

    y_pos_sp2 = range(len(sp2_eas))
    y_pos_sp3 = range(len(sp3_eas))

    ax2.barh([y + 0.15 for y in y_pos_sp2], sp2_eas, 0.3,
             label=f"C(sp\u00b2)-Li (n={len(sp2)})", color="#2171B5", edgecolor="white")
    ax2.barh([y - 0.15 + len(sp2) + 0.5 for y in y_pos_sp3], sp3_eas, 0.3,
             label=f"C(sp\u00b3)-Li (n={len(sp3)})", color="#E6550D", edgecolor="white")

    all_names = sp2_names + [""] + sp3_names
    all_y = list(range(len(sp2))) + [len(sp2) + 0.2] + [y + len(sp2) + 0.5 for y in range(len(sp3))]
    ax2.set_yticks(all_y)
    ax2.set_yticklabels(all_names, fontsize=8)
    ax2.invert_yaxis()
    ax2.axhline(len(sp2) + 0.2, color="gray", linewidth=0.5, linestyle="--")
    ax2.set_xlabel("Ea (kJ/mol)", fontsize=12)
    ax2.set_title("(b) C(sp\u00b2)-Li vs C(sp\u00b3)-Li\nDecomposition Ea", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9, loc="lower right")
    ax2.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    path = os.path.join(PLOT_DIR, "hammett_and_hybridization.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Plot 2: Stability ranking with reactor zones ──

def plot_stability_reactor_zones(data):
    fig, ax = plt.subplots(figsize=(11, 7))

    sorted_data = sorted(data, key=lambda d: d["t_half_m40"])
    names = [d["props"]["short"] for d in sorted_data]
    log_t = [np.log10(d["t_half_m40"]) for d in sorted_data]
    mechs = [d["props"]["mechanism"] for d in sorted_data]
    eas = [d["Ea"] for d in sorted_data]

    mech_colors = {
        "conjugation-stabilized": "#2171B5",
        "benzyne elimination": "#E6550D",
        "alpha-elimination": "#9E9AC8",
        "ring-opening": "#6A51A3",
        "protonation/polymerization": "#969696",
    }
    colors = [mech_colors.get(m, "#CCC") for m in mechs]

    bars = ax.barh(range(len(names)), log_t, color=colors, edgecolor="white", height=0.7)

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10, fontweight="bold")
    ax.set_xlabel("log\u2081\u2080(t\u00bd at -40\u00b0C)  [s]", fontsize=12)
    ax.set_title("Organolithium Intermediate Stability Ranking\nwith Reactor Type Zones & Decomposition Mechanism",
                  fontsize=13, fontweight="bold")

    # Reactor zones
    ax.axvspan(np.log10(60), 4, alpha=0.08, color="green")
    ax.axvspan(np.log10(1), np.log10(60), alpha=0.08, color="blue")
    ax.axvspan(-4, np.log10(1), alpha=0.08, color="red")

    ax.text(np.log10(200), len(names) - 0.3, "Batch\ncompatible",
            fontsize=9, color="green", fontweight="bold", ha="center")
    ax.text(np.log10(8), len(names) - 0.3, "Standard\nflow",
            fontsize=9, color="blue", fontweight="bold", ha="center")
    ax.text(np.log10(0.03), len(names) - 0.3, "Flash chemistry\nrequired",
            fontsize=9, color="red", fontweight="bold", ha="center")

    # Add Ea and t_half labels
    for i, (bar, ea, d) in enumerate(zip(bars, eas, sorted_data)):
        th = d["t_half_m40"]
        if th > 1:
            t_str = f"{th:.1f} s"
        else:
            t_str = f"{th*1000:.1f} ms"
        ax.text(bar.get_width() + 0.08, i, f"{t_str}  (Ea={ea:.1f})",
                va="center", fontsize=8, fontweight="bold")

    # Legend for mechanisms
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=m) for m, c in mech_colors.items()]
    ax.legend(handles=legend_elements, fontsize=8, loc="lower right",
              title="Decomposition mechanism", title_fontsize=9)

    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(PLOT_DIR, "stability_ranking_reactor_zones.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Plot 3: Position effect (o/m/p-CN) ──

def plot_position_effect(halflives):
    fig, ax = plt.subplots(figsize=(7, 5))

    cn_data = {}
    for r in halflives:
        if "lithiobenzonitrile" in r["intermediate"]:
            if "p-" in r["intermediate"]:
                pos = "para"
            elif "m-" in r["intermediate"]:
                pos = "meta"
            elif "o-" in r["intermediate"]:
                pos = "ortho"
            else:
                continue
            T = r["T_C"]
            if pos not in cn_data:
                cn_data[pos] = {}
            cn_data[pos][T] = r["t_half"]

    # Plot at 20°C (all three have data)
    positions = ["ortho", "meta", "para"]
    colors_pos = {"ortho": "#E6550D", "meta": "#2171B5", "para": "#31A354"}

    if 20.0 in cn_data.get("ortho", {}):
        t20 = [cn_data[p].get(20.0, 0) for p in positions]
        bars = ax.bar(positions, t20, color=[colors_pos[p] for p in positions],
                      edgecolor="black", width=0.5)
        for bar, t in zip(bars, t20):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                    f"{t:.1f} s", ha="center", fontsize=11, fontweight="bold")

    ax.set_ylabel("t\u00bd at 20\u00b0C (s)", fontsize=12)
    ax.set_title("Position Effect on Lithiobenzonitrile Stability\n(o- vs m- vs p-CN substituent)",
                  fontsize=12, fontweight="bold")
    ax.set_ylim(0, max(t20) * 1.3 if t20 else 150)
    ax.grid(axis="y", alpha=0.3)

    ax.text(0.95, 0.9, "ortho most stable\n(possible Li\u00b7\u00b7\u00b7N chelation)",
            transform=ax.transAxes, fontsize=9, ha="right", style="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    fig.tight_layout()
    path = os.path.join(PLOT_DIR, "position_effect_cn.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Plot 4: Ea vs ln_A (compensation effect) ──

def plot_ea_lna_compensation(data):
    fig, ax = plt.subplots(figsize=(8, 6))

    eas = [d["Ea"] for d in data]
    lnAs = [d["ln_A"] for d in data]
    names = [d["props"]["short"] for d in data]
    mechs = [d["props"]["mechanism"] for d in data]

    mech_colors = {
        "conjugation-stabilized": "#2171B5",
        "benzyne elimination": "#E6550D",
        "alpha-elimination": "#9E9AC8",
        "ring-opening": "#6A51A3",
        "protonation/polymerization": "#969696",
    }
    colors = [mech_colors.get(m, "#CCC") for m in mechs]

    ax.scatter(eas, lnAs, s=120, c=colors, edgecolors="black", zorder=5)

    for i, name in enumerate(names):
        ax.annotate(name, (eas[i], lnAs[i]), xytext=(6, 6),
                     textcoords="offset points", fontsize=7, fontweight="bold")

    # Fit line (isokinetic relationship)
    slope, intercept, r, p, se = linregress(eas, lnAs)
    x_line = np.linspace(min(eas) - 5, max(eas) + 5, 50)
    ax.plot(x_line, slope * x_line + intercept, "k--", alpha=0.4,
            label=f"Compensation: ln(A) = {slope:.2f}\u00b7Ea + {intercept:.1f}\n(r={r:.2f}, p={p:.3f})")

    ax.set_xlabel("Ea (kJ/mol)", fontsize=12)
    ax.set_ylabel("ln(A) (s\u207b\u00b9)", fontsize=12)
    ax.set_title("Ea vs ln(A): Enthalpy-Entropy Compensation\nin Organolithium Decomposition",
                  fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    path = os.path.join(PLOT_DIR, "ea_lna_compensation.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Main ──

def main():
    arrhenius = load_arrhenius()
    halflives = load_halflives()

    # Merge with properties
    data = []
    for d in arrhenius:
        props = INTERMEDIATE_PROPERTIES.get(d["intermediate"])
        if props:
            d["props"] = props
            data.append(d)
        else:
            print(f"  [WARN] No properties for: {d['intermediate']}")

    print(f"Loaded {len(data)} intermediates with properties")

    # Save electronic analysis CSV
    csv_rows = []
    for d in data:
        row = {
            "intermediate": d["intermediate"],
            "short_name": d["props"]["short"],
            "Ea_kJ_mol": d["Ea"],
            "ln_A": d["ln_A"],
            "t_half_m40C_s": d["t_half_m40"],
            "sigma_hammett": d["props"]["sigma"] if d["props"]["sigma"] is not None else "",
            "category": d["props"]["category"],
            "ewg": d["props"]["ewg"],
            "mechanism": d["props"]["mechanism"],
            "hybridization": d["props"]["hybridization"],
        }
        # Reactor recommendation
        if d["t_half_m40"] > 60:
            row["reactor_recommendation"] = "batch_compatible"
        elif d["t_half_m40"] > 1:
            row["reactor_recommendation"] = "standard_flow"
        else:
            row["reactor_recommendation"] = "flash_chemistry"
        csv_rows.append(row)

    fieldnames = list(csv_rows[0].keys())
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Saved: {OUTPUT_CSV}")

    # Generate plots
    print("\nGenerating plots...")
    plot_hammett(data)
    plot_stability_reactor_zones(data)
    plot_position_effect(halflives)
    plot_ea_lna_compensation(data)

    # Print summary statistics
    print(f"\n{'='*60}")
    print(f"ELECTRONIC ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"{'Intermediate':<25s} {'Ea':>6s} {'ln_A':>7s} {'t½@-40':>10s} {'Mechanism':>25s} {'Reactor':>15s}")
    print("-" * 95)
    for d in sorted(data, key=lambda x: x["t_half_m40"], reverse=True):
        th = d["t_half_m40"]
        t_str = f"{th:.1f}s" if th > 1 else f"{th*1000:.1f}ms"
        print(f"{d['props']['short']:<25s} {d['Ea']:>6.1f} {d['ln_A']:>7.2f} {t_str:>10s} "
              f"{d['props']['mechanism']:>25s} {csv_rows[[r['intermediate'] for r in csv_rows].index(d['intermediate'])]['reactor_recommendation']:>15s}")

    # Correlations
    arli = [d for d in data if d["props"]["sigma"] is not None]
    non_benzyne = [d for d in arli if d["props"]["mechanism"] != "benzyne elimination"]
    if len(non_benzyne) >= 3:
        s = [d["props"]["sigma"] for d in non_benzyne]
        e = [d["Ea"] for d in non_benzyne]
        r, p = pearsonr(s, e)
        print(f"\nHammett correlation (excl. benzyne): r={r:.3f}, p={p:.3f}")

    # Ea-lnA compensation
    all_eas = [d["Ea"] for d in data]
    all_lnAs = [d["ln_A"] for d in data]
    r_comp, p_comp = pearsonr(all_eas, all_lnAs)
    print(f"Ea-ln(A) compensation: r={r_comp:.3f}, p={p_comp:.4f}")

    print(f"{'='*60}")


if __name__ == "__main__":
    main()
