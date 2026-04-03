"""
Phase B: Arrhenius fitting of decomposition rate constants.

For each intermediate, fits ln(k_d) vs 1/T to extract:
  - Ea_decomp (activation energy of decomposition, kJ/mol)
  - ln_A (log of pre-exponential factor)

Input:  data/ml_lifetime/phase_a_halflives.csv
Output: data/ml_lifetime/phase_b_arrhenius.csv
        data/ml_lifetime/phase_b_arrhenius_plot.png

Usage:
  cd /Users/zhaowenyuan/Projects/FlowFigTabMiner
  python scripts/ml_lifetime/phase_b_arrhenius.py
"""

import os
import csv
import numpy as np
from scipy.stats import linregress
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

INPUT_CSV = os.path.join(PROJECT_ROOT, "data/ml_lifetime/phase_a_halflives.csv")
OUTPUT_CSV = os.path.join(PROJECT_ROOT, "data/ml_lifetime/phase_b_arrhenius.csv")
PLOT_PATH = os.path.join(PROJECT_ROOT, "data/ml_lifetime/phase_b_arrhenius_plot.png")

R = 8.314e-3  # kJ/(mol·K)

# Minimum number of temperature points for Arrhenius fitting
MIN_T_POINTS = 2


def load_phase_a():
    """Load Phase A results, grouped by intermediate."""
    by_inter = defaultdict(list)
    with open(INPUT_CSV) as f:
        for r in csv.DictReader(f):
            if r["model"] != "competing" or not r["k_d"]:
                continue
            smi = r["intermediate_smiles"]
            by_inter[smi].append({
                "intermediate": r["intermediate"],
                "T_K": float(r["T_K"]),
                "T_C": float(r["T_C"]),
                "k_d": float(r["k_d"]),
                "t_half_s": float(r["t_half_s"]),
                "r2_fit": float(r["fit_r2"]),
                "paper": r["paper"],
            })
    return by_inter


def arrhenius_fit(T_K_arr, k_d_arr):
    """Fit ln(k_d) = ln(A) - Ea/(R*T). Returns (Ea_kJ_mol, ln_A, r2)."""
    inv_T = 1.0 / np.array(T_K_arr)
    ln_k = np.log(np.array(k_d_arr))

    result = linregress(inv_T, ln_k)
    # slope = -Ea/R → Ea = -slope * R
    Ea = -result.slope * R
    ln_A = result.intercept
    r2 = result.rvalue ** 2

    return Ea, ln_A, r2


def plot_arrhenius(results):
    """Plot Arrhenius fits for all intermediates."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available, skipping plot")
        return

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    for (inter_name, data), color in zip(results.items(), colors):
        inv_T = 1000.0 / np.array(data["T_K"])  # 1000/T for readability
        ln_k = np.log(np.array(data["k_d"]))

        # Data points
        label = f"{inter_name[:35]} (Ea={data['Ea']:.1f} kJ/mol)"
        ax.scatter(inv_T, ln_k, color=color, s=40, zorder=5)

        # Fit line
        inv_T_fine = np.linspace(inv_T.min() - 0.2, inv_T.max() + 0.2, 50)
        ln_k_fit = data["ln_A"] - data["Ea"] / (R * 1000.0 / inv_T_fine)
        ax.plot(inv_T_fine, ln_k_fit, color=color, linewidth=1.5, label=label, alpha=0.8)

    ax.set_xlabel("1000 / T  (K⁻¹)", fontsize=12)
    ax.set_ylabel("ln(k_d)  (s⁻¹)", fontsize=12)
    ax.set_title("Arrhenius Plot: Organolithium Intermediate Decomposition", fontsize=13)
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)

    # Add top x-axis with temperature in °C
    ax2 = ax.twiny()
    temps_C = [-78, -60, -40, -20, 0, 20]
    temps_inv = [1000.0 / (t + 273.15) for t in temps_C]
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(temps_inv)
    ax2.set_xticklabels([f"{t}°C" for t in temps_C])

    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)
    plt.close(fig)
    print(f"  Saved Arrhenius plot to {PLOT_PATH}")


def main():
    by_inter = load_phase_a()
    print(f"[Phase B] Loaded decay data for {len(by_inter)} intermediates")

    output_rows = []
    plot_data = {}

    for smi, points in sorted(by_inter.items(), key=lambda x: -len(x[1])):
        inter_name = points[0]["intermediate"]
        paper = points[0]["paper"]

        # Use all Phase A competing-kinetics fits (Arrhenius R² is the real QC)
        good_pts = points
        if len(good_pts) < MIN_T_POINTS:
            print(f"  SKIP {inter_name[:40]}: only {len(good_pts)} good T points")
            continue

        T_K = [p["T_K"] for p in good_pts]
        k_d = [p["k_d"] for p in good_pts]
        T_C = [p["T_C"] for p in good_pts]

        Ea, ln_A, r2 = arrhenius_fit(T_K, k_d)

        # Predicted t₁/₂ at standard temperatures
        t_half_m78 = np.log(2) / np.exp(ln_A - Ea / (R * (-78 + 273.15)))
        t_half_m40 = np.log(2) / np.exp(ln_A - Ea / (R * (-40 + 273.15)))
        t_half_0 = np.log(2) / np.exp(ln_A - Ea / (R * (0 + 273.15)))
        t_half_25 = np.log(2) / np.exp(ln_A - Ea / (R * (25 + 273.15)))

        row = {
            "intermediate": inter_name,
            "intermediate_smiles": smi,
            "Ea_decomp_kJ_mol": f"{Ea:.2f}",
            "ln_A": f"{ln_A:.4f}",
            "arrhenius_r2": f"{r2:.4f}",
            "n_temperatures": len(good_pts),
            "T_range_C": f"{min(T_C):.0f} to {max(T_C):.0f}",
            "t_half_m78C_s": f"{t_half_m78:.4g}",
            "t_half_m40C_s": f"{t_half_m40:.4g}",
            "t_half_0C_s": f"{t_half_0:.4g}",
            "t_half_25C_s": f"{t_half_25:.4g}",
            "paper": paper,
        }
        output_rows.append(row)

        plot_data[inter_name] = {
            "T_K": np.array(T_K),
            "k_d": np.array(k_d),
            "Ea": Ea,
            "ln_A": ln_A,
        }

    # Write CSV
    fieldnames = [
        "intermediate", "intermediate_smiles",
        "Ea_decomp_kJ_mol", "ln_A", "arrhenius_r2", "n_temperatures", "T_range_C",
        "t_half_m78C_s", "t_half_m40C_s", "t_half_0C_s", "t_half_25C_s",
        "paper",
    ]
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    # Report
    print(f"\n{'=' * 70}")
    print(f"PHASE B RESULTS: {OUTPUT_CSV}")
    print(f"{'=' * 70}")
    print(f"Intermediates fitted: {len(output_rows)}")

    print(f"\n── Arrhenius parameters ──")
    print(f"{'Intermediate':<45s} {'Ea (kJ/mol)':>12s} {'R²':>8s} {'n_T':>4s} {'t₁/₂@-40°C':>12s} {'t₁/₂@0°C':>12s}")
    print("-" * 95)
    for r in sorted(output_rows, key=lambda x: float(x["Ea_decomp_kJ_mol"])):
        print(f"{r['intermediate'][:45]:<45s} "
              f"{float(r['Ea_decomp_kJ_mol']):>12.1f} "
              f"{float(r['arrhenius_r2']):>8.3f} "
              f"{r['n_temperatures']:>4} "
              f"{float(r['t_half_m40C_s']):>12.4g} "
              f"{float(r['t_half_0C_s']):>12.4g}")

    # Stability ranking at -40°C
    print(f"\n── Stability ranking at -40°C (by t₁/₂) ──")
    ranked = sorted(output_rows, key=lambda x: float(x["t_half_m40C_s"]), reverse=True)
    for i, r in enumerate(ranked):
        t_h = float(r["t_half_m40C_s"])
        if t_h > 60:
            t_str = f"{t_h/60:.1f} min"
        elif t_h > 1:
            t_str = f"{t_h:.1f} s"
        else:
            t_str = f"{t_h*1000:.1f} ms"
        print(f"  {i+1}. {r['intermediate'][:50]:<50s} t₁/₂ = {t_str}")

    print(f"{'=' * 70}")

    # Plot
    plot_arrhenius(plot_data)


if __name__ == "__main__":
    main()
