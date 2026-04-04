"""
Final Results Summary Figure — 4-panel publication figure.

  (a) Hammett σ vs Ea with regression line + validation predictions
  (b) Arrhenius plot (ln(k_d) vs 1/T) for all 11 intermediates
  (c) Ea vs ln(A) compensation effect
  (d) Stability ranking with reactor zones + experimental predictions

Output: data/ml_lifetime/analysis_figures/final_results_summary.png

Usage:
  cd /Users/zhaowenyuan/Projects/FlowFigTabMiner
  python scripts/ml_lifetime/plot_final_results_summary.py
"""

import os, csv
import numpy as np
from scipy.stats import linregress

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARRHENIUS_CSV = os.path.join(PROJECT_ROOT, "data/ml_lifetime/phase_b_arrhenius.csv")
ELECTRONIC_CSV = os.path.join(PROJECT_ROOT, "data/ml_lifetime/electronic_analysis.csv")
HALFLIFE_CSV = os.path.join(PROJECT_ROOT, "data/ml_lifetime/phase_a_halflives.csv")
PLOT_DIR = os.path.join(PROJECT_ROOT, "data/ml_lifetime/analysis_figures")

# Validation predictions (from Hammett regression: Ea = -48.5σ + 36.1)
VALIDATION = [
    {"name": "p-CF₃-PhLi", "sigma": 0.54, "Ea_pred": -48.5*0.54+36.1, "marker": "D"},
    {"name": "p-F-PhLi", "sigma": 0.06, "Ea_pred": -48.5*0.06+36.1, "marker": "D"},
    {"name": "p-CH₃-PhLi", "sigma": -0.17, "Ea_pred": -48.5*(-0.17)+36.1, "marker": "D"},
]

R = 8.314e-3  # kJ/(mol·K)

def load_data():
    electronic = []
    with open(ELECTRONIC_CSV) as f:
        for r in csv.DictReader(f):
            electronic.append(r)

    arrhenius = []
    with open(ARRHENIUS_CSV) as f:
        for r in csv.DictReader(f):
            arrhenius.append(r)

    halflives = []
    with open(HALFLIFE_CSV) as f:
        for r in csv.DictReader(f):
            if r["model"] == "competing" and r["t_half_s"]:
                halflives.append(r)

    return electronic, arrhenius, halflives


def main():
    electronic, arrhenius, halflives = load_data()

    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.28)

    # ── (a) Hammett σ vs Ea ──
    ax_a = fig.add_subplot(gs[0, 0])

    mech_colors = {
        "conjugation-stabilized": "#2171B5",
        "benzyne elimination": "#E6550D",
        "alpha-elimination": "#9E9AC8",
        "ring-opening": "#6A51A3",
        "protonation/polymerization": "#969696",
    }

    # Plot known intermediates
    arli_points = []
    non_benzyne = []
    for e in electronic:
        if e["sigma_hammett"]:
            sigma = float(e["sigma_hammett"])
            ea = float(e["Ea_kJ_mol"])
            color = mech_colors.get(e["mechanism"], "#999")
            is_ortho = "ortho" in e.get("ewg", "")
            marker = "s" if is_ortho else "o"
            ax_a.scatter(sigma, ea, s=120, c=color, edgecolors="black",
                         zorder=5, marker=marker)
            ax_a.annotate(e["short_name"], (sigma, ea), xytext=(6, 6),
                         textcoords="offset points", fontsize=6.5, fontweight="bold")
            arli_points.append((sigma, ea))
            if (e["mechanism"] != "benzyne elimination"
                    and not is_ortho):
                non_benzyne.append((sigma, ea))

    # Hammett fit line (excl benzyne)
    if non_benzyne:
        s_fit, e_fit = zip(*non_benzyne)
        slope, intercept, r_val, p_val, _ = linregress(s_fit, e_fit)
        x_line = np.linspace(-0.3, 0.75, 100)
        y_line = slope * x_line + intercept
        ax_a.plot(x_line, y_line, "b--", alpha=0.6, linewidth=2,
                  label=f"Ea = {slope:.1f}σ + {intercept:.1f}\n(r = {r_val:.2f}, p = {p_val:.3f})")

    # Plot validation predictions
    for v in VALIDATION:
        ax_a.scatter(v["sigma"], v["Ea_pred"], s=150, c="gold", edgecolors="red",
                     marker="D", zorder=6, linewidth=2)
        ax_a.annotate(v["name"], (v["sigma"], v["Ea_pred"]),
                     xytext=(-8, -18), textcoords="offset points",
                     fontsize=7.5, color="red", fontweight="bold")

    # Benzyne zone (Ea = 59.7–65.4 after tR correction)
    ax_a.axhspan(55, 72, alpha=0.08, color="red")
    ax_a.text(0.02, 68, "benzyne\nelimination", fontsize=8, color="red", style="italic")

    ax_a.scatter([], [], s=150, c="gold", edgecolors="red", marker="D",
                 label="Validation targets\n(to be measured)")
    ax_a.set_xlabel("Hammett σ", fontsize=12)
    ax_a.set_ylabel("Ea (kJ/mol)", fontsize=12)
    ax_a.set_title("(a) Hammett σ vs Decomposition Ea", fontsize=13, fontweight="bold")
    ax_a.legend(fontsize=8, loc="upper left")
    ax_a.grid(alpha=0.3)
    ax_a.set_xlim(-0.3, 0.75)
    ax_a.set_ylim(-5, 95)

    # ── (b) Arrhenius plot ──
    ax_b = fig.add_subplot(gs[0, 1])

    # Group halflives by intermediate
    from collections import defaultdict
    by_inter = defaultdict(list)
    for h in halflives:
        by_inter[h["intermediate"]].append(h)

    inter_to_mech = {e["intermediate"]: e["mechanism"] for e in electronic}
    inter_to_short = {e["intermediate"]: e["short_name"] for e in electronic}

    # Unique color per intermediate (not per mechanism)
    arrh_palette = [
        "#E6194B", "#3CB44B", "#4363D8", "#F58231", "#911EB4",
        "#42D4F4", "#F032E6", "#BFEF45", "#FABED4", "#469990",
        "#DCBEFF", "#9A6324", "#800000", "#AAFFC3", "#808000",
    ]
    arrh_intermediates = sorted(by_inter.keys(),
                                key=lambda x: -len([p for p in by_inter[x]
                                                    if float(p["k_d"]) > 0]))
    inter_color_map = {inter: arrh_palette[i % len(arrh_palette)]
                       for i, inter in enumerate(arrh_intermediates)}

    for inter in arrh_intermediates:
        pts = by_inter[inter]
        k_d = [float(p["k_d"]) for p in pts if float(p["k_d"]) > 0]
        T_K_valid = [float(p["T_K"]) for p in pts if float(p["k_d"]) > 0]

        if len(k_d) < 2:
            continue

        inv_T = 1000.0 / np.array(T_K_valid)
        ln_k = np.log(np.array(k_d))

        short = inter_to_short.get(inter, inter[:20])
        color = inter_color_map[inter]

        ax_b.scatter(inv_T, ln_k, s=40, c=[color], zorder=5, alpha=0.8)

        # Fit line
        sl, ic, _, _, _ = linregress(inv_T, ln_k)
        x_fit = np.linspace(inv_T.min() - 0.1, inv_T.max() + 0.1, 50)
        ax_b.plot(x_fit, sl * x_fit + ic, color=color, linewidth=1.5, alpha=0.7,
                  label=f"{short}")

    ax_b.set_xlabel("1000 / T  (K\u207b\u00b9)", fontsize=12)
    ax_b.set_ylabel("ln(k_d)  (s\u207b\u00b9)", fontsize=12)
    ax_b.set_title("(b) Arrhenius Plot: Decomposition Rate Constants", fontsize=13, fontweight="bold")
    ax_b.legend(fontsize=5.5, loc="upper left", ncol=2)
    ax_b.grid(alpha=0.3)

    # Top axis: temperature in °C
    ax_b2 = ax_b.twiny()
    temps_C = [-78, -60, -40, -20, 0, 20]
    temps_inv = [1000.0 / (t + 273.15) for t in temps_C]
    ax_b2.set_xlim(ax_b.get_xlim())
    ax_b2.set_xticks(temps_inv)
    ax_b2.set_xticklabels([f"{t}°C" for t in temps_C], fontsize=8)

    # ── (c) Ea vs ln(A) compensation ──
    ax_c = fig.add_subplot(gs[1, 0])

    eas_all = [float(a["Ea_decomp_kJ_mol"]) for a in arrhenius]
    lnAs_all = [float(a["ln_A"]) for a in arrhenius]
    names_all = [inter_to_short.get(a["intermediate"], a["intermediate"][:20]) for a in arrhenius]
    mechs_all = [inter_to_mech.get(a["intermediate"], "") for a in arrhenius]
    colors_all = [mech_colors.get(m, "#999") for m in mechs_all]

    ax_c.scatter(eas_all, lnAs_all, s=100, c=colors_all, edgecolors="black", zorder=5)
    for i, name in enumerate(names_all):
        ax_c.annotate(name, (eas_all[i], lnAs_all[i]), xytext=(5, 5),
                     textcoords="offset points", fontsize=7, fontweight="bold")

    # Fit line
    sl_c, ic_c, r_c, p_c, _ = linregress(eas_all, lnAs_all)
    T_iso = 1 / (R * sl_c) - 273.15
    x_fit_c = np.linspace(0, 90, 50)
    ax_c.plot(x_fit_c, sl_c * x_fit_c + ic_c, "k--", alpha=0.5, linewidth=2,
              label=f"ln(A) = {sl_c:.3f}·Ea + {ic_c:.2f}\nr = {r_c:.3f}\nT_iso = {T_iso:.0f}°C")

    # Plot validation predictions
    for v in VALIDATION:
        lnA_pred = sl_c * v["Ea_pred"] + ic_c
        ax_c.scatter(v["Ea_pred"], lnA_pred, s=130, c="gold", edgecolors="red",
                     marker="D", zorder=6, linewidth=2)
        ax_c.annotate(v["name"], (v["Ea_pred"], lnA_pred),
                     xytext=(5, -15), textcoords="offset points",
                     fontsize=7, color="red", fontweight="bold")

    ax_c.set_xlabel("Ea (kJ/mol)", fontsize=12)
    ax_c.set_ylabel("ln(A)  (s⁻¹)", fontsize=12)
    ax_c.set_title("(c) Enthalpy-Entropy Compensation", fontsize=13, fontweight="bold")
    ax_c.legend(fontsize=9, loc="upper left")
    ax_c.grid(alpha=0.3)

    # ── (d) Stability ranking with reactor zones ──
    ax_d = fig.add_subplot(gs[1, 1])

    # Existing intermediates
    sorted_data = sorted(arrhenius, key=lambda d: float(d["t_half_m40C_s"]))
    names_d = [inter_to_short.get(d["intermediate"], d["intermediate"][:20]) for d in sorted_data]
    log_t = [np.log10(float(d["t_half_m40C_s"])) for d in sorted_data]
    mechs_d = [inter_to_mech.get(d["intermediate"], "") for d in sorted_data]
    colors_d = [mech_colors.get(m, "#CCC") for m in mechs_d]

    # Add validation predictions
    for v in VALIDATION:
        ea = v["Ea_pred"]
        lnA = sl_c * ea + ic_c
        k_d_m40 = np.exp(lnA - ea / (R * (-40 + 273.15)))
        t_half = np.log(2) / k_d_m40
        v["t_half_m40"] = t_half
        names_d.append(v["name"])
        log_t.append(np.log10(t_half))
        colors_d.append("gold")

    # Sort all together
    order = np.argsort(log_t)
    names_d = [names_d[i] for i in order]
    log_t = [log_t[i] for i in order]
    colors_d = [colors_d[i] for i in order]

    bars = ax_d.barh(range(len(names_d)), log_t, color=colors_d,
                      edgecolor=["red" if c == "gold" else "white" for c in colors_d],
                      linewidth=[2 if c == "gold" else 0.5 for c in colors_d],
                      height=0.7)

    ax_d.set_yticks(range(len(names_d)))
    ax_d.set_yticklabels(names_d, fontsize=8.5, fontweight="bold")
    ax_d.set_xlabel("log₁₀(t½ at -40°C)  [s]", fontsize=12)
    ax_d.set_title("(d) Stability Ranking + Experimental Predictions", fontsize=13, fontweight="bold")

    # Reactor zones (Yoshida classification)
    ax_d.axvspan(np.log10(1), 4, alpha=0.08, color="blue")
    ax_d.axvspan(-4, np.log10(1), alpha=0.08, color="red")

    y_top = len(names_d) - 0.3
    ax_d.text(np.log10(10), y_top, "Flow", fontsize=8, color="blue", fontweight="bold", ha="center")
    ax_d.text(np.log10(0.03), y_top, "Flash", fontsize=8, color="red", fontweight="bold", ha="center")

    # Add t½ labels
    for i, (bar, lt) in enumerate(zip(bars, log_t)):
        th = 10 ** lt
        if th > 1:
            t_str = f"{th:.1f} s"
        else:
            t_str = f"{th*1000:.1f} ms"
        ax_d.text(bar.get_width() + 0.05, i, t_str, va="center", fontsize=7.5)

    # Legend
    legend_elements = [
        Patch(facecolor=c, label=m) for m, c in mech_colors.items()
    ] + [Patch(facecolor="gold", edgecolor="red", linewidth=2, label="Validation predictions")]
    ax_d.legend(handles=legend_elements, fontsize=7, loc="lower right")
    ax_d.grid(axis="x", alpha=0.3)

    # Save
    fig.savefig(os.path.join(PLOT_DIR, "final_results_summary.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(PLOT_DIR, "final_results_summary.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {PLOT_DIR}/final_results_summary.png")
    print(f"Saved: {PLOT_DIR}/final_results_summary.pdf")

    # Print validation predictions
    print("\n=== Validation Predictions ===")
    for v in VALIDATION:
        th = v["t_half_m40"]
        t_str = f"{th:.1f} s" if th > 1 else f"{th*1000:.1f} ms"
        print(f"  {v['name']:15s} | σ={v['sigma']:+.2f} | Ea={v['Ea_pred']:.1f} kJ/mol | t½@-40°C = {t_str}")


if __name__ == "__main__":
    main()
