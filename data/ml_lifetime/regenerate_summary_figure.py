#!/usr/bin/env python3
"""Regenerate final_results_summary — 4-panel figure for paper.

Updated to reflect the unified four-parameter LFER model (Encoding B):
  Ea  = -48.01σ + 8.00Es + 25.53δ_ortho + 44.44δ_benzyne + 35.89
  lnA = -24.17σ + 6.72Es + 15.55δ_ortho + 29.70δ_benzyne + 15.41

Panel layout:
  (a) Arrhenius plot (data)
  (b) Parity plot: Ea observed vs predicted (training + LOOCV)
  (c) Parity plot: t½ observed vs predicted (training + LOOCV)
  (d) Stability ranking + 2 validation predictions from 4-parameter model
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut

matplotlib.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 200,
})

BASE = Path(__file__).parent
arrhenius = pd.read_csv(BASE / "phase_b_arrhenius.csv")
electronic = pd.read_csv(BASE / "electronic_analysis.csv")
halflives = pd.read_csv(BASE / "phase_a_halflives.csv")

R_GAS = 8.314e-3  # kJ/(mol·K)

# ── 10 ArLi substrates with model features ──
SUBSTRATES = [
    # (short_name, σ, Es, δ_ortho, δ_benzyne, Ea, ln_A, t½@-40°C)
    ("m-CN-ArLi",       0.56,  0.00, 0, 0,  4.83, -0.35,  11.9),
    ("p-CN-ArLi",       0.66,  0.00, 0, 0,  7.06,  0.47,  16.5),
    ("p-CO₂ᵗBu-ArLi",  0.45,  0.00, 0, 0, 14.97,  5.78,   4.8),
    ("o-CO₂ᵗBu-ArLi",  0.45, -1.54, 1, 0, 26.80,  9.51,  51.9),
    ("PhLi",            0.00,  0.00, 0, 0, 36.54, 15.38,  22.4),
    ("o-CO₂Me-ArLi",    0.45,  0.00, 1, 0, 36.53, 18.99,   0.60),
    ("o-CO₂ⁱPr-ArLi",  0.45, -0.47, 1, 0, 38.05, 17.54,   5.6),
    ("o-CO₂Et-ArLi",    0.45, -0.07, 1, 0, 41.26, 20.31,   1.8),
    ("o-I-ArLi",        0.35,  0.00, 0, 1, 59.73, 36.16,   0.0033),
    ("o-Br-ArLi",       0.39,  0.00, 0, 1, 65.42, 36.17,   0.061),
]

names  = [s[0] for s in SUBSTRATES]
sigma  = np.array([s[1] for s in SUBSTRATES])
Es     = np.array([s[2] for s in SUBSTRATES])
d_ort  = np.array([s[3] for s in SUBSTRATES])
d_bz   = np.array([s[4] for s in SUBSTRATES])
Ea_obs = np.array([s[5] for s in SUBSTRATES])
lnA_obs = np.array([s[6] for s in SUBSTRATES])
thalf_obs = np.array([s[7] for s in SUBSTRATES])

X = np.column_stack([sigma, Es, d_ort, d_bz])
T_eval = -40 + 273.15

# Fit models
reg_Ea = LinearRegression().fit(X, Ea_obs)
reg_lnA = LinearRegression().fit(X, lnA_obs)
Ea_pred = reg_Ea.predict(X)
lnA_pred = reg_lnA.predict(X)
thalf_pred = np.log(2) / np.exp(lnA_pred - Ea_pred / (R_GAS * T_eval))

# LOOCV
loo = LeaveOneOut()
Ea_loo = np.zeros(10)
lnA_loo = np.zeros(10)
for tr_ix, te_ix in loo.split(X):
    Ea_loo[te_ix] = LinearRegression().fit(X[tr_ix], Ea_obs[tr_ix]).predict(X[te_ix])
    lnA_loo[te_ix] = LinearRegression().fit(X[tr_ix], lnA_obs[tr_ix]).predict(X[te_ix])
thalf_loo = np.log(2) / np.exp(lnA_loo - Ea_loo / (R_GAS * T_eval))

# Validation predictions (4-parameter model)
VAL_SUBSTRATES = [
    ("p-CF₃-PhLi",  0.54, 0.0, 0, 0),
    ("p-CH₃-PhLi", -0.17, 0.0, 0, 0),
]
val_names = [v[0] for v in VAL_SUBSTRATES]
X_val = np.array([v[1:] for v in VAL_SUBSTRATES])
Ea_val = reg_Ea.predict(X_val)
lnA_val = reg_lnA.predict(X_val)
thalf_val = np.log(2) / np.exp(lnA_val - Ea_val / (R_GAS * T_eval))

# Mechanism colors
MECH_COLORS = {
    "conjugation-stabilized": "#2166ac",
    "benzyne elimination": "#d6604d",
    "alpha-elimination": "#f4a582",
    "ring-opening": "#4dac26",
    "protonation/polymerization": "#878787",
}

# Substrate → mechanism mapping
SUBSTRATE_MECH = {
    "m-CN-ArLi": "conjugation-stabilized",
    "p-CN-ArLi": "conjugation-stabilized",
    "p-CO₂ᵗBu-ArLi": "conjugation-stabilized",
    "o-CO₂ᵗBu-ArLi": "conjugation-stabilized",
    "PhLi": "protonation/polymerization",
    "o-CO₂Me-ArLi": "conjugation-stabilized",
    "o-CO₂ⁱPr-ArLi": "conjugation-stabilized",
    "o-CO₂Et-ArLi": "conjugation-stabilized",
    "o-I-ArLi": "benzyne elimination",
    "o-Br-ArLi": "benzyne elimination",
}

# Substrate → marker (distinguish ortho/para/meta)
SUBSTRATE_MARKER = {
    "m-CN-ArLi": "^",       # meta = triangle up
    "p-CN-ArLi": "o",       # para = circle
    "p-CO₂ᵗBu-ArLi": "o",
    "o-CO₂ᵗBu-ArLi": "s",  # ortho = square
    "PhLi": "o",
    "o-CO₂Me-ArLi": "s",
    "o-CO₂ⁱPr-ArLi": "s",
    "o-CO₂Et-ArLi": "s",
    "o-I-ArLi": "s",
    "o-Br-ArLi": "s",
}


def fmt_thalf(t):
    if t < 0.01:
        return f"{t*1000:.1f} ms"
    elif t < 1:
        return f"{t*1000:.0f} ms"
    else:
        return f"{t:.1f} s"


# ─────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.subplots_adjust(hspace=0.32, wspace=0.30)

# ═══ Panel (a): Arrhenius Plot ═══════════════════════════════════════════
ax = axes[0, 0]

intermediates_arr = arrhenius.sort_values("Ea_decomp_kJ_mol")
cmap_arr = plt.cm.coolwarm
norm_arr = matplotlib.colors.Normalize(vmin=0, vmax=len(intermediates_arr) - 1)

for idx, (_, row) in enumerate(intermediates_arr.iterrows()):
    name = row["intermediate"]
    ea = row["Ea_decomp_kJ_mol"]
    lnA = row["ln_A"]

    T_range_str = row["T_range_C"]
    parts = T_range_str.split(" to ")
    T_lo = float(parts[0]) + 273.15
    T_hi = float(parts[1]) + 273.15
    T_arr = np.linspace(T_lo, T_hi, 50)
    inv_T = 1000.0 / T_arr
    ln_kd = lnA - (ea * 1000.0) / (8.314 * T_arr)

    color = cmap_arr(norm_arr(idx))
    short = electronic.loc[electronic["intermediate"] == name, "short_name"].values
    label = short[0] if len(short) > 0 else name[:15]
    ax.plot(inv_T, ln_kd, "-", color=color, linewidth=1.2, alpha=0.8, label=label)

    # Data points from halflives
    sub = halflives[halflives["intermediate"] == name]
    sub = sub[sub["t_half_s"] > 0]
    if len(sub) > 0:
        temps_K = sub["T_C"].values + 273.15
        kd_vals = np.log(2) / sub["t_half_s"].values
        ln_kd_pts = np.log(kd_vals)
        inv_T_pts = 1000.0 / temps_K
        ax.scatter(inv_T_pts, ln_kd_pts, c=[color], s=30, edgecolors="k",
                   linewidths=0.3, zorder=5)

ax2_top = ax.twiny()
temp_ticks_C = [20, 0, -20, -40, -60, -78]
temp_ticks_invT = [1000.0 / (t + 273.15) for t in temp_ticks_C]
ax2_top.set_xlim(ax.get_xlim())
ax2_top.set_xticks(temp_ticks_invT)
ax2_top.set_xticklabels([f"{t}°C" for t in temp_ticks_C], fontsize=8)

ax.set_xlabel("1000/T (K⁻¹)")
ax.set_ylabel("ln(k_d) (s⁻¹)")
ax.set_title("(a) Arrhenius Plot: Decomposition Rate Constants", fontweight="bold")
ax.legend(fontsize=6, ncol=2, loc="lower right", framealpha=0.8)


# ═══ Panel (b): Parity — Observed vs Predicted Ea ═══════════════════════
ax = axes[0, 1]

r2_Ea = 1 - np.sum((Ea_obs - Ea_pred)**2) / np.sum((Ea_obs - Ea_obs.mean())**2)

for i, n in enumerate(names):
    mech = SUBSTRATE_MECH[n]
    c = MECH_COLORS[mech]
    ax.scatter(Ea_obs[i], Ea_pred[i], c=c, s=80, edgecolors="k",
               linewidths=0.8, zorder=6)
    offsets = {
        "m-CN-ArLi": (6, 8),
        "p-CN-ArLi": (6, -12),
        "p-CO₂ᵗBu-ArLi": (6, -14),
        "o-CO₂ᵗBu-ArLi": (-90, 14),     # far upper-left
        "PhLi": (8, 18),                  # above — separated from o-CO₂Me
        "o-CO₂Me-ArLi": (-90, -18),      # far lower-left
        "o-CO₂ⁱPr-ArLi": (-90, 2),       # far left, mid
        "o-CO₂Et-ArLi": (8, 14),         # right, up
        "o-I-ArLi": (-65, -10),
        "o-Br-ArLi": (6, -14),
    }
    ofs = offsets.get(n, (6, 5))
    ax.annotate(n, (Ea_obs[i], Ea_pred[i]), fontsize=6.5,
                textcoords="offset points", xytext=ofs,
                arrowprops=dict(arrowstyle="-", color="gray", lw=0.5, alpha=0.5))

# 1:1 line
ea_range = np.linspace(0, 70, 100)
ax.plot(ea_range, ea_range, "k-", lw=1, alpha=0.4)

# Validation predictions on diagonal (no observed data)
val_offsets_b = {"p-CF₃-PhLi": (6, -14), "p-CH₃-PhLi": (-80, 10)}
for j, vn in enumerate(val_names):
    ax.scatter(Ea_val[j], Ea_val[j], marker="*", c="#ffd700", s=150,
               edgecolors="k", linewidths=0.6, zorder=7)
    vofs = val_offsets_b.get(vn, (6, 6))
    ax.annotate(vn, (Ea_val[j], Ea_val[j]), fontsize=6.5, fontstyle="italic",
                textcoords="offset points", xytext=vofs,
                arrowprops=dict(arrowstyle="-", color="goldenrod", lw=0.5, alpha=0.7))

ax.text(0.05, 0.95, f"R² = {r2_Ea:.3f}", transform=ax.transAxes,
        fontsize=9, va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

ax.set_xlabel("Observed Ea (kJ/mol)")
ax.set_ylabel("Predicted Ea (kJ/mol)")
ax.set_title("(b) Unified 4-Parameter Model: Ea Parity",
             fontweight="bold")
ax.set_xlim(-2, 72)
ax.set_ylim(-2, 72)
ax.set_aspect("equal")


# ═══ Panel (c): Parity plot — predicted vs observed t½ ══════════════════
ax = axes[1, 0]

# Training fit only (filled circles)
for i, n in enumerate(names):
    mech = SUBSTRATE_MECH[n]
    c = MECH_COLORS[mech]
    ax.scatter(thalf_obs[i], thalf_pred[i], c=c, s=80,
               edgecolors="k", linewidths=0.8, zorder=6)

# 1:1 line and 3× band
t_range = np.logspace(-4, 2.5, 100)
ax.plot(t_range, t_range, "k-", lw=1, alpha=0.5)
ax.fill_between(t_range, t_range / 3, t_range * 3, alpha=0.08, color="green")
ax.plot(t_range, t_range * 3, ":", color="green", lw=0.8, alpha=0.5)
ax.plot(t_range, t_range / 3, ":", color="green", lw=0.8, alpha=0.5)

# Labels — use arrowprops to avoid overlap
for i, n in enumerate(names):
    x, y = thalf_obs[i], thalf_pred[i]
    label_offsets = {
        "PhLi": (10, 8), "m-CN-ArLi": (10, 6), "p-CN-ArLi": (-60, 8),
        "p-CO₂ᵗBu-ArLi": (10, -10), "o-CO₂ᵗBu-ArLi": (-80, 8),
        "o-CO₂Me-ArLi": (10, 8), "o-CO₂Et-ArLi": (10, -10),
        "o-CO₂ⁱPr-ArLi": (-80, -8), "o-I-ArLi": (-60, -10), "o-Br-ArLi": (10, 6),
    }
    ofs = label_offsets.get(n, (8, 5))
    ax.annotate(n, (x, y), fontsize=6, textcoords="offset points", xytext=ofs,
                arrowprops=dict(arrowstyle="-", color="gray", lw=0.5, alpha=0.5))

# Validation predictions on diagonal
val_offsets_c = {"p-CF₃-PhLi": (10, -10), "p-CH₃-PhLi": (-80, 10)}
for j, vn in enumerate(val_names):
    ax.scatter(thalf_val[j], thalf_val[j], marker="*", c="#ffd700", s=150,
               edgecolors="k", linewidths=0.6, zorder=7)
    vofs = val_offsets_c.get(vn, (8, 6))
    ax.annotate(vn, (thalf_val[j], thalf_val[j]), fontsize=6, fontstyle="italic",
                textcoords="offset points", xytext=vofs,
                arrowprops=dict(arrowstyle="-", color="goldenrod", lw=0.5, alpha=0.7))

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(1e-4, 300)
ax.set_ylim(1e-4, 300)
ax.set_xlabel("Observed t½ at −40°C (s)")
ax.set_ylabel("Predicted t½ at −40°C (s)")

n_fit_3x = np.sum((thalf_pred / thalf_obs >= 1/3) & (thalf_pred / thalf_obs <= 3))

r2_logt = 1 - np.sum((np.log10(thalf_pred) - np.log10(thalf_obs))**2) / \
               np.sum((np.log10(thalf_obs) - np.log10(thalf_obs).mean())**2)

ax.text(0.05, 0.95, f"R² = {r2_logt:.3f}\n{n_fit_3x}/10 within 3×",
        transform=ax.transAxes, fontsize=8, va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

legend_c = [
    Patch(facecolor="green", alpha=0.1, edgecolor="green",
          label="3× accuracy band"),
]
ax.legend(handles=legend_c, loc="lower right", fontsize=7.5, framealpha=0.9)
ax.set_title("(c) Model Parity: Predicted vs Observed t½",
             fontweight="bold")
ax.set_aspect("equal")


# ═══ Panel (d): Stability Ranking + Validation Predictions ══════════════
ax = axes[1, 1]

# All 14 intermediates + validation predictions
ranking = electronic.sort_values("t_half_m40C_s", ascending=True).copy()
ranking["log_t_half"] = np.log10(ranking["t_half_m40C_s"])

val_preds = pd.DataFrame([
    {"short_name": vn, "t_half_m40C_s": tv, "mechanism": "validation_prediction",
     "Ea_kJ_mol": ev}
    for vn, tv, ev in zip(val_names, thalf_val, Ea_val)
])

combined = pd.concat([ranking, val_preds], ignore_index=True)
combined = combined.sort_values("t_half_m40C_s", ascending=True)
combined["log_t_half"] = np.log10(combined["t_half_m40C_s"])

mech_bar_colors = {
    "conjugation-stabilized": "#2166ac",
    "benzyne elimination": "#d6604d",
    "alpha-elimination": "#f4a582",
    "ring-opening": "#b2df8a",
    "protonation/polymerization": "#878787",
    "validation_prediction": "#ffd700",
}

y_pos = np.arange(len(combined))
bar_colors = [mech_bar_colors.get(m, "#cccccc") for m in combined["mechanism"].values]

bars = ax.barh(y_pos, combined["log_t_half"].values, color=bar_colors,
               edgecolor="k", linewidth=0.3, height=0.7)

for i, (_, row) in enumerate(combined.iterrows()):
    t_val = row["t_half_m40C_s"]
    label = fmt_thalf(t_val)
    ea_val_r = row.get("Ea_kJ_mol", None)
    if pd.notna(ea_val_r):
        label += f"  (Ea={ea_val_r:.1f})"
    x_pos = row["log_t_half"] + 0.05
    ax.text(x_pos, i, label, va="center", fontsize=7)

ax.set_yticks(y_pos)
ax.set_yticklabels(combined["short_name"].values, fontsize=8)
ax.set_xlabel("log₁₀(t½ at −40°C) [s]")
ax.set_title("(d) Stability Ranking + Model Predictions", fontweight="bold")

# Reactor zone shading
ax.axvspan(-4, 0, alpha=0.06, color="red")
ax.axvspan(0, 3, alpha=0.06, color="blue")
ax.axvline(x=0, color="gray", linewidth=0.8, linestyle=":", alpha=0.5)

ax.text(-2.0, len(combined) - 0.3, "Flash", fontsize=9, color="red",
        fontweight="bold", ha="center", va="bottom")
ax.text(1.0, len(combined) - 0.3, "Flow", fontsize=9, color="blue",
        fontweight="bold", ha="center", va="bottom")

legend_d = [
    Patch(facecolor="#2166ac", edgecolor="k", label="conjugation-stabilized"),
    Patch(facecolor="#d6604d", edgecolor="k", label="benzyne elimination"),
    Patch(facecolor="#f4a582", edgecolor="k", label="α-elimination"),
    Patch(facecolor="#b2df8a", edgecolor="k", label="ring-opening"),
    Patch(facecolor="#878787", edgecolor="k", label="protonation/polymerization"),
    Patch(facecolor="#ffd700", edgecolor="k", label="validation prediction"),
]
ax.legend(handles=legend_d, fontsize=6, loc="lower right", framealpha=0.8)


# ─── Save ────────────────────────────────────────────────────────────────
out_dir = BASE / "analysis_figures"
out_dir.mkdir(exist_ok=True)
fig.savefig(out_dir / "final_results_summary.png", dpi=200, bbox_inches="tight",
            facecolor="white")
fig.savefig(out_dir / "final_results_summary.pdf", bbox_inches="tight",
            facecolor="white")
print(f"Saved to {out_dir / 'final_results_summary.png'}")
print(f"Saved to {out_dir / 'final_results_summary.pdf'}")
plt.close()
