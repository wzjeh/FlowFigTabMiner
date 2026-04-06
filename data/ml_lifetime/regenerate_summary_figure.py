#!/usr/bin/env python3
"""Regenerate final_results_summary.png — 4-panel figure for paper.

Panel order: (a) Arrhenius, (b) Hammett σ (meta/para only), (c) Taft Es, (d) Stability ranking.
Reactor zones use Yoshida's classification: flash (t½ < 1 s) vs flow (t½ ≥ 1 s).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import pandas as pd
from pathlib import Path

matplotlib.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'figure.dpi': 200,
})

BASE = Path(__file__).parent
arrhenius = pd.read_csv(BASE / "phase_b_arrhenius.csv")
electronic = pd.read_csv(BASE / "electronic_analysis.csv")
halflives = pd.read_csv(BASE / "phase_a_halflives.csv")

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.subplots_adjust(hspace=0.32, wspace=0.30)

# Mechanism color map (shared)
mech_colors = {
    'conjugation-stabilized': '#2166ac',
    'benzyne elimination': '#d6604d',
    'alpha-elimination': '#f4a582',
    'ring-opening': '#4dac26',
    'protonation/polymerization': '#878787',
}

# ─── Panel (a): Arrhenius Plot ──────────────────────────────────────────
ax = axes[0, 0]

intermediates_arr = arrhenius.sort_values('Ea_decomp_kJ_mol')
cmap_arr = plt.cm.coolwarm
norm_arr = matplotlib.colors.Normalize(vmin=0, vmax=len(intermediates_arr) - 1)

for idx, (_, row) in enumerate(intermediates_arr.iterrows()):
    name = row['intermediate']
    ea = row['Ea_decomp_kJ_mol']
    lnA = row['ln_A']

    T_range_str = row['T_range_C']
    parts = T_range_str.split(' to ')
    T_lo = float(parts[0]) + 273.15
    T_hi = float(parts[1]) + 273.15
    T_arr = np.linspace(T_lo, T_hi, 50)
    inv_T = 1000.0 / T_arr
    ln_kd = lnA - (ea * 1000.0) / (8.314 * T_arr)

    color = cmap_arr(norm_arr(idx))
    short = electronic.loc[electronic['intermediate'] == name, 'short_name'].values
    label = short[0] if len(short) > 0 else name[:15]
    ax.plot(inv_T, ln_kd, '-', color=color, linewidth=1.2, alpha=0.8, label=label)

    # Data points
    sub = halflives[halflives['intermediate'] == name]
    sub = sub[sub['t_half_s'] > 0]
    if len(sub) > 0:
        temps_K = sub['T_C'].values + 273.15
        kd_vals = np.log(2) / sub['t_half_s'].values
        ln_kd_pts = np.log(kd_vals)
        inv_T_pts = 1000.0 / temps_K
        ax.scatter(inv_T_pts, ln_kd_pts, c=[color], s=30, edgecolors='k',
                   linewidths=0.3, zorder=5)

# Temperature axis on top
ax2_top = ax.twiny()
temp_ticks_C = [20, 0, -20, -40, -60, -78]
temp_ticks_invT = [1000.0 / (t + 273.15) for t in temp_ticks_C]
ax2_top.set_xlim(ax.get_xlim())
ax2_top.set_xticks(temp_ticks_invT)
ax2_top.set_xticklabels([f'{t}°C' for t in temp_ticks_C], fontsize=8)

ax.set_xlabel('1000/T (K⁻¹)')
ax.set_ylabel('ln(k_d) (s⁻¹)')
ax.set_title('(a) Arrhenius Plot: Decomposition Rate Constants', fontweight='bold')
ax.legend(fontsize=6, ncol=2, loc='lower right', framealpha=0.8)

# ─── Panel (b): Hammett σ vs Ea (meta/para only) ───────────────────────
ax = axes[0, 1]

# Exclude ortho substituents entirely — they don't follow Hammett
exclude_names = {
    'o-CO₂Me-ArLi', 'o-CO₂Et-ArLi', 'o-CO₂ⁱPr-ArLi', 'o-CO₂ᵗBu-ArLi',
    'o-Br-ArLi', 'o-I-ArLi',
}

for _, row in electronic.iterrows():
    sigma = row['sigma_hammett']
    short = row['short_name']
    if pd.isna(sigma) or short in exclude_names:
        continue
    mech = row['mechanism']
    c = mech_colors.get(mech, '#999999')
    ax.scatter(sigma, row['Ea_kJ_mol'], c=c, marker='o', s=80, edgecolors='k',
               linewidths=0.5, zorder=5)
    # Label offsets
    offset = (5, 5)
    if short == 'PhLi':
        offset = (-35, 8)
    elif short == 'p-CO₂ᵗBu-ArLi':
        offset = (5, -15)
    ax.annotate(short, (sigma, row['Ea_kJ_mol']), fontsize=7,
                textcoords='offset points', xytext=offset)

# Hammett regression line (meta/para n=4)
sigma_range = np.linspace(-0.3, 0.75, 100)
ea_line = -48.5 * sigma_range + 36.1
ax.plot(sigma_range, ea_line, '--', color='#2166ac', linewidth=1.5, alpha=0.7)
ax.fill_between(sigma_range, ea_line - 8, ea_line + 8, alpha=0.08, color='#d73027')

# Validation targets — yellow diamonds
val_targets = [
    (-0.17, 44.3, 'p-CH₃-PhLi'),
    (0.06, 33.2, 'p-F-PhLi'),
    (0.54, 9.9, 'p-CF₃-PhLi'),
]
for sigma_v, ea_v, name in val_targets:
    ax.scatter(sigma_v, ea_v, c='gold', marker='D', s=100, edgecolors='k',
               linewidths=1.0, zorder=6)
    ax.annotate(name, (sigma_v, ea_v), fontsize=7, color='#b35900',
                textcoords='offset points', xytext=(-10, -15))

legend_elements_h = [
    Line2D([0], [0], linestyle='--', color='#2166ac', linewidth=1.5,
           label='Ea = −48.5σ + 36.1\n(r = −0.98, p = 0.021)'),
    Line2D([0], [0], marker='D', color='w', markerfacecolor='gold',
           markeredgecolor='k', markersize=8, label='Validation targets\n(to be measured)'),
]
ax.legend(handles=legend_elements_h, loc='upper right', fontsize=8, framealpha=0.9)

ax.set_xlabel('Hammett σ')
ax.set_ylabel('Ea (kJ/mol)')
ax.set_title('(b) Hammett σ vs Decomposition Ea', fontweight='bold')
ax.set_xlim(-0.35, 0.80)
ax.set_ylim(-5, 50)

# ─── Panel (c): Taft Es vs log(t½) ─────────────────────────────────────
ax = axes[1, 0]

taft_data = {
    'Me':  {'Es': 0.00,  'short_name': 'o-CO₂Me-ArLi'},
    'Et':  {'Es': -0.07, 'short_name': 'o-CO₂Et-ArLi'},
    'iPr': {'Es': -0.47, 'short_name': 'o-CO₂ⁱPr-ArLi'},
    'tBu': {'Es': -1.54, 'short_name': 'o-CO₂ᵗBu-ArLi'},
}

es_vals, log_t_vals, labels_t = [], [], []
batch_yields = {'Me': 0, 'Et': 0, 'iPr': 12, 'tBu': 61}

for R, info in taft_data.items():
    row_e = electronic[electronic['short_name'] == info['short_name']]
    if len(row_e) == 0:
        continue
    t_half = row_e['t_half_m40C_s'].values[0]
    es_vals.append(info['Es'])
    log_t_vals.append(np.log10(t_half))
    labels_t.append(R)

es_vals = np.array(es_vals)
log_t_vals = np.array(log_t_vals)
yield_vals = [batch_yields[l] for l in labels_t]

cmap_taft = plt.cm.RdYlGn
norm_taft = matplotlib.colors.Normalize(vmin=0, vmax=80)

sc = ax.scatter(es_vals, log_t_vals, c=yield_vals, cmap=cmap_taft, norm=norm_taft,
                s=150, edgecolors='k', linewidths=1.0, zorder=5)

for i, R in enumerate(labels_t):
    offset = (8, 5) if R != 'tBu' else (8, -12)
    ax.annotate(R, (es_vals[i], log_t_vals[i]), fontsize=11, fontweight='bold',
                textcoords='offset points', xytext=offset)

# Regression line
es_range = np.linspace(-1.8, 0.3, 100)
log_t_line = -1.13 * es_range + 0.04
ax.plot(es_range, log_t_line, '--', color='#555555', linewidth=1.5, alpha=0.7)

ax.legend([f'log(t₁/₂) = −1.13·Es + 0.04\n(r = −0.969, p = 0.031, n = 4)'],
          loc='upper right', fontsize=8, framealpha=0.9)

cbar = fig.colorbar(sc, ax=ax, pad=0.02, shrink=0.8)
cbar.set_label('Batch yield at −78°C (%)', fontsize=9)

ax.set_xlabel('Taft Es')
ax.set_ylabel('log₁₀(t₁/₂ at −40°C / s)')
ax.set_title('(c) Taft Steric Effect on Ortho-Ester ArLi', fontweight='bold')

# ─── Panel (d): Stability Ranking + Experimental Predictions ───────────
ax = axes[1, 1]

ranking = electronic.sort_values('t_half_m40C_s', ascending=True).copy()
ranking['log_t_half'] = np.log10(ranking['t_half_m40C_s'])

val_preds = pd.DataFrame([
    {'short_name': 'p-CF₃-PhLi', 'log_t_half': np.log10(22.9), 'mechanism': 'validation_prediction',
     't_half_m40C_s': 22.9},
    {'short_name': 'p-F-PhLi', 'log_t_half': np.log10(1.7), 'mechanism': 'validation_prediction',
     't_half_m40C_s': 1.7},
    {'short_name': 'p-CH₃-PhLi', 'log_t_half': np.log10(0.497), 'mechanism': 'validation_prediction',
     't_half_m40C_s': 0.497},
])
combined = pd.concat([ranking, val_preds], ignore_index=True)
combined = combined.sort_values('t_half_m40C_s', ascending=True)
combined['log_t_half'] = np.log10(combined['t_half_m40C_s'])

mech_bar_colors = {
    'conjugation-stabilized': '#2166ac',
    'benzyne elimination': '#d6604d',
    'alpha-elimination': '#f4a582',
    'ring-opening': '#b2df8a',
    'protonation/polymerization': '#878787',
    'validation_prediction': '#ffd700',
}

y_pos = np.arange(len(combined))
bar_colors = [mech_bar_colors.get(m, '#cccccc') for m in combined['mechanism'].values]

bars = ax.barh(y_pos, combined['log_t_half'].values, color=bar_colors,
               edgecolor='k', linewidth=0.3, height=0.7)

for i, (_, row) in enumerate(combined.iterrows()):
    t_val = row['t_half_m40C_s']
    if t_val >= 1:
        label = f'{t_val:.1f} s'
    elif t_val >= 0.001:
        label = f'{t_val*1000:.1f} ms'
    else:
        label = f'{t_val*1e6:.0f} µs'
    x_pos = row['log_t_half'] + 0.05
    ax.text(x_pos, i, label, va='center', fontsize=7)

ax.set_yticks(y_pos)
ax.set_yticklabels(combined['short_name'].values, fontsize=8)
ax.set_xlabel('log₁₀(t½ at −40°C) [s]')
ax.set_title('(d) Stability Ranking + Experimental Predictions', fontweight='bold')

# Reactor zone shading — Yoshida classification: flash < 1 s, flow ≥ 1 s
# Boundary at log10(1) = 0
ax.axvspan(-4, 0, alpha=0.06, color='red')    # flash: t½ < 1 s
ax.axvspan(0, 3, alpha=0.06, color='blue')     # flow: t½ ≥ 1 s
ax.axvline(x=0, color='gray', linewidth=0.8, linestyle=':', alpha=0.5)

ax.text(-2.0, len(combined) - 0.3, 'Flash', fontsize=9, color='red',
        fontweight='bold', ha='center', va='bottom')
ax.text(1.0, len(combined) - 0.3, 'Flow', fontsize=9, color='blue',
        fontweight='bold', ha='center', va='bottom')

legend_elements_d = [
    Patch(facecolor='#2166ac', edgecolor='k', label='conjugation-stabilized'),
    Patch(facecolor='#d6604d', edgecolor='k', label='benzyne elimination'),
    Patch(facecolor='#f4a582', edgecolor='k', label='α-elimination'),
    Patch(facecolor='#b2df8a', edgecolor='k', label='ring-opening'),
    Patch(facecolor='#878787', edgecolor='k', label='protonation/polymerization'),
    Patch(facecolor='#ffd700', edgecolor='k', label='validation prediction'),
]
ax.legend(handles=legend_elements_d, fontsize=6, loc='lower right', framealpha=0.8)

# ─── Save ───────────────────────────────────────────────────────────────
out_dir = BASE / "analysis_figures"
fig.savefig(out_dir / "final_results_summary.png", dpi=200, bbox_inches='tight',
            facecolor='white')
fig.savefig(out_dir / "final_results_summary.pdf", bbox_inches='tight',
            facecolor='white')
print(f"Saved to {out_dir / 'final_results_summary.png'}")
print(f"Saved to {out_dir / 'final_results_summary.pdf'}")
plt.close()
