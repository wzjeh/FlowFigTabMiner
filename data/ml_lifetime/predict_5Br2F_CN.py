"""
Predict yield surface for 5-bromo-2-fluorobenzonitrile + n-BuLi.

Target ArLi: 3-cyano-4-fluoro-phenyllithium (m-CN-ArLi class with p-F & o-CN-F).
SMILES (ArLi): [Li]c1cc(F)c(C#N)cc1  -- wait, let me derive properly.

Substrate: 5-Br-2-F-C6H3-CN     (positions: 1=CN, 2=F, 5=Br)
After Li/Br exchange:           1=CN, 2=F, 5=Li
Relative to Li (pos 5):
  - CN at pos 1 = meta (1,3 relationship across ring)
  - F  at pos 2 = para (1,4 relationship across ring)
→ This is "m-CN-ArLi + p-F" — class m-ArLi (reactive, has CN).

Base analog: m-cyanophenyllithium (3-CN-PhLi), Tier 1 Arrhenius:
  Ea_f=34.69, lnA_f=22.11, Ea_d=27.3, lnA_d=9.08, y_max=83.1, r²=0.97

Perturbation from analog:
  - para-F (relative to Li, pos 2 above): σ_p = +0.06 (very weak EWG, inductive only)
  - F also ortho to CN: marginally increases CN electrophilicity
  Conservative estimate: no change to formation params (σ_m_sum unchanged at 0.56)
  Small change to Ea_d possible (~ −1 to −2 kJ/mol), but uncertain.

We report:
  (a) Primary: 3-CN-PhLi experimental parameters (no perturbation)
  (b) Optimistic: −2 kJ/mol on Ea_d (slightly faster decay)
  (c) Pessimistic: +2 kJ/mol on Ea_d (slightly slower decay)
  Plus v5.0 m-ArLi formation Hammett (σ_m=0.56, σ_p=0.06):
    Ea_f = 39.54 − 15.64·0.56 = 30.78
    lnA_f = 37.23 − 25.95·0.56 = 22.70

Output: 2D yield surface T × tR + diagnostic prints.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).parent
R = 8.314e-3  # kJ/mol/K

# ============================================================
# Base parameters: 3-CN-PhLi experimental (Tier 1, r²=0.97)
# ============================================================
PARAMS_BASE = dict(Ea_f=34.69, lnA_f=22.11, Ea_d=27.3, lnA_d=9.08, y_max=83.1)

# v5.0 m-ArLi formation Hammett (positive finding, LOO R²=0.87 for lnA_f)
sigma_m = 0.56  # CN only (F is para to Li)
PARAMS_V50 = dict(
    Ea_f  = 39.54 - 15.64 * sigma_m,   # ≈ 30.78
    lnA_f = 37.23 - 25.95 * sigma_m,   # ≈ 22.70
    Ea_d  = PARAMS_BASE['Ea_d'],        # v5.0 has no decay Hammett — fall back
    lnA_d = PARAMS_BASE['lnA_d'],
    y_max = PARAMS_BASE['y_max'],
)

# Scenarios (perturbations on Ea_d only)
SCENARIOS = {
    'A_primary_analog':   dict(**PARAMS_BASE),
    'B_optimistic_Ead-2': dict(PARAMS_BASE, Ea_d=PARAMS_BASE['Ea_d'] - 2.0),
    'C_pessimistic_Ead+2': dict(PARAMS_BASE, Ea_d=PARAMS_BASE['Ea_d'] + 2.0),
    'D_v50_formation_Hammett': PARAMS_V50,
}


def yield_5param(T_C, tR_s, p):
    """5-parameter bell-curve: yield(T,tR) = y_max·(1-exp(-k_f·tR))·exp(-k_d·tR)"""
    Tk = T_C + 273.15
    k_f = np.exp(p['lnA_f'] - p['Ea_f'] / (R * Tk))
    k_d = np.exp(p['lnA_d'] - p['Ea_d'] / (R * Tk))
    y = p['y_max'] * (1 - np.exp(-k_f * tR_s)) * np.exp(-k_d * tR_s)
    return y, k_f, k_d


def t_half(T_C, p):
    Tk = T_C + 273.15
    k_d = np.exp(p['lnA_d'] - p['Ea_d'] / (R * Tk))
    return np.log(2) / k_d


def t_peak(T_C, p):
    """tR that maximizes yield: t* = ln(1 + k_f/k_d) / k_f (approx; numerical scan)."""
    Tk = T_C + 273.15
    k_f = np.exp(p['lnA_f'] - p['Ea_f'] / (R * Tk))
    k_d = np.exp(p['lnA_d'] - p['Ea_d'] / (R * Tk))
    # Analytical: d/dtR = 0 → t* = ln((k_f+k_d)/k_d) / k_f
    t_star = np.log((k_f + k_d) / k_d) / k_f
    y_star, _, _ = yield_5param(T_C, t_star, p)
    return t_star, y_star


def main():
    print("=" * 80)
    print("Yield surface prediction: 5-bromo-2-fluorobenzonitrile + n-BuLi")
    print("Target ArLi: 3-cyano-4-fluoro-phenyllithium (m-CN class + p-F)")
    print("=" * 80)

    # Summary table per scenario
    print("\n=== Scenario parameters ===")
    rows = []
    for name, p in SCENARIOS.items():
        rows.append({'scenario': name, **{k: round(v, 2) for k, v in p.items()}})
    summ = pd.DataFrame(rows)
    print(summ.to_string(index=False))
    summ.to_csv(BASE / '5Br2F_CN_scenarios.csv', index=False)

    # === Diagnostic: t½, t_peak, max yield at key temperatures ===
    print("\n=== Diagnostic (primary scenario A: 3-CN-PhLi analog) ===")
    print(f"  {'T(°C)':>6}{'k_d (s⁻¹)':>14}{'t½ (s)':>12}{'t_peak (s)':>14}{'y_max@t_peak (%)':>20}")
    p = SCENARIOS['A_primary_analog']
    diag_rows = []
    for T in [-78, -50, -25, 0, 20]:
        Tk = T + 273.15
        k_d = np.exp(p['lnA_d'] - p['Ea_d'] / (R * Tk))
        th = np.log(2) / k_d
        tp, yp = t_peak(T, p)
        diag_rows.append({'T_C': T, 'k_d': k_d, 't_half_s': th, 't_peak_s': tp, 'yield_at_t_peak': yp})
        print(f"  {T:>+6}{k_d:>14.3e}{th:>12.3e}{tp:>14.3e}{yp:>20.2f}")
    pd.DataFrame(diag_rows).to_csv(BASE / '5Br2F_CN_diagnostic.csv', index=False)

    # === 2D yield surface (primary scenario) ===
    T_grid = np.linspace(-78, 20, 50)
    tR_grid = np.logspace(np.log10(0.05), np.log10(60), 60)
    Tm, tm = np.meshgrid(T_grid, tR_grid, indexing='xy')

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    levels = [5, 10, 20, 30, 40, 50, 60, 70, 75, 80, 82, 83]
    for ax, (name, p) in zip(axes.flat, SCENARIOS.items()):
        Y = np.zeros_like(Tm)
        for i in range(Tm.shape[0]):
            for j in range(Tm.shape[1]):
                Y[i, j], _, _ = yield_5param(Tm[i, j], tm[i, j], p)
        cf = ax.contourf(Tm, tm, Y, levels=levels, cmap='viridis')
        cs = ax.contour(Tm, tm, Y, levels=levels, colors='white', linewidths=0.5, alpha=0.7)
        ax.clabel(cs, inline=True, fontsize=7, fmt='%d')
        # Mark t_peak curve
        T_curve = np.linspace(-78, 20, 60)
        t_peak_curve = [t_peak(t, p)[0] for t in T_curve]
        ax.plot(T_curve, t_peak_curve, 'r-', lw=2, label='t_peak (optimal tR)')
        ax.set_yscale('log')
        ax.set_xlabel('Temperature (°C)')
        ax.set_ylabel('Residence time tR (s)')
        ax.set_title(f"{name}\n"
                     f"Ea_f={p['Ea_f']:.1f}, lnA_f={p['lnA_f']:.2f}, "
                     f"Ea_d={p['Ea_d']:.1f}, lnA_d={p['lnA_d']:.2f}")
        ax.legend(loc='lower right', fontsize=8)
        fig.colorbar(cf, ax=ax, label='Yield (%)')
    fig.suptitle('5-bromo-2-fluorobenzonitrile yield surface — 4 scenarios', fontsize=13)
    out = BASE / 'analysis_figures' / '5Br2F_CN_yield_surface.png'
    fig.savefig(out, dpi=140, bbox_inches='tight')
    print(f"\nSaved: {out}")

    # === Recommended experimental conditions ===
    print("\n" + "=" * 80)
    print("RECOMMENDED experimental conditions (primary scenario A)")
    print("=" * 80)
    p = SCENARIOS['A_primary_analog']
    print("\n  Optimum point per temperature (max yield):")
    print(f"  {'T(°C)':>6}{'t_peak (s)':>14}{'y_max (%)':>14}{'Note':>40}")
    for T in [-78, -65, -50, -35, -20, 0]:
        tp, yp = t_peak(T, p)
        note = ""
        if tp < 0.1: note = "very short tR — needs fast micromixer"
        elif tp > 30: note = "long tR — easy reactor"
        else: note = "standard flow regime"
        print(f"  {T:>+6}{tp:>14.3e}{yp:>14.2f}{note:>40}")

    # Recommended scan: discrete T × tR points worth running
    print("\n  Suggested experimental scan (3 T × 4 tR = 12 points minimum):")
    T_scan = [-78, -50, -25]
    for T in T_scan:
        Tk = T + 273.15
        k_d = np.exp(p['lnA_d'] - p['Ea_d'] / (R * Tk))
        th = np.log(2) / k_d
        # Scan around t_peak with ratios 0.3, 1, 3, 10 of t_peak (or t_half if t_peak < 0.05)
        tp, yp = t_peak(T, p)
        anchors = [0.3 * tp, tp, 3 * tp, 10 * tp]
        scan_str = ", ".join(f"{x:.2f}s" for x in anchors)
        print(f"    T={T:+3}°C  t_peak={tp:.2f}s  t½={th:.1e}s  scan tR={scan_str}")

    print("""
NOTES:
  • Primary base = m-cyanophenyllithium experimental Tier 1 (n_T=6, r²=0.97).
  • Para-F perturbation (σ_p=0.06) is small → minimal impact on m-ArLi class kinetics.
  • Ortho-F on CN (relative to CN) may marginally accelerate decay via inductive
    activation of CN electrophilicity — see scenarios B (Ea_d −2) and C (Ea_d +2).
  • If experimental yield falls between A and B, suggests F slightly accelerates decay.
  • If yield matches A or is HIGHER than A, F effects are negligible.
  • y_max ceiling = 83.1% (3-CN-PhLi value) — predicted absolute max yield.
  • GC analysis confirms: substrate (5-Br-2-F-C6H3-CN) + product (2-F-C6H4-CN) both
    volatile aromatic nitriles, BPs ~180-230°C, well separated by GC-FID/MS.
""")


if __name__ == '__main__':
    main()
