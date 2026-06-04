"""
Predict t_max and t_half(formation) and t_half(ArLi decay) for p-FC6H4Li.

Sources of rate constants:
  k_f(T) — from experimental Arrhenius fit (this work, swap+all 22 pts)
  k_d(T) — from v4.6 HYBRID v3 model prediction for p-FC6H4Li

Descriptors of p-FC6H4Li (from intermediates_master.csv + aggregation calc):
  d_LiC = 1.9053 Å,  B5 = 3.2404,  ΔVbur = 0.1683
  BDE = 418.5 kJ/mol,  dipole = 8.214 D,  vol = 122.66 Å³

v4.6 p-ArLi formulas:
  Ea_d  = −1254·d_LiC + 29.07·B5 + 159.96·ΔVbur + 2307.94 = 39.88 kJ/mol
  lnA_d = +1.6366·BDE − 6.9992·dipole + 0.2079·vol − 629.88 = 23.07
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = 8.314e-3  # kJ/(mol·K)
OUT_PNG = Path(__file__).parent / "analysis_figures" / "pFArLi_tmax_thalf.png"


# ---- Experimental Ea_f / lnA_f from 5-param fit on conversion (swap, 22 pts) ----
# Using results from fit_5param_conversion.py: Ea_f=24.04, lnA_f=16.38
# Alternative: 2-param k_f-only fit (Ea_f=22.50, lnA_f=15.20)
EA_F_EXP, LNA_F_EXP = 24.04, 16.38       # 5-param fit (preferred, more stable)
EA_F_EXP_2P, LNA_F_EXP_2P = 22.50, 15.20  # 2-param k_f Arrhenius

# ---- v4.6 model predictions for p-FC6H4Li ----
# Ea_d formula coefficients
d_LiC, B5, dVbur = 1.9053, 3.2404, 0.1683
EA_D_MODEL = -1254.0179 * d_LiC + 29.0706 * B5 + 159.9647 * dVbur + 2307.9437

# lnA_d formula coefficients
BDE, dipole, vol = 418.5, 8.214, 122.66
LNA_D_MODEL = 1.6366 * BDE - 6.9992 * dipole + 0.2079 * vol - 629.8836

print(f"Experimental k_f (from 5-param fit):")
print(f"  Ea_f  = {EA_F_EXP:.2f} kJ/mol")
print(f"  lnA_f = {LNA_F_EXP:.2f}")
print(f"\nv4.6 HYBRID v3 prediction for p-FC6H4Li:")
print(f"  Ea_d  = {EA_D_MODEL:.2f} kJ/mol")
print(f"  lnA_d = {LNA_D_MODEL:.2f}")


def kf(T_C):
    return np.exp(LNA_F_EXP - EA_F_EXP / (R * (T_C + 273.15)))


def kd(T_C):
    return np.exp(LNA_D_MODEL - EA_D_MODEL / (R * (T_C + 273.15)))


def t_half_form(T_C):  # substrate -> ArLi formation 50%
    return np.log(2) / kf(T_C)


def t_half_arli(T_C):  # ArLi natural decay 50%
    return np.log(2) / kd(T_C)


def t_max(T_C):
    kf_, kd_ = kf(T_C), kd(T_C)
    if abs(kd_ - kf_) < 1e-12:
        return 1.0 / kf_
    return np.log(kd_ / kf_) / (kd_ - kf_)


def y_max_at(T_C):
    """Maximum ArLi fraction at t_max."""
    kf_, kd_ = kf(T_C), kd(T_C)
    tm = t_max(T_C)
    return (kf_ / (kd_ - kf_)) * (np.exp(-kf_ * tm) - np.exp(-kd_ * tm))


# ---- Compute at key temperatures ----
T_list = [25, 0, -25, -50, -78]
print("\n" + "=" * 95)
print(f"{'T (°C)':>8}  {'k_f (s⁻¹)':>11}  {'k_d (s⁻¹)':>11}  "
      f"{'t½_form (s)':>13}  {'t½_ArLi (s)':>13}  {'t_max (s)':>11}  {'max ArLi %':>11}")
print("=" * 95)
for T in T_list:
    print(f"{T:>8}  {kf(T):>11.3f}  {kd(T):>11.3e}  "
          f"{t_half_form(T):>13.3f}  {t_half_arli(T):>13.3e}  "
          f"{t_max(T):>11.3f}  {y_max_at(T)*100:>11.2f}")

# ---- Plot ----
fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
T_grid = np.linspace(-100, 60, 200)

# Panel 1: rate constants
ax = axes[0]
ax.semilogy(1000/(T_grid+273.15), kf(T_grid), 'b-', linewidth=2,
            label=f'k_f (experimental)\nEa_f={EA_F_EXP:.1f}, lnA_f={LNA_F_EXP:.1f}')
ax.semilogy(1000/(T_grid+273.15), kd(T_grid), 'r-', linewidth=2,
            label=f'k_d (v4.6 prediction)\nEa_d={EA_D_MODEL:.1f}, lnA_d={LNA_D_MODEL:.1f}')
for T in T_list:
    ax.axvline(1000/(T+273.15), color='gray', alpha=0.3, linestyle=':')
    ax.text(1000/(T+273.15), 3e-3, f'{T}°C', ha='center', fontsize=8,
            color='gray', rotation=90)
ax.set_xlabel('1000 / T  (1/K)', fontsize=11)
ax.set_ylabel('Rate constant (s⁻¹)', fontsize=11)
ax.set_title('Arrhenius plot:  k_f (experiment) vs k_d (v4.6 model)', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3, which='both')
ax.legend(fontsize=10, loc='upper right')
ax.set_ylim(1e-4, 1e3)

# Panel 2: time constants
ax = axes[1]
ax.semilogy(T_grid, t_half_form(T_grid), 'b-', linewidth=2, label='t½ formation  (k_f)')
ax.semilogy(T_grid, t_half_arli(T_grid), 'r-', linewidth=2, label='t½ ArLi decay  (k_d)')
ax.semilogy(T_grid, np.array([t_max(t) for t in T_grid]), 'g--', linewidth=2.5,
            label='t_max  (ArLi peak)')
for T in T_list:
    ax.axvline(T, color='gray', alpha=0.3, linestyle=':')

# Reactor zones (visual hint)
ax.axhspan(1e-4, 0.1, alpha=0.10, color='red',    label=None)
ax.axhspan(0.1,  60,  alpha=0.10, color='orange', label=None)
ax.axhspan(60,   3600,alpha=0.10, color='green',  label=None)
ax.text(60, 0.03, 'flash (<0.1s)', fontsize=9, color='darkred')
ax.text(60, 3,    'flow (0.1-60s)', fontsize=9, color='darkorange')
ax.text(60, 300,  'batch (>60s)',   fontsize=9, color='darkgreen')

ax.set_xlabel('Temperature / °C', fontsize=11)
ax.set_ylabel('Characteristic time (s)', fontsize=11)
ax.set_title('Time constants vs T  (p-FC₆H₄Li)', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3, which='both')
ax.legend(fontsize=10, loc='upper right')
ax.set_xlim(-100, 60)
ax.set_ylim(1e-4, 3600)

# Annotate key T
for T in T_list:
    tf = t_half_form(T); td = t_half_arli(T); tm = t_max(T)
    ax.scatter([T], [tf], s=70, color='blue', zorder=5, edgecolor='black')
    ax.scatter([T], [td], s=70, color='red',  zorder=5, edgecolor='black')
    ax.scatter([T], [tm], s=90, color='green', marker='*', zorder=5, edgecolor='black')

plt.suptitle('p-FC₆H₄Li kinetics: experimental k_f + v4.6-predicted k_d',
             fontsize=13, fontweight='bold')
plt.tight_layout()
OUT_PNG.parent.mkdir(exist_ok=True)
plt.savefig(OUT_PNG, dpi=160, bbox_inches='tight')
print(f"\nSaved: {OUT_PNG}")
