"""
Verification: is the 5Br2FCN short-L deficit a MIXING effect or a CHEMICAL Ea_f error?

His rig (M1 = T-mixer ID 250 μm, R1 ID 0.5 mm, 7.5 mL/min, 0.11 M dilute) has a geometric
micromixing time τ_mix ≈ d/u ≈ 0.1 ms (×3 engulfment ≈ 0.3 ms) — Da << 1 → CHEMICAL regime.
The Path-A fit needed τ_mix ≈ 25 ms (~100–250× larger), which is unphysical for this mixer.
→ Hypothesis: the deficit is the v6.2 Bayesian formation Arrhenius (Ea_f/lnA_f) being wrong,
   not mixing. Since his rig is chemically controlled, Ea_f IS identifiable from his own
   5-temperature data. Refit it and compare.
"""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import minimize
from draw_paper_figures import VALIDATION, _load_exp, R_GAS
plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"]})
BASE = Path(__file__).parent

cfg = next(c for c in VALIDATION if c["key"] == "5Br2FCN")
p0 = cfg["pred"]
df = _load_exp(cfg)
tR, T, y = df["tR_s"].values, df["T_C"].values, df["val"].values
Tk = T + 273.15

def model(Ea_f, lnA_f, Ea_d, lnA_d, ymax):
    kf = np.exp(lnA_f - Ea_f / (R_GAS * Tk))
    kd = np.exp(lnA_d - Ea_d / (R_GAS * Tk))
    return ymax * (1 - np.exp(-kf * tR)) * np.exp(-kd * tR)

def mae(pred): return np.mean(np.abs(pred - y))

# baseline (v6.2 as-is)
base = mae(model(p0["Ea_f"], p0["lnA_f"], p0["Ea_d"], p0["lnA_d"], p0["y_max"]))

# refit (Ea_f, lnA_f) only, keep decomposition + y_max from model
def obj2(x): return np.mean((model(x[0], x[1], p0["Ea_d"], p0["lnA_d"], p0["y_max"]) - y)**2)
r2 = minimize(obj2, [p0["Ea_f"], p0["lnA_f"]], method="Nelder-Mead",
              options=dict(xatol=1e-4, fatol=1e-6, maxiter=5000))
Eaf2, lnAf2 = r2.x
mae2 = mae(model(Eaf2, lnAf2, p0["Ea_d"], p0["lnA_d"], p0["y_max"]))

# full 5-param refit
def obj5(x): return np.mean((model(*x) - y)**2)
r5 = minimize(obj5, [p0["Ea_f"], p0["lnA_f"], p0["Ea_d"], p0["lnA_d"], p0["y_max"]],
              method="Nelder-Mead", options=dict(xatol=1e-4, fatol=1e-6, maxiter=20000))
mae5 = mae(model(*r5.x))

print("="*70)
print(f"5Br2FCN — what actually fixes the short-L deficit?  (n={len(y)} points)")
print("="*70)
print(f"  baseline v6.2 (no change)         : MAE = {base:5.2f} pp"
      f"   [Ea_f={p0['Ea_f']:.1f}, lnA_f={p0['lnA_f']:.2f}]")
print(f"  Path A  mixing cap τ_mix=25.2 ms  : MAE =  5.71 pp   (UNPHYSICAL: geom τ_mix≈0.1 ms)")
print(f"  refit Ea_f,lnA_f (his own data)   : MAE = {mae2:5.2f} pp"
      f"   [Ea_f={Eaf2:.1f}, lnA_f={lnAf2:.2f}]")
print(f"  full 5-param refit                : MAE = {mae5:5.2f} pp")
print(f"      [Ea_f={r5.x[0]:.1f}, lnA_f={r5.x[1]:.2f}, Ea_d={r5.x[2]:.1f}, "
      f"lnA_d={r5.x[3]:.2f}, y_max={r5.x[4]:.1f}]")
# Da check at his rig with refit Ea_f
TAU_GEOM = 0.098e-3  # s, M1 250µm @7.5mL/min
for Tref in [-70, -25, 20]:
    kc = np.exp(lnAf2 - Eaf2 / (R_GAS * (Tref + 273.15)))
    print(f"      Da@{Tref:+d}°C (refit k_chem={kc:8.1f}/s, τ_mix=0.098ms) = {TAU_GEOM*kc:.4f}")
print("="*70)

# ---- figure: the REAL fix is the chemical Ea_f refit, not a mixing cap ----
Ts = sorted(set(np.round(T).astype(int)))
fig, axs = plt.subplots(1, len(Ts), figsize=(3.0*len(Ts), 3.0), sharey=True)
axs = np.atleast_1d(axs)
tg = np.logspace(np.log10(tR.min()*0.7), np.log10(tR.max()*1.2), 200)
Tkg = None
def curve(Ea_f, lnA_f, Ea_d, lnA_d, ymax, Tc):
    Tkk = Tc + 273.15
    kf = np.exp(lnA_f - Ea_f / (R_GAS * Tkk)); kd = np.exp(lnA_d - Ea_d / (R_GAS * Tkk))
    return ymax * (1 - np.exp(-kf * tg)) * np.exp(-kd * tg)
for ax, Tc in zip(axs, Ts):
    m = np.round(T).astype(int) == Tc
    ax.scatter(tR[m], y[m], s=45, c="k", zorder=5, label="exp")
    ax.plot(tg, curve(p0["Ea_f"], p0["lnA_f"], p0["Ea_d"], p0["lnA_d"], p0["y_max"], Tc),
            "--", c="#888", lw=1.6, label="v6.2 Bayesian $E_{a,f}$")
    ax.plot(tg, curve(Eaf2, lnAf2, p0["Ea_d"], p0["lnA_d"], p0["y_max"], Tc),
            "-", c="#1f6fb2", lw=2.0, label="refit $E_{a,f}$ (his data)")
    ax.set_xscale("log"); ax.set_title(f"{Tc} °C", fontsize=10)
    ax.set_xlabel("$t_R$ (s)"); ax.grid(alpha=0.25)
axs[0].set_ylabel("yield (%)"); axs[0].set_ylim(0, 100)
axs[-1].legend(fontsize=7.5, loc="lower right")
fig.suptitle("5-bromo-2-fluorobenzonitrile — the deficit is a chemical $E_{a,f}$ error, not mixing\n"
             f"v6.2 $E_{{a,f}}$=35.2 (MAE 7.64) → refit $E_{{a,f}}$=9.3, ln$A_f$=7.9 (MAE 5.19); Da≈0.003 → chemical regime",
             fontsize=10.5, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.90])
out = BASE / "analysis_figures" / "fig_formation_refit_5Br2FCN.png"
fig.savefig(out, dpi=160, bbox_inches="tight"); plt.close(fig)
print("✓", out)
