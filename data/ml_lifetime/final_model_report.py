"""
Final layered-model report for the two in-house substrates:
  • print the explicit model (equations + all constants per substrate)
  • prediction-vs-actual table + MAE/bias per substrate
  • parity figure (pred vs measured yield)

Model:
  k_chem(σ,T)  = exp(lnA0 + ρ·σ − Ea_f/(R·Tk))          fixed Charton LFER (intrinsic chemistry)
  k_f,obs      = k_chem / (1 + τ_eff·k_chem)             observation layer (resistance-in-series)
  kd(T)        = exp(lnA_d − Ea_d/(R·Tk))                decomposition (unchanged)
  yield(tR,T)  = y_max·(1 − exp(−k_f,obs·tR))·exp(−kd·tR)
"""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import formation_model as fm
from draw_paper_figures import VALIDATION, _load_exp

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"]})
BASE = Path(__file__).parent
R = fm.R_GAS
SIGMA = {"4BrFC6H4": 0.06, "5Br2FCN": 0.62}
TAU_EFF = 21.9e-3      # s — rig fingerprint, calibrated once (500 µm mixer, global fit)

def predict(p, sigma, tR, T_C):
    Tk = np.asarray(T_C, float) + 273.15
    kc = fm.k_chem(sigma, T_C)
    keff = fm.observation_model(kc, TAU_EFF, form="series")
    kd = np.exp(p["lnA_d"] - p["Ea_d"] / (R * Tk))
    return p["y_max"] * (1 - np.exp(-keff * np.asarray(tR, float))) * np.exp(-kd * np.asarray(tR, float))

print("="*78)
print("FINAL LAYERED MODEL  (intrinsic chemistry × observation layer)")
print("="*78)
print(f"  Formation (FIXED Charton eq.36, σ-in-lnA):")
print(f"     k_chem(σ,T) = exp( {fm.LNA0:.0f} + {fm.RHO_LNA:.2f}·σ − {fm.EA_F:.0f}/(R·T) )   [/s]")
print(f"  Observation layer (resistance-in-series):")
print(f"     k_f,obs = k_chem / (1 + τ_eff·k_chem),   τ_eff = {TAU_EFF*1e3:.1f} ms (500 µm rig)")
print(f"  Decomposition:  kd(T) = exp(lnA_d − Ea_d/(R·T));  R = {R} kJ/mol·K")
print(f"  Yield(tR,T) = y_max·(1−exp(−k_f,obs·tR))·exp(−kd·tR)\n")

rows = []
for cfg in VALIDATION:
    p, sig = cfg["pred"], SIGMA[cfg["key"]]
    df = _load_exp(cfg)
    yp = predict(p, sig, df["tR_s"].values, df["T_C"].values)
    ya = df["val"].values
    mae = np.mean(np.abs(yp - ya)); bias = np.mean(yp - ya)
    print("-"*78)
    print(f"  {cfg['name']}  ({cfg['key']})   σ={sig}")
    print(f"     decomposition: Ea_d={p['Ea_d']}, lnA_d={p['lnA_d']}, y_max={p['y_max']}")
    print(f"     → MAE = {mae:5.2f} pp,  bias = {bias:+5.2f} pp,  n = {len(ya)}")
    # condensed table: shortest (3 cm) & longest (100 cm) tR at each T
    d = df.copy(); d["pred"] = yp
    print(f"     {'T(°C)':>6} {'tR(s)':>7} {'L(cm)':>6} {'actual':>7} {'pred':>7} {'Δ':>6}")
    for T in sorted(set(np.round(d["T_C"]).astype(int))):
        sub = d[np.round(d["T_C"]).astype(int) == T].sort_values("tR_s")
        for _, r in sub.iloc[[0, -1]].iterrows():
            Lc = r.get("L_cm", np.nan)
            print(f"     {T:6d} {r['tR_s']:7.3f} {Lc:6.0f} {r['val']:7.1f} {r['pred']:7.1f} {r['pred']-r['val']:+6.1f}")
    rows.append(dict(key=cfg["key"], name=cfg["name"], sigma=sig, n=len(ya),
                     MAE=round(mae, 2), bias=round(bias, 2),
                     ya=ya, yp=yp))
print("="*78)

# ---- parity figure ----
fig, ax = plt.subplots(figsize=(5.6, 5.6))
col = {"4BrFC6H4": "#1f6fb2", "5Br2FCN": "#c0392b"}
for r in rows:
    ax.scatter(r["ya"], r["yp"], s=50, c=col[r["key"]], edgecolor="k", lw=0.6,
               label=f"{r['name'][:24]} (MAE {r['MAE']})", zorder=5)
ax.plot([0, 100], [0, 100], "k--", lw=1.2, alpha=0.6)
for off in (10, -10):
    ax.plot([0, 100], [off, 100+off], ":", c="0.6", lw=0.8)
ax.set_xlim(20, 100); ax.set_ylim(20, 100)
ax.set_xlabel("measured yield (%)", fontsize=11); ax.set_ylabel("predicted yield (%)", fontsize=11)
ax.set_title("Final layered model — prediction vs measured\n(fixed Charton χ × Da observation layer; ±10 pp dotted)",
             fontsize=10.5, fontweight="bold")
ax.legend(fontsize=8.5, loc="upper left"); ax.grid(alpha=0.25)
out = BASE / "analysis_figures" / "fig_final_model_parity.png"
fig.savefig(out, dpi=160, bbox_inches="tight"); plt.close(fig)
print("✓", out)
