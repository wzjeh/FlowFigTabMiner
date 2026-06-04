"""
Path A — mixing-corrected formation prediction for the two in-house validation substrates.

Keep the v6.2 model's INTRINSIC chemical formation rate k_chem(T) = exp(lnA_f − Ea_f/RT),
then fold in micromixing via a SINGLE apparatus parameter τ_mix (series-resistance):

    k_f,eff(T) = k_chem(T) / (1 + Da),   Da = τ_mix · k_chem(T)
    yield(tR,T) = y_max · (1 − exp(−k_f,eff·tR)) · exp(−kd·tR)

τ_mix is the rig fingerprint → fitted once, SHARED across both substrates and all T.
Equivalently a maximum effective formation rate  k_f,max = 1/τ_mix  (an apparatus cap).
Decomposition (kd) and y_max are unchanged. Reports baseline (τ_mix=0) vs corrected MAE,
plus a per-substrate τ_mix for diagnosis. Honest: shows where mixing helps and where it cannot.
"""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from pathlib import Path
from draw_paper_figures import VALIDATION, _load_exp, R_GAS   # single source of truth

BASE = Path(__file__).parent
plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"]})

def kf_eff(p, tR, T_C, tau_mix):
    Tk = np.asarray(T_C, float) + 273.15
    kchem = np.exp(p["lnA_f"] - p["Ea_f"] / (R_GAS * Tk))
    return kchem / (1.0 + tau_mix * kchem)               # series resistance

def predict(p, tR, T_C, tau_mix):
    Tk = np.asarray(T_C, float) + 273.15
    keff = kf_eff(p, tR, T_C, tau_mix)
    kd = np.exp(p["lnA_d"] - p["Ea_d"] / (R_GAS * Tk))
    tR = np.asarray(tR, float)
    return p["y_max"] * (1 - np.exp(-keff * tR)) * np.exp(-kd * tR)

# ---- load both substrates' cleaned experimental data (same prep as the figures) ----
DATASETS = []
for cfg in VALIDATION:
    df = _load_exp(cfg)
    DATASETS.append(dict(key=cfg["key"], name=cfg["name"], p=cfg["pred"],
                         tR=df["tR_s"].values, T=df["T_C"].values, y=df["val"].values))

def mae_for_tau(tau_mix, sets):
    err = []
    for d in sets:
        pr = predict(d["p"], d["tR"], d["T"], tau_mix)
        err.append(np.abs(pr - d["y"]))
    return np.mean(np.concatenate(err))

def fit_tau(sets):
    res = minimize_scalar(lambda lt: mae_for_tau(10**lt, sets),
                          bounds=(-5, 0), method="bounded")   # τ_mix ∈ [1e-5, 1] s
    return 10**res.x, res.fun

# ---- baseline (τ_mix→0, current model) vs single shared τ_mix ----
base = mae_for_tau(1e-12, DATASETS)
tau_shared, mae_shared = fit_tau(DATASETS)
print(f"{'='*68}\nBASELINE (no mixing, current v6.2):  global MAE = {base:5.2f} pp")
print(f"SHARED τ_mix fit:  τ_mix = {tau_shared*1e3:8.3f} ms  (k_f,max = {1/tau_shared:7.1f} /s)"
      f"   global MAE = {mae_shared:5.2f} pp\n{'='*68}")

# per-substrate diagnosis
for d in DATASETS:
    b = mae_for_tau(1e-12, [d]); t, m = fit_tau([d])
    pr_sh = predict(d["p"], d["tR"], d["T"], tau_shared)
    m_sh = np.mean(np.abs(pr_sh - d["y"]))
    print(f"[{d['key']:9s}] baseline MAE={b:5.2f} | own-τ_mix={t*1e3:7.3f}ms MAE={m:5.2f} "
          f"| shared-τ MAE={m_sh:5.2f}")

# ---- before/after curves, faceted by temperature ----
for d in DATASETS:
    Ts = sorted(set(np.round(d["T"]).astype(int)))
    n = len(Ts); fig, axs = plt.subplots(1, n, figsize=(3.0*n, 3.0), sharey=True)
    axs = np.atleast_1d(axs)
    tgrid = np.logspace(np.log10(d["tR"].min()*0.7), np.log10(d["tR"].max()*1.2), 200)
    for ax, T in zip(axs, Ts):
        mask = np.round(d["T"]).astype(int) == T
        ax.scatter(d["tR"][mask], d["y"][mask], s=45, c="k", zorder=5, label="exp")
        ax.plot(tgrid, predict(d["p"], tgrid, T, 1e-12), "--", c="#888", lw=1.6,
                label="v6.2 (no mixing)")
        ax.plot(tgrid, predict(d["p"], tgrid, T, tau_shared), "-", c="#c0392b", lw=2.0,
                label=f"+mixing (τ={tau_shared*1e3:.2f}ms)")
        ax.set_xscale("log"); ax.set_title(f"{T} °C", fontsize=10)
        ax.set_xlabel("$t_R$ (s)"); ax.grid(alpha=0.25)
    axs[0].set_ylabel("yield (%)"); axs[0].set_ylim(0, 100)
    axs[-1].legend(fontsize=7.5, loc="lower right")
    fig.suptitle(f"{d['name']} — mixing-corrected formation (Path A)", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = BASE / "analysis_figures" / f"fig_mixing_correction_{d['key']}.png"
    fig.savefig(out, dpi=160, bbox_inches="tight"); plt.close(fig)
    print("✓", out)
