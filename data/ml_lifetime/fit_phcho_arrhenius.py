"""
Fit Arrhenius parameters from the 2026-05-12 PhCHO-trap dataset (25 vials).

L=10 label correction (2026-05-13):
  Zhao used a half-inner-diameter tube for the L=10cm spool, so its
  effective volume = (1/4) × (10 cm at φ500μm), equivalent to
  L_eff = 2.5 cm at normal 500 μm bore.  → tR shorter than L=3 row.

Two models tested:
  (A) k_f Arrhenius only (2 params) on substrate conversion
        conv(tR,T) = 1 - exp(-k_f(T)·tR)
  (B) 4-param bell curve on trap-product yield (fixed y_max = 100%)
        yield(tR,T) = (1 - exp(-k_f·tR)) · exp(-k_d·tR)
        params: Ea_f, lnA_f, Ea_d, lnA_d
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize

CSV = Path(__file__).parent / "experiment_phcho_summary.csv"
OUT_PNG = Path(__file__).parent / "analysis_figures" / "phcho_arrhenius_fit.png"
R = 8.314e-3   # kJ/(mol·K)

# vial → (T, L_effective_cm) — note L=10 label maps to L_eff=2.5 cm
L_ORDER = [3, 10, 25, 50, 100]
L_EFF   = {3: 3.0, 10: 2.5, 25: 25.0, 50: 50.0, 100: 100.0}
T_ORDER = [20, 0, -25, -50, -65]
T_BATCH = [20, 0, -25, -50]
VIAL_MAP = {}
i = 1
for T in T_BATCH:
    for L in L_ORDER:
        VIAL_MAP[i] = (T, L)
        i += 1
VIAL_MAP[21] = (-65, 3)
VIAL_MAP[22] = (-65, 10)
VIAL_MAP[23] = (-65, 25)
VIAL_MAP[24] = (-65, 50)
VIAL_MAP[25] = (-65, 100)

TUBE_AREA_CM2 = np.pi * 0.025 ** 2
TOTAL_FLOW_ML_S = 7.5 / 60


# ---- models ----
def conv_2param(params, tR, T_K):
    Ea_f, lnA_f = params
    kf = np.exp(lnA_f - Ea_f / (R * T_K))
    return 1.0 - np.exp(-kf * tR)


def yield_4param(params, tR, T_K):
    Ea_f, lnA_f, Ea_d, lnA_d = params
    kf = np.exp(lnA_f - Ea_f / (R * T_K))
    kd = np.exp(lnA_d - Ea_d / (R * T_K))
    return (1 - np.exp(-kf * tR)) * np.exp(-kd * tR)


def fit_global(model_fn, x_tR, y_T, y_obs, bounds, label):
    def loss(p):
        return np.sum((model_fn(p, x_tR, y_T) - y_obs) ** 2)
    de = differential_evolution(loss, bounds, seed=42, tol=1e-11, maxiter=1500, polish=True)
    res = minimize(loss, de.x, method="Nelder-Mead",
                   options={"xatol": 1e-10, "fatol": 1e-10, "maxiter": 100000})
    p = res.x
    pred = model_fn(p, x_tR, y_T)
    rss = np.sum((y_obs - pred) ** 2)
    tss = np.sum((y_obs - y_obs.mean()) ** 2)
    R2 = 1 - rss / tss
    rmse = np.sqrt(rss / len(y_obs))
    print(f"\n>>> {label}  (n={len(y_obs)})")
    if len(p) == 2:
        print(f"  Ea_f  = {p[0]:.2f} kJ/mol   lnA_f = {p[1]:.2f}")
    elif len(p) == 4:
        print(f"  Ea_f  = {p[0]:.2f} kJ/mol   lnA_f = {p[1]:.2f}")
        print(f"  Ea_d  = {p[2]:.2f} kJ/mol   lnA_d = {p[3]:.2f}")
        kd_298 = np.exp(p[3] - p[2] / (R * 298.15))
        kf_298 = np.exp(p[1] - p[0] / (R * 298.15))
        print(f"  k_f(298K) = {kf_298:.3f} s⁻¹     k_d(298K) = {kd_298:.3f} s⁻¹")
    print(f"  R²   = {R2:.4f}   RMSE = {rmse*100:.2f} %-pts")
    return p, R2, rmse


def main():
    df = pd.read_csv(CSV)
    df["T_C"]  = df["vial"].map(lambda v: VIAL_MAP.get(v, (None, None))[0])
    df["L_cm_label"] = df["vial"].map(lambda v: VIAL_MAP.get(v, (None, None))[1])
    df["L_cm_eff"]   = df["L_cm_label"].map(L_EFF)
    df["tR_s"]       = TUBE_AREA_CM2 * df["L_cm_eff"] / TOTAL_FLOW_ML_S
    df = df.dropna(subset=["T_C","L_cm_eff","conversion_pct","trap_yield_pct"]).copy()

    print("Data table after L=10→L_eff=2.5 correction:")
    print(df[["vial","T_C","L_cm_label","L_cm_eff","tR_s",
              "conversion_pct","trap_yield_pct"]]
          .sort_values(["T_C","tR_s"], ascending=[False, True])
          .to_string(index=False))

    Tk  = (df["T_C"] + 273.15).to_numpy()
    tR  = df["tR_s"].to_numpy()
    conv = df["conversion_pct"].to_numpy() / 100.0
    yld  = df["trap_yield_pct"].to_numpy() / 100.0

    # ---- Model A: 2-param k_f Arrhenius on conversion ----
    pA, R2_A, rmse_A = fit_global(
        conv_2param, tR, Tk, conv,
        bounds=[(5, 100), (5, 35)],
        label="Model A: k_f Arrhenius (2 params) on conversion")

    # ---- Model B: 4-param bell curve on trap-yield ----
    # If RRF was estimated slightly low, yields > 100%. Clip to ≤100% for fit
    yld_fit = np.clip(yld, 0.001, 1.0)
    pB, R2_B, rmse_B = fit_global(
        yield_4param, tR, Tk, yld_fit,
        bounds=[(5, 100), (5, 35), (0.1, 100), (-5, 35)],
        label="Model B: 4-param bell curve (Ea_f,lnA_f,Ea_d,lnA_d) on trap yield")

    # ---- plot ----
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    cmap = plt.get_cmap("coolwarm_r")

    # Panel 1: conversion + model A
    ax = axes[0]
    Ts = sorted(df.T_C.unique())
    for i, T in enumerate(Ts):
        color = cmap(i / max(len(Ts)-1, 1))
        sub = df[df.T_C == T].sort_values("tR_s")
        ax.scatter(sub.tR_s, sub.conversion_pct, s=110, color=color,
                   edgecolor="black", linewidth=1.2, zorder=5,
                   label=f"T={T}°C (n={len(sub)})")
        xx = np.logspace(np.log10(0.015), np.log10(2.5), 200)
        yy = 100 * conv_2param(pA, xx, T + 273.15)
        ax.plot(xx, yy, "--", color=color, linewidth=1.6, alpha=0.85)
    ax.set_xscale("log")
    ax.set_xlabel("Residence time tR / s", fontsize=11)
    ax.set_ylabel("Substrate conversion %", fontsize=11)
    ax.set_title(f"Model A: 2-param k_f Arrhenius (on conversion)\n"
                 f"Ea_f={pA[0]:.1f} kJ/mol, lnA_f={pA[1]:.1f}, R²={R2_A:.3f}",
                 fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3); ax.legend(fontsize=9, loc="lower right")
    ax.set_ylim(0, 110)

    # Panel 2: yield + model B
    ax = axes[1]
    for i, T in enumerate(Ts):
        color = cmap(i / max(len(Ts)-1, 1))
        sub = df[df.T_C == T].sort_values("tR_s")
        ax.scatter(sub.tR_s, sub.trap_yield_pct, s=110, color=color,
                   edgecolor="black", linewidth=1.2, zorder=5,
                   label=f"T={T}°C (n={len(sub)})")
        xx = np.logspace(np.log10(0.015), np.log10(2.5), 200)
        yy = 100 * yield_4param(pB, xx, T + 273.15)
        ax.plot(xx, yy, "--", color=color, linewidth=1.6, alpha=0.85)
    ax.set_xscale("log")
    ax.set_xlabel("Residence time tR / s", fontsize=11)
    ax.set_ylabel("Trap product yield %", fontsize=11)
    ax.set_title(f"Model B: 4-param bell curve (on trap yield)\n"
                 f"Ea_f={pB[0]:.1f}, lnA_f={pB[1]:.1f}, "
                 f"Ea_d={pB[2]:.1f}, lnA_d={pB[3]:.1f}, R²={R2_B:.3f}",
                 fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3); ax.legend(fontsize=9, loc="lower right")
    ax.set_ylim(0, 120)

    plt.suptitle("p-FC₆H₄Li · PhCHO-trap experiment (n=25) · Arrhenius fits\n"
                 "L=10cm label corrected to L_eff=2.5cm (half-bore tube)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    OUT_PNG.parent.mkdir(exist_ok=True)
    plt.savefig(OUT_PNG, dpi=160, bbox_inches="tight")
    print(f"\nSaved: {OUT_PNG}")

    # ---- Compare to v4.6 HYBRID v3 predictions ----
    print("\n" + "=" * 60)
    print("Comparison vs v4.6 HYBRID v3 prediction for p-FC₆H₄Li:")
    print("=" * 60)
    # v4.6 p-ArLi formulas (from earlier compute):
    print("  v4.6 prediction:")
    print("    Ea_f ≈ 34.4 kJ/mol  (LOO-R²=−0.19, weak)")
    print("    Ea_d ≈ 39.9 kJ/mol  (LOO-R²=0.97)")
    print(f"  Experimental (model A, conversion only):  Ea_f = {pA[0]:.1f}")
    if len(pB) == 4:
        print(f"  Experimental (model B, bell curve):")
        print(f"    Ea_f = {pB[0]:.1f}    Ea_d = {pB[2]:.1f}")


if __name__ == "__main__":
    main()
