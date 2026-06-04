"""
Fit formation rate constant k_f from substrate-conversion data.

Model (no-side-product assumption):
  substrate(tR, T) = subst_0 × exp(-k_f(T)·tR)
  conversion(tR, T) = 1 - exp(-k_f(T)·tR)
  k_f(T) = exp(lnA_f - Ea_f / RT)

Apply Zhao's suspected vial-label swap: T=0°C L=3 ↔ L=10.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize

CSV = Path(__file__).parent / "experiment_yield_summary.csv"
OUT_PNG = Path(__file__).parent / "analysis_figures" / "experiment_kf_arrhenius.png"
R = 8.314e-3  # kJ/(mol·K)


def apply_label_swap(df):
    """Swap (T=0, L=3) and (T=0, L=10) values (Zhao's vial mix-up hypothesis)."""
    a = df[(df.T_C == 0) & (df.L_cm == 3)].index[0]
    b = df[(df.T_C == 0) & (df.L_cm == 10)].index[0]
    cols_to_swap = ['subst_area', 'area_ratio_obs',
                    'residual_subst_pct_ECN', 'conversion_pct_ECN',
                    'yield_pct_ECN']
    for c in cols_to_swap:
        if c in df.columns:
            df.loc[a, c], df.loc[b, c] = df.loc[b, c], df.loc[a, c]
    return df


def kf_global_model(params, tR, T_K):
    Ea_f, lnA_f = params
    kf = np.exp(lnA_f - Ea_f / (R * T_K))
    return 1.0 - np.exp(-kf * tR)


def fit_global_kf(df, label):
    """Global Arrhenius fit on conversion (single rate constant k_f)."""
    tR = df.tR_s.to_numpy()
    Tk = df.T_C.to_numpy() + 273.15
    Y = df.conversion_pct_ECN.to_numpy() / 100.0

    def loss(p):
        return np.sum((kf_global_model(p, tR, Tk) - Y) ** 2)

    de = differential_evolution(loss, [(5, 100), (5, 35)], seed=42,
                                tol=1e-11, maxiter=600, polish=True)
    res = minimize(loss, de.x, method="Nelder-Mead",
                   options={"xatol": 1e-9, "fatol": 1e-9, "maxiter": 50000})
    Ea_f, lnA_f = res.x
    pred = kf_global_model(res.x, tR, Tk)
    rss = np.sum((Y - pred) ** 2); tss = np.sum((Y - Y.mean()) ** 2)
    R2 = 1 - rss / tss
    rmse_pct = 100 * np.sqrt(rss / len(Y))
    print(f"\n>>> {label}  (n={len(Y)})")
    print(f"  Ea_f  = {Ea_f:.2f} kJ/mol")
    print(f"  lnA_f = {lnA_f:.2f}  (s⁻¹)")
    print(f"  R²    = {R2:.4f}")
    print(f"  RMSE  = {rmse_pct:.2f} % conversion")
    return Ea_f, lnA_f, R2, rmse_pct


def fit_per_T_kf(df):
    """Per-T k_f fit using ln(1-conv) = -k_f·tR (no intercept) at each T."""
    out = {}
    for T_C, sub in df.groupby('T_C'):
        if len(sub) < 2: continue
        tR = sub.tR_s.to_numpy()
        # bound conversion away from 100% to avoid log(0)
        conv = np.clip(sub.conversion_pct_ECN.to_numpy() / 100, 0.001, 0.9999)
        y_log = np.log(1 - conv)
        # OLS through origin
        kf = -np.sum(tR * y_log) / np.sum(tR ** 2) if np.sum(tR**2) > 0 else None
        out[T_C] = kf
    return out


def main():
    df = pd.read_csv(CSV).dropna(subset=['conversion_pct_ECN']).copy()

    # ---- 1. without swap ----
    Ea1, lnA1, R2_1, rmse1 = fit_global_kf(df, "No swap (raw 22 pts)")

    # ---- 2. with vial-swap (T=0 L=3↔L=10) ----
    df_sw = apply_label_swap(df.copy())
    Ea2, lnA2, R2_2, rmse2 = fit_global_kf(df_sw, "After vial-swap (T=0°C L=3↔L=10)")

    # ---- 3. day-2 only (drop 0°C and -25°C entire rows) ----
    df_d2 = df[~df.T_C.isin([0, -25])].copy()
    Ea3, lnA3, R2_3, rmse3 = fit_global_kf(df_d2, "Day-2 only (drop 0/-25°C)")

    # per-T quick check
    print("\nPer-T k_f (from ln(1-conv) vs tR):")
    for label, dat in [("raw", df), ("swap", df_sw), ("day-2", df_d2)]:
        kfs = fit_per_T_kf(dat)
        print(f"  [{label:6s}] " + "  ".join(f"T={T}°C: kf={kf:6.2f}" for T,kf in sorted(kfs.items(), reverse=True)))

    # ---- plot 3 scenarios ----
    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    scenarios = [
        ("Raw 22 pts",          df,    Ea1, lnA1, R2_1, rmse1),
        ("After vial-swap",     df_sw, Ea2, lnA2, R2_2, rmse2),
        ("Day-2 only (12 pts)", df_d2, Ea3, lnA3, R2_3, rmse3),
    ]
    cmap = plt.get_cmap("coolwarm_r")
    for ax, (label, ddf, Ea, lnA, R2, rmse) in zip(axes, scenarios):
        Ts = sorted(ddf.T_C.unique())
        for i, T in enumerate(Ts):
            color = cmap(i / max(len(Ts) - 1, 1))
            sub = ddf[ddf.T_C == T]
            ax.scatter(sub.tR_s, sub.conversion_pct_ECN, s=110,
                       color=color, edgecolor="black", linewidth=1.2,
                       zorder=5, label=f"T={T}°C (n={len(sub)})")
            xx = np.logspace(np.log10(0.02), np.log10(2.5), 200)
            yy = 100 * kf_global_model([Ea, lnA], xx, T + 273.15)
            ax.plot(xx, yy, "--", color=color, linewidth=1.6, alpha=0.85)
        ax.set_xscale("log")
        ax.set_xlabel("tR / s"); ax.set_ylabel("Conversion %")
        ax.set_title(f"{label}\nEa_f={Ea:.1f} kJ/mol, lnA_f={lnA:.1f}\n"
                     f"R²={R2:.3f}, RMSE={rmse:.1f}%", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc='lower right')
        ax.set_ylim(0, 105)
        text = (f"v4.6 p-ArLi Ea_f prediction\n"
                f"  Ea_f = 14.07·σ − 18.01·B1 + 0.20·ΔH + 60.67\n"
                f"       = 27.1 kJ/mol\n"
                f"Expt: {Ea:.1f} kJ/mol  (Δ = {Ea-27.1:+.1f})")
        ax.text(0.04, 0.96, text, transform=ax.transAxes, va="top",
                fontsize=9, family="monospace",
                bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", ec="gray"))

    plt.suptitle("Formation k_f fit: conversion(tR,T) = 1 − exp(−k_f·tR), assuming no side-product\n"
                 "4-bromofluorobenzene → 4-FC₆H₄Li · pre-experiment 4-溴氟苯",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    OUT_PNG.parent.mkdir(exist_ok=True)
    plt.savefig(OUT_PNG, dpi=160, bbox_inches="tight")
    print(f"\nSaved: {OUT_PNG}")


if __name__ == "__main__":
    main()
