"""
GLOBAL Arrhenius fit (the right way for n=12-15 sparse data):
  yield(tR, T) = y_max × (1 - exp(-k_f(T)·tR)) × exp(-k_d(T)·tR)
  k_f(T) = exp(lnA_f - Ea_f / RT)
  k_d(T) = exp(lnA_d - Ea_d / RT)

All temperatures share ONE y_max and ONE pair of (Ea, lnA) per channel.
Total parameters: 5  (y_max, Ea_f, lnA_f, Ea_d, lnA_d)
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize

CSV = Path(__file__).parent / "experiment_yield_with_flags.csv"
OUT_PNG = Path(__file__).parent / "analysis_figures" / "experiment_global_arrhenius.png"

R = 8.314e-3   # kJ/(mol K)
EXCLUDE_FLAGS = {"anomaly", "rebound"}


def model(params, tR_s, T_K):
    y_max, Ea_f, lnA_f, Ea_d, lnA_d = params
    kf = np.exp(lnA_f - Ea_f / (R * T_K))
    kd = np.exp(lnA_d - Ea_d / (R * T_K))
    return y_max * (1 - np.exp(-kf * tR_s)) * np.exp(-kd * tR_s)


def fit(df, label):
    tR = df["tR_s"].to_numpy()
    Tk = df["T_C"].to_numpy() + 273.15
    Y  = df["yield_pct_ECN"].to_numpy()

    def loss(p):
        return np.sum((model(p, tR, Tk) - Y) ** 2)

    # Global search first (DE), then local refine
    bounds = [(50, 200),     # y_max
              (5, 80),       # Ea_f kJ/mol
              (10, 30),      # lnA_f
              (5, 100),      # Ea_d
              (5, 30)]       # lnA_d
    res_de = differential_evolution(loss, bounds, seed=42, tol=1e-10,
                                     maxiter=500, polish=True)
    res = minimize(loss, res_de.x, method="Nelder-Mead",
                    options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 50000})
    p = res.x
    y_pred = model(p, tR, Tk)
    rss = np.sum((Y - y_pred) ** 2)
    tss = np.sum((Y - Y.mean()) ** 2)
    R2 = 1 - rss / tss
    rmse = np.sqrt(rss / len(Y))
    print(f"\n>>> {label}  (n={len(Y)})")
    print(f"  y_max  = {p[0]:.2f}%   (theoretical 100%, scaled by RF)")
    print(f"  Ea_f   = {p[1]:.2f} kJ/mol")
    print(f"  lnA_f  = {p[2]:.2f}")
    print(f"  Ea_d   = {p[3]:.2f} kJ/mol  ← key for HYBRID v2")
    print(f"  lnA_d  = {p[4]:.2f}")
    print(f"  R²     = {R2:.4f}")
    print(f"  RMSE   = {rmse:.2f} % yield")
    return p, R2, rmse


def main():
    df_full = pd.read_csv(CSV)
    df = df_full[~df_full["flag"].fillna("").isin(EXCLUDE_FLAGS)].dropna(subset=["yield_pct_ECN"])

    df_A = df.copy()                              # 15 pts: w/ -25°C
    df_B = df[df["T_C"] != -25].copy()            # 12 pts: only day-2

    pA, R2_A, rmse_A = fit(df_A, "Group A: 15 pts (w/ -25°C day-1)")
    pB, R2_B, rmse_B = fit(df_B, "Group B: 12 pts (only day-2)")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, df_, p, R2, label in [
        (axes[0], df_A, pA, R2_A, f"Group A: 15 pts (w/ -25°C)\nR²={R2_A:.3f}"),
        (axes[1], df_B, pB, R2_B, f"Group B: 12 pts (day-2 only)\nR²={R2_B:.3f}"),
    ]:
        T_unique = sorted(df_["T_C"].unique())
        cmap = plt.get_cmap("coolwarm_r")
        for i, T in enumerate(T_unique):
            color = cmap(i / max(len(T_unique) - 1, 1))
            sub = df_[df_["T_C"] == T]
            ax.scatter(sub["tR_s"], sub["yield_pct_ECN"], s=110,
                       color=color, edgecolor="black", linewidth=1.2,
                       zorder=5, label=f"T={T}°C  (n={len(sub)})")
            xx = np.logspace(np.log10(0.02), np.log10(2.5), 200)
            yy = model(p, xx, T + 273.15)
            ax.plot(xx, yy, "--", color=color, linewidth=1.6, alpha=0.85)

        ax.set_xscale("log")
        ax.set_xlabel("Residence time tR / s", fontsize=12)
        ax.set_ylabel("Yield % (ECN-est.)", fontsize=12)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc="upper right")

        # parameter box
        Ea_d_pred_lo, Ea_d_pred_hi = 55, 65
        in_range = Ea_d_pred_lo <= p[3] <= Ea_d_pred_hi
        verdict = "✓" if in_range else "≠"
        text = (f"GLOBAL FIT (5 params, all T share):\n"
                f"  y_max  = {p[0]:.1f}% (scale by RF later)\n"
                f"  Ea_f   = {p[1]:.1f} kJ/mol  (formation)\n"
                f"  lnA_f  = {p[2]:.1f}\n"
                f"  Ea_d   = {p[3]:.1f} kJ/mol  ← decay\n"
                f"  lnA_d  = {p[4]:.1f}\n"
                f"  R²     = {R2:.3f}\n\n"
                f"HYBRID v2 prediction:\n"
                f"  Ea_d ≈ 55–65 kJ/mol\n"
                f"  Verdict: {verdict}  experimental Ea_d = {p[3]:.1f}")
        ax.text(0.04, 0.05, text, transform=ax.transAxes, va="bottom",
                fontsize=10, family="monospace",
                bbox=dict(boxstyle="round,pad=0.5",
                          fc="#e8f4ea" if in_range else "#fbeaea",
                          ec="black", alpha=0.95))

    plt.suptitle("Global Arrhenius fit — yield = y_max·(1-exp(-k_f·tR))·exp(-k_d·tR)\n"
                 "p-FC₆H₄Br + n-BuLi flow validation",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    OUT_PNG.parent.mkdir(exist_ok=True)
    plt.savefig(OUT_PNG, dpi=160, bbox_inches="tight")
    print(f"\nSaved: {OUT_PNG}")


if __name__ == "__main__":
    main()
