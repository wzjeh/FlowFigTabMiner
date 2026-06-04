"""
Refit global Arrhenius dropping:
  - All T=+20°C points (3)
  - T=0°C L=100  (rebound)
  - T=-25°C L=50 (rebound)

Remaining: 17 points across 4 temperatures (-78, -50, -25, 0).
Now Arrhenius direction is monotonic: yield(L=3) decreases with T.

Model: yield = y_max × (1 - exp(-k_f·tR)) × exp(-k_d·tR)
      k_f(T) = exp(lnA_f - Ea_f/RT),   k_d(T) = exp(lnA_d - Ea_d/RT)
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize

CSV = Path(__file__).parent / "experiment_yield_summary.csv"
OUT_PNG = Path(__file__).parent / "analysis_figures" / "experiment_arrhenius_no20C.png"

R = 8.314e-3   # kJ/(mol K)


def model(params, tR_s, T_K):
    y_max, Ea_f, lnA_f, Ea_d, lnA_d = params
    kf = np.exp(lnA_f - Ea_f / (R * T_K))
    kd = np.exp(lnA_d - Ea_d / (R * T_K))
    return y_max * (1 - np.exp(-kf * tR_s)) * np.exp(-kd * tR_s)


def fit_global(df, label):
    tR = df["tR_s"].to_numpy()
    Tk = df["T_C"].to_numpy() + 273.15
    Y  = df["yield_pct_ECN"].to_numpy()

    def loss(p):
        return np.sum((model(p, tR, Tk) - Y) ** 2)

    bounds = [(50, 200),     # y_max
              (5, 100),      # Ea_f
              (10, 35),      # lnA_f
              (5, 120),      # Ea_d  (allow large)
              (5, 35)]       # lnA_d
    de = differential_evolution(loss, bounds, seed=42, tol=1e-11,
                                maxiter=800, polish=True)
    res = minimize(loss, de.x, method="Nelder-Mead",
                   options={"xatol": 1e-9, "fatol": 1e-9, "maxiter": 80000})
    p = res.x
    pred = model(p, tR, Tk)
    rss = np.sum((Y - pred) ** 2)
    tss = np.sum((Y - Y.mean()) ** 2)
    R2 = 1 - rss / tss
    rmse = np.sqrt(rss / len(Y))
    print(f"\n>>> {label}  (n={len(Y)})")
    print(f"  y_max  = {p[0]:.2f}%")
    print(f"  Ea_f   = {p[1]:.2f} kJ/mol  (formation)")
    print(f"  lnA_f  = {p[2]:.2f}")
    print(f"  Ea_d   = {p[3]:.2f} kJ/mol  (decay) ← key")
    print(f"  lnA_d  = {p[4]:.2f}")
    print(f"  R²     = {R2:.4f}")
    print(f"  RMSE   = {rmse:.2f} % yield")
    return p, R2, rmse


def main():
    df = pd.read_csv(CSV).dropna(subset=["yield_pct_ECN"]).copy()

    # ---- Three scenarios ----
    drop_A = df[
        (df["T_C"] == 20) |
        ((df["T_C"] == 0)   & (df["L_cm"] == 100)) |
        ((df["T_C"] == -25) & (df["L_cm"] == 50))
    ].index
    drop_B = df[
        (df["T_C"] == 20) |
        (df["T_C"] == 0)   |    # drop entire 0°C row (still day-1 suspect)
        ((df["T_C"] == -25) & (df["L_cm"] == 50))
    ].index
    drop_C = df[
        (df["T_C"] == 20) |
        (df["T_C"] == 0)   |
        (df["T_C"] == -25)      # drop entire -25 row too (full day-1 cleanup)
    ].index

    p_all, R2_all, rmse_all = [], [], []
    df_list = []
    labels = ["A: 17pts (drop +20°C only + 2 outl.)",
              "B: 13pts (also drop 0°C row)",
              "C: 10pts (drop all day-1 = day-2 only)"]
    for drop_idx, lab in zip([drop_A, drop_B, drop_C], labels):
        df_u = df.drop(index=drop_idx).reset_index(drop=True)
        df_list.append(df_u)
        p, R2, rmse = fit_global(df_u, lab)
        p_all.append(p); R2_all.append(R2); rmse_all.append(rmse)

    p, R2, rmse = p_all[0], R2_all[0], rmse_all[0]
    df_use = df_list[0]

    # ---- 3-panel plot, one per scenario ----
    fig, axes = plt.subplots(1, 3, figsize=(21, 6.5))
    cmap = plt.get_cmap("coolwarm_r")

    for ax, df_u, p_, R2_, rmse_, lab in zip(
            axes, df_list, p_all, R2_all, rmse_all, labels):
        Ts = sorted(df_u["T_C"].unique())
        for i, T in enumerate(Ts):
            color = cmap(i / max(len(Ts) - 1, 1))
            sub = df_u[df_u["T_C"] == T]
            ax.scatter(sub["tR_s"], sub["yield_pct_ECN"], s=110,
                       color=color, edgecolor="black", linewidth=1.2,
                       zorder=5, label=f"T={T}°C (n={len(sub)})")
            xx = np.logspace(np.log10(0.02), np.log10(2.5), 200)
            ax.plot(xx, model(p_, xx, T + 273.15), "--",
                    color=color, linewidth=1.6, alpha=0.85)

        ax.set_xscale("log")
        ax.set_xlabel("tR / s", fontsize=11)
        ax.set_ylabel("Yield %", fontsize=11)
        Ea_d_ok = 55 <= p_[3] <= 65
        title_color = "darkgreen" if Ea_d_ok else (
                      "darkblue" if p_[3] > 65 else "darkorange")
        ax.set_title(
            f"{lab}\nEa_d = {p_[3]:.1f} kJ/mol  ·  R²={R2_:.3f}  RMSE={rmse_:.1f}%",
            fontsize=11, color=title_color, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="upper right")

        # parameter box inside subplot
        text = (f"y_max ={p_[0]:5.1f}%\n"
                f"Ea_f  ={p_[1]:5.1f} kJ/mol\n"
                f"lnA_f ={p_[2]:5.1f}\n"
                f"Ea_d  ={p_[3]:5.1f} kJ/mol\n"
                f"lnA_d ={p_[4]:5.1f}")
        ax.text(0.04, 0.05, text, transform=ax.transAxes, va="bottom",
                fontsize=9, family="monospace",
                bbox=dict(boxstyle="round,pad=0.4",
                          fc="lightyellow", ec="gray"))

    plt.suptitle("Effect of progressively dropping day-1 data on Arrhenius fit\n"
                 "(HYBRID v2 prediction: Ea_d ≈ 55–65 kJ/mol)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    OUT_PNG.parent.mkdir(exist_ok=True)
    plt.savefig(OUT_PNG, dpi=160, bbox_inches="tight")
    print(f"\nSaved: {OUT_PNG}")


if __name__ == "__main__":
    main()
