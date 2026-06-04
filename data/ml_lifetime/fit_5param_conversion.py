"""
5-parameter global Arrhenius fit on substrate conversion data,
treating it as if it were ArH yield (Zhao's mass-balance interpretation
under the no-side-product assumption).

Model: conv(tR, T) = y_max × (1 − exp(−k_f(T)·tR)) × exp(−k_d(T)·tR)
       k_f(T) = exp(lnA_f − Ea_f / RT)
       k_d(T) = exp(lnA_d − Ea_d / RT)

Includes Zhao's suspected vial swap (T=0°C L=3 ↔ L=10).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize

CSV = Path(__file__).parent / "experiment_yield_summary.csv"
OUT_PNG = Path(__file__).parent / "analysis_figures" / "experiment_5param_conversion.png"
R_GAS = 8.314e-3  # kJ/(mol·K)


def apply_label_swap(df):
    a = df[(df.T_C == 0) & (df.L_cm == 3)].index[0]
    b = df[(df.T_C == 0) & (df.L_cm == 10)].index[0]
    for c in ['subst_area', 'area_ratio_obs',
              'residual_subst_pct_ECN', 'conversion_pct_ECN', 'yield_pct_ECN']:
        if c in df.columns:
            df.loc[a, c], df.loc[b, c] = df.loc[b, c], df.loc[a, c]
    return df


def model_5p(params, tR, T_K):
    y_max, Ea_f, lnA_f, Ea_d, lnA_d = params
    kf = np.exp(lnA_f - Ea_f / (R_GAS * T_K))
    kd = np.exp(lnA_d - Ea_d / (R_GAS * T_K))
    return y_max * (1 - np.exp(-kf * tR)) * np.exp(-kd * tR)


def fit_5param(df, label):
    tR = df.tR_s.to_numpy()
    Tk = df.T_C.to_numpy() + 273.15
    Y = df.conversion_pct_ECN.to_numpy() / 100.0

    def loss(p):
        return np.sum((model_5p(p, tR, Tk) - Y) ** 2)

    bounds = [
        (0.5, 1.5),     # y_max
        (5, 100),       # Ea_f kJ/mol
        (5, 35),        # lnA_f
        (1, 80),        # Ea_d kJ/mol
        (-5, 20),       # lnA_d
    ]
    de = differential_evolution(loss, bounds, seed=42, tol=1e-11,
                                maxiter=1000, polish=True)
    res = minimize(loss, de.x, method="Nelder-Mead",
                   options={"xatol": 1e-9, "fatol": 1e-9, "maxiter": 100000})
    p = res.x
    y_max, Ea_f, lnA_f, Ea_d, lnA_d = p
    pred = model_5p(p, tR, Tk)
    rss = np.sum((Y - pred) ** 2); tss = np.sum((Y - Y.mean()) ** 2)
    R2 = 1 - rss / tss
    rmse = 100 * np.sqrt(rss / len(Y))
    # derive k_d at 298 K and -78 °C for interpretation
    kd_298 = np.exp(lnA_d - Ea_d / (R_GAS * 298.15))
    kd_m78 = np.exp(lnA_d - Ea_d / (R_GAS * 195.15))
    print(f"\n>>> {label} (n={len(Y)})")
    print(f"  y_max = {y_max:.3f}  ({y_max*100:.1f} %)")
    print(f"  Ea_f  = {Ea_f:.2f} kJ/mol   lnA_f = {lnA_f:.2f}")
    print(f"  Ea_d  = {Ea_d:.2f} kJ/mol   lnA_d = {lnA_d:.2f}")
    print(f"  k_d(298K) = {kd_298:.4f} s⁻¹     k_d(-78°C) = {kd_m78:.4f} s⁻¹")
    print(f"  R²    = {R2:.4f}   RMSE = {rmse:.2f}% conversion")
    return p, R2, rmse


def main():
    df = pd.read_csv(CSV).dropna(subset=['conversion_pct_ECN']).copy()
    df_sw = apply_label_swap(df.copy())
    df_d2 = df_sw[~df_sw.T_C.isin([0, -25])].copy()

    scenarios = []
    for lab, dat in [("All 22 pts (swap applied)", df_sw),
                     ("Day-2 only (12 pts)",       df_d2)]:
        p, R2, rmse = fit_5param(dat, lab)
        scenarios.append((lab, dat, p, R2, rmse))

    # plot 2 panels
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    cmap = plt.get_cmap("coolwarm_r")
    for ax, (label, dat, p, R2, rmse) in zip(axes, scenarios):
        Ts = sorted(dat.T_C.unique())
        for i, T in enumerate(Ts):
            color = cmap(i / max(len(Ts) - 1, 1))
            sub = dat[dat.T_C == T]
            ax.scatter(sub.tR_s, sub.conversion_pct_ECN, s=110,
                       color=color, edgecolor="black", linewidth=1.2,
                       zorder=5, label=f"T={T}°C (n={len(sub)})")
            xx = np.logspace(np.log10(0.02), np.log10(2.5), 200)
            yy = 100 * model_5p(p, xx, T + 273.15)
            ax.plot(xx, yy, "--", color=color, linewidth=1.6, alpha=0.85)
        ax.set_xscale("log")
        ax.set_xlabel("tR / s"); ax.set_ylabel("Conversion %")
        ax.set_ylim(0, 110)
        ax.grid(True, alpha=0.3); ax.legend(fontsize=9, loc='lower right')

        y_max, Ea_f, lnA_f, Ea_d, lnA_d = p
        ax.set_title(f"{label}\nR²={R2:.3f}   RMSE={rmse:.1f}%",
                     fontsize=12, fontweight="bold")

        txt = (f"y_max = {y_max*100:.1f}%\n"
               f"Ea_f  = {Ea_f:.1f} kJ/mol\n"
               f"lnA_f = {lnA_f:.1f}\n"
               f"Ea_d  = {Ea_d:.1f} kJ/mol\n"
               f"lnA_d = {lnA_d:.1f}\n\n"
               f"v4.6 p-ArLi predictions:\n"
               f"  Ea_f ≈ 27.1 kJ/mol\n"
               f"  Ea_d ≈ 39.9 kJ/mol")
        ax.text(0.04, 0.95, txt, transform=ax.transAxes, va="top",
                fontsize=10, family="monospace",
                bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", ec="gray"))

    plt.suptitle("5-param Arrhenius bell-curve fit on substrate conversion data\n"
                 "Treating conversion as proxy for ArH yield (no-side-product assumption)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    OUT_PNG.parent.mkdir(exist_ok=True)
    plt.savefig(OUT_PNG, dpi=160, bbox_inches="tight")
    print(f"\nSaved: {OUT_PNG}")


if __name__ == "__main__":
    main()
