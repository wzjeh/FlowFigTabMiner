"""
Unified Yoshida 4-parameter fit for three experimental validation cases:

  Case A: 4-Br-FC6H4 + MeOH quench   (conversion-only, treat as 100% selective → yield)
  Case B: 4-Br-FC6H4 + PhCHO quench  (trap yield from Ar(Ph)CHOH peak)
  Case C: 3,5-Br2-C6H3-CN + TMSCl    (trap yield from Ar-TMS peak)

Yoshida model:
   yield(tR, T) [%] = 100 · [1 − exp(−k_f·tR)] · exp(−k_d·tR)
   with k_f, k_d = exp(lnA − Ea / (R·T))

Fit:  4 params globally per dataset (Ea_f, lnA_f, Ea_d, lnA_d), using DE + Nelder-Mead.
Output: parameters, R², figure, comparison vs v4.6 prediction (for case C, m-ArLi).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize

R = 8.314e-3   # kJ/(mol·K)
BASE = Path(__file__).parent


def yoshida(p, tR, Tk):
    """5-parameter bell-curve model:
       yield(tR, T) = y_max · [1 - exp(-k_f(T)·tR)] · exp(-k_d(T)·tR)
       k_i(T) = exp(lnA_i - Ea_i / (R·T))
    """
    Ea_f, lnA_f, Ea_d, lnA_d, y_max = p
    kf = np.exp(lnA_f - Ea_f / (R * Tk))
    kd = np.exp(lnA_d - Ea_d / (R * Tk))
    return y_max * (1 - np.exp(-kf * tR)) * np.exp(-kd * tR)


def fit_yoshida(tR, Tk, y_obs, constrain_positive_Ead=True):
    """5-param fit; physically-motivated bounds.
    constrain_positive_Ead: enforce Ea_d ≥ 0 (positive activation barrier for decay)."""
    def loss(p):
        pred = yoshida(p, tR, Tk)
        return np.sum((pred - y_obs) ** 2)
    Ead_lo = 0.1 if constrain_positive_Ead else -50.0
    bounds = [(5.0, 150.0),   # Ea_f
              (5.0, 40.0),    # lnA_f
              (Ead_lo, 150.0),# Ea_d
              (-10.0, 40.0),  # lnA_d
              (30.0, 110.0)]  # y_max (allow >100 if RRF underestimated)
    de = differential_evolution(loss, bounds, seed=42, tol=1e-12,
                                maxiter=3000, polish=False, workers=1, popsize=40)
    res = minimize(loss, de.x, method="L-BFGS-B", bounds=bounds,
                   options={"ftol": 1e-12, "gtol": 1e-10, "maxiter": 50000})
    p = res.x
    pred = yoshida(p, tR, Tk)
    rss = np.sum((y_obs - pred) ** 2)
    tss = np.sum((y_obs - y_obs.mean()) ** 2)
    R2 = 1 - rss / tss if tss > 0 else np.nan
    rmse = np.sqrt(rss / len(y_obs))
    return p, R2, rmse


# ---- load three datasets ----
def load_caseA():
    df = pd.read_csv(BASE / "experiment_yield_summary.csv")
    df = df[df["yield_pct_ECN"].notna()].copy()
    # Use conversion_pct_ECN as yield (100% selectivity assumption)
    df["yield"] = df["conversion_pct_ECN"].clip(0, 100)
    df["Tk"] = df["T_C"] + 273.15
    return df[["T_C", "Tk", "L_cm", "tR_s", "yield"]].copy(), "Case A: 4-Br-FC6H4 + MeOH (conv→yield)"


def load_caseB():
    df = pd.read_csv(BASE / "experiment_phcho_summary_v2.csv")
    df = df[df["trap_yield_pct"].notna()].copy()
    df["yield"] = df["trap_yield_pct"].clip(0, 100)
    df["Tk"] = df["T_C"] + 273.15
    return df[["T_C", "Tk", "L_cm", "tR_s", "yield"]].copy(), "Case B: 4-Br-FC6H4 + PhCHO"


def load_caseC():
    df = pd.read_csv(BASE / "experiment_35brcn_summary.csv")
    df = df[df["trap_yield_pct"].notna()].copy()
    df["yield"] = df["trap_yield_pct"].clip(0, 100)
    df["Tk"] = df["T_C"] + 273.15
    return df[["T_C", "Tk", "L_cm", "tR_s", "yield"]].copy(), "Case C: 3,5-Br2-CN + TMSCl (TMS-only)"


def load_caseC_pool():
    """Case C variant: use (TMS+ArH)/area_ratio_at_100 as ArLi pool yield."""
    df = pd.read_csv(BASE / "experiment_35brcn_summary.csv")
    df = df[df["prod_area"].notna() & df["arh_area"].notna() & df["c12_area"].notna()].copy()
    AREA_RATIO_AT_100 = 3.445
    df["yield"] = ((df["prod_area"] + df["arh_area"]) / df["c12_area"] / AREA_RATIO_AT_100 * 100).clip(0, 110)
    df["Tk"] = df["T_C"] + 273.15
    return df[["T_C", "Tk", "L_cm", "tR_s", "yield"]].copy(), "Case C′: 3,5-Br2-CN (TMS+ArH pool)"


def k_at_T(Ea, lnA, T_C):
    return np.exp(lnA - Ea / (R * (T_C + 273.15)))


def main():
    cases = [("A", *load_caseA()),
             ("B", *load_caseB()),
             ("C", *load_caseC()),
             ("C'", *load_caseC_pool())]
    results = []
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5))
    cmap = plt.get_cmap("coolwarm_r")

    for ax, (tag, df, label) in zip(axes, cases):
        tR = df["tR_s"].to_numpy()
        Tk = df["Tk"].to_numpy()
        y  = df["yield"].to_numpy()
        p, R2, rmse = fit_yoshida(tR, Tk, y)
        Ea_f, lnA_f, Ea_d, lnA_d, y_max = p
        kf25 = k_at_T(Ea_f, lnA_f, 25)
        kd25 = k_at_T(Ea_d, lnA_d, 25)
        kf_m65 = k_at_T(Ea_f, lnA_f, -65)
        kd_m65 = k_at_T(Ea_d, lnA_d, -65)
        # t_max and t_half at 25°C and -65°C
        def t_max(T_C):
            kf = k_at_T(Ea_f, lnA_f, T_C); kd = k_at_T(Ea_d, lnA_d, T_C)
            if abs(kf - kd) < 1e-10: return np.nan
            return np.log(kf/kd) / (kf - kd)
        def t_half(T_C):
            kd = k_at_T(Ea_d, lnA_d, T_C)
            return np.log(2) / kd if kd > 1e-10 else np.inf
        results.append({
            "case": tag, "label": label, "n": len(df),
            "Ea_f": Ea_f, "lnA_f": lnA_f, "Ea_d": Ea_d, "lnA_d": lnA_d, "y_max": y_max,
            "R2": R2, "RMSE_%": rmse,
            "k_f(25C)": kf25, "k_d(25C)": kd25,
            "k_f(-65C)": kf_m65, "k_d(-65C)": kd_m65,
            "t_max(25C)": t_max(25), "t_half(25C)": t_half(25),
        })

        # Plot
        Ts = sorted(df["T_C"].unique())
        tR_range = np.logspace(np.log10(max(df["tR_s"].min()*0.5, 0.005)),
                               np.log10(df["tR_s"].max()*1.5), 200)
        for i, T_C in enumerate(Ts):
            color = cmap(i / max(len(Ts)-1, 1))
            sub = df[df["T_C"] == T_C].sort_values("tR_s")
            ax.scatter(sub["tR_s"], sub["yield"], s=110, color=color,
                       edgecolor="black", linewidth=1.2, zorder=5,
                       label=f"T={int(T_C):+}°C")
            yy = yoshida(p, tR_range, T_C + 273.15)
            ax.plot(tR_range, yy, "--", color=color, linewidth=1.6, alpha=0.85)
        ax.set_xscale("log")
        ax.set_xlabel("Residence time tR / s", fontsize=11)
        ax.set_ylabel("yield (%)", fontsize=11)
        ax.set_title(
            f"{label}\n"
            f"Ea_f={Ea_f:.1f}  lnA_f={lnA_f:.1f}  Ea_d={Ea_d:.1f}  lnA_d={lnA_d:.1f}  y_max={y_max:.1f}  R²={R2:.3f}",
            fontsize=9.5, fontweight="bold")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower center", ncol=3)
        ax.set_ylim(-5, 110)

    out = BASE / "analysis_figures" / "yoshida_three_cases.png"
    out.parent.mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=160, bbox_inches="tight")
    print(f"\nSaved: {out}")

    res_df = pd.DataFrame(results)
    out_csv = BASE / "yoshida_three_cases.csv"
    res_df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}\n")

    # Pretty-print
    for r in results:
        print(f"--- {r['label']}  (n={r['n']}) ---")
        print(f"  Ea_f  = {r['Ea_f']:6.2f} kJ/mol   lnA_f = {r['lnA_f']:6.2f}")
        print(f"  Ea_d  = {r['Ea_d']:6.2f} kJ/mol   lnA_d = {r['lnA_d']:6.2f}")
        print(f"  y_max = {r['y_max']:.2f} %")
        print(f"  R² = {r['R2']:.4f}   RMSE = {r['RMSE_%']:.2f} %")
        print(f"  k_f(+25°C)  = {r['k_f(25C)']:.2e} /s    k_d(+25°C)  = {r['k_d(25C)']:.2e} /s")
        print(f"  k_f(-65°C)  = {r['k_f(-65C)']:.2e} /s    k_d(-65°C)  = {r['k_d(-65C)']:.2e} /s")
        print(f"  t_max(+25°C) = {r['t_max(25C)']:.4f} s    t_½(+25°C) = {r['t_half(25C)']:.3f} s")
        print()

    # ---- v4.6 prediction overlay for Case C (m-ArLi) ----
    pred_csv = BASE / "35brcn_v46_prediction.csv"
    if pred_csv.exists():
        pred = pd.read_csv(pred_csv).iloc[0]
        print("=== Case C vs v4.6 m-ArLi prediction (3,5-Br2-CN) ===")
        rC = results[2]
        print(f"  Param     experiment   v4.6      diff")
        print(f"  Ea_f      {rC['Ea_f']:7.2f}   {pred['Ea_f_pred']:7.2f}   {rC['Ea_f']-pred['Ea_f_pred']:+.2f}")
        print(f"  lnA_f     {rC['lnA_f']:7.2f}   {pred['lnA_f_pred']:7.2f}   {rC['lnA_f']-pred['lnA_f_pred']:+.2f}")
        print(f"  Ea_d      {rC['Ea_d']:7.2f}   {pred['Ea_d_pred']:7.2f}   {rC['Ea_d']-pred['Ea_d_pred']:+.2f}")
        print(f"  lnA_d     {rC['lnA_d']:7.2f}   {pred['lnA_d_pred']:7.2f}   {rC['lnA_d']-pred['lnA_d_pred']:+.2f}")


if __name__ == "__main__":
    main()
