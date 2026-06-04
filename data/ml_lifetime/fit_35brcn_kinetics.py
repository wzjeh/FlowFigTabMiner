"""
Fit ArLi decay kinetics from 3,5-Dibromobenzonitrile · n-BuLi · TMSCl-trap.

Key insight: TMS-product yield mixes ArLi decay AND quench efficiency.
   ArLi_alive(tR, T) = (TMS_area + ArH_area) / C12_area
   because: ArLi + TMSCl → Ar-TMS  (main quench path)
            ArLi + H+   → Ar-H     (proto-de-Li quench path)
   Both paths originate from the SAME ArLi pool at the quench moment.

Model:
   ArLi_alive(tR, T) = ArLi_max(T) · exp(-k_d(T) · tR)
   ln[ArLi_alive] = ln[ArLi_max] − k_d(T) · tR     (linear fit per T)

Then Arrhenius:
   ln(k_d) = lnA_d − Ea_d / (R·T)

Note: Li-Br exchange formation (Ea_f) cannot be fitted — already ~95%
complete at minimum tR (0.047 s, -65°C) due to strong CN+Br EWG.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

CSV = Path(__file__).parent / "experiment_35brcn_summary.csv"
PRED_CSV = Path(__file__).parent / "35brcn_v46_prediction.csv"
OUT_PNG = Path(__file__).parent / "analysis_figures" / "35brcn_arrhenius_fit.png"
R = 8.314e-3   # kJ/(mol·K)


def main():
    df = pd.read_csv(CSV)
    df = df.dropna(subset=["prod_area", "arh_area", "c12_area", "tR_s", "T_C"]).copy()
    df["r_pool"] = (df["prod_area"] + df["arh_area"]) / df["c12_area"]
    df["Tk"] = df["T_C"] + 273.15

    # Per-T linear fit: ln(r_pool) = ln(A_max) − k_d · tR
    fits = []
    for T_C, sub in df.groupby("T_C"):
        sub = sub.sort_values("tR_s")
        if len(sub) < 3:
            continue
        tR = sub["tR_s"].to_numpy()
        ln_pool = np.log(sub["r_pool"].to_numpy())
        # Linear: y = a + b·x  with b = -k_d, a = ln(A_max)
        b, a = np.polyfit(tR, ln_pool, 1)
        k_d = -b
        pool_max = np.exp(a)
        # R² of linear fit
        ss_res = np.sum((ln_pool - (a + b*tR))**2)
        ss_tot = np.sum((ln_pool - ln_pool.mean())**2)
        R2 = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan
        fits.append({
            "T_C": T_C, "Tk": T_C + 273.15,
            "n": len(sub),
            "k_d": k_d, "pool_max": pool_max, "R2_linear": R2,
        })
        print(f"T={T_C:+4}°C  n={len(sub)}  k_d={k_d:+.4f}/s  pool_max={pool_max:.3f}  R²={R2:.3f}")

    fits = pd.DataFrame(fits)
    # Quality filter: only T points with clean monotonic decay (R² > 0.65)
    # — at low T, decay is too small relative to noise to fit reliably
    fits["valid"] = (fits["k_d"] > 0) & (fits["R2_linear"] > 0.65)
    print(f"\n{fits['valid'].sum()} valid T points for Arrhenius (R² > 0.65, k_d > 0)")

    arr = fits[fits["valid"]].copy()
    if len(arr) >= 2:
        inv_T = 1.0 / arr["Tk"].to_numpy()
        ln_k  = np.log(arr["k_d"].to_numpy())
        slope, intercept = np.polyfit(inv_T, ln_k, 1)
        Ea_d_exp = -slope * R
        lnA_d_exp = intercept
        # R²
        pred_lnk = intercept + slope*inv_T
        ss_res = np.sum((ln_k - pred_lnk)**2)
        ss_tot = np.sum((ln_k - ln_k.mean())**2)
        R2_arr = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan
        print(f"\nArrhenius fit (n={len(arr)}):")
        print(f"  Ea_d  = {Ea_d_exp:.2f} kJ/mol")
        print(f"  lnA_d = {lnA_d_exp:.2f}")
        print(f"  R²    = {R2_arr:.3f}")
    else:
        Ea_d_exp = lnA_d_exp = R2_arr = np.nan
        print("\n[!] Not enough valid points for Arrhenius fit.")

    # Pull v4.6 prediction if available
    v46_Ea_d = v46_lnA_d = None
    if PRED_CSV.exists():
        pred = pd.read_csv(PRED_CSV).iloc[0]
        v46_Ea_d = pred.get("Ea_d_pred", None)
        v46_lnA_d = pred.get("lnA_d_pred", None)
        print(f"\nv4.6 prediction: Ea_d={v46_Ea_d}  lnA_d={v46_lnA_d}")
    else:
        print(f"\n[!] {PRED_CSV.name} not yet generated (xTB still running).")

    # ---- Plot ----
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    cmap = plt.get_cmap("coolwarm_r")
    Ts = sorted(df["T_C"].unique())

    # Panel 1: per-T r_pool vs tR with fitted exp decay
    ax = axes[0]
    for i, T_C in enumerate(Ts):
        color = cmap(i / max(len(Ts)-1, 1))
        sub = df[df["T_C"] == T_C].sort_values("tR_s")
        ax.scatter(sub["tR_s"], sub["r_pool"], s=110, color=color,
                   edgecolor="black", linewidth=1.2, zorder=5,
                   label=f"T={T_C:+}°C (n={len(sub)})")
        row = fits[fits["T_C"] == T_C]
        if len(row):
            r = row.iloc[0]
            xx = np.linspace(sub["tR_s"].min(), sub["tR_s"].max(), 100)
            yy = r["pool_max"] * np.exp(-r["k_d"] * xx)
            ax.plot(xx, yy, "--", color=color, linewidth=1.6, alpha=0.85)
    ax.set_xscale("log")
    ax.set_xlabel("Residence time tR / s", fontsize=11)
    ax.set_ylabel("[ArLi alive] proxy = (TMS+ArH)/C12", fontsize=11)
    ax.set_title("Per-T exponential decay fit", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="best")
    ax.grid(alpha=0.3)

    # Panel 2: Arrhenius plot ln(k_d) vs 1000/T
    ax = axes[1]
    if len(fits) > 0:
        ax.scatter(1000/fits["Tk"], np.log(np.abs(fits["k_d"])), s=140,
                   c=fits["T_C"], cmap="coolwarm_r",
                   edgecolor="black", linewidth=1.5, zorder=5)
        for _, r in fits.iterrows():
            if r["valid"]:
                ax.annotate(f"{r['T_C']:+}°C", (1000/r["Tk"], np.log(r["k_d"])),
                            xytext=(5, 5), textcoords="offset points", fontsize=9)
    if len(arr) >= 2:
        xx = np.linspace(1000/df["Tk"].max() * 0.95, 1000/df["Tk"].min() * 1.05, 100)
        yy = lnA_d_exp - Ea_d_exp/R / (1000/xx)
        ax.plot(xx, yy, "k-", linewidth=2,
                label=f"Exp: Ea_d={Ea_d_exp:.1f}, lnA_d={lnA_d_exp:.1f}, R²={R2_arr:.2f}")
    if v46_Ea_d is not None:
        xx = np.linspace(1000/df["Tk"].max() * 0.95, 1000/df["Tk"].min() * 1.05, 100)
        yy = v46_lnA_d - v46_Ea_d/R / (1000/xx)
        ax.plot(xx, yy, "g--", linewidth=2,
                label=f"v4.6: Ea_d={v46_Ea_d:.1f}, lnA_d={v46_lnA_d:.1f}")
    ax.set_xlabel("1000 / T  (1/K)", fontsize=11)
    ax.set_ylabel("ln(k_d / s⁻¹)", fontsize=11)
    ax.set_title("Arrhenius plot of ArLi decay", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, loc="best")
    ax.grid(alpha=0.3)

    plt.suptitle("3,5-Dibromobenzonitrile · n-BuLi · TMSCl trap  ·  ArLi decay kinetics",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    OUT_PNG.parent.mkdir(exist_ok=True)
    plt.savefig(OUT_PNG, dpi=160, bbox_inches="tight")
    print(f"\nSaved: {OUT_PNG}")

    # ---- Mass balance check ----
    print("\n=== Quench efficiency η_q = TMS/(TMS+ArH) per vial ===")
    df["eta_q"] = df["prod_area"] / (df["prod_area"] + df["arh_area"])
    print(df[["vial","T_C","L_cm","tR_s","r_pool","eta_q"]].to_string(index=False))

    # Save fit results
    fits.to_csv(Path(__file__).parent / "35brcn_arrhenius_fits.csv", index=False)
    print(f"\nSaved: 35brcn_arrhenius_fits.csv")


if __name__ == "__main__":
    main()
