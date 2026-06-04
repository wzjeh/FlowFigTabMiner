"""
Fit kinetic models to validated p-FC6H4Br + n-BuLi flow data and
compare against HYBRID v2 model predictions.

Two model variants:
  Model 1 (simple decay):    yield = y_max × exp(-k_d × tR)
  Model 2 (formation+decay): yield = y_max × (1 - exp(-k_f × tR)) × exp(-k_d × tR)

Per-temperature fit + Arrhenius regression of ln k_d vs 1/T → Ea_d, lnA_d.

HYBRID v2 prediction for p-FC6H4Li (from /Users/.../EXPERIMENTAL_VALIDATION_PLAN.md):
  Ea_d ≈ 55-65 kJ/mol (predicted)
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

CSV = Path(__file__).parent / "experiment_yield_with_flags.csv"
OUT_PNG = Path(__file__).parent / "analysis_figures" / "experiment_arrhenius_fit.png"

R_GAS = 8.314e-3   # kJ/(mol K)

EXCLUDE_FLAGS = {"anomaly", "rebound"}


def load_data(exclude_t25=False):
    df = pd.read_csv(CSV)
    df = df[~df["flag"].fillna("").isin(EXCLUDE_FLAGS)].copy()
    if exclude_t25:
        df = df[df["T_C"] != -25].copy()
    df = df.dropna(subset=["yield_pct_ECN"])
    return df


def fit_per_T_simple(df):
    """Fit yield = y_max * exp(-k_d * tR) per T. Returns dict {T_C: (y_max, k_d, n)}."""
    out = {}
    for T_C, sub in df.groupby("T_C"):
        if len(sub) < 2:
            continue
        tR = sub["tR_s"].to_numpy()
        Y  = sub["yield_pct_ECN"].to_numpy()

        def model(tR, ymax, kd):
            return ymax * np.exp(-kd * tR)

        try:
            p0 = [Y.max() * 1.1, 1.0]
            popt, _ = curve_fit(model, tR, Y, p0=p0,
                                bounds=([1, 0], [200, 1e4]),
                                maxfev=20000)
            out[T_C] = (popt[0], popt[1], len(sub))
        except Exception as e:
            print(f"  fit failed at T={T_C}: {e}")
    return out


def fit_per_T_full(df):
    """yield = y_max * (1 - exp(-k_f tR)) * exp(-k_d tR). Joint fit per T."""
    out = {}
    for T_C, sub in df.groupby("T_C"):
        if len(sub) < 3:
            continue
        tR = sub["tR_s"].to_numpy()
        Y  = sub["yield_pct_ECN"].to_numpy()

        def model(tR, ymax, kf, kd):
            return ymax * (1 - np.exp(-kf * tR)) * np.exp(-kd * tR)

        try:
            p0 = [Y.max() * 1.5, 50.0, 1.0]
            popt, _ = curve_fit(model, tR, Y, p0=p0,
                                bounds=([1, 0.1, 0], [300, 1e5, 1e4]),
                                maxfev=30000)
            out[T_C] = (*popt, len(sub))
        except Exception as e:
            print(f"  fit failed at T={T_C}: {e}")
    return out


def arrhenius(per_T_kd):
    """ln k_d vs 1/T → Ea, lnA."""
    Ts = np.array(sorted(per_T_kd.keys()))
    kds = np.array([per_T_kd[T] for T in Ts])
    invT = 1.0 / (Ts + 273.15)
    ln_kd = np.log(kds)
    # weighted linear fit
    p, cov = np.polyfit(invT, ln_kd, 1, cov=True)
    slope, intercept = p
    Ea = -slope * 8.314e-3                          # kJ/mol
    Ea_se = np.sqrt(cov[0, 0]) * 8.314e-3
    lnA = intercept                                  # ln s^-1
    lnA_se = np.sqrt(cov[1, 1])
    # R^2
    pred = slope * invT + intercept
    ss_res = np.sum((ln_kd - pred) ** 2)
    ss_tot = np.sum((ln_kd - np.mean(ln_kd)) ** 2)
    R2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return Ea, Ea_se, lnA, lnA_se, R2, Ts, kds


def main():
    print("=" * 70)
    print("Group A: 15 points (exclude day-1 0°C, exclude rebound)")
    print("=" * 70)
    dfA = load_data(exclude_t25=False)
    print(dfA[["T_C","L_cm","tR_s","yield_pct_ECN"]].to_string(index=False))

    print("\nGroup B: 12 points (also exclude -25°C day-1)")
    print("=" * 70)
    dfB = load_data(exclude_t25=True)
    print(dfB[["T_C","L_cm","tR_s","yield_pct_ECN"]].to_string(index=False))

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    for row_idx, (label, df) in enumerate([("A: w/ -25°C (15 pts)", dfA),
                                            ("B: only day-2 (12 pts)", dfB)]):
        print(f"\n\n>>> Group {label}")
        # Fit Model 1 (simple decay)
        fits1 = fit_per_T_simple(df)
        kd1 = {T: f[1] for T, f in fits1.items() if f[1] > 0}
        ymax1 = {T: f[0] for T, f in fits1.items()}

        print(f"\nModel 1: yield = y_max·exp(-k_d·tR)")
        for T in sorted(fits1):
            ymax, kd, n = fits1[T]
            print(f"  T={T:>4}°C  y_max={ymax:6.2f}%  k_d={kd:8.3f} s⁻¹  (n={n})")

        if len(kd1) >= 3:
            Ea, Ea_se, lnA, lnA_se, R2, Ts, kds = arrhenius(kd1)
            print(f"\n  Arrhenius (Model 1):")
            print(f"    Ea_d = {Ea:.1f} ± {Ea_se:.1f} kJ/mol")
            print(f"    lnA_d = {lnA:.2f} ± {lnA_se:.2f} (A in s⁻¹)")
            print(f"    R² = {R2:.4f}")
        else:
            print("  too few T for Arrhenius")
            Ea = lnA = R2 = np.nan; Ts = np.array([]); kds = np.array([])

        # Fit Model 2 (formation+decay)
        fits2 = fit_per_T_full(df)
        kd2 = {T: f[2] for T, f in fits2.items() if f[2] > 0}
        kf2 = {T: f[1] for T, f in fits2.items()}
        ymax2 = {T: f[0] for T, f in fits2.items()}

        print(f"\nModel 2: yield = y_max·(1-exp(-k_f·tR))·exp(-k_d·tR)")
        for T in sorted(fits2):
            ymax, kf, kd, n = fits2[T]
            print(f"  T={T:>4}°C  y_max={ymax:6.2f}%  k_f={kf:8.2f}  k_d={kd:7.3f}  (n={n})")

        # plot per-T fits
        ax = axes[row_idx, 0]
        for T in sorted(df["T_C"].unique()):
            sub = df[df["T_C"] == T]
            ax.scatter(sub["tR_s"], sub["yield_pct_ECN"], s=80,
                       label=f"{T}°C (n={len(sub)})", zorder=5)
            if T in fits1:
                ymax, kd, _ = fits1[T]
                xx = np.linspace(0.01, sub["tR_s"].max()*1.1, 100)
                ax.plot(xx, ymax * np.exp(-kd * xx), "--", alpha=0.7)
        ax.set_xlabel("tR / s"); ax.set_ylabel("Yield %")
        ax.set_title(f"{label}\nModel 1: y=y_max·exp(-k_d·tR)")
        ax.set_xscale("log"); ax.legend(fontsize=9); ax.grid(alpha=0.3)

        # Arrhenius plot
        ax = axes[row_idx, 1]
        if len(kd1) >= 2:
            Ts_arr = np.array(sorted(kd1.keys()))
            invT = 1000.0 / (Ts_arr + 273.15)
            kds_arr = np.array([kd1[T] for T in Ts_arr])
            ax.scatter(invT, np.log(kds_arr), s=120, color="tab:blue",
                       edgecolor="black", zorder=5)
            for T_, x_, y_ in zip(Ts_arr, invT, np.log(kds_arr)):
                ax.annotate(f"{T_}°C", (x_, y_), xytext=(8, 4),
                            textcoords="offset points", fontsize=10)
            if len(kd1) >= 3:
                xx = np.linspace(invT.min()*0.95, invT.max()*1.05, 100)
                # slope is in units of 1000/T; convert
                slope = -Ea / (R_GAS * 1000)   # 1/K, but x is *1000
                ax.plot(xx, slope * xx + lnA, "r--",
                        label=f"Ea={Ea:.1f}±{Ea_se:.1f} kJ/mol\nlnA={lnA:.2f}\nR²={R2:.3f}")
                ax.legend(fontsize=10, loc="best")
        ax.set_xlabel("1000/T  /  K⁻¹"); ax.set_ylabel("ln(k_d / s⁻¹)")
        ax.set_title("Arrhenius plot (Model 1)")
        ax.grid(alpha=0.3)

        # Predicted vs HYBRID v2 box
        ax = axes[row_idx, 2]
        ax.axis("off")
        text = []
        text.append(f"Group {label}")
        text.append("=" * 36)
        text.append("")
        if not np.isnan(Ea):
            text.append("Experimental fit (Model 1):")
            text.append(f"  Ea_d  = {Ea:.1f} ± {Ea_se:.1f} kJ/mol")
            text.append(f"  lnA_d = {lnA:.2f} ± {lnA_se:.2f}  (s⁻¹)")
            text.append(f"  R²    = {R2:.4f}")
        text.append("")
        text.append("HYBRID v2 prediction:")
        text.append("  Ea_d ≈ 55–65 kJ/mol")
        text.append("  (p-FC6H4Li, m-ArLi class)")
        text.append("")
        if not np.isnan(Ea):
            in_range = 55 <= Ea <= 65
            mark = "✓ within 55-65" if in_range else "✗ outside 55-65"
            text.append(f"Verdict: {mark}")
            if not in_range:
                if Ea < 55:
                    text.append("  → smaller Ea than predicted →")
                    text.append("    may indicate different decay pathway")
                else:
                    text.append("  → larger Ea than predicted")
        text.append("")
        text.append("⚠ yield% uses ECN-estimated RRF.")
        text.append("  Absolute Ea is invariant to RRF")
        text.append("  (RRF only scales y_max, not k_d).")
        ax.text(0.02, 0.98, "\n".join(text), transform=ax.transAxes,
                va="top", ha="left", family="monospace", fontsize=11,
                bbox=dict(boxstyle="round,pad=0.5", fc="#f0f4f8", ec="gray"))

    plt.suptitle("p-FC6H4Br + n-BuLi → p-FC6H4Li : kinetic model fit vs HYBRID v2",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    OUT_PNG.parent.mkdir(exist_ok=True)
    plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {OUT_PNG}")


if __name__ == "__main__":
    main()
