"""
4-panel comparison heatmap for 5-bromo-2-fluorobenzonitrile (substrate 2):
  (a) PREVIOUS experiment, original (non-D) calibration   — historical, miscalibrated
  (b) NEW experiment (260602 MeOH quench)                  — same chemistry, fresh run
  (c) PREVIOUS experiment re-calibrated with D-product cal — Zhao 2026-06-04 update
  (d) Layered-model prediction                              — paper Fig 5(e) params

Panel (a) uses experiment_fbrcn_summary.csv         (untouched)
Panel (b) uses experiment_fbrcn1_summary.csv         (untouched, vial 1 + 11 dropped)
Panel (c) uses experiment_fbrcn_d_calibrated_summary.csv (new this round)
Panel (d) draws the model surface; numeric labels show the predicted yield
          at each (T, t_R) cell of panel (b)'s grid.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patheffects as patheffects
from matplotlib.colors import Normalize
from scipy.interpolate import Rbf
from pathlib import Path

import formation_model as fm
from draw_paper_figures import _bayes_5Br2FCN, _TAU_REF

BASE = Path(__file__).parent
OUT  = BASE / "analysis_figures"
R_GAS = 8.314e-3   # kJ mol⁻¹ K⁻¹

PRED_PARAMS = {**_bayes_5Br2FCN(), "sigma": 0.62, "tau_eff": _TAU_REF}


def yld_layered(p, tR, T_C):
    Tk = np.asarray(T_C, float) + 273.15
    tau = fm.tau_eff_T(p["tau_eff"], T_C)
    kf  = fm.observation_model(fm.k_chem(p["sigma"], T_C), tau, form="series")
    kd  = np.exp(p["lnA_d"] - p["Ea_d"] / (R_GAS * Tk))
    return p["y_max"] * (1 - np.exp(-kf * np.asarray(tR, float))) * np.exp(-kd * np.asarray(tR, float))


def wada_panel(ax, df, title, vmin=0, vmax=100, cmap="bwr", pred_p=None):
    """Same convention as plot_fbrcn_meoh_vs_meod_vs_pred.py.

    pred_p=None: RBF contour from experimental yields; labels = measured yield.
    pred_p given: contour from layered model; labels = predicted yield.
    """
    d = df.dropna(subset=["tR_s", "T_C", "yield_pct"])
    x = np.log10(d["tR_s"].values); y = d["T_C"].values; z = d["yield_pct"].values

    xi = np.linspace(x.min() - 0.05, x.max() + 0.05, 400)
    yi = np.linspace(y.min() - 3, y.max() + 3, 400)
    XI, YI = np.meshgrid(xi, yi)

    if pred_p is None:
        sx, sy = np.std(x) or 1, np.std(y) or 1
        Z = np.clip(Rbf(x / sx, y / sy, z, function="linear", smooth=0.3)(XI / sx, YI / sy),
                    0, 110)
    else:
        Z = np.clip(yld_layered(pred_p, 10**XI, YI), 0, 110)

    cs = ax.contourf(XI, YI, Z, levels=np.linspace(vmin, vmax, 21), cmap=cmap,
                     norm=Normalize(vmin, vmax), extend="both")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, n: f"$10^{{{v:.1f}}}$"))
    ax.scatter(x, y, s=28, facecolor="white", edgecolors="black", zorder=5, clip_on=False)
    xm, ym = x.mean(), y.mean()
    for _, r in d.iterrows():
        xv = np.log10(r["tR_s"]); yv = r["T_C"]
        zv = yld_layered(pred_p, r["tR_s"], r["T_C"]) if pred_p is not None else r["yield_pct"]
        if np.isnan(zv): continue
        ha = "left" if xv < xm else "right"
        va = "top" if yv > ym else "bottom"
        ax.text(xv + (0.04 if ha == "left" else -0.04),
                yv + (1.0 if va == "bottom" else -1.0),
                f"{zv:.0f}", ha=ha, va=va, fontsize=8, color="black", clip_on=False,
                path_effects=[patheffects.withStroke(linewidth=3, foreground="white")])
    ax.set_xlabel(r"$t_R$ (s)")
    ax.set_ylabel(r"$T$ ($^\circ$C)")
    ax.set_title(title, fontsize=10, fontweight="bold", pad=8)
    return cs


def main():
    df_old      = pd.read_csv(BASE / "experiment_fbrcn_summary.csv")
    df_new_meoh = pd.read_csv(BASE / "experiment_fbrcn1_summary.csv")
    df_old_dcal = pd.read_csv(BASE / "experiment_fbrcn_d_calibrated_summary.csv")

    # Drop the same two anomalous new-batch vials as in the 3-panel comparison
    df_new_meoh = df_new_meoh[~df_new_meoh["sample"].isin({"WY-260602-1", "WY-260602-11"})].copy()

    for df in (df_old, df_new_meoh, df_old_dcal):
        df.dropna(subset=["tR_s", "T_C", "yield_pct"], inplace=True)

    # Common T-axis for visual alignment across all 4 panels
    T_lo = min(df_old["T_C"].min(), df_new_meoh["T_C"].min(),
               df_old_dcal["T_C"].min()) - 3
    T_hi = max(df_old["T_C"].max(), df_new_meoh["T_C"].max(),
               df_old_dcal["T_C"].max()) + 3

    fig, axes = plt.subplots(1, 4, figsize=(22, 5.2), sharey=True)

    cs = wada_panel(axes[0], df_old,
                    title="(a) Old MeOD experiment\n"
                          "calibration: non-D 2-F-CN, $y=0.5070\\,x{-}0.0701$")
    wada_panel(axes[1], df_new_meoh,
               title="(b) New MeOH experiment (260602)\n"
                     "calibration: non-D 2-F-CN, $y=0.5070\\,x{-}0.0701$")
    wada_panel(axes[2], df_old_dcal,
               title="(c) Old MeOD re-calibrated with D-product\n"
                     "calibration: 2-F-CN-5-d, $y=0.9092\\,x{-}0.1201$")
    wada_panel(axes[3], df_new_meoh, pred_p=PRED_PARAMS,
               title=r"(d) Layered-model prediction" + "\n"
                     r"Charton $\chi$ × Da, $\tau_{ref}=15.9$ ms, $\sigma=0.62$")

    for ax in axes:
        ax.set_ylim(T_lo, T_hi); ax.grid(alpha=0.15)
    for ax in axes[1:]:
        ax.set_ylabel("")

    cax = fig.add_axes([0.94, 0.18, 0.010, 0.66])
    fig.colorbar(cs, cax=cax, ticks=np.arange(0, 101, 20)).set_label(
        "yield (%)", rotation=270, labelpad=14)

    fig.suptitle(
        "5-bromo-2-fluorobenzonitrile (strongly activated, $\\sigma=0.62$) — "
        "old vs new MeOH vs D-calibrated old vs layered prediction",
        fontsize=12, fontweight="bold", y=1.00)
    fig.tight_layout(rect=[0, 0, 0.935, 0.96])

    out = OUT / "fbrcn_4panel_old_dcal_meoh_pred.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ {out}")

    # Quantitative summary
    print("\n=== (a) Old MeOD, non-D cal — yield (%) ===")
    print(df_old.pivot(index="T_C", columns="L_cm", values="yield_pct").round(1).to_string())
    print("\n=== (c) Old MeOD, D cal — yield (%) ===")
    print(df_old_dcal.pivot(index="T_C", columns="L_cm", values="yield_pct").round(1).to_string())
    print("\n=== (c) − (a)  systematic shift from D recalibration (pp) ===")
    diff = (df_old_dcal.pivot(index="T_C", columns="L_cm", values="yield_pct") -
            df_old.pivot(index="T_C", columns="L_cm", values="yield_pct"))
    print(diff.round(1).to_string())
    print("\n=== summary statistics for (c)/(a) ratio ===")
    r = (df_old_dcal["yield_pct"] / df_old["yield_pct"]).dropna()
    print(f"  mean = {r.mean():.3f}  std = {r.std():.3f}  range = [{r.min():.3f}, {r.max():.3f}]")


if __name__ == "__main__":
    main()
