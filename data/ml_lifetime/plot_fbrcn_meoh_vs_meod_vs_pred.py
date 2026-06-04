"""
3-panel comparison heatmap for 5-bromo-2-fluorobenzonitrile (substrate 2):
  (a) PREVIOUS experimental yield (260522 + 260523, MeOD label per paper)
  (b) NEW experimental yield      (260602 batch, MeOH quench)
  (c) Layered-model prediction    (Charton × Damköhler × τ_eff(T), σ=0.62)

Same Wada-style contour rendering as plot_fbrcn_wada.py: log-t_R on x, T on y,
yield contour clipped to 0–110 %, white-edged dots at experimental points with
numeric yield labelled. Shared colorbar across panels.

Run:
  python plot_fbrcn_meoh_vs_meod_vs_pred.py
Output:
  analysis_figures/fbrcn_meoh_vs_meod_vs_pred.png
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

BASE = Path(__file__).parent
OUT  = BASE / "analysis_figures"
OUT.mkdir(exist_ok=True)
R_GAS = 8.314e-3   # kJ mol⁻¹ K⁻¹

# ---- layered-model parameters for substrate 2 (paper Fig 5(e) values) ----
# Decomposition (lnA_d, Ea_d, y_max) come from the v6.0 ArLi Bayesian posterior
# used in draw_paper_figures.py via _bayes_5Br2FCN().  We import directly.
from draw_paper_figures import _bayes_5Br2FCN, _TAU_REF
PRED_PARAMS = {**_bayes_5Br2FCN(), "sigma": 0.62, "tau_eff": _TAU_REF}


def yld_layered(p, tR, T_C):
    """Layered model: kf,obs = k_chem / (1 + τ_eff·k_chem); kd from Arrhenius."""
    Tk = np.asarray(T_C, float) + 273.15
    tau = fm.tau_eff_T(p["tau_eff"], T_C)
    kf  = fm.observation_model(fm.k_chem(p["sigma"], T_C), tau, form="series")
    kd  = np.exp(p["lnA_d"] - p["Ea_d"] / (R_GAS * Tk))
    return p["y_max"] * (1 - np.exp(-kf * np.asarray(tR, float))) * np.exp(-kd * np.asarray(tR, float))


def wada_panel(ax, df, title, vmin=0, vmax=100, cmap="bwr", pred_p=None):
    """Render one heatmap panel.

    pred_p=None: contour is RBF-interpolated from experimental yields; numeric
        overlays show the measured yield at each experimental (t_R, T) point.
    pred_p given: contour is the layered-model surface; numeric overlays show
        the *predicted* yield at each (t_R, T) cell — the cross-comparison with
        panel (b) reveals the model's per-cell residual at a glance."""
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
    ax.scatter(x, y, s=30, facecolor="white", edgecolors="black", zorder=5, clip_on=False)
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
    ax.set_title(title, fontsize=10.5, fontweight="bold", pad=8)
    return cs


def main():
    df_old = pd.read_csv(BASE / "experiment_fbrcn_summary.csv")
    df_new = pd.read_csv(BASE / "experiment_fbrcn1_summary.csv")

    # Drop two anomalous vials from the new (260602) batch (Zhao 2026-06-04):
    #   vial 1  — +20°C, L=3 cm: yield 2.2%, mass balance 40% (startup transient)
    #   vial 11 — −25°C, L=3 cm: yield 109%, mass balance 113% (calibration overshoot)
    DROPPED = {"WY-260602-1", "WY-260602-11"}
    df_new = df_new[~df_new["sample"].isin(DROPPED)].copy()

    # Align column expectations
    for df in (df_old, df_new):
        df.dropna(subset=["tR_s", "T_C", "yield_pct"], inplace=True)

    # Common T axis for visual alignment
    T_lo = min(df_old["T_C"].min(), df_new["T_C"].min()) - 3
    T_hi = max(df_old["T_C"].max(), df_new["T_C"].max()) + 3

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), sharey=True)

    cs = wada_panel(axes[0], df_old,
                    title=f"(a) Previous experiment\n5-Br-2-F-C$_6$H$_3$CN + n-BuLi, MeOD quench (260522/523)")
    wada_panel(axes[1], df_new,
               title=f"(b) New experiment\n5-Br-2-F-C$_6$H$_3$CN + n-BuLi, MeOH quench (260602)")
    wada_panel(axes[2], df_new, pred_p=PRED_PARAMS,
               title=r"(c) Layered model prediction (labels = predicted yield)"+"\n"+
                     r"Charton $\chi$ × Da, viscosity-scaled $\tau_{eff}(T)$, $\sigma=0.62$")

    for ax in axes:
        ax.set_ylim(T_lo, T_hi)
        ax.grid(alpha=0.15)
    # Hide redundant y-labels on shared axes
    for ax in axes[1:]:
        ax.set_ylabel("")

    cax = fig.add_axes([0.92, 0.18, 0.012, 0.66])
    fig.colorbar(cs, cax=cax, ticks=np.arange(0, 101, 20)).set_label(
        "yield (%)", rotation=270, labelpad=14)

    fig.suptitle(
        "5-bromo-2-fluorobenzonitrile (strongly activated, $\\sigma=0.62$) "
        "— old vs new flow quench experiments vs layered-model prediction",
        fontsize=12, fontweight="bold", y=1.00)
    fig.tight_layout(rect=[0, 0, 0.91, 0.96])

    out = OUT / "fbrcn_meoh_vs_meod_vs_pred.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ {out}")

    # numeric overlap table: side-by-side (T, t_R) -> yield in both experiments
    print("\n=== old vs new yield comparison (T, t_R common cells) ===")
    pivot_old = df_old.pivot_table(index="T_C", columns="L_cm", values="yield_pct", aggfunc="mean")
    pivot_new = df_new.pivot_table(index="T_C", columns="L_cm", values="yield_pct", aggfunc="mean")
    print("Old (MeOD):\n", pivot_old.round(1).to_string())
    print("\nNew (MeOH):\n", pivot_new.round(1).to_string())
    common_T = sorted(set(pivot_old.index) & set(pivot_new.index))
    if common_T:
        diff = pivot_new.loc[common_T] - pivot_old.loc[common_T]
        print("\nNew − Old (yield pp diff at shared (T, L)):\n", diff.round(1).to_string())


if __name__ == "__main__":
    main()
