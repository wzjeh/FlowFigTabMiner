"""
Wada-style contour of ArLi-derived TRAP PRODUCT yield (14 min peak only).

This is the bell-curve metric Zhao expects:
  - main trap product = (4-FC6H4)(Ph)CHOH at ~14 min
  - secondary 11.6 min peak (Bu-CH(OH)Ph) is n-BuLi-residual byproduct, excluded
  - 15.4 min minor side product, excluded

Calibration: anchor vial #18 (-50°C, L=25, conv=96%) to yield = 96%.
  i.e. assume at this point all substrate-consumed → trap product captured.
  → empirical RRF_effective = (area_14/C12)_vial18 / (0.96 × 1.143)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patheffects as patheffects
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from scipy.interpolate import Rbf

CSV = Path(__file__).parent / "experiment_phcho_summary.csv"
OUT_PNG = Path(__file__).parent / "analysis_figures" / "experiment_phcho_yield_wada.png"
OUT_TIF = Path(__file__).parent / "analysis_figures" / "experiment_phcho_yield_wada.tif"

L_ORDER = [3, 10, 25, 50, 100]
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
EXCLUDE_VIALS = set()

TUBE_AREA_CM2 = np.pi * 0.025 ** 2
TOTAL_FLOW_ML_S = 7.5 / 60


def main():
    df = pd.read_csv(CSV)
    df["T_C"]  = df["vial"].map(lambda v: VIAL_MAP.get(v, (None, None))[0])
    df["L_cm"] = df["vial"].map(lambda v: VIAL_MAP.get(v, (None, None))[1])
    df = df[~df["vial"].isin(EXCLUDE_VIALS)].copy()
    df = df.dropna(subset=["T_C", "L_cm", "prod_area", "c12_area"]).copy()
    df["prod_over_c12"] = df["prod_area"] / df["c12_area"]
    df["tR_s"] = TUBE_AREA_CM2 * df["L_cm"] / TOTAL_FLOW_ML_S

    # ---- Calibration: anchor vial #18 to yield = 96% (= its conversion) ----
    anchor = df[df["vial"] == 18].iloc[0]
    anchor_ratio = anchor["prod_over_c12"]
    anchor_yield = anchor["conversion_pct"]
    yield_per_unit = anchor_yield / anchor_ratio
    df["yield_pct"] = df["prod_over_c12"] * yield_per_unit

    print(f"Anchor: vial #18 (-50°C, L=25)  conv={anchor_yield:.1f}%  "
          f"(prod/C12)={anchor_ratio:.3f}")
    print(f"=> yield_pct = (prod/C12) × {yield_per_unit:.3f}\n")
    print(df[["vial","T_C","L_cm","conversion_pct","prod_over_c12","yield_pct"]]
          .sort_values(["T_C","L_cm"], ascending=[False, True])
          .to_string(index=False))

    # Build 5x5 grid
    z0 = np.full((len(T_ORDER), len(L_ORDER)), np.nan)
    for _, r in df.iterrows():
        i = T_ORDER.index(int(r["T_C"]))
        j = L_ORDER.index(int(r["L_cm"]))
        z0[i, j] = r["yield_pct"]

    x0 = TUBE_AREA_CM2 * np.array(L_ORDER, dtype=float) / TOTAL_FLOW_ML_S
    y0 = np.array(T_ORDER, dtype=float)
    x1 = np.log10(x0)
    X, Y = np.meshgrid(x1, y0)

    xi = np.linspace(x1.min(), x1.max(), 500)
    yi = np.linspace(y0.min(), y0.max(), 500)
    XI, YI = np.meshgrid(xi, yi)

    mask = ~np.isnan(z0)
    x_train = X[mask]; y_train = Y[mask]; z_train = z0[mask]
    sx = np.std(x_train) or 1.0
    sy = np.std(y_train) or 1.0
    rbf = Rbf(x_train/sx, y_train/sy, z_train, function='linear', smooth=0.2)
    ZI = rbf(XI/sx, YI/sy)
    ZI = np.clip(ZI, 0, 100)

    def format_func(value, tick_number):
        return f"$10^{{{value:.1f}}}$"

    plt.rcParams["font.family"] = 'Helvetica'
    plt.rcParams["font.size"] = 12
    plt.rcParams['mathtext.fontset'] = 'dejavusans'

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_func))

    levels = np.linspace(0, 100, 21)
    norm = Normalize(vmin=0, vmax=100)
    cntr = plt.contourf(XI, YI, ZI, levels=levels, cmap='bwr',
                        norm=norm, extend='both')

    cbar = plt.colorbar(cntr, ax=ax)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label("yield(%)", rotation=0, va='bottom', ha='left', labelpad=15)
    cbar_label = cbar.ax.yaxis.get_label()
    cbar_label.set_verticalalignment('bottom')
    cbar_label.set_horizontalalignment('right')
    cbar_label.set_position((0, 1.05))
    cbar.set_ticks(np.arange(0, 101, 10))

    plt.scatter(X[mask], Y[mask], s=30, facecolor='white',
                edgecolors='black', zorder=5, clip_on=False)

    offset_x = 0.05
    offset_y = 1
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if not np.isnan(z0[i, j]):
                ha = 'left' if X[i, j] < X.mean() else 'right'
                va = 'top'  if Y[i, j] > Y.mean() else 'bottom'
                ax.text(X[i, j] + (offset_x if ha == 'left' else -offset_x),
                        Y[i, j] + (offset_y if va == 'bottom' else -offset_y),
                        f'{z0[i, j]:.0f}',
                        verticalalignment=va, horizontalalignment=ha,
                        color='black', fontsize=14, fontname='Helvetica',
                        clip_on=False,
                        path_effects=[patheffects.withStroke(linewidth=3,
                                      foreground='white', capstyle="round")])

    plt.xlabel('$t_{1}$(s)')
    plt.ylabel('$T$(°C)')

    y_label = ax.yaxis.get_label()
    y_label.set_rotation(0)
    y_label.set_verticalalignment('bottom')
    y_label.set_horizontalalignment('left')
    y_label.set_position((0, 1.05))
    ax.yaxis.labelpad = -10

    OUT_PNG.parent.mkdir(exist_ok=True)
    plt.savefig(OUT_PNG, dpi=200, bbox_inches='tight')
    plt.savefig(OUT_TIF, format='tif', dpi=300, bbox_inches='tight')
    print(f"\nSaved: {OUT_PNG}")
    print(f"Saved: {OUT_TIF}")


if __name__ == "__main__":
    main()
