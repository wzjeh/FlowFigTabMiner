"""
Wada-style contour plots for 3,5-Dibromobenzonitrile / n-BuLi / TMSCl-trap.
Generates 4 figures:
  1. Substrate conversion %
  2. TMS-product yield %
  3. proto-de-Li by-product (ArH) %
  4. ortho-deprot/TMS by-product %
Vial 5 (-65°C, L=100) already excluded in CSV.
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

CSV = Path(__file__).parent / "experiment_35brcn_summary.csv"
OUT_DIR = Path(__file__).parent / "analysis_figures"


def make_wada_plot(df, metric, label, vmin, vmax, out_png, cmap='bwr'):
    plt.rcParams["font.family"] = 'Helvetica'
    plt.rcParams["font.size"] = 12
    plt.rcParams['mathtext.fontset'] = 'dejavusans'

    df = df.dropna(subset=[metric, "tR_s", "T_C"]).copy()
    x_data = np.log10(df["tR_s"].to_numpy())
    y_data = df["T_C"].to_numpy()
    z_data = df[metric].to_numpy()

    xi = np.linspace(x_data.min() - 0.05, x_data.max() + 0.05, 500)
    yi = np.linspace(y_data.min() - 3, y_data.max() + 3, 500)
    XI, YI = np.meshgrid(xi, yi)

    sx = np.std(x_data) or 1.0
    sy = np.std(y_data) or 1.0
    rbf = Rbf(x_data/sx, y_data/sy, z_data, function='linear', smooth=0.5)
    ZI = np.clip(rbf(XI/sx, YI/sy), vmin - 5, vmax + 5)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda v, n: f"$10^{{{v:.1f}}}$"))

    levels = np.linspace(vmin, vmax, 21)
    norm = Normalize(vmin=vmin, vmax=vmax)
    cntr = plt.contourf(XI, YI, ZI, levels=levels, cmap=cmap,
                        norm=norm, extend='both')

    cbar = plt.colorbar(cntr, ax=ax)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label(label, rotation=0, va='bottom', ha='left', labelpad=15)
    cbar_label = cbar.ax.yaxis.get_label()
    cbar_label.set_verticalalignment('bottom')
    cbar_label.set_horizontalalignment('right')
    cbar_label.set_position((0, 1.05))
    nticks = max(2, int((vmax - vmin) / 10) + 1)
    cbar.set_ticks(np.linspace(vmin, vmax, nticks))

    plt.scatter(x_data, y_data, s=30, facecolor='white',
                edgecolors='black', zorder=5, clip_on=False)

    x_mean = x_data.mean(); y_mean = y_data.mean()
    for _, r in df.iterrows():
        xv = np.log10(r["tR_s"]); yv = r["T_C"]; zv = r[metric]
        if zv is None or np.isnan(zv): continue
        ha = 'left' if xv < x_mean else 'right'
        va = 'top'  if yv > y_mean else 'bottom'
        ax.text(xv + (0.04 if ha == 'left' else -0.04),
                yv + (1.0 if va == 'bottom' else -1.0),
                f'{zv:.0f}',
                verticalalignment=va, horizontalalignment=ha,
                color='black', fontsize=11, fontname='Helvetica',
                clip_on=False,
                path_effects=[patheffects.withStroke(linewidth=3,
                              foreground='white', capstyle="round")])

    plt.xlabel('$t_{R}$(s)')
    plt.ylabel('$T$(°C)')
    y_label = ax.yaxis.get_label()
    y_label.set_rotation(0)
    y_label.set_verticalalignment('bottom')
    y_label.set_horizontalalignment('left')
    y_label.set_position((0, 1.05))
    ax.yaxis.labelpad = -10

    out_png = Path(out_png)
    out_png.parent.mkdir(exist_ok=True)
    plt.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.savefig(out_png.with_suffix('.tif'), format='tif', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_png}")


def main():
    df = pd.read_csv(CSV)
    print(f"Plotting {len(df)} vials (vial 5 excluded).\n")

    make_wada_plot(df, "conversion_pct",
                   label="conversion(%)", vmin=80, vmax=100,
                   out_png=OUT_DIR / "experiment_35brcn_conversion_wada.png")
    make_wada_plot(df, "trap_yield_pct",
                   label="yield(%)", vmin=0, vmax=100,
                   out_png=OUT_DIR / "experiment_35brcn_yield_wada.png")
    make_wada_plot(df, "proto_deLi_pct",
                   label="ArH(%)", vmin=0, vmax=70,
                   out_png=OUT_DIR / "experiment_35brcn_arh_wada.png")
    make_wada_plot(df, "ortho_deprot_pct",
                   label="ortho-TMS(%)", vmin=0, vmax=12,
                   out_png=OUT_DIR / "experiment_35brcn_ortho_wada.png")


if __name__ == "__main__":
    main()
