"""Side-by-side comparison: MeOH quench (260509) vs PhCHO trap (260512)."""
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

OUT_PNG = Path(__file__).parent / "analysis_figures" / "compare_meoh_phcho.png"

L_ORDER = [3, 10, 25, 50, 100]
T_ORDER = [20, 0, -25, -50, -78, -65]    # union; -78 only MeOH, -65 only PhCHO


def build_grid(df, t_axis, l_axis, col):
    z = np.full((len(t_axis), len(l_axis)), np.nan)
    for _, r in df.iterrows():
        if int(r["T_C"]) not in t_axis: continue
        if int(r["L_cm"]) not in l_axis: continue
        i = t_axis.index(int(r["T_C"]))
        j = l_axis.index(int(r["L_cm"]))
        z[i, j] = r[col]
    return z


def plot_panel(ax, z0, x1, y0, title):
    X, Y = np.meshgrid(x1, y0)
    xi = np.linspace(x1.min(), x1.max(), 400)
    yi = np.linspace(y0.min(), y0.max(), 400)
    XI, YI = np.meshgrid(xi, yi)
    mask = ~np.isnan(z0)
    if mask.sum() >= 4:
        sx = np.std(X[mask]) or 1.0
        sy = np.std(Y[mask]) or 1.0
        rbf = Rbf(X[mask]/sx, Y[mask]/sy, z0[mask], function='linear', smooth=0.2)
        ZI = np.clip(rbf(XI/sx, YI/sy), 0, 100)
        levels = np.linspace(0, 100, 21)
        ax.contourf(XI, YI, ZI, levels=levels, cmap='bwr',
                    norm=Normalize(vmin=0, vmax=100), extend='both')
    ax.scatter(X[mask], Y[mask], s=30, facecolor='white',
               edgecolors='black', zorder=5, clip_on=False)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if not np.isnan(z0[i, j]):
                ha = 'left' if X[i, j] < X.mean() else 'right'
                va = 'top' if Y[i, j] > Y.mean() else 'bottom'
                ax.text(X[i, j] + (0.04 if ha == 'left' else -0.04),
                        Y[i, j] + (1.0 if va == 'bottom' else -1.0),
                        f'{z0[i, j]:.0f}', verticalalignment=va, horizontalalignment=ha,
                        color='black', fontsize=11, fontname='Helvetica', clip_on=False,
                        path_effects=[patheffects.withStroke(linewidth=3,
                                      foreground='white', capstyle="round")])
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, n: f"$10^{{{v:.1f}}}$"))
    ax.set_xlabel('$t_{1}$(s)')
    ax.set_title(title, fontsize=12)


def main():
    plt.rcParams["font.family"] = 'Helvetica'
    plt.rcParams["font.size"] = 11
    plt.rcParams['mathtext.fontset'] = 'dejavusans'

    # ---- MeOH data ----
    meoh = pd.read_csv(Path(__file__).parent / "experiment_yield_summary.csv")
    meoh = meoh.rename(columns={'conversion_pct_ECN': 'conv'})

    # ---- PhCHO data ----
    phcho = pd.read_csv(Path(__file__).parent / "experiment_phcho_summary.csv")
    phcho['T_C']  = phcho['T_C_guess']
    phcho['L_cm'] = phcho['L_cm_guess']
    phcho = phcho[phcho['vial'] != 21]
    phcho = phcho.rename(columns={'conversion_pct': 'conv'})

    # union of T axes
    t_meoh = [20, 0, -25, -50, -78]
    t_phcho = [20, 0, -25, -50, -65]
    L_arr = np.array(L_ORDER, dtype=float)
    tube = np.pi * 0.025 ** 2
    x0 = tube * L_arr / (7.5/60)
    x1 = np.log10(x0)

    z_meoh  = build_grid(meoh,  t_meoh,  L_ORDER, 'conv')
    z_phcho = build_grid(phcho, t_phcho, L_ORDER, 'conv')

    # Difference grid (common temperatures)
    t_common = [20, 0, -25, -50]
    z_m = build_grid(meoh,  t_common, L_ORDER, 'conv')
    z_p = build_grid(phcho, t_common, L_ORDER, 'conv')
    z_diff = z_p - z_m

    # 3 panels
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    plot_panel(axes[0], z_meoh,  x1, np.array(t_meoh, dtype=float),
               "MeOH quench  (260509)")
    plot_panel(axes[1], z_phcho, x1, np.array(t_phcho, dtype=float),
               "PhCHO trap   (260512)")

    # difference panel uses different cmap
    X, Y = np.meshgrid(x1, np.array(t_common, dtype=float))
    xi = np.linspace(x1.min(), x1.max(), 400)
    yi = np.linspace(min(t_common), max(t_common), 400)
    XI, YI = np.meshgrid(xi, yi)
    mask = ~np.isnan(z_diff)
    if mask.sum() >= 4:
        sx = np.std(X[mask]) or 1.0
        sy = np.std(Y[mask]) or 1.0
        rbf = Rbf(X[mask]/sx, Y[mask]/sy, z_diff[mask],
                  function='linear', smooth=0.2)
        ZI = np.clip(rbf(XI/sx, YI/sy), -80, 20)
        levels = np.arange(-80, 21, 5)
        axes[2].contourf(XI, YI, ZI, levels=levels, cmap='RdBu',
                         norm=Normalize(vmin=-80, vmax=20), extend='both')
    axes[2].scatter(X[mask], Y[mask], s=30, facecolor='white',
                    edgecolors='black', zorder=5, clip_on=False)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if not np.isnan(z_diff[i, j]):
                v = z_diff[i, j]
                ha = 'left' if X[i, j] < X.mean() else 'right'
                va = 'top' if Y[i, j] > Y.mean() else 'bottom'
                axes[2].text(X[i, j] + (0.04 if ha == 'left' else -0.04),
                             Y[i, j] + (1.0 if va == 'bottom' else -1.0),
                             f'{v:+.0f}', verticalalignment=va, horizontalalignment=ha,
                             color='black', fontsize=11, fontname='Helvetica', clip_on=False,
                             path_effects=[patheffects.withStroke(linewidth=3,
                                           foreground='white', capstyle="round")])
    axes[2].xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, n: f"$10^{{{v:.1f}}}$"))
    axes[2].set_xlabel('$t_{1}$(s)')
    axes[2].set_title("Difference: PhCHO − MeOH  (negative = PhCHO lower)", fontsize=12)

    for ax in axes:
        ax.set_ylabel('$T$(°C)')

    plt.suptitle("Substrate conversion comparison · 4-Br-FC₆H₄ + n-BuLi · MeOH quench vs PhCHO trap",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    OUT_PNG.parent.mkdir(exist_ok=True)
    plt.savefig(OUT_PNG, dpi=200, bbox_inches='tight')
    print(f"Saved: {OUT_PNG}")


if __name__ == "__main__":
    main()
