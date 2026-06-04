"""Single-panel wada-style contour of experimental yield only."""
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patheffects as patheffects
from matplotlib.colors import Normalize
from scipy.interpolate import Rbf
from pathlib import Path

BASE = Path(__file__).parent
df = pd.read_csv(BASE / 'experiment_fbrcn_summary.csv')

plt.rcParams["font.family"] = 'Helvetica'; plt.rcParams["font.size"] = 11
fig, ax = plt.subplots(figsize=(8.5, 6))

x_data = np.log10(df['tR_s'].to_numpy())
y_data = df['T_C'].to_numpy()
z_data = df['yield_pct'].to_numpy()

xi = np.linspace(x_data.min()-0.05, x_data.max()+0.05, 400)
yi = np.linspace(y_data.min()-3, y_data.max()+3, 400)
XI, YI = np.meshgrid(xi, yi)
sx, sy = np.std(x_data) or 1, np.std(y_data) or 1
rbf = Rbf(x_data/sx, y_data/sy, z_data, function='linear', smooth=0.3)
ZI = np.clip(rbf(XI/sx, YI/sy), 0, 110)

cntr = ax.contourf(XI, YI, ZI, levels=np.linspace(0, 100, 21), cmap='bwr',
                   norm=Normalize(vmin=0, vmax=100), extend='both')
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, n: f"$10^{{{v:.1f}}}$"))
ax.scatter(x_data, y_data, s=40, facecolor='white', edgecolors='black',
           linewidth=1.2, zorder=5, clip_on=False)

x_mean, y_mean = x_data.mean(), y_data.mean()
for _, r in df.iterrows():
    xv = np.log10(r['tR_s']); yv = r['T_C']; zv = r['yield_pct']
    ha = 'left' if xv < x_mean else 'right'
    va = 'top' if yv > y_mean else 'bottom'
    ax.text(xv + (0.04 if ha=='left' else -0.04),
            yv + (1.2 if va=='bottom' else -1.2),
            f'{zv:.0f}', ha=ha, va=va, fontsize=10, color='black',
            clip_on=False,
            path_effects=[patheffects.withStroke(linewidth=3, foreground='white')])

ax.set_xlabel('$t_{R}$ (s)', fontsize=12)
ax.set_ylabel('$T$ (°C)', fontsize=12)
ax.set_title("5-bromo-2-fluorobenzonitrile + n-BuLi + MeOH quench\n"
             "Experimental yield (%) — calibrated, n=25, side peak included",
             fontsize=12, fontweight='bold')

cbar = fig.colorbar(cntr, ax=ax, fraction=0.045, pad=0.02)
cbar.set_label('yield (%)', rotation=270, labelpad=18)
cbar.set_ticks(np.arange(0, 101, 10))

best = df.loc[df['yield_pct'].idxmax()]
ax.annotate(f"peak: {best['yield_pct']:.1f}% @ T={int(best['T_C'])}°C, L={int(best['L_cm'])}cm",
            xy=(np.log10(best['tR_s']), best['T_C']),
            xytext=(0.55, 0.95), textcoords='axes fraction',
            fontsize=10, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='black', lw=1.2),
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='black'))

plt.tight_layout()
out = BASE / 'analysis_figures' / 'fbrcn_wada_exp_only.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f"Saved: {out}")
