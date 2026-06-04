"""
Wada-style yield surface plot: 5-bromo-2-fluorobenzonitrile + n-BuLi + MeOH quench.

Panel (a): experimental yield (calibrated, side-peak included)
Panel (b): prediction using 3-CN-PhLi Tier 1 Arrhenius as analog
           (Ea_f=34.69, lnA_f=22.11, Ea_d=27.3, lnA_d=9.08, y_max=83.1).
           Each experimental (T, tR) position is overlaid with predicted yield text.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patheffects as patheffects
from matplotlib.colors import Normalize
from scipy.interpolate import Rbf
from pathlib import Path

BASE = Path(__file__).parent
R_GAS = 8.314e-3

# Prediction params: 3-CN-PhLi Tier 1 (m-cyanophenyllithium, n_T=6, r²=0.97)
PRED = dict(Ea_f=34.69, lnA_f=22.11, Ea_d=27.3, lnA_d=9.08, y_max=83.1)

def yld(p, tR, T_C):
    Tk = T_C + 273.15
    kf = np.exp(p['lnA_f'] - p['Ea_f']/(R_GAS*Tk))
    kd = np.exp(p['lnA_d'] - p['Ea_d']/(R_GAS*Tk))
    return p['y_max'] * (1 - np.exp(-kf*tR)) * np.exp(-kd*tR)


def wada_panel(ax, df, x_col, y_col, z_col, vmin, vmax, title,
               annotate_col=None, cmap='bwr'):
    df = df.dropna(subset=[x_col, y_col, z_col]).copy()
    x_data = np.log10(df[x_col].to_numpy())
    y_data = df[y_col].to_numpy()
    z_data = df[z_col].to_numpy()
    xi = np.linspace(x_data.min()-0.05, x_data.max()+0.05, 400)
    yi = np.linspace(y_data.min()-3, y_data.max()+3, 400)
    XI, YI = np.meshgrid(xi, yi)
    sx, sy = np.std(x_data) or 1, np.std(y_data) or 1
    rbf = Rbf(x_data/sx, y_data/sy, z_data, function='linear', smooth=0.3)
    ZI = np.clip(rbf(XI/sx, YI/sy), 0, 110)
    cntr = ax.contourf(XI, YI, ZI, levels=np.linspace(vmin, vmax, 21), cmap=cmap,
                       norm=Normalize(vmin=vmin, vmax=vmax), extend='both')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, n: f"$10^{{{v:.1f}}}$"))
    ax.scatter(x_data, y_data, s=30, facecolor='white', edgecolors='black',
               zorder=5, clip_on=False)
    x_mean, y_mean = x_data.mean(), y_data.mean()
    label_col = annotate_col if annotate_col else z_col
    for _, r in df.iterrows():
        xv = np.log10(r[x_col]); yv = r[y_col]; zv = r[label_col]
        if pd.isna(zv): continue
        ha = 'left' if xv < x_mean else 'right'
        va = 'top' if yv > y_mean else 'bottom'
        ax.text(xv + (0.04 if ha=='left' else -0.04),
                yv + (1.0 if va=='bottom' else -1.0),
                f'{zv:.0f}', ha=ha, va=va, fontsize=9, color='black',
                clip_on=False,
                path_effects=[patheffects.withStroke(linewidth=3, foreground='white')])
    ax.set_xlabel('$t_{R}$ (s)'); ax.set_ylabel('$T$ (°C)')
    ax.set_title(title, fontsize=10, fontweight='bold')
    return cntr


def main():
    plt.rcParams["font.family"] = 'Helvetica'
    plt.rcParams["font.size"] = 11

    df = pd.read_csv(BASE / 'experiment_fbrcn_summary.csv')
    df['pred'] = df.apply(lambda r: yld(PRED, r['tR_s'], r['T_C']), axis=1)
    df['resid'] = (df['yield_pct'] - df['pred']).round(2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel (a): experimental yield
    wada_panel(axes[0], df, 'tR_s', 'T_C', 'yield_pct', 0, 100,
               '(a) 5-Br-2-F-C$_6$H$_3$-CN + n-BuLi + MeOH quench: experimental yield (%)')

    # Panel (b): prediction surface with predicted yield text at each exp point
    ax = axes[1]
    T_vals = np.linspace(df['T_C'].min()-3, df['T_C'].max()+3, 80)
    tR_vals = np.logspace(np.log10(df['tR_s'].min()*0.8),
                          np.log10(df['tR_s'].max()*1.2), 80)
    T_grid, tR_grid = np.meshgrid(T_vals, tR_vals)
    Y = np.array([[yld(PRED, tR, T) for T in T_vals] for tR in tR_vals])
    log_tR = np.log10(tR_grid)
    cs = ax.contourf(log_tR, T_grid, Y, levels=np.linspace(0,100,21), cmap='bwr',
                     norm=Normalize(vmin=0, vmax=100), extend='both')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, n: f"$10^{{{v:.1f}}}$"))
    # Overlay exp positions + predicted yield text
    log_tR_exp = np.log10(df['tR_s'].to_numpy())
    T_exp = df['T_C'].to_numpy()
    x_mean = log_tR_exp.mean(); y_mean = T_exp.mean()
    ax.scatter(log_tR_exp, T_exp, s=30, facecolor='white',
               edgecolors='black', zorder=5, clip_on=False)
    for _, r in df.iterrows():
        xv = np.log10(r['tR_s']); yv = r['T_C']; pv = r['pred']
        ha = 'left' if xv < x_mean else 'right'
        va = 'top' if yv > y_mean else 'bottom'
        ax.text(xv + (0.04 if ha=='left' else -0.04),
                yv + (1.0 if va=='bottom' else -1.0),
                f'{pv:.0f}', ha=ha, va=va, fontsize=9, color='black',
                clip_on=False,
                path_effects=[patheffects.withStroke(linewidth=3, foreground='white')])
    ax.set_xlabel('$t_{R}$ (s)'); ax.set_ylabel('$T$ (°C)')
    ax.set_title('(b) 5-Br-2-F-C$_6$H$_3$-CN prediction (3-CN-PhLi analog)\n'
                 f"Ea_f={PRED['Ea_f']}, lnA_f={PRED['lnA_f']}, "
                 f"Ea_d={PRED['Ea_d']}, lnA_d={PRED['lnA_d']}, y_max={PRED['y_max']}",
                 fontsize=10, fontweight='bold')

    cbar = fig.colorbar(cs, ax=axes, fraction=0.025, pad=0.01)
    cbar.set_label('yield (%)', rotation=270, labelpad=18)
    cbar.set_ticks(np.arange(0, 101, 10))

    mae  = (df['yield_pct'] - df['pred']).abs().mean()
    mbe  = (df['yield_pct'] - df['pred']).mean()
    plt.suptitle("5-bromo-2-fluorobenzonitrile + n-BuLi + MeOH quench:  "
                 f"experiment (a) vs prediction (b)\n"
                 f"3-CN-PhLi Tier 1 analog params  |  n=25, MAE={mae:.1f} pp, "
                 f"mean(exp-pred)={mbe:+.1f} pp",
                 fontsize=12, fontweight='bold', y=1.04)

    out = BASE / 'analysis_figures' / 'fbrcn_wada_exp_vs_pred.png'
    fig.savefig(out, dpi=140, bbox_inches='tight')
    print(f"Saved: {out}")
    print(f"\nMAE = {mae:.2f} pp, mean(exp-pred) = {mbe:+.2f} pp, n={len(df)}")
    print("\nPer-row residuals:")
    print(df[['sample','T_C','L_cm','tR_s','yield_pct','pred','resid']].to_string(index=False))


if __name__ == '__main__':
    main()
