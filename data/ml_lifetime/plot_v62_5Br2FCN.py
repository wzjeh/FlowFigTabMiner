"""v6.2 blind validation figure for 5-Br-2-F-CN — 2 panels (no residual)."""
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

# v6.2 prediction for 5-Br-2-F-CN (m-CN-p-F-PhLi)
PRED_v62 = dict(Ea_f=35.23, lnA_f=23.49, Ea_d=27.51, lnA_d=9.82, y_max=84.10)


def yld(p, tR, T_C):
    Tk = T_C + 273.15
    kf = np.exp(p['lnA_f'] - p['Ea_f']/(R_GAS*Tk))
    kd = np.exp(p['lnA_d'] - p['Ea_d']/(R_GAS*Tk))
    return p['y_max'] * (1 - np.exp(-kf*tR)) * np.exp(-kd*tR)


def wada_panel(ax, df, x_col, y_col, z_col, vmin, vmax, title,
               annotate_col=None, fmt='{:.0f}'):
    df = df.dropna(subset=[x_col, y_col, z_col]).copy()
    x_data = np.log10(df[x_col].to_numpy()); y_data = df[y_col].to_numpy(); z_data = df[z_col].to_numpy()
    xi = np.linspace(x_data.min()-0.05, x_data.max()+0.05, 400)
    yi = np.linspace(y_data.min()-3, y_data.max()+3, 400)
    XI, YI = np.meshgrid(xi, yi)
    sx, sy = np.std(x_data) or 1, np.std(y_data) or 1
    rbf = Rbf(x_data/sx, y_data/sy, z_data, function='linear', smooth=0.3)
    ZI = np.clip(rbf(XI/sx, YI/sy), 0, 110)
    cf = ax.contourf(XI, YI, ZI, levels=np.linspace(vmin, vmax, 21), cmap='bwr',
                      norm=Normalize(vmin=vmin, vmax=vmax), extend='both')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, n: f"$10^{{{v:.1f}}}$"))
    ax.scatter(x_data, y_data, s=40, facecolor='white', edgecolors='black',
               linewidth=1.2, zorder=5, clip_on=False)
    x_mean, y_mean = x_data.mean(), y_data.mean()
    label_col = annotate_col if annotate_col else z_col
    for _, r in df.iterrows():
        xv = np.log10(r[x_col]); yv = r[y_col]; zv = r[label_col]
        if pd.isna(zv): continue
        ha = 'left' if xv < x_mean else 'right'; va = 'top' if yv > y_mean else 'bottom'
        ax.text(xv + (0.04 if ha=='left' else -0.04),
                yv + (1.0 if va=='bottom' else -1.0),
                fmt.format(zv), ha=ha, va=va, fontsize=8.5, color='black',
                clip_on=False,
                path_effects=[patheffects.withStroke(linewidth=3, foreground='white')])
    ax.set_xlabel('$t_{R}$ (s)'); ax.set_ylabel('$T$ (°C)')
    ax.set_title(title, fontsize=10, fontweight='bold')
    return cf


def main():
    plt.rcParams["font.family"] = 'Helvetica'
    plt.rcParams["font.size"] = 11

    df = pd.read_csv(BASE / 'experiment_fbrcn_summary.csv')
    df['pred'] = df.apply(lambda r: yld(PRED_v62, r['tR_s'], r['T_C']), axis=1)
    mae = (df['yield_pct'] - df['pred']).abs().mean()
    bias = (df['yield_pct'] - df['pred']).mean()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    wada_panel(axes[0], df, 'tR_s', 'T_C', 'yield_pct', 0, 100,
               '(a) Experiment — 5-Br-2-F-CN + n-BuLi + MeOH (25 points, calibrated)')

    # Panel (b): smooth prediction surface from v6.2 params + experimental points
    # with PREDICTED yield text at each (T, tR)
    ax = axes[1]
    T_vals = np.linspace(df['T_C'].min()-3, df['T_C'].max()+3, 80)
    tR_vals = np.logspace(np.log10(df['tR_s'].min()*0.8),
                          np.log10(df['tR_s'].max()*1.2), 80)
    T_grid, tR_grid = np.meshgrid(T_vals, tR_vals)
    Y = np.array([[yld(PRED_v62, tR, T) for T in T_vals] for tR in tR_vals])
    log_tR = np.log10(tR_grid)
    cs = ax.contourf(log_tR, T_grid, Y, levels=np.linspace(0,100,21), cmap='bwr',
                     norm=Normalize(vmin=0, vmax=100), extend='both')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, n: f"$10^{{{v:.1f}}}$"))
    log_tR_exp = np.log10(df['tR_s'].to_numpy())
    T_exp = df['T_C'].to_numpy()
    x_mean = log_tR_exp.mean(); y_mean = T_exp.mean()
    ax.scatter(log_tR_exp, T_exp, s=40, facecolor='white',
               edgecolors='black', linewidth=1.2, zorder=5, clip_on=False)
    for _, r in df.iterrows():
        xv = np.log10(r['tR_s']); yv = r['T_C']; pv = r['pred']
        ha = 'left' if xv < x_mean else 'right'
        va = 'top' if yv > y_mean else 'bottom'
        ax.text(xv + (0.04 if ha=='left' else -0.04),
                yv + (1.0 if va=='bottom' else -1.0),
                f'{pv:.0f}', ha=ha, va=va, fontsize=8.5, color='black',
                clip_on=False,
                path_effects=[patheffects.withStroke(linewidth=3, foreground='white')])
    ax.set_xlabel('$t_{R}$ (s)'); ax.set_ylabel('$T$ (°C)')
    ax.set_title(f'(b) v6.2 prediction (σ + EWG_type + mol_volume Bayesian)\n'
                 f"Ea_d={PRED_v62['Ea_d']:.1f}, lnA_d={PRED_v62['lnA_d']:.1f}, "
                 f"Ea_f={PRED_v62['Ea_f']:.1f}, lnA_f={PRED_v62['lnA_f']:.1f}, "
                 f"y_max={PRED_v62['y_max']:.0f}  |  MAE={mae:.1f} pp",
                 fontsize=10, fontweight='bold')

    cbar = fig.colorbar(cs, ax=axes, fraction=0.025, pad=0.01)
    cbar.set_label('yield (%)', rotation=270, labelpad=18)
    cbar.set_ticks(np.arange(0, 101, 10))

    plt.suptitle("v6.2 blind validation on 5-Br-2-F-CN  "
                 f"(C2/CN, m-CN-p-F-PhLi)  |  n=25, MAE={mae:.2f} pp, bias={bias:+.2f} pp",
                 fontsize=13, fontweight='bold', y=1.04)

    out = BASE / 'analysis_figures' / 'v62_blind_5Br2F_CN.png'
    fig.savefig(out, dpi=140, bbox_inches='tight')
    print(f"Saved: {out}")
    print(f"MAE={mae:.2f} pp, bias={bias:+.2f} pp, n={len(df)}")


if __name__ == '__main__':
    main()
