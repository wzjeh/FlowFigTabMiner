"""
v6.0 Phase 5: Blind validation on 4-Br-FC6H4 and 5-Br-2-F-CN.

Uses v60_subclass_anchors.csv (empirical (class × EWG_type) anchors).

  4-Br-FC6H4  → p-F-PhLi   → (C1, none)  → empirical anchor (Ea_d=76.6, lnA_d=18.2, ...)
  5-Br-2-F-CN → m-CN-p-F-PhLi → (C2, CN) → empirical anchor (Ea_d=28.1, lnA_d=9.5, ...)

Compares to:
  - Experimental yield matrix (4-Br-FC6H4: experiment_phcho_summary_v2.csv)
  - Experimental yield matrix (5-Br-2-F-CN: experiment_fbrcn_summary.csv)
  - v4.7 lit-anchor (Ea_f=32, lnA_f=21, Ea_d=78, lnA_d=22) for C1
  - 3-CN-PhLi analog (Ea_d=27.3, lnA_d=9.08, ...) for C2 (essentially the C2/CN anchor)
  - v4.6 m-ArLi formula prediction (the disastrous −48 Ea_d) — for honesty

Generates yield surface comparison figure.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patheffects as patheffects
from matplotlib.colors import Normalize
from scipy.interpolate import Rbf

BASE = Path(__file__).parent
R_GAS = 8.314e-3


def yld(p, tR, T_C):
    Tk = T_C + 273.15
    kf = np.exp(p['lnA_f'] - p['Ea_f']/(R_GAS*Tk))
    kd = np.exp(p['lnA_d'] - p['Ea_d']/(R_GAS*Tk))
    return p['y_max'] * (1 - np.exp(-kf*tR)) * np.exp(-kd*tR)


def get_anchor(anchors_df, cls, ewg):
    sub = anchors_df[(anchors_df['class']==cls) & (anchors_df['ewg_type']==ewg)]
    if len(sub) == 0:
        return None
    r = sub.iloc[0]
    return dict(
        Ea_f=r['Ea_f_mean'], lnA_f=r['lnA_f_mean'],
        Ea_d=r['Ea_d_mean'], lnA_d=r['lnA_d_mean'],
        y_max=r['y_max_mean'],
        Ea_f_sd=r['Ea_f_sd'], lnA_f_sd=r['lnA_f_sd'],
        Ea_d_sd=r['Ea_d_sd'], lnA_d_sd=r['lnA_d_sd'],
        n=int(r['n']),
    )


def main():
    anchors = pd.read_csv(BASE / 'v60_subclass_anchors.csv')

    # ============================================================
    # SUBSTRATE 1: 4-Br-FC6H4 → p-F-PhLi (C1, none)
    # ============================================================
    print("=" * 80)
    print("Blind validation 1: 4-Br-FC6H4 + n-BuLi + PhCHO")
    print("class = C1 (inert / proto-de-Li), ewg_type = none")
    print("=" * 80)

    p_C1 = get_anchor(anchors, 'C1', 'none')
    p_v47 = dict(Ea_f=32.0, lnA_f=21.0, Ea_d=78.0, lnA_d=22.0, y_max=100.0)

    print(f"\nv6.0 (C1/none, n={p_C1['n']}):")
    print(f"  Ea_d={p_C1['Ea_d']:.1f}±{p_C1['Ea_d_sd']:.1f}  "
          f"lnA_d={p_C1['lnA_d']:.1f}±{p_C1['lnA_d_sd']:.1f}  "
          f"y_max={p_C1['y_max']:.0f}")
    print(f"v4.7 lit anchor: Ea_d=78, lnA_d=22, y_max=100")

    # Load 4-Br-FC6H4 experimental (matches Step 11.8a in notebook)
    df1 = pd.read_csv(BASE / 'experiment_phcho_summary_v2.csv')
    df1 = df1[df1['date'] == 260512].copy()
    df1['val'] = df1['Yeild'] * 100
    df1 = df1[~((df1['L_cm']==2.5) & (df1['T_C']==-65))]
    mask = (df1['L_cm']==2.5)
    df1.loc[mask, 'L_cm'] = 10
    df1.loc[mask, 'tR_s'] = 0.1571
    df1['pred_v60'] = df1.apply(lambda r: yld(p_C1, r['tR_s'], r['T_C']), axis=1)
    df1['pred_v47'] = df1.apply(lambda r: yld(p_v47, r['tR_s'], r['T_C']), axis=1)
    df1['resid_v60'] = df1['val'] - df1['pred_v60']
    df1['resid_v47'] = df1['val'] - df1['pred_v47']

    mae_v60_1 = df1['resid_v60'].abs().mean()
    mae_v47_1 = df1['resid_v47'].abs().mean()
    print(f"\n  v6.0 MAE = {mae_v60_1:.2f} pp,  mean residual = {df1['resid_v60'].mean():+.2f} pp")
    print(f"  v4.7 MAE = {mae_v47_1:.2f} pp,  mean residual = {df1['resid_v47'].mean():+.2f} pp")

    # ============================================================
    # SUBSTRATE 2: 5-Br-2-F-CN → m-CN-p-F-PhLi (C2, CN)
    # ============================================================
    print("\n" + "=" * 80)
    print("Blind validation 2: 5-bromo-2-fluorobenzonitrile + n-BuLi + MeOH")
    print("class = C2 (remote-EWG), ewg_type = CN")
    print("=" * 80)

    p_C2 = get_anchor(anchors, 'C2', 'CN')
    p_analog = dict(Ea_f=34.69, lnA_f=22.11, Ea_d=27.3, lnA_d=9.08, y_max=83.1)
    p_v46_disaster = dict(Ea_f=43.23, lnA_f=21.83, Ea_d=-48.15, lnA_d=19.76, y_max=83.1)

    print(f"\nv6.0 (C2/CN, n={p_C2['n']}):")
    print(f"  Ea_d={p_C2['Ea_d']:.1f}±{p_C2['Ea_d_sd']:.1f}  "
          f"lnA_d={p_C2['lnA_d']:.1f}±{p_C2['lnA_d_sd']:.1f}  "
          f"y_max={p_C2['y_max']:.0f}")
    print(f"3-CN-PhLi analog: Ea_d=27.3, lnA_d=9.08, y_max=83.1")
    print(f"v4.6 disaster:    Ea_d=-48.15 (NEGATIVE), lnA_d=19.76, y_max=83.1")

    df2 = pd.read_csv(BASE / 'experiment_fbrcn_summary.csv')
    df2['pred_v60'] = df2.apply(lambda r: yld(p_C2, r['tR_s'], r['T_C']), axis=1)
    df2['pred_analog'] = df2.apply(lambda r: yld(p_analog, r['tR_s'], r['T_C']), axis=1)
    df2['pred_v46'] = df2.apply(lambda r: yld(p_v46_disaster, r['tR_s'], r['T_C']), axis=1)
    df2['resid_v60'] = df2['yield_pct'] - df2['pred_v60']
    df2['resid_analog'] = df2['yield_pct'] - df2['pred_analog']
    df2['resid_v46'] = df2['yield_pct'] - df2['pred_v46']

    mae_v60_2 = df2['resid_v60'].abs().mean()
    mae_analog = df2['resid_analog'].abs().mean()
    mae_v46 = df2['resid_v46'].abs().mean()
    print(f"\n  v6.0 MAE = {mae_v60_2:.2f} pp,  mean residual = {df2['resid_v60'].mean():+.2f} pp")
    print(f"  3-CN-PhLi analog MAE = {mae_analog:.2f} pp,  mean residual = {df2['resid_analog'].mean():+.2f} pp")
    print(f"  v4.6 disaster MAE = {mae_v46:.2f} pp")

    # Save predictions
    df1.to_csv(BASE / 'v60_4BrFC6H4_predictions.csv', index=False)
    df2.to_csv(BASE / 'v60_5Br2F_CN_predictions.csv', index=False)

    # ============================================================
    # Summary table
    # ============================================================
    print("\n" + "=" * 80)
    print("BLIND VALIDATION SUMMARY")
    print("=" * 80)
    print(f"\n{'Substrate':<20}  {'Class':<10}  {'Model':<25}  {'MAE (pp)':>10}  {'Bias':>8}")
    print("-" * 80)
    print(f"{'4-Br-FC6H4':<20}  {'C1/none':<10}  {'v6.0 anchor':<25}  {mae_v60_1:>10.2f}  {df1['resid_v60'].mean():>+8.2f}")
    print(f"{'(p-F-PhLi)':<20}  {'(inert)':<10}  {'v4.7 lit anchor':<25}  {mae_v47_1:>10.2f}  {df1['resid_v47'].mean():>+8.2f}")
    print()
    print(f"{'5-Br-2-F-CN':<20}  {'C2/CN':<10}  {'v6.0 anchor':<25}  {mae_v60_2:>10.2f}  {df2['resid_v60'].mean():>+8.2f}")
    print(f"{'(m-CN-p-F-PhLi)':<20}  {'(reactive)':<10}  {'3-CN-PhLi analog':<25}  {mae_analog:>10.2f}  {df2['resid_analog'].mean():>+8.2f}")
    print(f"{'':<20}  {'':<10}  {'v4.6 disaster':<25}  {mae_v46:>10.2f}  (Ea_d=-48 invalid)")

    # ============================================================
    # Figures (3 panels for each substrate: exp / v6.0 / comparison anchor)
    # ============================================================
    plt.rcParams["font.family"] = 'Helvetica'
    plt.rcParams["font.size"] = 11

    def wada_subplot(ax, df, x_col, y_col, z_col, vmin, vmax, title,
                     annotate_col=None, fmt='{:.0f}', cmap='bwr'):
        df = df.dropna(subset=[x_col, y_col, z_col]).copy()
        x_data = np.log10(df[x_col].to_numpy()); y_data = df[y_col].to_numpy(); z_data = df[z_col].to_numpy()
        xi = np.linspace(x_data.min()-0.05, x_data.max()+0.05, 400)
        yi = np.linspace(y_data.min()-3, y_data.max()+3, 400)
        XI, YI = np.meshgrid(xi, yi)
        sx, sy = np.std(x_data) or 1, np.std(y_data) or 1
        rbf = Rbf(x_data/sx, y_data/sy, z_data, function='linear', smooth=0.3)
        ZI = np.clip(rbf(XI/sx, YI/sy), 0, 110)
        cf = ax.contourf(XI, YI, ZI, levels=np.linspace(vmin, vmax, 21), cmap=cmap,
                          norm=Normalize(vmin=vmin, vmax=vmax), extend='both')
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, n: f"$10^{{{v:.1f}}}$"))
        ax.scatter(x_data, y_data, s=30, facecolor='white', edgecolors='black',
                   zorder=5, clip_on=False)
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

    # === Figure 1: 4-Br-FC6H4 ===
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
    wada_subplot(axes[0], df1, 'tR_s', 'T_C', 'val', 0, 100,
                 '(a) Experiment — 4-Br-FC6H4 + n-BuLi + PhCHO (corrected yield)')
    wada_subplot(axes[1], df1, 'tR_s', 'T_C', 'pred_v60', 0, 100,
                 f'(b) v6.0 prediction (C1/none anchor)\n'
                 f'Ea_d={p_C1["Ea_d"]:.0f}±{p_C1["Ea_d_sd"]:.0f}, lnA_d={p_C1["lnA_d"]:.1f}, '
                 f'y_max={p_C1["y_max"]:.0f}  |  MAE={mae_v60_1:.1f} pp')
    wada_subplot(axes[2], df1, 'tR_s', 'T_C', 'resid_v60', -30, 30,
                 f'(c) Residual (exp − v6.0)  mean={df1["resid_v60"].mean():+.1f}',
                 fmt='{:+.0f}', cmap='RdBu_r')
    plt.suptitle("v6.0 blind validation on 4-Br-FC6H4 (C1/none, inert)",
                 fontsize=13, fontweight='bold', y=1.04)
    out1 = BASE / 'analysis_figures' / 'v60_blind_4BrFC6H4.png'
    plt.savefig(out1, dpi=140, bbox_inches='tight')
    print(f"\nSaved: {out1}")

    # === Figure 2: 5-Br-2-F-CN ===
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
    wada_subplot(axes[0], df2, 'tR_s', 'T_C', 'yield_pct', 0, 100,
                 '(a) Experiment — 5-Br-2-F-CN + n-BuLi + MeOH (25 points, calibrated)')
    wada_subplot(axes[1], df2, 'tR_s', 'T_C', 'pred_v60', 0, 100,
                 f'(b) v6.0 prediction (C2/CN anchor)\n'
                 f'Ea_d={p_C2["Ea_d"]:.0f}±{p_C2["Ea_d_sd"]:.0f}, lnA_d={p_C2["lnA_d"]:.1f}, '
                 f'y_max={p_C2["y_max"]:.0f}  |  MAE={mae_v60_2:.1f} pp')
    wada_subplot(axes[2], df2, 'tR_s', 'T_C', 'resid_v60', -30, 30,
                 f'(c) Residual (exp − v6.0)  mean={df2["resid_v60"].mean():+.1f}',
                 fmt='{:+.0f}', cmap='RdBu_r')
    plt.suptitle("v6.0 blind validation on 5-Br-2-F-CN (C2/CN, m-CN-attack)",
                 fontsize=13, fontweight='bold', y=1.04)
    out2 = BASE / 'analysis_figures' / 'v60_blind_5Br2F_CN.png'
    plt.savefig(out2, dpi=140, bbox_inches='tight')
    print(f"Saved: {out2}")

    # === Verification ===
    print("\n" + "=" * 80)
    print("Verification (Phase 5 acceptance criteria)")
    print("=" * 80)
    n_neg_anchors = (anchors[anchors['use_for_pred']]['Ea_d_mean'] < 0).sum() + \
                    (anchors[anchors['use_for_pred']]['Ea_f_mean'] < 0).sum()
    print(f"  1. Negative Ea in sub-anchors: {n_neg_anchors} (must be 0) — {'PASS' if n_neg_anchors==0 else 'FAIL'}")
    print(f"  2. 5-Br-2-F-CN Ea_d ∈ [20, 40]: {p_C2['Ea_d']:.1f} — {'PASS' if 20<=p_C2['Ea_d']<=40 else 'FAIL'}")
    print(f"  3. 4-Br-FC6H4 Ea_d ≈ 78: {p_C1['Ea_d']:.1f} — {'PASS' if 70<=p_C1['Ea_d']<=85 else 'FAIL'}")
    print(f"  4. 4-Br-FC6H4 yield MAE < 15 pp: {mae_v60_1:.1f} — {'PASS' if mae_v60_1<15 else 'FAIL'}")
    print(f"  5. 5-Br-2-F-CN yield MAE < 15 pp: {mae_v60_2:.1f} — {'PASS' if mae_v60_2<15 else 'FAIL'}")


if __name__ == '__main__':
    main()
