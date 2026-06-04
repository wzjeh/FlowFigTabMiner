"""
Phase 3: Validate v5.0 vs v4.7 on:
1. Per-substrate parity plot (training set, 24 reactive ArLi)
2. Blind test on 4-F-PhLi (Case A/B) and 3,5-Br2-CN (Case C/D)
3. Specific m-ArLi formation Hammett finding (the v5.0 win)

v5.0 model recipe:
  - Inert ArLi: lit anchor (Ea_f=32, lnA_f=21, Ea_d=78, lnA_d=22, y_max=100)
  - Reactive m-ArLi formation: Hammett correction
        Ea_f = 39.54 - 15.64·σ_m   (R²_LOO 0.47)
        lnA_f = 37.23 - 25.95·σ_m  (R²_LOO 0.87)
  - All other reactive (p/o-ArLi formation + all decay + all y_max): v4.6 公式 (unchanged)
"""
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).parent

# Load data
df = pd.read_csv(BASE / 'v50_training_set.csv')
v46 = pd.read_csv(BASE / 'analysis_figures' / 'class_fitted_models_v46.csv')

DESC_COLS = {
    'q_C':'q_C','q_Li':'q_Li','HOMO':'HOMO','LUMO':'LUMO','eta':'eta','fukui':'fukui',
    'd_LiC':'d_LiC','BDE':'BDE','dipole':'dipole','Gsolv':'Gsolv',
    'B1':'B1','B5':'B5','L':'L','pVbur':'pVbur','vol':'vol','Es':'Es','sigma':'sigma',
    'dE_dim':'dE_dim','dH_dim':'dH_dim','dG_dim':'dG_dim','dS_dim':'dS_dim','dVbur_dim':'dVbur_dim',
}

# Merge needed v4.6 descriptors from intermediates_master (only relevant cols)
m_full = pd.read_csv(BASE / 'intermediates_master.csv').rename(columns={'intermediate_smiles_canonical':'smi'})
m_cols = ['smi','sigma_hammett','dft_dipole_D','dft_Gsolv_kJ','sterimol_B1','sterimol_B5','sterimol_L',
          'mol_volume','fukui_f_minus_C','dft_LiC_bond_A','dft_HOMO_eV','dft_LUMO_eV',
          'HOMO_LUMO_gap_eV','dft_charge_Li','dft_charge_C_ipso','Es_taft','dft_LiC_BDE_kJ','buried_vol_Li']
m = m_full[[c for c in m_cols if c in m_full.columns]].drop_duplicates(subset='smi')
df = df.drop(columns=['Es'], errors='ignore')   # drop pre-existing Es from training set (substituent-level)
df = df.merge(m, on='smi', how='left')
df = df.rename(columns={'sigma_hammett':'sigma','dft_dipole_D':'dipole','dft_Gsolv_kJ':'Gsolv',
    'sterimol_B1':'B1','sterimol_B5':'B5','sterimol_L':'L','mol_volume':'vol',
    'fukui_f_minus_C':'fukui',
    'dft_LiC_bond_A':'d_LiC','dft_HOMO_eV':'HOMO','dft_LUMO_eV':'LUMO',
    'HOMO_LUMO_gap_eV':'eta','dft_charge_Li':'q_Li','dft_charge_C_ipso':'q_C',
    'Es_taft':'Es','dft_LiC_BDE_kJ':'BDE','buried_vol_Li':'pVbur'})
df = df.drop(columns=['dG_dim'], errors='ignore')  # avoid duplicate
agg_full = pd.read_csv(BASE / 'aggregation_descriptors.csv')
agg = agg_full[['smi','dE_dim_kJ','dH_dim_kJ','dG_dim_kJ','dS_dim_J_K']].drop_duplicates(subset='smi')
df = df.merge(agg, on='smi', how='left')
df = df.rename(columns={'dE_dim_kJ':'dE_dim','dH_dim_kJ':'dH_dim','dG_dim_kJ':'dG_dim','dS_dim_J_K':'dS_dim'})


def v46_predict(cls, param, row):
    f = v46[(v46['class']==cls) & (v46['param']==param)]
    if len(f)==0: return None
    f = f.iloc[0]
    descs = [d.strip() for d in f['descriptors'].split('+')]
    coefs = [float(c.strip()) for c in f['coefficients'].split(',')]
    b = float(f['intercept'])
    val = b
    for d, c in zip(descs, coefs):
        col = DESC_COLS.get(d, d)
        if col not in row.index or pd.isna(row[col]): return None
        val += c * row[col]
    return val


REACTIVE_PATTERNS = ['C#N','[N+](=O)[O-]','C(=O)']
HALIDE_REACTIVE = ['Br','I']
def has_reactive_R(smi):
    s = str(smi).replace('[Li]','')
    return any(p in s for p in REACTIVE_PATTERNS) or any(h in s for h in HALIDE_REACTIVE)

INERT_ANCHOR = {'Ea_f':32.0, 'lnA_f':21.0, 'Ea_d':78.0, 'lnA_d':22.0, 'y_max':100.0}

def v47_predict(row):
    """v4.7: v4.6 + inert override."""
    cls = row['cls']
    pred = {}
    if cls == 'oxiranylLi':
        for p in ['Ea_f','lnA_f','Ea_d','lnA_d','y_max']:
            pred[p] = v46_predict('oxiranylLi', p, row)
        return pred
    if not has_reactive_R(row['smi']):
        return INERT_ANCHOR.copy()
    # Reactive p/m/o
    for p in ['Ea_f','lnA_f','Ea_d','lnA_d','y_max']:
        pred[p] = v46_predict(cls, p, row)
    return pred


def v50_predict(row):
    """v5.0: v4.7 + m-ArLi formation Hammett correction."""
    cls = row['cls']
    pred = v47_predict(row).copy()
    # Apply m-ArLi formation Hammett correction for reactive m-substrates
    if cls == 'm-ArLi' and has_reactive_R(row['smi']):
        sigma_m = row.get('sigma_m_sum', 0)
        if pd.notna(sigma_m):
            pred['Ea_f'] = 39.54 - 15.64 * sigma_m         # R²_LOO 0.47
            pred['lnA_f'] = 37.23 - 25.95 * sigma_m        # R²_LOO 0.87
    return pred


# Compute predictions for all training set
v47_rows = []
v50_rows = []
for _, r in df.iterrows():
    v47 = v47_predict(r)
    v50 = v50_predict(r)
    for params, label, store in [(v47, 'v4.7', v47_rows), (v50, 'v5.0', v50_rows)]:
        store.append({
            'smi': r['smi'], 'cls': r['cls'],
            **{f'{p}_exp': r[p] for p in ['Ea_f','lnA_f','Ea_d','lnA_d','y_max']},
            **{f'{p}_pred': params.get(p, np.nan) for p in ['Ea_f','lnA_f','Ea_d','lnA_d','y_max']},
        })

v47_df = pd.DataFrame(v47_rows)
v50_df = pd.DataFrame(v50_rows)

# Per-param MAE comparison
print("="*88)
print("v5.0 vs v4.7 — per-parameter MAE on training set (n=24)")
print("="*88)
print(f"\n  {'param':<10}{'v4.7 MAE':>12}{'v5.0 MAE':>12}{'Δ (v5.0-v4.7)':>16}")
for p in ['Ea_f','lnA_f','Ea_d','lnA_d','y_max']:
    err47 = (v47_df[f'{p}_exp'] - v47_df[f'{p}_pred']).dropna()
    err50 = (v50_df[f'{p}_exp'] - v50_df[f'{p}_pred']).dropna()
    mae47 = err47.abs().mean()
    mae50 = err50.abs().mean()
    d = mae50 - mae47
    arrow = "↓ better" if d < -0.5 else ("↑ worse" if d > 0.5 else "≈ same")
    print(f"  {p:<10}{mae47:>12.2f}{mae50:>12.2f}{d:>+16.2f}  {arrow}")

# Specifically: m-ArLi formation predictions
print("\n" + "="*88)
print("m-ArLi formation (Ea_f, lnA_f) — where v5.0 Hammett helps")
print("="*88)
m_df = df[df['cls']=='m-ArLi']
print(f"\n  {'substrate':<32}{'σ_m':>6}{'Ea_f exp':>10}{'v4.6 Ea_f':>11}{'v5.0 Ea_f':>11}{'lnA_f exp':>11}{'v4.6 lnA_f':>12}{'v5.0 lnA_f':>12}")
for _, r in m_df.iterrows():
    if not has_reactive_R(r['smi']): continue
    v47 = v47_predict(r); v50 = v50_predict(r)
    def fmt(v):
        return f"{v:.2f}" if v is not None and not pd.isna(v) else "—"
    print(f"  {r['smi'][:30]:<32}{r['sigma_m_sum']:>6.2f}"
          f"{r['Ea_f']:>10.2f}{fmt(v47.get('Ea_f')):>11}{fmt(v50.get('Ea_f')):>11}"
          f"{r['lnA_f']:>11.2f}{fmt(v47.get('lnA_f')):>12}{fmt(v50.get('lnA_f')):>12}")

# Blind test on 4-F-PhLi
print("\n" + "="*88)
print("Blind test: 4-F-PhLi (Case A/B, inert)")
print("="*88)
F_row = m[m['smi']=='[Li]c1ccc(F)cc1'].iloc[0]
F_row['cls'] = 'p-ArLi'
F_row['sigma_p_sum'] = 0.06; F_row['sigma_m_sum'] = 0; F_row['sigma_o_sum'] = 0
v47_F = v47_predict(F_row)
v50_F = v50_predict(F_row)
print(f"  v4.7: Ea_f={v47_F['Ea_f']:.1f}, lnA_f={v47_F['lnA_f']:.1f}, Ea_d={v47_F['Ea_d']:.1f}, lnA_d={v47_F['lnA_d']:.1f}, y_max={v47_F['y_max']:.1f}")
print(f"  v5.0: Ea_f={v50_F['Ea_f']:.1f}, lnA_f={v50_F['lnA_f']:.1f}, Ea_d={v50_F['Ea_d']:.1f}, lnA_d={v50_F['lnA_d']:.1f}, y_max={v50_F['y_max']:.1f}")
print(f"  → No difference (4-F-PhLi is inert, both use v4.7 anchor)")

# Blind test on 3,5-Br2-CN
print("\n" + "="*88)
print("Blind test: 3,5-Br2-CN-ArLi (Case C/D, m-reactive, dual-EWG)")
print("="*88)
br_pred = pd.read_csv(BASE / '35brcn_v46_prediction.csv').iloc[0]
# Build descriptor row for 3,5-Br2-CN
row_35 = pd.Series({
    'smi': '[Li]c1cc(Br)cc(C#N)c1',
    'cls': 'm-ArLi',
    'sigma_p_sum': 0.0, 'sigma_m_sum': 0.95, 'sigma_o_sum': 0.0,  # CN σ_m=0.56 + Br σ_m=0.39
    'Es_ortho': 0, 'delta_5exo_o': 0, 'delta_chelation_o': 0, 'delta_benzyne_o': 0,
    'dVbur_dim': 0.3059, 'n_EWG': 2,
    # v4.6 descriptors (proxy from earlier compute)
    'd_LiC': 1.9105, 'B5': 3.2404, 'pVbur': 0.2303,
    'BDE': 419.5, 'vol': 136.34, 'LUMO': -6.23, 'Gsolv': -71.69,
    'q_C': 0, 'q_Li': 0.42, 'HOMO': -8, 'eta': 1.8, 'fukui': -0.13,
    'dipole': 8, 'B1': 1.8, 'L': 4.1, 'Es': 0,
    'dE_dim': -103.67, 'dH_dim': -100.94, 'dG_dim': -38.75, 'dS_dim': -208.61,
})
v47_35 = v47_predict(row_35)
v50_35 = v50_predict(row_35)
case_c_prime = dict(Ea_f=32.81, lnA_f=34.57, Ea_d=10.42, lnA_d=2.97, y_max=90.48)
print(f"  v4.7:  Ea_f={v47_35.get('Ea_f',np.nan):.1f}, lnA_f={v47_35.get('lnA_f',np.nan):.1f}, "
      f"Ea_d={v47_35.get('Ea_d',np.nan):.1f}, lnA_d={v47_35.get('lnA_d',np.nan):.1f}")
print(f"  v5.0:  Ea_f={v50_35.get('Ea_f',np.nan):.1f}, lnA_f={v50_35.get('lnA_f',np.nan):.1f}, "
      f"Ea_d={v50_35.get('Ea_d',np.nan):.1f}, lnA_d={v50_35.get('lnA_d',np.nan):.1f}")
print(f"  Case C' fit (experimental): Ea_f=32.81, lnA_f=34.57, Ea_d=10.42, lnA_d=2.97")
print(f"\n  v5.0 m-ArLi Hammett correction effect:")
print(f"    Ea_f: v4.7 → {v47_35['Ea_f']:.1f}, v5.0 → {v50_35['Ea_f']:.1f}, exp ≈ 32.8")
print(f"    lnA_f: v4.7 → {v47_35['lnA_f']:.1f}, v5.0 → {v50_35['lnA_f']:.1f}, exp ≈ 34.6")

# Save predictions
v50_df.to_csv(BASE / 'v50_predictions.csv', index=False)

# Generate parity plot
fig, axes = plt.subplots(1, 5, figsize=(22, 5))
for i, p in enumerate(['Ea_f','lnA_f','Ea_d','lnA_d','y_max']):
    ax = axes[i]
    x_v47 = v47_df[f'{p}_exp']; y_v47 = v47_df[f'{p}_pred']
    x_v50 = v50_df[f'{p}_exp']; y_v50 = v50_df[f'{p}_pred']
    ax.scatter(x_v47, y_v47, c='red', alpha=0.5, label='v4.7', s=60, edgecolor='darkred')
    ax.scatter(x_v50, y_v50, c='green', alpha=0.5, label='v5.0', s=60, edgecolor='darkgreen', marker='^')
    # 1:1 line
    valid = v47_df.dropna(subset=[f'{p}_exp', f'{p}_pred'])
    if len(valid):
        lo = min(valid[f'{p}_exp'].min(), valid[f'{p}_pred'].min())
        hi = max(valid[f'{p}_exp'].max(), valid[f'{p}_pred'].max())
        ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.5)
    ax.set_xlabel(f'{p} experimental')
    ax.set_ylabel(f'{p} predicted')
    ax.set_title(p)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.suptitle('v5.0 (with m-ArLi formation Hammett) vs v4.7 — parity plot on n=24 reactive ArLi training', fontweight='bold')
plt.tight_layout()
plt.savefig(BASE / 'analysis_figures' / 'v50_vs_v47_comparison.png', dpi=140, bbox_inches='tight')
print(f"\nSaved: analysis_figures/v50_vs_v47_comparison.png")
