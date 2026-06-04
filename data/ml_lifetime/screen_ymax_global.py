"""
Exhaustive y_max screen across ALL 41 in-house substrates (no mechanism class split).

Hypothesis (Zhao 2026-05-24): y_max is a global molecular property (quench efficiency +
side reaction loss + aggregation state), not strictly mechanism-bound. Test if a
descriptor-only model gives better LOO R² than the per-class empirical mean.

Exhaustive: k=1, k=2, k=3 over 25-feature pool. Physical filter: predicted y_max ∈ [30, 100].
"""
import itertools
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.linear_model import LinearRegression

BASE = Path(__file__).parent

DESC_POOL = [
    'dft_charge_Li', 'dft_charge_C_ipso', 'dft_HOMO_eV', 'dft_LUMO_eV',
    'HOMO_LUMO_gap_eV', 'fukui_f_minus_C',
    'sterimol_B1', 'sterimol_B5', 'sterimol_L', 'buried_vol_Li', 'mol_volume',
    'dft_LiC_bond_A', 'dft_LiC_BDE_kJ',
    'dft_Gsolv_kJ', 'dft_dipole_D',
    'dE_dim_kJ', 'dH_dim_kJ', 'dG_dim_kJ', 'dS_dim_J_K', 'dVbur',
]

# 5-Br-2-F-CN test point descriptors
TEST_5Br2FCN = {
    'dft_LUMO_eV': -7.0793, 'dft_LiC_bond_A': 1.9141,
    'mol_volume': 139.33, 'dft_Gsolv_kJ': -81.19, 'dft_LiC_BDE_kJ': 434.41,
    'dE_dim_kJ': -82.58, 'dH_dim_kJ': -70.38, 'dG_dim_kJ': -19.92,
    'dS_dim_J_K': -169.23, 'dVbur': 0.407,
}


def canon(s):
    try:
        m = Chem.MolFromSmiles(str(s))
        return Chem.MolToSmiles(m, canonical=True) if m else None
    except: return None


def loo_r2(X, y):
    n = len(y)
    if n < X.shape[1] + 2: return np.nan, np.array([])
    preds = np.empty(n)
    for i in range(n):
        m = LinearRegression().fit(np.delete(X, i, 0), np.delete(y, i))
        preds[i] = m.predict(X[i:i+1])[0]
    rss = np.sum((y - preds)**2); tss = np.sum((y - y.mean())**2)
    return (1 - rss/tss if tss > 0 else np.nan), preds


def main():
    # ---- Merge all 41 substrates with descriptors ----
    df = pd.read_csv(BASE / 'global_arrhenius.csv')
    df['smi_can'] = df['smi'].apply(canon)
    master = pd.read_csv(BASE / 'intermediates_master.csv')
    master['smi_can'] = master['intermediate_smiles_canonical'].apply(canon)
    master_keep = ['smi_can'] + [d for d in DESC_POOL if d in master.columns]
    df = df.merge(master[master_keep].drop_duplicates('smi_can'),
                  on='smi_can', how='left')
    agg = pd.read_csv(BASE / 'aggregation_descriptors.csv')
    agg['smi_can'] = agg['smi'].apply(canon)
    agg_keep = ['smi_can'] + [d for d in DESC_POOL if d in agg.columns]
    df = df.merge(agg[agg_keep].drop_duplicates('smi_can'),
                  on='smi_can', how='left')

    print(f"All substrates: n={len(df)}")
    nn = df[DESC_POOL].notna().all(axis=1)
    print(f"  with complete descriptors: {nn.sum()}/{len(df)}")
    df = df[nn].dropna(subset=['y_max']).reset_index(drop=True)
    print(f"  with y_max + descriptors: n={len(df)}")
    print(f"\ny_max range: [{df['y_max'].min():.1f}, {df['y_max'].max():.1f}]  "
          f"mean={df['y_max'].mean():.1f}  std={df['y_max'].std():.1f}")

    # Baseline: just the mean (LOO would just be each held-out's deviation from mean)
    ymax_vals = df['y_max'].to_numpy()
    loo_mean_preds = np.array([np.delete(ymax_vals, i).mean() for i in range(len(ymax_vals))])
    rss_mean = np.sum((ymax_vals - loo_mean_preds)**2)
    tss = np.sum((ymax_vals - ymax_vals.mean())**2)
    r2_baseline_mean = 1 - rss_mean / tss
    print(f"\nBaseline (LOO grand mean): R² = {r2_baseline_mean:+.3f}")

    # ---- Exhaustive k=1, k=2, k=3 ----
    y = ymax_vals
    print("\nExhaustive descriptor screen for y_max (all substrates):")
    all_results = []
    for k in [1, 2, 3]:
        best = []
        for combo in itertools.combinations(DESC_POOL, k):
            X = df[list(combo)].to_numpy()
            r2, preds = loo_r2(X, y)
            if np.isnan(r2): continue
            # Test prediction (if test point has all descriptors)
            test_pred = None
            if all(d in TEST_5Br2FCN for d in combo):
                m = LinearRegression().fit(X, y)
                test_x = np.array([[TEST_5Br2FCN[d] for d in combo]])
                test_pred = float(m.predict(test_x)[0])
            best.append({'k': k, 'descs': combo, 'R2_LOO': r2, 'test_pred': test_pred})
        best.sort(key=lambda x: -x['R2_LOO'])
        print(f"\n--- Top 10 k={k} (out of {len(best)} valid combos) ---")
        for b in best[:10]:
            tp = f" → test pred = {b['test_pred']:6.1f}" if b['test_pred'] is not None else " (test: N/A)"
            phys_flag = ""
            if b['test_pred'] is not None and (b['test_pred'] < 30 or b['test_pred'] > 100):
                phys_flag = "  ⚠ unphysical"
            desc_str = " + ".join(b['descs'])
            print(f"  LOO R²={b['R2_LOO']:+.3f}  {desc_str}{tp}{phys_flag}")
            all_results.append({'k': b['k'], 'descriptors': '+'.join(b['descs']),
                               'R2_LOO': round(b['R2_LOO'], 3),
                               'test_pred_5Br2FCN': round(b['test_pred'], 1) if b['test_pred'] is not None else None})

    # ---- Per-class breakdown of best global model ----
    if all_results:
        top = max(all_results, key=lambda r: r['R2_LOO'])
        if top['R2_LOO'] > 0.3:
            print("\n" + "=" * 80)
            print(f"BEST GLOBAL y_max model: LOO R²={top['R2_LOO']}")
            print(f"  Descriptors: {top['descriptors']}")
            print(f"  5-Br-2-F-CN pred: {top['test_pred_5Br2FCN']}")
            print(f"  vs C2/CN sub-anchor (v6.0/v6.2): 84.1")
            print("=" * 80)

    out = pd.DataFrame(all_results)
    out.to_csv(BASE / 'v6_ymax_global_screen.csv', index=False)
    print(f"\nSaved: v6_ymax_global_screen.csv ({len(out)} entries)")

    # ---- Summary: what's better than baseline? ----
    print("\n" + "=" * 80)
    print(f"Threshold: LOO R² > {r2_baseline_mean + 0.1:.2f}  (baseline + 0.1)")
    print("=" * 80)
    good = out[out['R2_LOO'] > r2_baseline_mean + 0.1].sort_values('R2_LOO', ascending=False)
    if len(good):
        print(good.head(20).to_string(index=False))
    else:
        print("  NO descriptor combo improves over baseline by ≥ 0.1.")


if __name__ == '__main__':
    main()
