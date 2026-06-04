"""
Exhaustive descriptor screening for v6.x class-specific Arrhenius models.

Loads the FULL descriptor pool from:
  - intermediates_master.csv (xTB monomer + Hammett/Taft/δ)
  - aggregation_descriptors.csv (xTB dimer thermodynamics)
  - v60_training_set.csv adds σ_p_sum, σ_m_sum from new classifier
  - fbrcn_descriptors.csv (5-Br-2-F-CN test point computed earlier)

Pool: 26 numeric descriptors covering Electronic, Steric, Bonding, Solvation,
Aggregation, Hammett/Taft categories.

For each (class, target) where n >= 5:
  - Exhaustive k=2 and k=3 OLS + LOO search
  - Physical filter: predicted Ea > 5 kJ/mol on ALL training + on 5-Br-2-F-CN test
  - Within physically-valid combos, rank by LOO R²
  - Report top 10 per (class, target)

Goal: see if descriptors beyond σ + EWG_type can improve C2 prediction.
"""
import sys, itertools
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.linear_model import LinearRegression

BASE = Path(__file__).parent

# ============================================================
# Descriptor pool (26 features)
# ============================================================
ELECTRONIC = ['dft_charge_Li', 'dft_charge_C_ipso', 'dft_HOMO_eV', 'dft_LUMO_eV',
              'HOMO_LUMO_gap_eV', 'fukui_f_minus_C']
STERIC     = ['sterimol_B1', 'sterimol_B5', 'sterimol_L', 'buried_vol_Li', 'mol_volume']
BONDING    = ['dft_LiC_bond_A', 'dft_LiC_BDE_kJ']
SOLVATION  = ['dft_Gsolv_kJ', 'dft_dipole_D']
AGGREG     = ['dE_dim_kJ', 'dH_dim_kJ', 'dG_dim_kJ', 'dS_dim_J_K', 'dVbur']
EMPIRICAL  = ['sigma_p_sum', 'sigma_m_sum', 'Es_taft', 'delta_ortho', 'delta_benzyne']

ALL_DESC = ELECTRONIC + STERIC + BONDING + SOLVATION + AGGREG + EMPIRICAL
print(f"Descriptor pool: {len(ALL_DESC)} total")
print(f"  Electronic: {len(ELECTRONIC)}  Steric: {len(STERIC)}  Bonding: {len(BONDING)}")
print(f"  Solvation: {len(SOLVATION)}  Aggregation: {len(AGGREG)}  Empirical: {len(EMPIRICAL)}")


def canon_smi(s):
    try:
        m = Chem.MolFromSmiles(str(s))
        if m is None: return None
        return Chem.MolToSmiles(m, canonical=True)
    except Exception:
        return None


def load_features():
    """Merge all descriptor sources keyed by canonical SMILES."""
    # 1) v60 training set
    df = pd.read_csv(BASE / 'v60_training_set.csv')
    df = df[~df['is_virtual']].copy()  # only real data (virtuals lack descriptors)
    df['smi_can'] = df['smi'].apply(canon_smi)

    # 2) intermediates_master — has xTB monomer + Hammett
    master = pd.read_csv(BASE / 'intermediates_master.csv')
    master['smi_can'] = master['intermediate_smiles_canonical'].apply(canon_smi)
    master_cols = ['smi_can'] + ELECTRONIC + STERIC + BONDING + SOLVATION + \
                  ['Es_taft', 'delta_ortho', 'delta_benzyne']
    master = master[master_cols].drop_duplicates('smi_can')

    # 3) aggregation_descriptors — xTB dimer
    agg = pd.read_csv(BASE / 'aggregation_descriptors.csv')
    agg['smi_can'] = agg['smi'].apply(canon_smi)
    agg_cols = ['smi_can'] + AGGREG
    agg = agg[agg_cols].drop_duplicates('smi_can')

    # Merge
    df = df.merge(master, on='smi_can', how='left', suffixes=('', '_m'))
    df = df.merge(agg, on='smi_can', how='left', suffixes=('', '_a'))

    print(f"\nMerged training set: n={len(df)}")
    print(f"  with class: {df['class_v60'].value_counts().to_dict()}")
    # Drop rows with any missing descriptor (for screen)
    nn_count = df[ALL_DESC].notna().all(axis=1).sum()
    print(f"  with complete descriptors: {nn_count}/{len(df)}")
    return df


def load_test_point():
    """Load 5-Br-2-F-CN computed descriptors (from compute_fbrcn_descriptors.py)."""
    f = pd.read_csv(BASE / 'fbrcn_descriptors.csv').iloc[0]
    # Map to our descriptor names
    return {
        # xTB monomer
        'dft_LUMO_eV': f['LUMO_eV'],
        'dft_LiC_bond_A': f['d_LiC_A'],
        'mol_volume': f['vol_A3'],
        'dft_Gsolv_kJ': f['Gsolv_kJ'],
        'dft_LiC_BDE_kJ': f['BDE_kJ'],
        # Aggregation
        'dE_dim_kJ': f['dE_dim_kJ'],
        'dH_dim_kJ': f['dH_dim_kJ'],
        'dG_dim_kJ': f['dG_dim_kJ'],
        'dS_dim_J_K': f['dS_dim_J'],
        'dVbur': f['dVbur_dim'],
        'Vbur_mono': f['Vbur_mono'],
        # Hammett (computed by classifier)
        'sigma_p_sum': 0.06,    # F at para
        'sigma_m_sum': 0.56,    # CN at meta
        'Es_taft': 0.0,
        'delta_ortho': 0.0,
        'delta_benzyne': 0.0,
        # Missing from fbrcn_descriptors.csv (not computed for new substrate):
        # dft_charge_Li, dft_charge_C_ipso, dft_HOMO_eV, HOMO_LUMO_gap_eV, fukui_f_minus_C,
        # sterimol_B1/B5/L, buried_vol_Li, dft_dipole_D
        # → These combos will be skipped for test prediction
    }


def loo_r2(X, y):
    """LOO cross-validated R² via OLS."""
    n = len(y)
    if n < X.shape[1] + 1: return np.nan, np.array([])
    preds = np.empty(n)
    for i in range(n):
        Xi = np.delete(X, i, 0); yi = np.delete(y, i)
        m = LinearRegression().fit(Xi, yi)
        preds[i] = m.predict(X[i:i+1])[0]
    rss = np.sum((y - preds)**2); tss = np.sum((y - y.mean())**2)
    r2 = 1 - rss/tss if tss > 0 else np.nan
    return r2, preds


def screen(df, class_label, targets, test_point=None, min_n=5):
    """For one mechanism class, exhaustively screen descriptor combos."""
    sub = df[df['class_v60'] == class_label].copy()
    sub = sub.dropna(subset=ALL_DESC).copy()
    n = len(sub)
    print(f"\n{'='*80}\n{class_label}: n={n} (complete descriptors)")
    if n < min_n:
        print(f"  → too few samples (<{min_n}), skip")
        return None
    print('='*80)

    all_results = []
    for target in targets:
        y = sub[target].dropna().to_numpy()
        rows = sub[sub[target].notna()].copy()
        if len(y) < min_n:
            print(f"  {target}: n_obs={len(y)} < {min_n}, skip")
            continue

        # Single descriptor
        best_k1 = []
        for d in ALL_DESC:
            X = rows[[d]].to_numpy()
            r2, _ = loo_r2(X, y)
            if np.isnan(r2): continue
            m = LinearRegression().fit(X, y)
            best_k1.append({'k': 1, 'descs': (d,), 'R2_LOO': r2})
        # k=2 combos
        best_k2 = []
        for combo in itertools.combinations(ALL_DESC, 2):
            X = rows[list(combo)].to_numpy()
            r2, _ = loo_r2(X, y)
            if np.isnan(r2): continue
            best_k2.append({'k': 2, 'descs': combo, 'R2_LOO': r2})
        # k=3 combos
        best_k3 = []
        for combo in itertools.combinations(ALL_DESC, 3):
            X = rows[list(combo)].to_numpy()
            r2, _ = loo_r2(X, y)
            if np.isnan(r2): continue
            best_k3.append({'k': 3, 'descs': combo, 'R2_LOO': r2})

        # Sort by LOO R²
        for lst in [best_k1, best_k2, best_k3]:
            lst.sort(key=lambda x: -x['R2_LOO'])

        # Apply physical test predict filter (if test_point available)
        def predict_test(descs):
            if test_point is None: return None
            # Check all descriptors present in test point
            if not all(d in test_point for d in descs): return None
            X_tr = rows[list(descs)].to_numpy()
            m = LinearRegression().fit(X_tr, y)
            X_test = np.array([[test_point[d] for d in descs]])
            return float(m.predict(X_test)[0])

        print(f"\n  --- {target} (n_obs={len(y)}) ---")
        for label, lst in [('k=1', best_k1[:5]), ('k=2', best_k2[:10]), ('k=3', best_k3[:10])]:
            print(f"    Top {label}:")
            for r in lst:
                test_pred = predict_test(r['descs'])
                test_pred_str = f"  → test pred = {test_pred:+7.2f}" if test_pred is not None else "  (test: missing desc)"
                # Flag negative Ea
                flag = ""
                if target.startswith('Ea') and test_pred is not None and test_pred < 5:
                    flag = "  ⚠ NEGATIVE"
                desc_str = " + ".join(r['descs'])
                print(f"      LOO R²={r['R2_LOO']:+.3f}  {desc_str}{test_pred_str}{flag}")
                all_results.append({
                    'class': class_label, 'target': target, 'k': r['k'],
                    'descriptors': '+'.join(r['descs']),
                    'R2_LOO': round(r['R2_LOO'], 3),
                    'test_pred': round(test_pred, 2) if test_pred is not None else None,
                })
    return all_results


def main():
    df = load_features()
    test = load_test_point()

    print("\n=== 5-Br-2-F-CN test point descriptors ===")
    for k, v in test.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    targets = ['Ea_d', 'lnA_d', 'Ea_f', 'lnA_f', 'y_max']

    all_results = []
    for cls in ['C1', 'C2', 'C3', 'OXI']:
        res = screen(df, cls, targets, test_point=test if cls == 'C2' else None, min_n=5)
        if res:
            all_results.extend(res)

    out = pd.DataFrame(all_results)
    out_csv = BASE / 'v6_descriptor_screen_results.csv'
    out.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}  ({len(out)} entries)")

    # ---- Summary: best per (class, target) ----
    print("\n" + "=" * 80)
    print("SUMMARY: Best descriptor combo per (class, target) with LOO R² > 0.5")
    print("=" * 80)
    if len(out) > 0:
        for (cls, tgt), grp in out.groupby(['class', 'target']):
            best = grp.loc[grp['R2_LOO'].idxmax()]
            if best['R2_LOO'] > 0.5:
                test_str = f"  test_pred={best['test_pred']}" if best['test_pred'] is not None else ""
                print(f"  {cls} | {tgt}: LOO R²={best['R2_LOO']:.3f}  k={best['k']}  "
                      f"[{best['descriptors']}]{test_str}")


if __name__ == '__main__':
    main()
