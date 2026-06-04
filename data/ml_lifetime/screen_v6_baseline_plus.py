"""
Refined screen: does adding ANY descriptor beyond (σ + EWG_type) baseline improve LOO R² on C2?
Also: within each C2 sub-class (CN/NO2/ester), can descriptors capture residual variation?

Tests the hypothesis: v6.1 σ + EWG_type is already optimal, OR there's additional descriptor signal.
"""
import itertools
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.linear_model import LinearRegression

BASE = Path(__file__).parent

# ============================================================
# Descriptor pool (excl. categorical EWG flags — used directly)
# ============================================================
DESC_POOL = [
    # Electronic
    'dft_charge_Li', 'dft_charge_C_ipso', 'dft_HOMO_eV', 'dft_LUMO_eV',
    'HOMO_LUMO_gap_eV', 'fukui_f_minus_C',
    # Steric
    'sterimol_B1', 'sterimol_B5', 'sterimol_L', 'buried_vol_Li', 'mol_volume',
    # Bonding
    'dft_LiC_bond_A', 'dft_LiC_BDE_kJ',
    # Solvation
    'dft_Gsolv_kJ', 'dft_dipole_D',
    # Aggregation
    'dE_dim_kJ', 'dH_dim_kJ', 'dG_dim_kJ', 'dS_dim_J_K', 'dVbur',
]

# Test point (5-Br-2-F-CN) — descriptors we have
TEST_5Br2FCN = {
    'dft_LUMO_eV': -7.0793, 'dft_LiC_bond_A': 1.9141,
    'mol_volume': 139.33, 'dft_Gsolv_kJ': -81.19, 'dft_LiC_BDE_kJ': 434.41,
    'dE_dim_kJ': -82.58, 'dH_dim_kJ': -70.38, 'dG_dim_kJ': -19.92,
    'dS_dim_J_K': -169.23, 'dVbur': 0.407,
    'sigma_p_sum': 0.06, 'sigma_m_sum': 0.56,
    # EWG_type = CN
    'is_CN': 1, 'is_NO2': 0, 'is_ester': 0,
}


def canon_smi(s):
    try:
        m = Chem.MolFromSmiles(str(s))
        return Chem.MolToSmiles(m, canonical=True) if m else None
    except: return None


def load_features():
    df = pd.read_csv(BASE / 'v60_training_set.csv')
    df = df[~df['is_virtual']].copy()
    df['smi_can'] = df['smi'].apply(canon_smi)
    master = pd.read_csv(BASE / 'intermediates_master.csv')
    master['smi_can'] = master['intermediate_smiles_canonical'].apply(canon_smi)
    master = master[['smi_can'] + DESC_POOL[:-5]].drop_duplicates('smi_can')
    agg = pd.read_csv(BASE / 'aggregation_descriptors.csv')
    agg['smi_can'] = agg['smi'].apply(canon_smi)
    agg = agg[['smi_can'] + DESC_POOL[-5:]].drop_duplicates('smi_can')
    df = df.merge(master, on='smi_can', how='left').merge(agg, on='smi_can', how='left')
    df['is_CN'] = (df['ewg_type'] == 'CN').astype(int)
    df['is_NO2'] = (df['ewg_type'] == 'NO2').astype(int)
    df['is_ester'] = (df['ewg_type'] == 'ester').astype(int)
    df['sigma_eff'] = df['sigma_p_sum'].fillna(0) + df['sigma_m_sum'].fillna(0)
    return df


def loo_r2(X, y):
    n = len(y)
    if n < X.shape[1] + 2: return np.nan
    preds = np.empty(n)
    for i in range(n):
        m = LinearRegression().fit(np.delete(X, i, 0), np.delete(y, i))
        preds[i] = m.predict(X[i:i+1])[0]
    rss = np.sum((y-preds)**2); tss = np.sum((y-y.mean())**2)
    return 1 - rss/tss if tss > 0 else np.nan


def main():
    df = load_features()
    c2 = df[df['class_v60'] == 'C2'].dropna(subset=DESC_POOL).copy()
    print(f"C2 with complete descriptors: n={len(c2)}")
    print(c2['ewg_type'].value_counts().to_string())

    BASELINE_FEAT = ['sigma_eff', 'is_CN', 'is_NO2', 'is_ester']

    # ============================================================
    # PART 1: Baseline (σ + EWG_type) LOO performance
    # ============================================================
    print("\n" + "=" * 80)
    print("PART 1: Baseline (σ_eff + is_CN + is_NO2 + is_ester) LOO performance")
    print("=" * 80)
    baseline = {}
    for target in ['Ea_d', 'lnA_d', 'Ea_f', 'lnA_f', 'y_max']:
        sub = c2.dropna(subset=[target])
        X = sub[BASELINE_FEAT].to_numpy()
        y = sub[target].to_numpy()
        r2 = loo_r2(X, y)
        m = LinearRegression().fit(X, y)
        # Test on 5-Br-2-F-CN
        test_X = np.array([[TEST_5Br2FCN['sigma_eff'] if False else (TEST_5Br2FCN['sigma_p_sum']+TEST_5Br2FCN['sigma_m_sum']),
                            TEST_5Br2FCN['is_CN'], TEST_5Br2FCN['is_NO2'], TEST_5Br2FCN['is_ester']]])
        test_pred = float(m.predict(test_X)[0])
        baseline[target] = {'R2_LOO': r2, 'test_pred': test_pred}
        print(f"  {target}: LOO R²={r2:+.3f}  →  5-Br-2-F-CN pred = {test_pred:+.2f}")

    # ============================================================
    # PART 2: Baseline + 1 descriptor (does ANY add value?)
    # ============================================================
    print("\n" + "=" * 80)
    print("PART 2: Baseline + 1 extra descriptor — does it improve LOO?")
    print("=" * 80)

    results = []
    for target in ['Ea_d', 'lnA_d', 'Ea_f', 'lnA_f', 'y_max']:
        sub = c2.dropna(subset=[target])
        y = sub[target].to_numpy()
        base_r2 = baseline[target]['R2_LOO']
        improvements = []
        for d in DESC_POOL:
            X = sub[BASELINE_FEAT + [d]].to_numpy()
            r2 = loo_r2(X, y)
            if np.isnan(r2): continue
            # Test prediction (only if test has this descriptor)
            test_pred = None
            if d in TEST_5Br2FCN:
                m = LinearRegression().fit(X, y)
                test_x = np.array([[TEST_5Br2FCN['sigma_p_sum']+TEST_5Br2FCN['sigma_m_sum'],
                                    TEST_5Br2FCN['is_CN'], TEST_5Br2FCN['is_NO2'],
                                    TEST_5Br2FCN['is_ester'], TEST_5Br2FCN[d]]])
                test_pred = float(m.predict(test_x)[0])
            improvements.append({
                'desc': d, 'R2_LOO': r2,
                'delta_R2': r2 - base_r2, 'test_pred': test_pred,
            })
        improvements.sort(key=lambda x: -x['R2_LOO'])
        print(f"\n  --- {target}  (baseline LOO R²={base_r2:+.3f}) ---")
        for imp in improvements[:8]:
            delta = imp['delta_R2']
            sign = '↑' if delta > 0 else '↓'
            tp_str = f"  test={imp['test_pred']:+7.2f}" if imp['test_pred'] is not None else "  test=N/A"
            improved = " ⭐" if delta > 0.05 else ""
            print(f"    + {imp['desc']:<22} LOO={imp['R2_LOO']:+.3f}  Δ={delta:+.3f} {sign}{tp_str}{improved}")
            results.append({
                'target': target, 'desc_added': imp['desc'],
                'R2_LOO_baseline': round(base_r2, 3),
                'R2_LOO_with_desc': round(imp['R2_LOO'], 3),
                'delta_R2_LOO': round(imp['delta_R2'], 3),
                'test_pred': round(imp['test_pred'], 2) if imp['test_pred'] is not None else None,
            })

    pd.DataFrame(results).to_csv(BASE / 'v6_baseline_plus_screen.csv', index=False)

    # ============================================================
    # PART 3: Within-ester sub-class screen (n=8, where mechanism is homogeneous)
    # ============================================================
    print("\n" + "=" * 80)
    print("PART 3: WITHIN C2/ester sub-class (n=8) — pure mechanism, no EWG_type confounding")
    print("=" * 80)
    ester = c2[c2['ewg_type'] == 'ester'].copy()
    print(f"  C2/ester n={len(ester)}")
    for target in ['Ea_d', 'lnA_d', 'Ea_f', 'lnA_f', 'y_max']:
        sub = ester.dropna(subset=[target])
        y = sub[target].to_numpy()
        if len(y) < 5: continue
        # k=1 only (n=8 too few for k>=2)
        best = []
        for d in DESC_POOL + ['sigma_p_sum', 'sigma_m_sum']:
            X = sub[[d]].to_numpy()
            r2 = loo_r2(X, y)
            if not np.isnan(r2):
                best.append({'desc': d, 'R2_LOO': r2})
        best.sort(key=lambda x: -x['R2_LOO'])
        print(f"\n  --- {target}  (n={len(y)}) — top 5 single descriptors ---")
        for b in best[:5]:
            print(f"    {b['desc']:<22} LOO={b['R2_LOO']:+.3f}")

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 80)
    print("SUMMARY: Does any descriptor improve over (σ + EWG_type) baseline by ≥ 0.05?")
    print("=" * 80)
    res = pd.DataFrame(results)
    significant = res[res['delta_R2_LOO'] >= 0.05].sort_values(['target', 'delta_R2_LOO'], ascending=[True, False])
    if len(significant) > 0:
        print(significant.to_string(index=False))
    else:
        print("  NO descriptor adds ≥ 0.05 LOO R² improvement over baseline.")
        print("  Conclusion: v6.1 (σ + EWG_type) is essentially optimal for C2.")


if __name__ == '__main__':
    main()
