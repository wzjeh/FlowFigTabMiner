"""
Diagnose m-ArLi Ea_d formula: compare v4.5 (HYBRID v2 with forced Agg)
versus v4.0 (free k=2-3 across categories) on LOO-R².

Approach:
  1. Build modeling set: merge global_arrhenius + intermediates_master + agg_descriptors
  2. Classify by SMILES pattern; pick m-ArLi rows
  3. For each candidate descriptor combo, refit OLS leave-one-out
  4. Report LOO-R² so the comparison is apples-to-apples

Two specific formulas evaluated:
  v4.5 (current HYBRID v2): fukui + B1 + dS_dim
  v4.0 (free):              dipole + B5 + sigma
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

DATA = Path(__file__).parent


def classify(smi):
    smi = str(smi)
    if any(p in smi for p in ['CO1', 'C1CO1', 'OC1', 'C1OC1']):
        return 'oxiranylLi'
    if '[Li]c1ccccc1' in smi and smi != '[Li]c1ccccc1':
        return 'o-ArLi'
    if '[Li]c1ccc(' in smi:
        return 'p-ArLi'
    if '[Li]c1cccc(' in smi:
        return 'm-ArLi'
    if '[Li]c1' in smi:
        return 'hetero-ArLi'
    return 'other'


def loo_r2(X, y):
    """Leave-one-out R² for OLS."""
    n = len(y)
    if n < len(X.columns) + 2:
        return np.nan
    preds = np.empty(n)
    for i in range(n):
        mask = np.arange(n) != i
        m = LinearRegression().fit(X.iloc[mask], y[mask])
        preds[i] = m.predict(X.iloc[[i]])[0]
    ss_res = np.sum((y - preds) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1 - ss_res / ss_tot


def insample_r2(X, y):
    m = LinearRegression().fit(X, y)
    return m.score(X, y)


def main():
    g = pd.read_csv(DATA / 'global_arrhenius.csv')
    im = pd.read_csv(DATA / 'intermediates_master.csv')
    agg = pd.read_csv(DATA / 'aggregation_descriptors.csv')

    # merge
    df = g.merge(im, left_on='smi', right_on='intermediate_smiles_canonical',
                 how='left', suffixes=('', '_im'))
    df = df.merge(agg.rename(columns={'smi': 'smi_agg'}),
                  left_on='smi', right_on='smi_agg', how='left')
    df['cls'] = df['smi'].apply(classify)

    m_df = df[df['cls'] == 'm-ArLi'].copy()
    print(f"\nm-ArLi class members ({len(m_df)}):")
    print(m_df[['smi','Ea_d','sigma_hammett','sterimol_B5','dft_dipole_D']].to_string(index=False))

    # rename / unify column names
    rename = {
        'sigma_hammett': 'sigma',
        'Es_taft': 'Es',
        'dft_charge_Li': 'q_Li',
        'dft_charge_C_ipso': 'q_C',
        'dft_HOMO_eV': 'HOMO',
        'dft_LUMO_eV': 'LUMO',
        'HOMO_LUMO_gap_eV': 'eta',
        'dft_LiC_bond_A': 'd_LiC',
        'dft_LiC_BDE_kJ': 'BDE',
        'dft_dipole_D': 'dipole',
        'dft_Gsolv_kJ': 'Gsolv',
        'sterimol_B1': 'B1',
        'sterimol_B5': 'B5',
        'sterimol_L': 'L',
        'buried_vol_Li': 'pVbur',
        'mol_volume': 'vol',
        'fukui_f_minus_C': 'fukui',
        'dE_dim_kJ': 'dE_dim',
        'dH_dim_kJ': 'dH_dim',
        'dG_dim_kJ': 'dG_dim',
        'dS_dim_J_K': 'dS_dim',
        'dVbur': 'dVbur_dim',
    }
    m_df = m_df.rename(columns=rename)

    y = m_df['Ea_d'].to_numpy()

    print("\n" + "=" * 60)
    print("Evaluating m-ArLi Ea_d formulas")
    print("=" * 60)

    # v4.5 / HYBRID v2 current
    v45_descs = ['fukui', 'B1', 'dS_dim']
    # v4.0 free
    v40_descs = ['dipole', 'B5', 'sigma']

    for label, descs in [
        ("v4.5 (HYBRID v2 current)", v45_descs),
        ("v4.0 (free, no Agg)",      v40_descs),
    ]:
        sub = m_df.dropna(subset=descs + ['Ea_d'])
        X = sub[descs]
        y_ = sub['Ea_d'].to_numpy()
        n = len(y_)
        r2_in = insample_r2(X, y_)
        r2_lo = loo_r2(X, y_)
        m = LinearRegression().fit(X, y_)
        coef_str = " + ".join(f"{c:+.3f}·{d}" for c, d in zip(m.coef_, descs))
        print(f"\n{label}")
        print(f"  descriptors  : {descs}")
        print(f"  data rows    : n = {n} (after dropna)")
        print(f"  formula      : Ea_d = {coef_str} {m.intercept_:+.2f}")
        print(f"  R² in-sample : {r2_in:.4f}")
        print(f"  R² LOO       : {r2_lo:.4f}")

    # broader scan: all 3-descriptor combos with at least one descriptor from
    # 3 distinct categories, NO Agg constraint
    cat_map = {
        'sigma': 'E', 'Es': 'E', 'q_Li': 'E', 'q_C': 'E',
        'HOMO': 'E', 'LUMO': 'E', 'eta': 'E', 'fukui': 'E',
        'd_LiC': 'B', 'BDE': 'B',
        'dipole': 'Sv', 'Gsolv': 'Sv',
        'B1': 'St', 'B5': 'St', 'L': 'St',
        'pVbur': 'St', 'vol': 'St',
        'dE_dim': 'A', 'dH_dim': 'A', 'dG_dim': 'A',
        'dS_dim': 'A', 'dVbur_dim': 'A',
    }
    avail = [d for d in cat_map if d in m_df.columns]

    from itertools import combinations
    print(f"\n\n{'='*60}\nFull 3-descriptor scan (no forced Agg)")
    print(f"  candidate descriptors: {len(avail)}")
    print("=" * 60)
    results = []
    for combo in combinations(avail, 3):
        cats = {cat_map[d] for d in combo}
        if len(cats) < 3:    # require 3 distinct categories
            continue
        sub = m_df.dropna(subset=list(combo) + ['Ea_d'])
        if len(sub) < 5:
            continue
        # max intra-category |r|
        try:
            r2_lo = loo_r2(sub[list(combo)], sub['Ea_d'].to_numpy())
            r2_in = insample_r2(sub[list(combo)], sub['Ea_d'].to_numpy())
        except Exception:
            continue
        results.append({
            'combo': '+'.join(combo),
            'cats': ','.join(sorted(cats)),
            'has_Agg': 'A' in cats,
            'n': len(sub),
            'r2_in': r2_in,
            'r2_loo': r2_lo,
        })
    res_df = pd.DataFrame(results).sort_values('r2_loo', ascending=False)
    print("\nTop 15 combinations by LOO-R²:")
    print(res_df.head(15).to_string(index=False))

    print("\nTop 5 NO-Agg combinations (Plan B candidate):")
    print(res_df[~res_df['has_Agg']].head(5).to_string(index=False))

    print("\nTop 5 WITH-Agg combinations (current rule):")
    print(res_df[res_df['has_Agg']].head(5).to_string(index=False))

    res_df.to_csv(DATA / 'mArLi_Ead_diagnostic.csv', index=False)
    print(f"\nSaved full scan to: mArLi_Ead_diagnostic.csv")


if __name__ == "__main__":
    main()
