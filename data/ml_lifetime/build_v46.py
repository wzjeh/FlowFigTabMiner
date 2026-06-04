"""
v4.6 (HYBRID v3) builder.

For each of 16 (class × param) targets, scan all 3-descriptor combinations
under several rule sets and rank by LOO-R²:

  R0 = v4.5 reproduction          : k=3, 3 distinct cats, |r|<0.7, ArLi含Agg, oxLi排Agg, rank by in-sample R²
  R1 = same constraints, rank LOO : k=3, 3 distinct cats, |r|<0.7, same Agg rule, rank by LOO-R²
  R2 = relax collinearity         : k=3, 3 distinct cats, |r|<0.95, same Agg rule, rank by LOO-R²
  R3 = drop Agg constraint        : k=3, 3 distinct cats, |r|<0.95, no Agg rule,  rank by LOO-R²

Outputs:
  v4.6 spec = best LOO under R3 (the most permissive). Save formula CSV.
  diagnostic table comparing R0/R1/R2/R3 LOO for each (class × param).
"""
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

DATA = Path(__file__).parent
OUT_CSV = DATA / "analysis_figures" / "class_fitted_models_v46.csv"
DIAG_CSV = DATA / "analysis_figures" / "v46_diagnostic.csv"


def classify(smi):
    smi = str(smi)
    if any(p in smi for p in ['CO1','C1CO1','OC1','C1OC1']): return 'oxiranylLi'
    if '[Li]c1ccccc1' in smi and smi != '[Li]c1ccccc1': return 'o-ArLi'
    if '[Li]c1ccc(' in smi: return 'p-ArLi'
    if '[Li]c1cccc(' in smi: return 'm-ArLi'
    return 'other'


CAT = {  # descriptor → physical category
    # Electronic
    'sigma':'E','Es':'E','q_Li':'E','q_C':'E','HOMO':'E','LUMO':'E','eta':'E','fukui':'E',
    # Bonding
    'd_LiC':'B','BDE':'B',
    # Solvation (electronic environment of solvent)
    'dipole':'Sv','Gsolv':'Sv',
    # Steric
    'B1':'St','B5':'St','L':'St','pVbur':'St','vol':'St',
    # Aggregation
    'dE_dim':'A','dH_dim':'A','dG_dim':'A','dS_dim':'A','dVbur_dim':'A',
}

def loo_r2(X, y):
    n = len(y)
    if n < X.shape[1] + 2: return np.nan
    preds = np.empty(n)
    for i in range(n):
        m = LinearRegression().fit(np.delete(X, i, 0), np.delete(y, i))
        preds[i] = m.predict(X[i:i+1])[0]
    rss = np.sum((y - preds)**2); tss = np.sum((y - y.mean())**2)
    return 1 - rss/tss if tss > 0 else np.nan


def in_r2(X, y):
    return LinearRegression().fit(X, y).score(X, y)


def fit_coeffs(X, y, descs):
    m = LinearRegression().fit(X, y)
    return list(m.coef_), float(m.intercept_)


def best_under_rule(df, target, descs_avail, *, max_r, require_agg, exclude_agg,
                    rank_by, require_3cats=True):
    """Scan combos. Returns sorted list of dicts."""
    y_col = target
    rows = []
    y_full = df[y_col].to_numpy()
    for combo in combinations(descs_avail, 3):
        cats = {CAT[d] for d in combo}
        if require_3cats and len(cats) < 3: continue
        if require_agg and 'A' not in cats: continue
        if exclude_agg and 'A' in cats: continue
        sub = df[list(combo) + [y_col]].dropna()
        if len(sub) < 5: continue
        X = sub[list(combo)].to_numpy()
        y = sub[y_col].to_numpy()
        # max pairwise correlation
        c = np.abs(np.corrcoef(X.T))
        np.fill_diagonal(c, 0)
        r = c.max()
        if r > max_r: continue
        try:
            r2_in = in_r2(X, y)
            r2_lo = loo_r2(X, y)
        except Exception: continue
        rows.append({
            'combo': '+'.join(combo),
            'descs': list(combo),
            'cats': '+'.join(sorted(cats)),
            'has_Agg': 'A' in cats,
            'n': len(sub),
            'r2_in': r2_in, 'r2_loo': r2_lo,
            'max_r': r,
        })
    if not rows: return []
    return sorted(rows, key=lambda r: r[rank_by], reverse=True)


def build_master_df():
    g = pd.read_csv(DATA / 'global_arrhenius.csv')
    im = pd.read_csv(DATA / 'intermediates_master.csv')
    agg = pd.read_csv(DATA / 'aggregation_descriptors.csv')
    # Restrict to Tier C modeling set (matches HYBRID v2 CSV n counts)
    tier = pd.read_csv(DATA / 'model_comparison_L1_L2.csv')
    tier['tier_clean'] = tier['tier'].apply(
        lambda x: 'Tier A' if 'Tier A' in x else
                  ('Tier B' if 'Tier B' in x else 'Tier C'))
    excluded = set(tier[tier['tier_clean'].isin(['Tier A','Tier B'])]['smi'])
    # also apply notebook filters: r²_global > 0.6 and Ea_d < 140 kJ/mol
    g = g[~g['smi'].isin(excluded)]
    g = g[(g['r2_global'] > 0.6) & (g['Ea_d'] < 140)]
    df = g.merge(im, left_on='smi', right_on='intermediate_smiles_canonical', how='left')
    df = df.merge(agg.rename(columns={'smi':'smi_agg'}),
                  left_on='smi', right_on='smi_agg', how='left')
    df = df.rename(columns={
        'sigma_hammett':'sigma','dft_dipole_D':'dipole','dft_Gsolv_kJ':'Gsolv',
        'sterimol_B1':'B1','sterimol_B5':'B5','sterimol_L':'L','mol_volume':'vol',
        'fukui_f_minus_C':'fukui','dS_dim_J_K':'dS_dim','dVbur':'dVbur_dim',
        'dG_dim_kJ':'dG_dim','dH_dim_kJ':'dH_dim','dE_dim_kJ':'dE_dim',
        'dft_LiC_bond_A':'d_LiC','dft_HOMO_eV':'HOMO','dft_LUMO_eV':'LUMO',
        'HOMO_LUMO_gap_eV':'eta','dft_charge_Li':'q_Li','dft_charge_C_ipso':'q_C',
        'Es_taft':'Es','dft_LiC_BDE_kJ':'BDE','buried_vol_Li':'pVbur'})
    df['cls'] = df['smi'].apply(classify)
    return df


def main():
    df = build_master_df()
    descs_all = list(CAT.keys())

    classes = ['p-ArLi', 'm-ArLi', 'o-ArLi', 'oxiranylLi']
    targets = ['Ea_f', 'Ea_d', 'lnA_f', 'lnA_d']

    diag_rows = []
    final_rows = []

    for cls in classes:
        sub_df = df[df['cls'] == cls]
        avail = [d for d in descs_all if d in sub_df.columns]
        for target in targets:
            if target not in sub_df.columns: continue
            sub2 = sub_df.dropna(subset=[target])
            n_max = len(sub2)
            require_agg = (cls != 'oxiranylLi')
            exclude_agg = (cls == 'oxiranylLi')
            # v4.5 rule: oxiranylLi does NOT require 3 distinct cats and uses |r|<0.99
            r_max = 0.7 if cls != 'oxiranylLi' else 0.99
            need_3 = (cls != 'oxiranylLi')

            results = {}
            for label, args in [
                ('R0_v45_repro',  dict(max_r=r_max, require_agg=require_agg, exclude_agg=exclude_agg, rank_by='r2_in', require_3cats=need_3)),
                ('R1_strict_LOO', dict(max_r=r_max, require_agg=require_agg, exclude_agg=exclude_agg, rank_by='r2_loo', require_3cats=need_3)),
                ('R2_relaxR_LOO', dict(max_r=0.95, require_agg=require_agg, exclude_agg=exclude_agg, rank_by='r2_loo', require_3cats=need_3)),
                ('R3_no_Agg_LOO', dict(max_r=0.95, require_agg=False,       exclude_agg=False,       rank_by='r2_loo', require_3cats=need_3)),
            ]:
                cands = best_under_rule(sub2, target, avail, **args)
                results[label] = cands[0] if cands else None

            for label, top in results.items():
                if top is None: continue
                diag_rows.append({
                    'class': cls, 'param': target, 'rule': label,
                    'combo': top['combo'], 'cats': top['cats'],
                    'has_Agg': top['has_Agg'], 'n': top['n'],
                    'r2_in': round(top['r2_in'], 4), 'r2_loo': round(top['r2_loo'], 4),
                    'max_r': round(top['max_r'], 4),
                })

            # v4.6 = pick the rule that gives the BEST LOO-R² for this row.
            best_rule = None
            best_top = None
            for label in ['R0_v45_repro','R1_strict_LOO','R2_relaxR_LOO','R3_no_Agg_LOO']:
                t = results.get(label)
                if t is None: continue
                if best_top is None or t['r2_loo'] > best_top['r2_loo']:
                    best_top = t
                    best_rule = label
            if best_top is None: continue
            chosen = best_top
            chosen['rule_used'] = best_rule

            sub3 = sub2[chosen['descs'] + [target]].dropna()
            X = sub3[chosen['descs']].to_numpy()
            y = sub3[target].to_numpy()
            coefs, intercept = fit_coeffs(X, y, chosen['descs'])
            formula = " + ".join(f"{c:+.4f}·{d}" for c, d in zip(coefs, chosen['descs']))
            formula = f"{formula} {intercept:+.4f}"
            final_rows.append({
                'class': cls, 'param': target,
                'n': chosen['n'],
                'descriptors': chosen['combo'],
                'categories': chosen['cats'],
                'coefficients': ", ".join(f"{c:+.4f}" for c in coefs),
                'intercept': round(intercept, 4),
                'formula': formula,
                'r2_loo': round(chosen['r2_loo'], 4),
                'r2_in': round(chosen['r2_in'], 4),
                'max_intra_r': round(chosen['max_r'], 4),
                'rule_used': chosen['rule_used'],
            })

    diag = pd.DataFrame(diag_rows)
    fin  = pd.DataFrame(final_rows)
    diag.to_csv(DIAG_CSV, index=False)
    fin.to_csv(OUT_CSV, index=False)

    # ----- summary printout -----
    print("=" * 88)
    print("v4.6 builder: per-target rule comparison")
    print("=" * 88)
    cols = ['class','param','rule','combo','has_Agg','n','r2_in','r2_loo','max_r']
    print(diag[cols].to_string(index=False))

    print("\n" + "=" * 88)
    print(f"v4.6 final selections (rule used per row in 'rule_used' col)")
    print("=" * 88)
    print(fin[['class','param','descriptors','r2_loo','r2_in','max_intra_r','rule_used']].to_string(index=False))

    # Compare to v4.5 numbers
    v45 = pd.read_csv(DATA / 'analysis_figures' / 'class_fitted_models_HYBRID.csv')
    print("\n" + "=" * 88)
    print("v4.5 → v4.6 LOO-R² comparison")
    print("=" * 88)
    cmp = v45.merge(fin[['class','param','r2_loo','rule_used']],
                    on=['class','param'], suffixes=('_v45','_v46'))
    cmp['delta'] = cmp['r2_loo_v46'] - cmp['r2_loo_v45']
    cmp = cmp.sort_values('delta', ascending=False)
    print(cmp[['class','param','r2_loo_v45','r2_loo_v46','delta','rule_used']].to_string(index=False))

    print(f"\nSaved: {OUT_CSV}")
    print(f"Saved: {DIAG_CSV}")


if __name__ == "__main__":
    main()
