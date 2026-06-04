"""
Phase 1 of v5.0: Build training set for unified Hammett-LFER framework.

Filters global_arrhenius.csv to Tier B+ (r²>0.6), holds out 4-F-PhLi and
3,5-Br2-CN validation substrates, computes per-substrate Hammett σ_p/σ_m,
Taft Es, mechanism flags (δ_5exo, δ_chelation, δ_benzyne), n_EWG, and
merges aggregation descriptors (ΔG_dim, Δ%V_bur).

Output: v50_training_set.csv
"""
import sys, re
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))
from build_v46 import classify

# ============================================================
# Substituent Hammett σ table (extended, Hansch/Leo + supplements)
# σ_p = Hammett para constant; σ_m = meta constant; σ_o ≈ σ_p (empirically)
# ============================================================
SIGMA_TABLE = {
    # Strong EWG
    'CN':       {'sigma_p': 0.66, 'sigma_m': 0.56, 'Es': -0.51},
    'NO2':      {'sigma_p': 0.78, 'sigma_m': 0.71, 'Es': -1.01},
    # Carbonyl-class (ester/ketone/acid)
    'COOMe':    {'sigma_p': 0.45, 'sigma_m': 0.37, 'Es': -1.31},  # methyl ester
    'COOEt':    {'sigma_p': 0.45, 'sigma_m': 0.37, 'Es': -1.32},
    'COOiPr':   {'sigma_p': 0.45, 'sigma_m': 0.37, 'Es': -1.71},
    'COOtBu':   {'sigma_p': 0.45, 'sigma_m': 0.37, 'Es': -2.78},
    'COMe':     {'sigma_p': 0.50, 'sigma_m': 0.38, 'Es': -1.24},
    'COH':      {'sigma_p': 0.42, 'sigma_m': 0.35, 'Es': -0.55},
    'COOH':     {'sigma_p': 0.45, 'sigma_m': 0.37, 'Es': -1.31},
    # Halides
    'F':        {'sigma_p': 0.06, 'sigma_m': 0.34, 'Es': -0.46},
    'Cl':       {'sigma_p': 0.23, 'sigma_m': 0.37, 'Es': -0.97},
    'Br':       {'sigma_p': 0.23, 'sigma_m': 0.39, 'Es': -1.16},
    'I':        {'sigma_p': 0.18, 'sigma_m': 0.35, 'Es': -1.40},
    # EDG
    'Me':       {'sigma_p': -0.17,'sigma_m': -0.07,'Es': 0.00},
    'tBu':      {'sigma_p': -0.20,'sigma_m': -0.10,'Es': -1.54},
    'OMe':      {'sigma_p': -0.27,'sigma_m': 0.12, 'Es': -0.55},
    'OEt':      {'sigma_p': -0.24,'sigma_m': 0.10, 'Es': -0.55},
    'NMe2':     {'sigma_p': -0.83,'sigma_m': -0.16,'Es': -2.10},
    'CF3':      {'sigma_p': 0.54, 'sigma_m': 0.43, 'Es': -2.40},
    # Aryl-only (biphenyl, naphthyl)
    'Ph':       {'sigma_p': -0.01,'sigma_m': 0.06, 'Es': -1.01},
    'Ph(Br)':   {'sigma_p': 0.07, 'sigma_m': 0.07, 'Es': -1.01},   # 4-Br-biphenyl (treated as reactive due to Br)
    # H / placeholder
    'H':        {'sigma_p': 0.0, 'sigma_m': 0.0, 'Es': 0.0},
    # Default for unrecognized
    'unknown':  {'sigma_p': 0.0, 'sigma_m': 0.0, 'Es': 0.0},
}

# ============================================================
# SMILES → substituent classification
# Returns dict: position (p/m/o) → substituent type
# ============================================================
SMI_TO_SUBSTITUENTS = {
    # p-ArLi (4-substituted phenyllithium)
    '[Li]c1ccc(C#N)cc1':              {'p': 'CN'},
    '[Li]c1ccc(C(=O)OC)cc1':           {'p': 'COOMe'},
    '[Li]c1ccc(C(=O)OCC)cc1':          {'p': 'COOEt'},
    '[Li]c1ccc(C(=O)OC(C)C)cc1':       {'p': 'COOiPr'},
    '[Li]c1ccc(C(=O)OC(C)(C)C)cc1':    {'p': 'COOtBu'},
    '[Li]c1ccc([N+](=O)[O-])cc1':      {'p': 'NO2'},
    '[Li]c1ccc(Br)cc1':                {'p': 'Br'},
    '[Li]c1ccc(F)cc1':                 {'p': 'F'},
    '[Li]c1ccc(Cl)cc1':                {'p': 'Cl'},
    '[Li]c1ccc(OC)cc1':                {'p': 'OMe'},
    '[Li]c1ccc(C(F)(F)F)cc1':          {'p': 'CF3'},
    '[Li]c1ccc(-c2ccc(Br)cc2)cc1':     {'p': 'Ph(Br)'},  # 4-(4-Br-Ph)-PhLi
    # m-ArLi
    '[Li]c1cccc(C#N)c1':               {'m': 'CN'},
    '[Li]c1cccc(C(=O)OC)c1':           {'m': 'COOMe'},
    '[Li]c1cccc(C(=O)OCC)c1':          {'m': 'COOEt'},
    '[Li]c1cccc(C(=O)OC(C)C)c1':       {'m': 'COOiPr'},
    '[Li]c1cccc(C(=O)OC(C)(C)C)c1':    {'m': 'COOtBu'},
    '[Li]c1cccc([N+](=O)[O-])c1':      {'m': 'NO2'},
    '[Li]c1cccc(Br)c1':                {'m': 'Br'},
    '[Li]c1cc(Br)cc(C#N)c1':           {'m': 'CN', 'm2': 'Br'},  # 3,5-Br-CN (dual EWG)
    # o-ArLi (2-substituted)
    '[Li]c1ccccc1C#N':                 {'o': 'CN'},
    '[Li]c1ccccc1[N+](=O)[O-]':        {'o': 'NO2'},
    '[Li]c1ccccc1C(=O)OC':             {'o': 'COOMe'},
    '[Li]c1ccccc1C(=O)OCC':            {'o': 'COOEt'},
    '[Li]c1ccccc1C(=O)OC(C)C':         {'o': 'COOiPr'},
    '[Li]c1ccccc1C(=O)OC(C)(C)C':      {'o': 'COOtBu'},
    '[Li]c1ccccc1I':                   {'o': 'I'},
    '[Li]c1ccccc1Br':                  {'o': 'Br'},
    '[Li]c1ccccc1-c1ccccc1Br':         {'o': 'Ph(Br)'},  # 2-(2-Br-Ph)-PhLi
    '[Li]c1ccccc1OC':                  {'o': 'OMe'},
    # 2-pyridyllithium (special — heteroaryl, will be excluded if heteroaryl class)
    '[Li]c1ccccn1':                    {'p': 'CN'},  # placeholder, treats N as p-CN-like
}


def get_substituents(smi):
    """Return dict of {position: substituent_name} for given SMILES."""
    return SMI_TO_SUBSTITUENTS.get(smi, {})


def sum_sigma(subs, position_key):
    """Sum Hammett σ for all substituents at given position class.
    Convention: σ_p for para; σ_m for meta; σ_p (field-effect approx) for ortho electronic."""
    val = 0.0
    for pos, sub in subs.items():
        pos_norm = 'm' if pos.startswith('m') else pos
        if pos_norm != position_key:
            continue
        sub_data = SIGMA_TABLE.get(sub, SIGMA_TABLE['unknown'])
        if pos_norm == 'p' or pos_norm == 'o':   # ortho electronic ≈ para σ
            val += sub_data['sigma_p']
        else:  # meta
            val += sub_data['sigma_m']
    return val


def es_ortho(subs):
    """Taft Es for ortho substituent (only one allowed by convention)."""
    for pos, sub in subs.items():
        if pos == 'o':
            return SIGMA_TABLE.get(sub, SIGMA_TABLE['unknown'])['Es']
    return 0.0


def has_5exo_ortho(subs):
    """ortho-CN / NO2 / C=O can attack ArLi via 5-exo pathway."""
    if 'o' not in subs: return 0
    sub = subs['o']
    return int(sub in {'CN', 'NO2'} or sub.startswith('COO') or sub.startswith('COM') or sub == 'COH')


def has_chelation_ortho(subs):
    """ortho-OR / NR2 provides Li-O/N chelation."""
    if 'o' not in subs: return 0
    return int(subs['o'] in {'OMe', 'OEt', 'NMe2'})


def has_benzyne_ortho(subs):
    """ortho-Br/I undergoes benzyne elimination."""
    if 'o' not in subs: return 0
    return int(subs['o'] in {'Br', 'I'})


def has_ortho_aryl(subs):
    """ortho-aryl (e.g., 2-(2-Br-Ph)-PhLi) — special case, treat as benzyne-active if Br/I present."""
    if 'o' not in subs: return 0
    return int(subs['o'].startswith('Ph') and ('Br' in subs['o'] or 'I' in subs['o']))


def count_EWG(subs):
    """Count EWG substituents (any position)."""
    EWGs = {'CN','NO2','COOMe','COOEt','COOiPr','COOtBu','COMe','COH','COOH','CF3'}
    return sum(1 for sub in subs.values() if sub in EWGs)


def is_reactive(subs):
    """Reactive if any reactive substituent (CN, NO2, C=O, Br, I) at any position."""
    REACTIVE = {'CN','NO2','COOMe','COOEt','COOiPr','COOtBu','COMe','COH','COOH','Br','I','Ph(Br)'}
    return int(any(sub in REACTIVE for sub in subs.values()))


def main():
    # Load primary 5-param fit data
    g = pd.read_csv(BASE / 'global_arrhenius.csv')
    print(f"Total global_arrhenius entries: {len(g)}")

    # Filter Tier B+ (r²>0.6)
    g = g[g['r2_global'] > 0.6].copy()
    print(f"After r²>0.6 filter: {len(g)}")

    # Classify by v4.6 positional class
    g['cls'] = g['smi'].apply(classify)

    # Hold out validation substrates
    VALIDATION = {'[Li]c1ccc(F)cc1', '[Li]c1cc(Br)cc(C#N)c1'}
    g_validation = g[g['smi'].isin(VALIDATION)].copy()
    g_train = g[~g['smi'].isin(VALIDATION)].copy()
    print(f"Held out validation: {len(g_validation)}")
    print(f"Training set: {len(g_train)}")
    if len(g_validation):
        print(f"  Validation substrates: {g_validation['smi'].tolist()}")

    # Filter to known-substituent ArLi (drop substrates without SMI_TO_SUBSTITUENTS entries)
    # These would need manual assignment of substituents
    unknown_subs = []
    for smi in g_train['smi'].unique():
        if smi not in SMI_TO_SUBSTITUENTS:
            unknown_subs.append(smi)
    if unknown_subs:
        print(f"\n[WARNING] Substrates without substituent map (will be excluded):")
        for s in unknown_subs:
            cls = classify(s)
            row = g_train[g_train['smi']==s].iloc[0]
            print(f"  {s:<48}  cls={cls}  Ea_d={row['Ea_d']:.1f}")

    g_train = g_train[g_train['smi'].isin(SMI_TO_SUBSTITUENTS)].copy()
    print(f"\nFinal training set (with substituent assignments): {len(g_train)}")

    # Compute descriptors per substrate
    rows = []
    for _, r in g_train.iterrows():
        subs = get_substituents(r['smi'])
        rec = {
            'smi': r['smi'],
            'intermediate': r['intermediate'],
            'cls': r['cls'],
            'Ea_f': r['Ea_f'], 'lnA_f': r['lnA_f'],
            'Ea_d': r['Ea_d'], 'lnA_d': r['lnA_d'],
            'y_max': r['y_max'],
            'r2_global': r['r2_global'], 'n_temps': r['n_temps'],
            # Hammett descriptors
            'sigma_p_sum': sum_sigma(subs, 'p'),
            'sigma_m_sum': sum_sigma(subs, 'm'),
            'sigma_o_sum': sum_sigma(subs, 'o'),
            'Es_ortho': es_ortho(subs),
            # Mechanism flags
            'delta_5exo_o':    has_5exo_ortho(subs),
            'delta_chelation_o': has_chelation_ortho(subs),
            'delta_benzyne_o': has_benzyne_ortho(subs) or has_ortho_aryl(subs),
            'n_EWG': count_EWG(subs),
            'is_reactive': is_reactive(subs),
            # Substituent identities (for inspection)
            'subs_str': '+'.join(f'{p}:{s}' for p,s in subs.items()),
        }
        rows.append(rec)

    df_train = pd.DataFrame(rows)

    # Merge aggregation descriptors (ΔG_dim, Δ%V_bur)
    try:
        agg = pd.read_csv(BASE / 'aggregation_descriptors.csv')
        df_train = df_train.merge(agg[['smi','dG_dim_kJ','dVbur']], on='smi', how='left')
        df_train = df_train.rename(columns={'dG_dim_kJ':'dG_dim', 'dVbur':'dVbur_dim'})
        print(f"\nMerged aggregation descriptors. {df_train['dG_dim'].notna().sum()}/{len(df_train)} have ΔG_dim")
    except Exception as e:
        print(f"WARNING: aggregation merge failed: {e}")

    # Final cleanup: drop rows missing core targets
    df_train = df_train.dropna(subset=['Ea_d','lnA_d','Ea_f','lnA_f'])
    print(f"\nFinal training set rows (all 4 Arrhenius params present): {len(df_train)}")

    # Summary statistics
    print("\n=== Training set summary ===")
    print(f"Class breakdown:")
    print(df_train['cls'].value_counts().to_string())
    print(f"\nReactivity flag:")
    print(df_train['is_reactive'].value_counts().to_string())
    print(f"\nMechanism flag distribution:")
    for flag in ['delta_5exo_o','delta_chelation_o','delta_benzyne_o']:
        print(f"  {flag}: {df_train[flag].sum()}/{len(df_train)}")
    print(f"\nn_EWG distribution:")
    print(df_train['n_EWG'].value_counts().sort_index().to_string())
    print(f"\nHammett σ ranges:")
    print(f"  σ_p_sum range: {df_train['sigma_p_sum'].min():.2f} to {df_train['sigma_p_sum'].max():.2f}")
    print(f"  σ_m_sum range: {df_train['sigma_m_sum'].min():.2f} to {df_train['sigma_m_sum'].max():.2f}")
    print(f"  Es_ortho range: {df_train['Es_ortho'].min():.2f} to {df_train['Es_ortho'].max():.2f}")

    # Save training set
    out = BASE / 'v50_training_set.csv'
    df_train.to_csv(out, index=False)
    print(f"\nSaved: {out}")

    # Also save validation set for downstream use
    val_rows = []
    for _, r in g_validation.iterrows():
        if r['smi'] not in SMI_TO_SUBSTITUENTS: continue
        subs = get_substituents(r['smi'])
        rec = {
            'smi': r['smi'], 'intermediate': r['intermediate'],
            'cls': classify(r['smi']),
            'Ea_f_exp': r['Ea_f'], 'lnA_f_exp': r['lnA_f'],
            'Ea_d_exp': r['Ea_d'], 'lnA_d_exp': r['lnA_d'],
            'y_max_exp': r['y_max'],
            'sigma_p_sum': sum_sigma(subs, 'p'),
            'sigma_m_sum': sum_sigma(subs, 'm'),
            'sigma_o_sum': sum_sigma(subs, 'o'),
            'Es_ortho': es_ortho(subs),
            'delta_5exo_o': has_5exo_ortho(subs),
            'delta_chelation_o': has_chelation_ortho(subs),
            'delta_benzyne_o': has_benzyne_ortho(subs) or has_ortho_aryl(subs),
            'n_EWG': count_EWG(subs),
            'is_reactive': is_reactive(subs),
            'subs_str': '+'.join(f'{p}:{s}' for p,s in subs.items()),
        }
        val_rows.append(rec)
    df_val = pd.DataFrame(val_rows)
    if len(df_val):
        df_val.to_csv(BASE / 'v50_validation_set.csv', index=False)
        print(f"Saved: v50_validation_set.csv (validation targets — for blind testing)")


if __name__ == '__main__':
    main()
