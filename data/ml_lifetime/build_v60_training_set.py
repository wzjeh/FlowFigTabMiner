"""
v6.0 Phase 2: Build training set with literature anchors + Charton virtual augmentation.

Sources merged:
  1. In-house: v60_classified_substrates.csv (45 substrates with class labels)
     - Exclude OTHER, BENZYL, CARBENOID, OXI (not in v6.0 main fit)
     - Exclude known outliers (p-bromophenyllithium Ea_d=149 per v4.4)
  2. Literature C1 anchors (Stanetty 1997 / Honeycutt 1971 / Fitt 1984):
     - n-BuLi/THF, n-BuLi/Et2O — proto-de-Li mechanism (C1 class)
  3. Charton 2004 Eq.36 virtual Ea_f points (C2 class):
     - log k(ArLi+ArBr exchange) = 5.07σ_X − 2080/T + 6.84  → Ea_f = 40 − 21σ
     - Generate 15 virtual points covering σ ∈ [-0.3, 0.7]
     - Weight 0.3 vs real data weight 1.0

Output columns:
  intermediate, smi, class_v60, source, is_virtual, weight, sigma_p_sum, sigma_m_sum,
  Ea_f, lnA_f, Ea_d, lnA_d, y_max, r2_global, n_temps, notes
"""
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).parent
np.random.seed(42)

# ============================================================
# Outliers to exclude from in-house fit
# ============================================================
EXCLUDE_INTERMEDIATES = {
    'p-bromophenyllithium',                   # Ea_d=149 outlier (v4.4)
    'monolithiated 4',                        # truncated name (biphenyl)
    "monolithiated 4,4'-dibromobiphenyl",     # Ea_d=38 anomaly (biphenyl→biphenyl benzyne)
    '2-bromo-2\'-lithiobiphenyl',             # Ea_d=49 (biphenyl ortho-Br benzyne)
    "2-bromo-2'-lithiobiphenyl",              # robust quote variant
    '4-lithio-5-methyl-2-phenylthiazole',     # OTHER anyway
}


# ============================================================
# Literature anchors for C1 (inert proto-de-Li class)
# Source: v4.7 / CITATIONS_ARLI_STABILITY.md
# ============================================================
C1_LIT_ANCHORS = [
    # (name, SMILES, Ea_d, lnA_d, Ea_f, lnA_f, y_max, source)
    # Stanetty 1997 JOC 62, 1514 — n-BuLi in THF, multi-T
    ('n-BuLi/THF (Stanetty 1997)',     'CCCC[Li]',  75.7, 21.91, 32.0, 21.0, 95.0, 'Stanetty1997'),
    # Honeycutt 1971 JOMC 29, 1 — n-BuLi in Et2O, multi-T
    ('n-BuLi/Et2O (Honeycutt 1971)',   'CCCC[Li]',  79.9, 19.21, 32.0, 21.0, 95.0, 'Honeycutt1971'),
    # Stanetty 1997 — n-BuLi/THP, multi-T
    ('n-BuLi/THP (Stanetty 1997)',     'CCCC[Li]',  75.5, 19.40, 32.0, 21.0, 95.0, 'Stanetty1997b'),
    # In-house p-OMe-PhLi (already in dataset but reinforced as C1 anchor reference)
    # — already counted as in-house; not duplicated here
    # Luisi 2014 Table 18.3 — PhLi/THF, t½ ≈ 100h at +20°C
    # Derived from t½: k_d = ln(2)/360000 = 1.93e-6 s⁻¹ at 293K
    # → Ea_d ≈ 80 (assuming lnA = 22) — consistent with proto-de-Li
    ('PhLi/THF (Luisi 2014)',          '[Li]c1ccccc1', 80.0, 22.0, 32.0, 21.0, 95.0, 'Luisi2014'),
]


# ============================================================
# Charton 2004 Eq.36 virtual augmentation for C2 (remote-EWG class)
# log k(ArLi+ArBr exchange) = 5.07σ_X − 2080/T + 6.84   [PhLi base]
# Convert to Arrhenius: Ea_f = 8.314e-3 × 2080 × ln(10) = 39.83 kJ/mol per σ=0 baseline
# Adjusted for n-BuLi (Schlosser Ch.9): subtract ~8 kJ/mol → Ea_f(σ=0) = 32 kJ/mol
# Hammett: ρ_f → Ea_f = 32 - 21·σ_m (or σ_p, taking σ_p for para-EWG)
# ============================================================
def charton_virtual_points(n=15, weight=0.3):
    """Generate virtual C2 substrates using Charton ρ_f=5.07."""
    sigmas = np.linspace(-0.3, 0.7, n)
    rows = []
    for i, sigma in enumerate(sigmas):
        Ea_f_virt = 32.0 - 21.0 * sigma  # Charton-derived
        # lnA_f roughly constant (formation pre-exponential ~ 21-23)
        # plus minor Hammett-compensated drift (-25.95·σ_m_sum per v5.0 finding)
        # use ρ_lnA = -8 for virtual (less extreme to avoid over-anchoring)
        lnA_f_virt = 21.0 - 8.0 * sigma
        rows.append({
            'intermediate':   f'Charton_virtual_sigma_{sigma:+.2f}',
            'smi':            f'VIRTUAL_σ={sigma:+.2f}',
            'class_v60':      'C2',
            'source':         'Charton2004_virtual',
            'is_virtual':     True,
            'weight':         weight,
            'sigma_p_sum':    sigma if sigma >= 0 else sigma,  # treat as σ
            'sigma_m_sum':    0.0,
            'Ea_f':           Ea_f_virt,
            'lnA_f':          lnA_f_virt,
            'Ea_d':           np.nan,    # virtual only constrains formation
            'lnA_d':          np.nan,
            'y_max':          np.nan,
            'r2_global':      np.nan,
            'n_temps':        np.nan,
            'notes':          'Charton ρ_f=5.07 virtual point (formation Ea_f/lnA_f only)',
        })
    return rows


def main():
    # ---- Load in-house classifications ----
    df = pd.read_csv(BASE / 'v60_classified_substrates.csv')

    print("=" * 80)
    print("v6.0 Phase 2 — Build training set with literature + Charton augmentation")
    print("=" * 80)

    # Apply exclusions
    n_before = len(df)
    df = df[~df['intermediate'].isin(EXCLUDE_INTERMEDIATES)].copy()
    print(f"\nIn-house: {n_before} → {len(df)} after excluding outliers: {EXCLUDE_INTERMEDIATES}")

    # Restrict to v6.0 main classes (C1, C2, C3)
    main_classes = {'C1', 'C2', 'C3', 'C4', 'C5'}
    df_main = df[df['class_v60'].isin(main_classes)].copy()
    print(f"\nv6.0 main fit classes (C1-C5): n={len(df_main)}")
    print(df_main['class_v60'].value_counts().to_string())

    # Mark in-house data
    df_main['source'] = 'inhouse'
    df_main['is_virtual'] = False
    df_main['weight'] = 1.0
    df_main['notes'] = ''

    # ---- Add literature anchors (C1) ----
    lit_rows = []
    for name, smi, Ea_d, lnA_d, Ea_f, lnA_f, y_max, src in C1_LIT_ANCHORS:
        lit_rows.append({
            'intermediate':   name,
            'smi':            smi,
            'class_v60':      'C1',
            'is_aromatic':    'PhLi' in name or 'Ph' in name,
            'sigma_p_sum':    0.0,
            'sigma_m_sum':    0.0,
            'ortho_types':    '',
            'remote_types':   '',
            'notes':          'literature anchor',
            'Ea_f':           Ea_f,
            'lnA_f':          lnA_f,
            'Ea_d':           Ea_d,
            'lnA_d':          lnA_d,
            'y_max':          y_max,
            'r2_global':      np.nan,
            'n_temps':        np.nan,
            'source':         src,
            'is_virtual':     False,
            'weight':         0.5,   # half-weight (different mechanism context but same class)
        })
    df_lit = pd.DataFrame(lit_rows)
    print(f"\nLiterature C1 anchors: n={len(df_lit)} (weight=0.5)")

    # ---- Add Charton virtual points (C2) ----
    virt_rows = charton_virtual_points(n=15, weight=0.3)
    df_virt = pd.DataFrame(virt_rows)
    for col in ['is_aromatic', 'ortho_types', 'remote_types']:
        if col not in df_virt.columns:
            df_virt[col] = '' if 'types' in col else False
    print(f"\nCharton virtual C2 (Ea_f only): n={len(df_virt)} (weight=0.3)")

    # ---- Combine ----
    common_cols = ['intermediate','smi','class_v60','source','is_virtual','weight',
                   'sigma_p_sum','sigma_m_sum','ortho_types','remote_types',
                   'Ea_f','lnA_f','Ea_d','lnA_d','y_max','r2_global','n_temps','notes']
    for col in common_cols:
        for d in [df_main, df_lit, df_virt]:
            if col not in d.columns:
                d[col] = np.nan
    combined = pd.concat([
        df_main[common_cols], df_lit[common_cols], df_virt[common_cols]
    ], ignore_index=True)

    # ---- Add EWG_type sub-classification (for C2 mostly) ----
    # 'CN' if remote includes EWG_CN; 'ester' if EWG_ester; 'NO2' if EWG_NO2;
    # 'CF3' if EWG_CF3; otherwise 'none'.
    def ewg_type(r):
        rt = str(r.get('remote_types', '') or '')
        ot = str(r.get('ortho_types', '') or '')
        types = (rt + ';' + ot).split(';')
        if any('EWG_CN' in t for t in types):     return 'CN'
        if any('EWG_NO2' in t for t in types):    return 'NO2'
        if any('EWG_ester' in t for t in types):  return 'ester'
        if any('EWG_CF3' in t for t in types):    return 'CF3'
        if any('EWG_ketone' in t for t in types): return 'ketone'
        if any('EWG_CHO' in t for t in types):    return 'CHO'
        return 'none'
    combined['ewg_type'] = combined.apply(ewg_type, axis=1)
    # Virtual Charton points: treat as 'mixed' (don't fit to a specific EWG sub-anchor)
    combined.loc[combined['is_virtual'] == True, 'ewg_type'] = 'mixed_virtual'

    # ---- Summary ----
    print(f"\n{'='*80}\nFinal v6.0 training set: n={len(combined)}\n{'='*80}")
    print("\nClass × source breakdown:")
    print(combined.groupby(['class_v60', 'source']).size().to_string())
    print(f"\nClass × is_virtual:")
    print(combined.groupby(['class_v60', 'is_virtual']).size().to_string())
    print(f"\nWeight distribution:")
    print(combined.groupby(['class_v60', 'weight']).size().to_string())

    # Sanity: effective n per class (sum of weights)
    print(f"\nEffective n per class (weighted):")
    eff_n = combined.groupby('class_v60')['weight'].sum().round(2)
    print(eff_n.to_string())

    # Save
    out = BASE / 'v60_training_set.csv'
    combined.to_csv(out, index=False)
    print(f"\nSaved: {out}  ({len(combined)} rows)")


if __name__ == '__main__':
    main()
