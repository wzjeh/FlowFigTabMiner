"""
v6.0 Phase 3.5: Empirical sub-class anchors (class × EWG_type).

The Bayesian C2 anchor smooths across CN/NO2/ester (mean Ea_d=49). But mechanistically:
  CN-class:    Ea_d ~ 28  (5-exo CN attack, fast decay)
  ester-class: Ea_d ~ 50  (chelation-buffered)
  NO2-class:   Ea_d ~ 55  (slower decay)

These are MECHANISTICALLY DISTINCT. For prediction we should use the EWG-type-specific anchor.

Output: v60_subclass_anchors.csv
  per (class, ewg_type, n>=2): mean ± SD for 5 Arrhenius parameters
  flagged "use_for_pred=True" if n>=2 (empirical anchor reliable)
"""
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).parent

df = pd.read_csv(BASE / 'v60_training_set.csv')
df = df[~df['is_virtual']].copy()   # empirical only (no Charton virtual)

params = ['Ea_d', 'lnA_d', 'Ea_f', 'lnA_f', 'y_max']

rows = []
for (cls, ewg), sub in df.groupby(['class_v60', 'ewg_type']):
    row = {'class': cls, 'ewg_type': ewg, 'n': len(sub),
           'substrates': '; '.join(sub['intermediate'].astype(str).str[:30])}
    for p in params:
        vals = sub[p].dropna()
        if len(vals) >= 1:
            row[f'{p}_mean'] = float(vals.mean())
            row[f'{p}_sd']   = float(vals.std(ddof=1)) if len(vals) > 1 else 5.0  # default sd
            row[f'{p}_min']  = float(vals.min())
            row[f'{p}_max']  = float(vals.max())
        else:
            row[f'{p}_mean'] = np.nan
            row[f'{p}_sd'] = np.nan
            row[f'{p}_min'] = np.nan
            row[f'{p}_max'] = np.nan
    rows.append(row)

out = pd.DataFrame(rows).sort_values(['class', 'ewg_type']).reset_index(drop=True)
out['use_for_pred'] = out['n'] >= 2

# Add literature anchor row for C1 explicitly (n_BuLi reference)
out_csv = BASE / 'v60_subclass_anchors.csv'
out.to_csv(out_csv, index=False)

print("=" * 110)
print("v6.0 Sub-class empirical anchors (class × EWG_type)")
print("=" * 110)
display_cols = ['class', 'ewg_type', 'n', 'Ea_d_mean', 'Ea_d_sd',
                'lnA_d_mean', 'lnA_d_sd', 'Ea_f_mean', 'Ea_f_sd',
                'lnA_f_mean', 'lnA_f_sd', 'y_max_mean', 'use_for_pred']
print(out[display_cols].round(2).to_string(index=False))

print(f"\nSaved: {out_csv}")

# ---- Print compact paper-ready table ----
print("\n" + "=" * 80)
print("Paper-ready sub-class anchor table:")
print("=" * 80)
print(f"{'Class':<6}{'EWG type':<10}{'n':>3}  {'Ea_d':>14}  {'lnA_d':>14}  {'y_max':>10}")
for _, r in out.iterrows():
    if r['n'] < 1: continue
    ea_d = f"{r['Ea_d_mean']:.1f} ± {r['Ea_d_sd']:.1f}" if not pd.isna(r['Ea_d_mean']) else "--"
    lna_d = f"{r['lnA_d_mean']:.1f} ± {r['lnA_d_sd']:.1f}" if not pd.isna(r['lnA_d_mean']) else "--"
    y_max = f"{r['y_max_mean']:.0f}" if not pd.isna(r['y_max_mean']) else "--"
    print(f"{r['class']:<6}{r['ewg_type']:<10}{r['n']:>3}  {ea_d:>14}  {lna_d:>14}  {y_max:>10}")
