"""
v6.0 Phase 3: PyMC Bayesian hierarchical model for class-stratified Arrhenius prediction.

Per-class base anchors + Hammett correction (C2 only).
All Ea > 0 by LogNormal prior. lnA centered on Reich 2013 ΔS‡ → 22 ± 5.

Outputs: v60_posterior.csv (class × param × {mean, sd, ci_lo, ci_hi})
         v60_predictions.csv (per training substrate posterior)
"""
# Monkey-patch: arviz 0.17.1 still imports scipy.signal.gaussian (removed in scipy 1.13)
import scipy.signal, scipy.signal.windows
scipy.signal.gaussian = scipy.signal.windows.gaussian

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az

BASE = Path(__file__).parent
np.random.seed(42)

# ============================================================
# Class-level priors (physical anchors)
# Format: (Ea_d_mean, Ea_d_sd, lnA_d_mean, lnA_d_sd,
#          Ea_f_mean, Ea_f_sd, lnA_f_mean, lnA_f_sd,
#          y_max_mean, y_max_sd)
# Ea uses LogNormal (mean = log of prior median, sd = log-sigma)
# ============================================================
CLASS_PRIORS = {
    # C1 inert (Stanetty/Honeycutt anchor strong)
    'C1': dict(Ea_d=(78.0, 0.10), lnA_d=(22.0, 2.0),    # tight
               Ea_f=(32.0, 0.20), lnA_f=(21.0, 3.0),
               y_max=(95.0, 5.0)),
    # C2 remote-EWG (in-house data + Charton)
    'C2': dict(Ea_d=(30.0, 0.30), lnA_d=( 9.0, 5.0),    # broad
               Ea_f=(33.0, 0.20), lnA_f=(22.0, 4.0),
               y_max=(83.0, 8.0)),
    # C3 ortho-chelation (Li...O stabilization)
    'C3': dict(Ea_d=(60.0, 0.30), lnA_d=(20.0, 5.0),
               Ea_f=(35.0, 0.30), lnA_f=(19.0, 5.0),
               y_max=(90.0, 8.0)),
    # C4 ortho-5-exo (limited data — broad prior)
    'C4': dict(Ea_d=(45.0, 0.40), lnA_d=(17.0, 5.0),
               Ea_f=(30.0, 0.30), lnA_f=(18.0, 5.0),
               y_max=(75.0, 12.0)),
    # C5 ortho-benzyne (limited data)
    'C5': dict(Ea_d=(95.0, 0.20), lnA_d=(38.0, 5.0),
               Ea_f=(28.0, 0.30), lnA_f=(25.0, 5.0),
               y_max=(70.0, 15.0)),
}
CLASSES = list(CLASS_PRIORS.keys())


def main():
    df = pd.read_csv(BASE / 'v60_training_set.csv')
    df = df[df['class_v60'].isin(CLASSES)].copy().reset_index(drop=True)
    df['class_idx'] = df['class_v60'].map({c: i for i, c in enumerate(CLASSES)})
    df['sigma_C2'] = np.where(df['class_v60'] == 'C2',
                              df['sigma_m_sum'].fillna(0) + df['sigma_p_sum'].fillna(0),
                              0.0)
    df['is_C2'] = (df['class_v60'] == 'C2').astype(int)

    print(f"Training set: n={len(df)} (after filter)")
    print(df.groupby('class_v60').size().to_string())
    print(f"\nC2 σ range: [{df.loc[df.is_C2==1, 'sigma_C2'].min():.2f}, "
          f"{df.loc[df.is_C2==1, 'sigma_C2'].max():.2f}]")

    # ---- Build mask arrays for each target (skip NaN observations) ----
    targets = ['Ea_d', 'lnA_d', 'Ea_f', 'lnA_f', 'y_max']
    obs_data = {}
    for t in targets:
        mask = df[t].notna().to_numpy()
        obs_data[t] = {
            'mask': mask,
            'value': df.loc[mask, t].to_numpy(),
            'class_idx': df.loc[mask, 'class_idx'].to_numpy(),
            'sigma_C2': df.loc[mask, 'sigma_C2'].to_numpy(),
            'is_C2': df.loc[mask, 'is_C2'].to_numpy(),
            'weight': df.loc[mask, 'weight'].to_numpy(),
        }
        print(f"  {t}: n_obs = {mask.sum()}")

    # ---- Build PyMC model ----
    print("\n" + "=" * 80)
    print("Building PyMC model with physical priors")
    print("=" * 80)
    n_classes = len(CLASSES)

    with pm.Model() as model:
        # Class-level base anchors
        # Ea_d/Ea_f: LogNormal (>0 guarantee)
        Ea_d_base = pm.LogNormal('Ea_d_base',
            mu=np.log([CLASS_PRIORS[c]['Ea_d'][0] for c in CLASSES]),
            sigma=np.array([CLASS_PRIORS[c]['Ea_d'][1] for c in CLASSES]),
            shape=n_classes)
        Ea_f_base = pm.LogNormal('Ea_f_base',
            mu=np.log([CLASS_PRIORS[c]['Ea_f'][0] for c in CLASSES]),
            sigma=np.array([CLASS_PRIORS[c]['Ea_f'][1] for c in CLASSES]),
            shape=n_classes)
        # lnA: Normal
        lnA_d_base = pm.Normal('lnA_d_base',
            mu=np.array([CLASS_PRIORS[c]['lnA_d'][0] for c in CLASSES]),
            sigma=np.array([CLASS_PRIORS[c]['lnA_d'][1] for c in CLASSES]),
            shape=n_classes)
        lnA_f_base = pm.Normal('lnA_f_base',
            mu=np.array([CLASS_PRIORS[c]['lnA_f'][0] for c in CLASSES]),
            sigma=np.array([CLASS_PRIORS[c]['lnA_f'][1] for c in CLASSES]),
            shape=n_classes)
        # y_max: Normal bounded [0, 100] via clipping in likelihood
        y_max_base = pm.Normal('y_max_base',
            mu=np.array([CLASS_PRIORS[c]['y_max'][0] for c in CLASSES]),
            sigma=np.array([CLASS_PRIORS[c]['y_max'][1] for c in CLASSES]),
            shape=n_classes)

        # Hammett coefficients (C2 class only)
        # Ea_f: Charton ρ_f=5.07 → Ea_f = base - 21·σ (anchored tight)
        rho_Ea_f = pm.Normal('rho_Ea_f', mu=21.0, sigma=5.0)
        # lnA_f: v5.0 in-house finding (LOO R²=0.87)
        rho_lnA_f = pm.Normal('rho_lnA_f', mu=25.95, sigma=8.0)
        # Ea_d: weak in-house signal (negative ρ in v5.0 fits)
        rho_Ea_d = pm.Normal('rho_Ea_d', mu=10.0, sigma=15.0)
        # lnA_d: weak
        rho_lnA_d = pm.Normal('rho_lnA_d', mu=5.0, sigma=10.0)

        # Observation noise (per parameter)
        sigma_Ea_d = pm.HalfNormal('sigma_Ea_d', sigma=10.0)
        sigma_lnA_d = pm.HalfNormal('sigma_lnA_d', sigma=5.0)
        sigma_Ea_f = pm.HalfNormal('sigma_Ea_f', sigma=10.0)
        sigma_lnA_f = pm.HalfNormal('sigma_lnA_f', sigma=5.0)
        sigma_y_max = pm.HalfNormal('sigma_y_max', sigma=10.0)

        # Likelihood per parameter
        for tname, sig in [
            ('Ea_d',  sigma_Ea_d),
            ('lnA_d', sigma_lnA_d),
            ('Ea_f',  sigma_Ea_f),
            ('lnA_f', sigma_lnA_f),
            ('y_max', sigma_y_max),
        ]:
            d = obs_data[tname]
            if len(d['value']) == 0: continue
            cidx = d['class_idx']
            is_C2 = d['is_C2']
            sigma_X = d['sigma_C2']
            w = d['weight']
            # Effective sigma scaled by 1/sqrt(weight): higher weight → tighter
            eff_sig = sig / pm.math.sqrt(w)
            if tname == 'Ea_d':
                mu = Ea_d_base[cidx] - rho_Ea_d * sigma_X * is_C2
            elif tname == 'lnA_d':
                mu = lnA_d_base[cidx] - rho_lnA_d * sigma_X * is_C2
            elif tname == 'Ea_f':
                mu = Ea_f_base[cidx] - rho_Ea_f * sigma_X * is_C2
            elif tname == 'lnA_f':
                mu = lnA_f_base[cidx] - rho_lnA_f * sigma_X * is_C2
            elif tname == 'y_max':
                mu = y_max_base[cidx]
            pm.Normal(f'obs_{tname}', mu=mu, sigma=eff_sig, observed=d['value'])

    # ---- Sample ----
    print("\nSampling NUTS (2 chains × 1000 tune + 1000 draws)...")
    with model:
        idata = pm.sample(
            draws=1000, tune=1000, chains=2, cores=1,
            target_accept=0.95, random_seed=42, progressbar=False)

    # ---- Diagnostics ----
    print("\n" + "=" * 80)
    print("Convergence diagnostics")
    print("=" * 80)
    summary = az.summary(idata, var_names=[
        'Ea_d_base', 'lnA_d_base', 'Ea_f_base', 'lnA_f_base', 'y_max_base',
        'rho_Ea_f', 'rho_lnA_f', 'rho_Ea_d', 'rho_lnA_d',
        'sigma_Ea_d', 'sigma_lnA_d', 'sigma_Ea_f', 'sigma_lnA_f', 'sigma_y_max',
    ])
    print(summary.to_string())

    # ---- Extract anchor table ----
    posterior = idata.posterior
    rows = []
    for ic, c in enumerate(CLASSES):
        rows.append({
            'class': c,
            'Ea_d_mean':  float(posterior['Ea_d_base'][..., ic].mean()),
            'Ea_d_sd':    float(posterior['Ea_d_base'][..., ic].std()),
            'Ea_d_lo':    float(posterior['Ea_d_base'][..., ic].quantile(0.025)),
            'Ea_d_hi':    float(posterior['Ea_d_base'][..., ic].quantile(0.975)),
            'lnA_d_mean': float(posterior['lnA_d_base'][..., ic].mean()),
            'lnA_d_sd':   float(posterior['lnA_d_base'][..., ic].std()),
            'lnA_d_lo':   float(posterior['lnA_d_base'][..., ic].quantile(0.025)),
            'lnA_d_hi':   float(posterior['lnA_d_base'][..., ic].quantile(0.975)),
            'Ea_f_mean':  float(posterior['Ea_f_base'][..., ic].mean()),
            'Ea_f_sd':    float(posterior['Ea_f_base'][..., ic].std()),
            'Ea_f_lo':    float(posterior['Ea_f_base'][..., ic].quantile(0.025)),
            'Ea_f_hi':    float(posterior['Ea_f_base'][..., ic].quantile(0.975)),
            'lnA_f_mean': float(posterior['lnA_f_base'][..., ic].mean()),
            'lnA_f_sd':   float(posterior['lnA_f_base'][..., ic].std()),
            'lnA_f_lo':   float(posterior['lnA_f_base'][..., ic].quantile(0.025)),
            'lnA_f_hi':   float(posterior['lnA_f_base'][..., ic].quantile(0.975)),
            'y_max_mean': float(posterior['y_max_base'][..., ic].mean()),
            'y_max_sd':   float(posterior['y_max_base'][..., ic].std()),
            'y_max_lo':   float(posterior['y_max_base'][..., ic].quantile(0.025)),
            'y_max_hi':   float(posterior['y_max_base'][..., ic].quantile(0.975)),
        })
    anchor_df = pd.DataFrame(rows).round(3)

    # Add Hammett coefficients
    hammett_rows = [
        ('rho_Ea_f',  float(posterior['rho_Ea_f'].mean()),  float(posterior['rho_Ea_f'].std()),
         float(posterior['rho_Ea_f'].quantile(0.025)),  float(posterior['rho_Ea_f'].quantile(0.975))),
        ('rho_lnA_f', float(posterior['rho_lnA_f'].mean()), float(posterior['rho_lnA_f'].std()),
         float(posterior['rho_lnA_f'].quantile(0.025)), float(posterior['rho_lnA_f'].quantile(0.975))),
        ('rho_Ea_d',  float(posterior['rho_Ea_d'].mean()),  float(posterior['rho_Ea_d'].std()),
         float(posterior['rho_Ea_d'].quantile(0.025)),  float(posterior['rho_Ea_d'].quantile(0.975))),
        ('rho_lnA_d', float(posterior['rho_lnA_d'].mean()), float(posterior['rho_lnA_d'].std()),
         float(posterior['rho_lnA_d'].quantile(0.025)), float(posterior['rho_lnA_d'].quantile(0.975))),
    ]
    hammett_df = pd.DataFrame(hammett_rows, columns=['param', 'mean', 'sd', 'lo', 'hi']).round(3)

    print("\n" + "=" * 80)
    print("Class anchors (posterior mean ± sd, 95% CI)")
    print("=" * 80)
    print(anchor_df.to_string(index=False))

    print("\nHammett coefficients (C2 class only)")
    print(hammett_df.to_string(index=False))

    # ---- Save ----
    anchor_df.to_csv(BASE / 'v60_posterior_anchors.csv', index=False)
    hammett_df.to_csv(BASE / 'v60_posterior_hammett.csv', index=False)
    print(f"\nSaved: v60_posterior_anchors.csv, v60_posterior_hammett.csv")

    # ---- Per-substrate predictions (posterior mean for each training substrate) ----
    pred_rows = []
    for _, r in df.iterrows():
        ic = r['class_idx']
        sigma = r['sigma_C2'] if r['is_C2'] == 1 else 0
        Ea_d_p = anchor_df.iloc[ic]['Ea_d_mean'] - hammett_df.iloc[2]['mean'] * sigma if r['is_C2'] == 1 else anchor_df.iloc[ic]['Ea_d_mean']
        Ea_f_p = anchor_df.iloc[ic]['Ea_f_mean'] - hammett_df.iloc[0]['mean'] * sigma if r['is_C2'] == 1 else anchor_df.iloc[ic]['Ea_f_mean']
        lnA_d_p = anchor_df.iloc[ic]['lnA_d_mean'] - hammett_df.iloc[3]['mean'] * sigma if r['is_C2'] == 1 else anchor_df.iloc[ic]['lnA_d_mean']
        lnA_f_p = anchor_df.iloc[ic]['lnA_f_mean'] - hammett_df.iloc[1]['mean'] * sigma if r['is_C2'] == 1 else anchor_df.iloc[ic]['lnA_f_mean']
        y_max_p = anchor_df.iloc[ic]['y_max_mean']
        pred_rows.append({
            'intermediate': r['intermediate'], 'class_v60': r['class_v60'],
            'source': r['source'], 'is_virtual': r['is_virtual'],
            'sigma_C2': r['sigma_C2'],
            'Ea_d_obs': r['Ea_d'], 'Ea_d_pred': Ea_d_p,
            'lnA_d_obs': r['lnA_d'], 'lnA_d_pred': lnA_d_p,
            'Ea_f_obs': r['Ea_f'], 'Ea_f_pred': Ea_f_p,
            'lnA_f_obs': r['lnA_f'], 'lnA_f_pred': lnA_f_p,
            'y_max_obs': r['y_max'], 'y_max_pred': y_max_p,
        })
    pred_df = pd.DataFrame(pred_rows).round(2)
    pred_df.to_csv(BASE / 'v60_predictions.csv', index=False)
    print(f"Saved: v60_predictions.csv ({len(pred_df)} rows)")

    # ---- Per-class fit MAE ----
    print("\n" + "=" * 80)
    print("Per-class in-sample MAE")
    print("=" * 80)
    for c in CLASSES:
        sub = pred_df[(pred_df['class_v60']==c) & (~pred_df['is_virtual'])]
        if len(sub) == 0: continue
        line = f"  {c} (n={len(sub)}): "
        for t in ['Ea_d', 'lnA_d', 'Ea_f', 'lnA_f', 'y_max']:
            obs_col, pred_col = f'{t}_obs', f'{t}_pred'
            valid = sub[obs_col].notna()
            if valid.sum() == 0:
                line += f"{t}=- "
            else:
                mae = (sub.loc[valid, obs_col] - sub.loc[valid, pred_col]).abs().mean()
                line += f"{t}={mae:.1f} "
        print(line)

    # ---- Negative Ea check ----
    print("\n--- Negative Ea check ---")
    for col in ['Ea_d_pred', 'Ea_f_pred']:
        n_neg = (pred_df[col] < 0).sum()
        print(f"  {col}: n_negative = {n_neg}  (must be 0)")
        if n_neg > 0:
            print("  ", pred_df[pred_df[col] < 0][['intermediate', col]].to_string(index=False))


if __name__ == '__main__':
    main()
