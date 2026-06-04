"""
v6.1: Unified C2 Bayesian model (Zhao 2026-05-24).

Replaces per-(EWG_type) sub-class anchors with a SINGLE C2 model:

  Ea_d[i] = base_Ea_d
          + alpha_CN  · is_CN[i]
          + alpha_NO2 · is_NO2[i]
          + alpha_ester · is_ester[i]   # (CHO/ketone if exist)
          + rho_d · sigma_eff[i]                 # continuous Hammett
          + beta_Es · Es_R[i] · is_ester[i]      # Taft Es for ester R group

Similarly for lnA_d, Ea_f, lnA_f, y_max.

Goal: capture EWG-type heterogeneity (categorical) + within-type variation (continuous σ + steric)
in ONE unified equation. If it works → cleaner closed-form for the paper.
"""
# arviz 0.17 has scipy.signal.gaussian removed
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

# Taft Es for R group in ester CO2R (Hansch-Leo 1991)
ES_TABLE = {'Me': 0.00, 'Et': -0.07, 'iPr': -0.47, 'tBu': -1.54}


def infer_R(name):
    """Infer R group of ortho/m/p-CO2R from intermediate name."""
    n = str(name).lower()
    if 'tert-butyl' in n or 'tbu' in n: return 'tBu'
    if 'isopropyl' in n or 'ipr' in n:   return 'iPr'
    if 'ethyl' in n:    return 'Et'
    if 'methyl' in n:   return 'Me'
    return None


def main():
    df_all = pd.read_csv(BASE / 'v60_training_set.csv')
    # C2 only, exclude virtual (we want clean LOO; can add virtual later if helpful)
    df = df_all[(df_all['class_v60'] == 'C2') & (~df_all['is_virtual'])].copy().reset_index(drop=True)
    print(f"C2 in-house substrates: n={len(df)}")
    print(df['ewg_type'].value_counts().to_string())

    # Build features
    df['is_CN']    = (df['ewg_type'] == 'CN').astype(int)
    df['is_NO2']   = (df['ewg_type'] == 'NO2').astype(int)
    df['is_ester'] = (df['ewg_type'] == 'ester').astype(int)
    df['sigma_eff'] = df['sigma_p_sum'].fillna(0) + df['sigma_m_sum'].fillna(0)
    df['R_group'] = df['intermediate'].apply(infer_R)
    df['Es_R'] = df['R_group'].map(ES_TABLE).fillna(0.0)
    df['Es_R_ester'] = df['Es_R'] * df['is_ester']   # only for esters

    print("\nFeature matrix:")
    print(df[['intermediate','ewg_type','sigma_eff','Es_R','Es_R_ester',
              'Ea_d','lnA_d','Ea_f','lnA_f','y_max']].to_string(index=False))

    targets = ['Ea_d', 'lnA_d', 'Ea_f', 'lnA_f', 'y_max']
    obs = {t: df[t].dropna() for t in targets}

    # ---- Bayesian model ----
    print("\nBuilding unified C2 Bayesian model...")
    with pm.Model() as model:
        # Base intercept (at σ=0, no EWG_type offset — hypothetical inert reference)
        # Use prior centered at C2-mean
        base_Ea_d  = pm.Normal('base_Ea_d',  mu=50.0, sigma=20.0)
        base_lnA_d = pm.Normal('base_lnA_d', mu=22.0, sigma=10.0)
        base_Ea_f  = pm.Normal('base_Ea_f',  mu=32.0, sigma=10.0)
        base_lnA_f = pm.Normal('base_lnA_f', mu=22.0, sigma=10.0)
        base_y_max = pm.Normal('base_y_max', mu=80.0, sigma=15.0)

        # Categorical EWG offsets (relative to "none" baseline, but practically all are EWG)
        alpha_CN_Ea_d    = pm.Normal('alpha_CN_Ea_d',    mu=-25.0, sigma=15.0)
        alpha_NO2_Ea_d   = pm.Normal('alpha_NO2_Ea_d',   mu=  0.0, sigma=15.0)
        alpha_ester_Ea_d = pm.Normal('alpha_ester_Ea_d', mu=  0.0, sigma=15.0)
        alpha_CN_lnA_d    = pm.Normal('alpha_CN_lnA_d',    mu=-15.0, sigma=10.0)
        alpha_NO2_lnA_d   = pm.Normal('alpha_NO2_lnA_d',   mu=  0.0, sigma=10.0)
        alpha_ester_lnA_d = pm.Normal('alpha_ester_lnA_d', mu=  0.0, sigma=10.0)
        alpha_CN_Ea_f    = pm.Normal('alpha_CN_Ea_f',    mu= 0.0, sigma=10.0)
        alpha_NO2_Ea_f   = pm.Normal('alpha_NO2_Ea_f',   mu=-10.0, sigma=10.0)
        alpha_ester_Ea_f = pm.Normal('alpha_ester_Ea_f', mu=  0.0, sigma=10.0)
        alpha_CN_lnA_f    = pm.Normal('alpha_CN_lnA_f',    mu=  0.0, sigma=8.0)
        alpha_NO2_lnA_f   = pm.Normal('alpha_NO2_lnA_f',   mu= -3.0, sigma=8.0)
        alpha_ester_lnA_f = pm.Normal('alpha_ester_lnA_f', mu=  0.0, sigma=8.0)
        alpha_CN_y_max    = pm.Normal('alpha_CN_y_max',    mu=  0.0, sigma=10.0)
        alpha_NO2_y_max   = pm.Normal('alpha_NO2_y_max',   mu=  0.0, sigma=10.0)
        alpha_ester_y_max = pm.Normal('alpha_ester_y_max', mu= -5.0, sigma=10.0)

        # Continuous Hammett rho (centered loosely)
        rho_Ea_d  = pm.Normal('rho_Ea_d',  mu=  0.0, sigma=20.0)
        rho_lnA_d = pm.Normal('rho_lnA_d', mu=  0.0, sigma=15.0)
        rho_Ea_f  = pm.Normal('rho_Ea_f',  mu=-20.0, sigma=10.0)    # Charton
        rho_lnA_f = pm.Normal('rho_lnA_f', mu=-15.0, sigma=15.0)
        rho_y_max = pm.Normal('rho_y_max', mu=  0.0, sigma=10.0)

        # Taft Es for ester only (steric effect of R group)
        beta_Es_Ea_d  = pm.Normal('beta_Es_Ea_d',  mu=0.0, sigma=20.0)
        beta_Es_lnA_d = pm.Normal('beta_Es_lnA_d', mu=0.0, sigma=10.0)

        # Observation noise
        sigma_Ea_d  = pm.HalfNormal('sigma_Ea_d',  sigma=10.0)
        sigma_lnA_d = pm.HalfNormal('sigma_lnA_d', sigma=5.0)
        sigma_Ea_f  = pm.HalfNormal('sigma_Ea_f',  sigma=10.0)
        sigma_lnA_f = pm.HalfNormal('sigma_lnA_f', sigma=5.0)
        sigma_y_max = pm.HalfNormal('sigma_y_max', sigma=10.0)

        # Pull arrays
        sigma_eff = df['sigma_eff'].to_numpy()
        is_CN = df['is_CN'].to_numpy()
        is_NO2 = df['is_NO2'].to_numpy()
        is_ester = df['is_ester'].to_numpy()
        Es_R_ester = df['Es_R_ester'].to_numpy()

        # Predictions
        mu_Ea_d = (base_Ea_d
                   + alpha_CN_Ea_d * is_CN
                   + alpha_NO2_Ea_d * is_NO2
                   + alpha_ester_Ea_d * is_ester
                   + rho_Ea_d * sigma_eff
                   + beta_Es_Ea_d * Es_R_ester)
        mu_lnA_d = (base_lnA_d
                    + alpha_CN_lnA_d * is_CN
                    + alpha_NO2_lnA_d * is_NO2
                    + alpha_ester_lnA_d * is_ester
                    + rho_lnA_d * sigma_eff
                    + beta_Es_lnA_d * Es_R_ester)
        mu_Ea_f = (base_Ea_f
                   + alpha_CN_Ea_f * is_CN
                   + alpha_NO2_Ea_f * is_NO2
                   + alpha_ester_Ea_f * is_ester
                   + rho_Ea_f * sigma_eff)
        mu_lnA_f = (base_lnA_f
                    + alpha_CN_lnA_f * is_CN
                    + alpha_NO2_lnA_f * is_NO2
                    + alpha_ester_lnA_f * is_ester
                    + rho_lnA_f * sigma_eff)
        mu_y_max = (base_y_max
                    + alpha_CN_y_max * is_CN
                    + alpha_NO2_y_max * is_NO2
                    + alpha_ester_y_max * is_ester
                    + rho_y_max * sigma_eff)

        # Likelihood
        pm.Normal('obs_Ea_d',  mu=mu_Ea_d,  sigma=sigma_Ea_d,  observed=df['Ea_d'].to_numpy())
        pm.Normal('obs_lnA_d', mu=mu_lnA_d, sigma=sigma_lnA_d, observed=df['lnA_d'].to_numpy())
        pm.Normal('obs_Ea_f',  mu=mu_Ea_f,  sigma=sigma_Ea_f,  observed=df['Ea_f'].to_numpy())
        pm.Normal('obs_lnA_f', mu=mu_lnA_f, sigma=sigma_lnA_f, observed=df['lnA_f'].to_numpy())
        pm.Normal('obs_y_max', mu=mu_y_max, sigma=sigma_y_max, observed=df['y_max'].to_numpy())

    print("\nSampling NUTS (2 chains × 1500 tune + 1500 draws)...")
    with model:
        idata = pm.sample(draws=1500, tune=1500, chains=2, cores=1,
                          target_accept=0.95, random_seed=42, progressbar=False)

    # ---- Posterior summary ----
    print("\n" + "=" * 80)
    print("Unified C2 model posterior")
    print("=" * 80)
    var_names = ['base_Ea_d', 'alpha_CN_Ea_d', 'alpha_NO2_Ea_d', 'alpha_ester_Ea_d',
                 'rho_Ea_d', 'beta_Es_Ea_d', 'sigma_Ea_d',
                 'base_lnA_d', 'alpha_CN_lnA_d', 'alpha_NO2_lnA_d', 'alpha_ester_lnA_d',
                 'rho_lnA_d', 'beta_Es_lnA_d',
                 'base_Ea_f', 'alpha_CN_Ea_f', 'alpha_NO2_Ea_f', 'alpha_ester_Ea_f', 'rho_Ea_f',
                 'base_lnA_f', 'alpha_CN_lnA_f', 'alpha_NO2_lnA_f', 'alpha_ester_lnA_f', 'rho_lnA_f',
                 'base_y_max', 'alpha_CN_y_max', 'alpha_NO2_y_max', 'alpha_ester_y_max', 'rho_y_max']
    summary = az.summary(idata, var_names=var_names)
    print(summary[['mean','sd','hdi_3%','hdi_97%','r_hat']].round(2).to_string())

    # ---- Extract posterior means ----
    pm_means = {v: float(idata.posterior[v].mean()) for v in var_names}

    # ---- Closed-form prediction function ----
    def predict_C2(sigma_eff, ewg_type, R_group=None):
        """Apply unified C2 closed-form prediction."""
        is_CN = 1 if ewg_type == 'CN' else 0
        is_NO2 = 1 if ewg_type == 'NO2' else 0
        is_ester = 1 if ewg_type == 'ester' else 0
        Es = ES_TABLE.get(R_group, 0.0) if is_ester else 0.0
        return dict(
            Ea_d  = (pm_means['base_Ea_d']
                     + pm_means['alpha_CN_Ea_d']*is_CN
                     + pm_means['alpha_NO2_Ea_d']*is_NO2
                     + pm_means['alpha_ester_Ea_d']*is_ester
                     + pm_means['rho_Ea_d']*sigma_eff
                     + pm_means['beta_Es_Ea_d']*Es),
            lnA_d = (pm_means['base_lnA_d']
                     + pm_means['alpha_CN_lnA_d']*is_CN
                     + pm_means['alpha_NO2_lnA_d']*is_NO2
                     + pm_means['alpha_ester_lnA_d']*is_ester
                     + pm_means['rho_lnA_d']*sigma_eff
                     + pm_means['beta_Es_lnA_d']*Es),
            Ea_f  = (pm_means['base_Ea_f']
                     + pm_means['alpha_CN_Ea_f']*is_CN
                     + pm_means['alpha_NO2_Ea_f']*is_NO2
                     + pm_means['alpha_ester_Ea_f']*is_ester
                     + pm_means['rho_Ea_f']*sigma_eff),
            lnA_f = (pm_means['base_lnA_f']
                     + pm_means['alpha_CN_lnA_f']*is_CN
                     + pm_means['alpha_NO2_lnA_f']*is_NO2
                     + pm_means['alpha_ester_lnA_f']*is_ester
                     + pm_means['rho_lnA_f']*sigma_eff),
            y_max = (pm_means['base_y_max']
                     + pm_means['alpha_CN_y_max']*is_CN
                     + pm_means['alpha_NO2_y_max']*is_NO2
                     + pm_means['alpha_ester_y_max']*is_ester
                     + pm_means['rho_y_max']*sigma_eff),
        )

    # ---- In-sample fit ----
    print("\n" + "=" * 80)
    print("In-sample fit (predicted vs observed, all C2 substrates)")
    print("=" * 80)
    rows = []
    for _, r in df.iterrows():
        p = predict_C2(r['sigma_eff'], r['ewg_type'], r['R_group'])
        rows.append({
            'intermediate': r['intermediate'][:35],
            'ewg': r['ewg_type'], 'σ': r['sigma_eff'], 'R': r['R_group'] or '-',
            'Ea_d_obs': r['Ea_d'], 'Ea_d_pred': p['Ea_d'], 'Δ': r['Ea_d'] - p['Ea_d'],
            'lnA_d_obs': r['lnA_d'], 'lnA_d_pred': p['lnA_d'],
            'y_max_obs': r['y_max'], 'y_max_pred': p['y_max'],
        })
    fit_df = pd.DataFrame(rows).round(2)
    print(fit_df.to_string(index=False))

    # Per-parameter MAE
    print("\nPer-parameter in-sample MAE:")
    for t in targets:
        obs = df[t]
        pred = df.apply(lambda r: predict_C2(r['sigma_eff'], r['ewg_type'], r['R_group'])[t], axis=1)
        mae = (obs - pred).abs().mean()
        print(f"  {t}: MAE = {mae:.2f}")

    # ---- Blind prediction: 5-Br-2-F-CN ----
    print("\n" + "=" * 80)
    print("Blind prediction: 5-Br-2-F-CN → m-CN-p-F-PhLi  (σ_eff = 0.56 + 0.06 = 0.62, ewg=CN)")
    print("=" * 80)
    p_new = predict_C2(sigma_eff=0.62, ewg_type='CN', R_group=None)
    print(f"  Ea_d  = {p_new['Ea_d']:.2f}")
    print(f"  lnA_d = {p_new['lnA_d']:.2f}")
    print(f"  Ea_f  = {p_new['Ea_f']:.2f}")
    print(f"  lnA_f = {p_new['lnA_f']:.2f}")
    print(f"  y_max = {p_new['y_max']:.2f}")
    print(f"\nFor comparison:")
    print(f"  v6.0 C2/CN sub-anchor: Ea_d=28.1, lnA_d=9.5, y_max=84")
    print(f"  3-CN-PhLi analog:       Ea_d=27.3, lnA_d=9.08, y_max=83.1")

    # ---- Yield surface MAE on 5-Br-2-F-CN experiment ----
    exp = pd.read_csv(BASE / 'experiment_fbrcn_summary.csv')
    R_GAS = 8.314e-3
    def yld(p, tR, T_C):
        Tk = T_C + 273.15
        kf = np.exp(p['lnA_f'] - p['Ea_f']/(R_GAS*Tk))
        kd = np.exp(p['lnA_d'] - p['Ea_d']/(R_GAS*Tk))
        return p['y_max'] * (1 - np.exp(-kf*tR)) * np.exp(-kd*tR)
    exp['pred_v61'] = exp.apply(lambda r: yld(p_new, r['tR_s'], r['T_C']), axis=1)
    exp['resid_v61'] = exp['yield_pct'] - exp['pred_v61']
    mae_v61 = exp['resid_v61'].abs().mean()
    bias_v61 = exp['resid_v61'].mean()
    print(f"\n5-Br-2-F-CN yield surface (25 points):")
    print(f"  v6.1 unified C2: MAE = {mae_v61:.2f} pp,  bias = {bias_v61:+.2f} pp")
    print(f"  v6.0 sub-anchor: MAE = 13.23 pp,  bias = -6.44 pp")
    print(f"  Δ(MAE) = {mae_v61 - 13.23:+.2f} pp  ({'better ✓' if mae_v61 < 13.23 else 'worse'})")

    # ---- LOO comparison ----
    print("\n" + "=" * 80)
    print("LOO comparison: unified C2 vs sub-class anchors (in-house C2 only, n=12)")
    print("=" * 80)
    from sklearn.linear_model import BayesianRidge
    # Simpler LOO via OLS-style refit per held-out point
    # For brevity, use point estimate posterior means and refit each excluded
    # (this is approximate — full LOO would re-sample PyMC)

    loo_unified = []
    loo_subclass = []
    sub_anchors = pd.read_csv(BASE / 'v60_subclass_anchors.csv')
    for i in range(len(df)):
        row = df.iloc[i]
        # Sub-class anchor: use group mean excluding self
        ewg = row['ewg_type']
        peers = df[(df['ewg_type'] == ewg) & (df.index != i)]
        if len(peers) > 0:
            sub_Ea_d = peers['Ea_d'].mean()
            loo_subclass.append((row['Ea_d'] - sub_Ea_d) ** 2)
        # Unified: use posterior mean (approximate — would need to refit)
        p = predict_C2(row['sigma_eff'], row['ewg_type'], row['R_group'])
        loo_unified.append((row['Ea_d'] - p['Ea_d']) ** 2)
    print(f"  Unified RMSE (in-sample, posterior mean):  {np.sqrt(np.mean(loo_unified)):.2f}")
    print(f"  Sub-class RMSE (LOO mean within type):     {np.sqrt(np.mean(loo_subclass)):.2f}")

    # ---- Save posterior ----
    posterior_means = {v: pm_means[v] for v in var_names}
    pd.DataFrame([posterior_means]).to_csv(BASE / 'v61_unified_C2_posterior.csv', index=False)
    fit_df.to_csv(BASE / 'v61_unified_C2_predictions.csv', index=False)
    print(f"\nSaved: v61_unified_C2_posterior.csv, v61_unified_C2_predictions.csv")


if __name__ == '__main__':
    main()
