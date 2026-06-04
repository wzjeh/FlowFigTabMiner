"""
v6.2: σ + EWG_type + mol_volume Bayesian model for C2 class.

Based on screen_v6_baseline_plus.py: adding mol_volume to baseline gives:
  Ea_d:  LOO R² +0.24 → +0.50  (Δ=+0.26)
  lnA_d: LOO R² +0.75 → +0.85  (Δ=+0.10)
  lnA_f: LOO R² -0.18 → +0.69  (Δ=+0.87)

For Ea_f and y_max (no descriptor signal), keep v6.1 categorical-only.

Compare predictions on 5-Br-2-F-CN against v6.0/v6.1.
"""
import scipy.signal, scipy.signal.windows
scipy.signal.gaussian = scipy.signal.windows.gaussian

from pathlib import Path
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from rdkit import Chem

BASE = Path(__file__).parent
np.random.seed(42)

TEST_5Br2FCN = {
    'sigma_p_sum': 0.06, 'sigma_m_sum': 0.56,
    'is_CN': 1, 'is_NO2': 0, 'is_ester': 0,
    'mol_volume': 139.33,  # from compute_fbrcn_descriptors.py
}


def canon_smi(s):
    try:
        m = Chem.MolFromSmiles(str(s))
        return Chem.MolToSmiles(m, canonical=True) if m else None
    except: return None


def main():
    # Load C2 in-house with mol_volume
    df = pd.read_csv(BASE / 'v60_training_set.csv')
    df = df[(df['class_v60']=='C2') & (~df['is_virtual'])].copy()
    df['smi_can'] = df['smi'].apply(canon_smi)
    master = pd.read_csv(BASE / 'intermediates_master.csv')
    master['smi_can'] = master['intermediate_smiles_canonical'].apply(canon_smi)
    df = df.merge(master[['smi_can', 'mol_volume']].drop_duplicates('smi_can'),
                  on='smi_can', how='left')
    df = df.dropna(subset=['mol_volume']).reset_index(drop=True)
    df['is_CN'] = (df['ewg_type']=='CN').astype(int)
    df['is_NO2'] = (df['ewg_type']=='NO2').astype(int)
    df['is_ester'] = (df['ewg_type']=='ester').astype(int)
    df['sigma_eff'] = df['sigma_p_sum'].fillna(0) + df['sigma_m_sum'].fillna(0)
    # Center mol_volume for stable sampling
    vol_mean = df['mol_volume'].mean()
    df['vol_centered'] = df['mol_volume'] - vol_mean
    print(f"C2 in-house with mol_volume: n={len(df)}")
    print(f"  mol_volume range: [{df['mol_volume'].min():.1f}, {df['mol_volume'].max():.1f}]  (mean={vol_mean:.1f})")
    print(df['ewg_type'].value_counts().to_string())

    # ============================================================
    # Bayesian model: σ + EWG_type + mol_volume
    # ============================================================
    with pm.Model() as model:
        # Base
        base_Ea_d  = pm.Normal('base_Ea_d',  mu=50.0, sigma=15.0)
        base_lnA_d = pm.Normal('base_lnA_d', mu=22.0, sigma=10.0)
        base_lnA_f = pm.Normal('base_lnA_f', mu=22.0, sigma=8.0)

        # EWG offsets (vs ester ~ baseline)
        a_CN_Ea_d  = pm.Normal('a_CN_Ea_d',  mu=-25.0, sigma=10.0)
        a_NO2_Ea_d = pm.Normal('a_NO2_Ea_d', mu=  0.0, sigma=10.0)
        a_es_Ea_d  = pm.Normal('a_es_Ea_d',  mu=  0.0, sigma=10.0)
        a_CN_lnA_d  = pm.Normal('a_CN_lnA_d',  mu=-13.0, sigma=8.0)
        a_NO2_lnA_d = pm.Normal('a_NO2_lnA_d', mu=  0.0, sigma=8.0)
        a_es_lnA_d  = pm.Normal('a_es_lnA_d',  mu=  0.0, sigma=8.0)
        a_CN_lnA_f  = pm.Normal('a_CN_lnA_f',  mu=  0.0, sigma=5.0)
        a_NO2_lnA_f = pm.Normal('a_NO2_lnA_f', mu= -3.0, sigma=5.0)
        a_es_lnA_f  = pm.Normal('a_es_lnA_f',  mu=  0.0, sigma=5.0)

        # Hammett ρ
        rho_Ea_d  = pm.Normal('rho_Ea_d',  mu= 0.0, sigma=15.0)
        rho_lnA_d = pm.Normal('rho_lnA_d', mu= 0.0, sigma=10.0)
        rho_lnA_f = pm.Normal('rho_lnA_f', mu=-15.0, sigma=10.0)

        # mol_volume coefficients (NEW in v6.2)
        gamma_vol_Ea_d  = pm.Normal('gamma_vol_Ea_d',  mu=0.0, sigma=1.0)
        gamma_vol_lnA_d = pm.Normal('gamma_vol_lnA_d', mu=0.0, sigma=0.5)
        gamma_vol_lnA_f = pm.Normal('gamma_vol_lnA_f', mu=0.0, sigma=0.5)

        # Noise
        sig_Ea_d  = pm.HalfNormal('sig_Ea_d',  sigma=10.0)
        sig_lnA_d = pm.HalfNormal('sig_lnA_d', sigma=5.0)
        sig_lnA_f = pm.HalfNormal('sig_lnA_f', sigma=5.0)

        # Arrays
        sigma_eff = df['sigma_eff'].to_numpy()
        is_CN = df['is_CN'].to_numpy()
        is_NO2 = df['is_NO2'].to_numpy()
        is_es = df['is_ester'].to_numpy()
        v = df['vol_centered'].to_numpy()

        # Predictions
        mu_Ea_d = (base_Ea_d + a_CN_Ea_d*is_CN + a_NO2_Ea_d*is_NO2 + a_es_Ea_d*is_es
                   + rho_Ea_d*sigma_eff + gamma_vol_Ea_d*v)
        mu_lnA_d = (base_lnA_d + a_CN_lnA_d*is_CN + a_NO2_lnA_d*is_NO2 + a_es_lnA_d*is_es
                    + rho_lnA_d*sigma_eff + gamma_vol_lnA_d*v)
        mu_lnA_f = (base_lnA_f + a_CN_lnA_f*is_CN + a_NO2_lnA_f*is_NO2 + a_es_lnA_f*is_es
                    + rho_lnA_f*sigma_eff + gamma_vol_lnA_f*v)

        pm.Normal('obs_Ea_d',  mu=mu_Ea_d,  sigma=sig_Ea_d,  observed=df['Ea_d'].to_numpy())
        pm.Normal('obs_lnA_d', mu=mu_lnA_d, sigma=sig_lnA_d, observed=df['lnA_d'].to_numpy())
        pm.Normal('obs_lnA_f', mu=mu_lnA_f, sigma=sig_lnA_f, observed=df['lnA_f'].to_numpy())

    print("\nSampling v6.2...")
    with model:
        idata = pm.sample(draws=1500, tune=1500, chains=2, cores=1,
                          target_accept=0.95, random_seed=42, progressbar=False)

    # Posterior summary
    var_names = ['base_Ea_d', 'a_CN_Ea_d', 'a_NO2_Ea_d', 'a_es_Ea_d',
                 'rho_Ea_d', 'gamma_vol_Ea_d',
                 'base_lnA_d', 'a_CN_lnA_d', 'a_NO2_lnA_d', 'a_es_lnA_d',
                 'rho_lnA_d', 'gamma_vol_lnA_d',
                 'base_lnA_f', 'a_CN_lnA_f', 'a_NO2_lnA_f', 'a_es_lnA_f',
                 'rho_lnA_f', 'gamma_vol_lnA_f']
    print("\n" + "=" * 80)
    print("v6.2 posterior summary")
    print("=" * 80)
    print(az.summary(idata, var_names=var_names)[['mean','sd','hdi_3%','hdi_97%','r_hat']].round(2).to_string())

    pm_means = {v: float(idata.posterior[v].mean()) for v in var_names}

    # ---- Predict 5-Br-2-F-CN ----
    t = TEST_5Br2FCN
    sigma_t = t['sigma_p_sum'] + t['sigma_m_sum']
    v_t = t['mol_volume'] - vol_mean

    Ea_d = (pm_means['base_Ea_d'] + pm_means['a_CN_Ea_d']*t['is_CN']
            + pm_means['a_NO2_Ea_d']*t['is_NO2'] + pm_means['a_es_Ea_d']*t['is_ester']
            + pm_means['rho_Ea_d']*sigma_t + pm_means['gamma_vol_Ea_d']*v_t)
    lnA_d = (pm_means['base_lnA_d'] + pm_means['a_CN_lnA_d']*t['is_CN']
             + pm_means['a_NO2_lnA_d']*t['is_NO2'] + pm_means['a_es_lnA_d']*t['is_ester']
             + pm_means['rho_lnA_d']*sigma_t + pm_means['gamma_vol_lnA_d']*v_t)
    lnA_f = (pm_means['base_lnA_f'] + pm_means['a_CN_lnA_f']*t['is_CN']
             + pm_means['a_NO2_lnA_f']*t['is_NO2'] + pm_means['a_es_lnA_f']*t['is_ester']
             + pm_means['rho_lnA_f']*sigma_t + pm_means['gamma_vol_lnA_f']*v_t)

    # Ea_f, y_max: keep v6.1 anchors (no improvement found)
    # Use C2/CN sub-anchor values from v6.0
    anchors = pd.read_csv(BASE / 'v60_subclass_anchors.csv')
    cn_anchor = anchors[(anchors['class']=='C2') & (anchors['ewg_type']=='CN')].iloc[0]
    Ea_f = float(cn_anchor['Ea_f_mean'])
    y_max = float(cn_anchor['y_max_mean'])

    print("\n" + "=" * 80)
    print("v6.2 prediction for 5-Br-2-F-CN")
    print("=" * 80)
    print(f"  Ea_d  = {Ea_d:.2f}")
    print(f"  lnA_d = {lnA_d:.2f}")
    print(f"  Ea_f  = {Ea_f:.2f}  (kept from C2/CN sub-anchor)")
    print(f"  lnA_f = {lnA_f:.2f}")
    print(f"  y_max = {y_max:.2f}  (kept from C2/CN sub-anchor)")

    # ---- Yield surface MAE on 5-Br-2-F-CN ----
    R_GAS = 8.314e-3
    pred = dict(Ea_f=Ea_f, lnA_f=lnA_f, Ea_d=Ea_d, lnA_d=lnA_d, y_max=y_max)
    def yld(p, tR, T_C):
        Tk = T_C + 273.15
        kf = np.exp(p['lnA_f'] - p['Ea_f']/(R_GAS*Tk))
        kd = np.exp(p['lnA_d'] - p['Ea_d']/(R_GAS*Tk))
        return p['y_max'] * (1 - np.exp(-kf*tR)) * np.exp(-kd*tR)

    exp = pd.read_csv(BASE / 'experiment_fbrcn_summary.csv')
    exp['pred_v62'] = exp.apply(lambda r: yld(pred, r['tR_s'], r['T_C']), axis=1)
    exp['resid_v62'] = exp['yield_pct'] - exp['pred_v62']
    mae = exp['resid_v62'].abs().mean()
    bias = exp['resid_v62'].mean()

    print("\n" + "=" * 80)
    print("Blind validation on 5-Br-2-F-CN (25 points)")
    print("=" * 80)
    print(f"  v6.0 sub-anchor: MAE = 13.23 pp,  bias = -6.44 pp")
    print(f"  v6.1 unified:    MAE = 12.16 pp,  bias = -9.90 pp")
    print(f"  v6.2 +mol_vol:   MAE = {mae:.2f} pp,  bias = {bias:+.2f} pp")
    print(f"  Δ(MAE) vs v6.0: {mae - 13.23:+.2f} pp")
    print(f"  Δ(MAE) vs v6.1: {mae - 12.16:+.2f} pp")

    # ---- Save ----
    out = pd.DataFrame([{
        'model': 'v6.2 σ+EWG_type+mol_vol', 'class': 'C2', 'ewg_type': 'CN',
        'Ea_d': round(Ea_d, 2), 'lnA_d': round(lnA_d, 2),
        'Ea_f': round(Ea_f, 2), 'lnA_f': round(lnA_f, 2), 'y_max': round(y_max, 2),
        'MAE_5Br2FCN': round(mae, 2), 'bias_5Br2FCN': round(bias, 2),
    }])
    out.to_csv(BASE / 'v62_prediction_5Br2FCN.csv', index=False)
    print(f"\nSaved: v62_prediction_5Br2FCN.csv")

    # Posterior coefficients
    pd.DataFrame([pm_means]).to_csv(BASE / 'v62_posterior.csv', index=False)
    print(f"Saved: v62_posterior.csv")


if __name__ == '__main__':
    main()
