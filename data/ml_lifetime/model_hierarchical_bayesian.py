"""
Stage 2: Hierarchical Bayesian Model for Organolithium Lifetime Prediction

Three-level model:
  Level 1 (substrate): Ea_i, lnA_i ~ Normal(β·desc_i, σ)
  Level 2 (Arrhenius): ln(k_d)_ij = lnA_i - Ea_i/(R·T_j)
  Level 3 (observation): observed ln(k_d) ~ Normal(predicted, σ_obs)

Fits on phase_a k_d data (15 substrates, 53 temperature-specific observations).
Predicts t½ with 95% credible intervals for all 97 substrates with DFT descriptors.

Usage:
  cd /Users/zhaowenyuan/Projects/FlowFigTabMiner
  source flowfigtabminer/bin/activate
  python data/ml_lifetime/model_hierarchical_bayesian.py
"""

import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Fix scipy.signal.gaussian removal in scipy>=1.13
from scipy.signal.windows import gaussian as _gaussian
import scipy.signal
scipy.signal.gaussian = _gaussian

import pymc as pm
import arviz as az
from pathlib import Path
from rdkit import Chem

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).resolve().parent
FIG_DIR = DATA_DIR / "analysis_figures"
FIG_DIR.mkdir(exist_ok=True)

R_GAS = 8.314e-3  # kJ/(mol·K)


# ── 1. Load and prepare data ─────────────────────────────────────────

def load_kd_data():
    """Load phase_a k_d observations and merge with DFT descriptors.

    Only includes substrates that are Tier 1 in phase_b_arrhenius.csv.
    """
    ha = pd.read_csv(DATA_DIR / "phase_a_halflives.csv")
    desc = pd.read_csv(DATA_DIR / "clean_organolithium_unified_descriptors.csv")
    cache_path = DATA_DIR / "xtb_cache.json"

    # Filter to Tier 1 substrates only
    arr = pd.read_csv(DATA_DIR / "phase_b_arrhenius.csv")
    if "quality_tier" in arr.columns:
        tier1_smiles_raw = set(arr[arr.quality_tier == "tier1"].intermediate_smiles)
        tier1_can = set()
        for s in tier1_smiles_raw:
            m = Chem.MolFromSmiles(s)
            if m:
                tier1_can.add(Chem.MolToSmiles(m))
    else:
        tier1_can = None  # no filter

    # Canonicalize SMILES
    ha["can_smi"] = ha.intermediate_smiles.apply(
        lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s)) if Chem.MolFromSmiles(s) else s
    )

    # Filter to Tier 1 substrates
    if tier1_can is not None:
        ha = ha[ha.can_smi.isin(tier1_can)].copy()

    # Keep only rows with k_d > 0
    kd = ha[ha.k_d.notna() & (ha.k_d > 0)].copy()
    kd["log_kd"] = np.log(kd["k_d"])
    kd["T_K"] = kd["T_C"] + 273.15
    kd["inv_RT"] = 1.0 / (R_GAS * kd["T_K"])  # 1/(kJ/mol)

    # Get unique descriptors per canonical SMILES
    desc_unique = desc.drop_duplicates(subset="intermediate_smiles_canonical")

    # Merge
    kd_merged = kd.merge(desc_unique, left_on="can_smi",
                         right_on="intermediate_smiles_canonical", how="left")

    # For unmatched: try xtb cache
    # After merge, intermediate_smiles may be renamed to intermediate_smiles_x
    smi_col = "intermediate_smiles_x" if "intermediate_smiles_x" in kd_merged.columns else "intermediate_smiles"
    if cache_path.exists():
        cache = json.load(open(cache_path))
        for idx, row in kd_merged.iterrows():
            if pd.isna(row.get("dft_dipole_D")):
                smi = row[smi_col]
                if smi in cache:
                    d = cache[smi]
                    kd_merged.at[idx, "dft_dipole_D"] = d.get("dipole_D")
                    kd_merged.at[idx, "dft_LiC_BDE_kJ"] = d.get("bde_kJ")
                    kd_merged.at[idx, "dft_Gsolv_kJ"] = d.get("gsolv_kJ")
                    kd_merged.at[idx, "dft_HOMO_eV"] = d.get("homo_eV")
                    kd_merged.at[idx, "dft_LUMO_eV"] = d.get("lumo_eV")
                    kd_merged.at[idx, "dft_charge_Li"] = d.get("charge_Li")
                    kd_merged.at[idx, "dft_charge_C_ipso"] = d.get("charge_C_ipso")
                    kd_merged.at[idx, "dft_LiC_bond_A"] = d.get("lic_bond_A")

    print(f"k_d observations: {len(kd_merged)}")
    print(f"  unique substrates: {kd_merged.can_smi.nunique()}")
    print(f"  with DFT dipole: {kd_merged.dft_dipole_D.notna().sum()}")
    print(f"  T range: {kd_merged.T_C.min():.0f} to {kd_merged.T_C.max():.0f} °C")

    return kd_merged


def prepare_features(kd_data, desc_df, features):
    """Prepare feature matrices for observed and all substrates.

    Returns:
      obs_data: dict with arrays for MCMC (substrate_idx, inv_RT, log_kd, X_obs)
      all_data: dict with arrays for prediction (X_all, smiles_all, class_all)
    """
    # Observed substrates (those with k_d data + DFT features)
    valid_mask = kd_data[features].notna().all(axis=1)
    kd_valid = kd_data[valid_mask].copy()

    # Substrate index mapping
    unique_smiles = kd_valid.can_smi.unique()
    smi_to_idx = {s: i for i, s in enumerate(unique_smiles)}
    kd_valid["sub_idx"] = kd_valid.can_smi.map(smi_to_idx)

    # Feature matrix for observed substrates (one row per unique substrate)
    X_obs_list = []
    for smi in unique_smiles:
        row = kd_valid[kd_valid.can_smi == smi].iloc[0]
        X_obs_list.append([row[f] for f in features])
    X_obs = np.array(X_obs_list, dtype=float)

    # Standardize features
    feat_mean = X_obs.mean(axis=0)
    feat_std = X_obs.std(axis=0)
    feat_std[feat_std == 0] = 1.0
    X_obs_z = (X_obs - feat_mean) / feat_std

    # All substrates for prediction
    desc_unique = desc_df.drop_duplicates(subset="intermediate_smiles_canonical")
    valid_all = desc_unique[features].notna().all(axis=1)
    all_subs = desc_unique[valid_all].copy()
    X_all = all_subs[features].values.astype(float)
    X_all_z = (X_all - feat_mean) / feat_std

    obs_data = {
        "sub_idx": kd_valid["sub_idx"].values.astype(int),
        "inv_RT": kd_valid["inv_RT"].values,
        "log_kd": kd_valid["log_kd"].values,
        "X_obs": X_obs_z,
        "n_obs_subs": len(unique_smiles),
        "n_obs": len(kd_valid),
        "obs_smiles": unique_smiles,
        "kd_valid": kd_valid,
    }

    all_data = {
        "X_all": X_all_z,
        "smiles_all": all_subs["intermediate_smiles_canonical"].values,
        "class_all": all_subs["intermediate_class"].values,
        "feat_mean": feat_mean,
        "feat_std": feat_std,
    }

    return obs_data, all_data


# ── 2. Hierarchical Bayesian model ───────────────────────────────────

def build_and_sample(obs_data, features, n_samples=2000, n_tune=2000, target_accept=0.9):
    """Build hierarchical model and run NUTS sampling."""
    n_subs = obs_data["n_obs_subs"]
    n_feats = len(features)
    X = obs_data["X_obs"]  # (n_subs, n_feats), standardized
    sub_idx = obs_data["sub_idx"]
    inv_RT = obs_data["inv_RT"]
    log_kd = obs_data["log_kd"]

    print(f"\nBuilding model: {n_subs} substrates, {n_feats} features, {len(log_kd)} observations")

    with pm.Model() as model:
        # ── Hyperpriors (regression coefficients) ──
        # Ea = beta_Ea_0 + X @ beta_Ea  (in kJ/mol)
        beta_Ea_0 = pm.Normal("beta_Ea_0", mu=35, sigma=20)
        beta_Ea = pm.Normal("beta_Ea", mu=0, sigma=15, shape=n_feats)

        # lnA = beta_lnA_0 + X @ beta_lnA
        beta_lnA_0 = pm.Normal("beta_lnA_0", mu=15, sigma=15)
        beta_lnA = pm.Normal("beta_lnA", mu=0, sigma=10, shape=n_feats)

        # Residual variance across substrates
        sigma_Ea = pm.HalfNormal("sigma_Ea", sigma=15)
        sigma_lnA = pm.HalfNormal("sigma_lnA", sigma=10)

        # ── Substrate-level (partially pooled) ──
        mu_Ea = beta_Ea_0 + pm.math.dot(X, beta_Ea)
        mu_lnA = beta_lnA_0 + pm.math.dot(X, beta_lnA)

        Ea = pm.Normal("Ea", mu=mu_Ea, sigma=sigma_Ea, shape=n_subs)
        lnA = pm.Normal("lnA", mu=mu_lnA, sigma=sigma_lnA, shape=n_subs)

        # ── Arrhenius link ──
        # ln(k_d) = lnA - Ea / (R*T) = lnA - Ea * inv_RT
        predicted_log_kd = lnA[sub_idx] - Ea[sub_idx] * inv_RT

        # ── Observation noise ──
        sigma_obs = pm.HalfNormal("sigma_obs", sigma=2.0)
        pm.Normal("obs_log_kd", mu=predicted_log_kd, sigma=sigma_obs,
                  observed=log_kd)

    print("Sampling...")
    with model:
        trace = pm.sample(
            draws=n_samples,
            tune=n_tune,
            target_accept=target_accept,
            cores=2,
            chains=4,
            return_inferencedata=True,
            progressbar=True,
            random_seed=42,
        )

    return model, trace


# ── 3. Posterior prediction ───────────────────────────────────────────

def predict_new_substrates(trace, all_data, features, temps_C=[-78, -40, 0, 25]):
    """Predict Ea, lnA, and t½ for all substrates using posterior samples."""
    X_all = all_data["X_all"]  # (n_all, n_feats)
    n_all = len(X_all)

    # Extract posterior samples
    beta_Ea_0 = trace.posterior["beta_Ea_0"].values.flatten()
    beta_Ea = trace.posterior["beta_Ea"].values.reshape(-1, len(features))
    beta_lnA_0 = trace.posterior["beta_lnA_0"].values.flatten()
    beta_lnA = trace.posterior["beta_lnA"].values.reshape(-1, len(features))
    sigma_Ea = trace.posterior["sigma_Ea"].values.flatten()
    sigma_lnA = trace.posterior["sigma_lnA"].values.flatten()

    n_samples = len(beta_Ea_0)

    # For new substrates, draw from predictive prior
    # Ea_new ~ Normal(beta_Ea_0 + X @ beta_Ea, sigma_Ea)
    # beta_Ea: (n_samples, n_feats), X_all: (n_all, n_feats)
    # Result: (n_samples, n_all)
    mu_Ea_all = beta_Ea_0[:, None] + (X_all @ beta_Ea.T).T
    mu_lnA_all = beta_lnA_0[:, None] + (X_all @ beta_lnA.T).T

    rng = np.random.default_rng(42)
    Ea_samples = mu_Ea_all + sigma_Ea[:, None] * rng.standard_normal((n_samples, n_all))
    lnA_samples = mu_lnA_all + sigma_lnA[:, None] * rng.standard_normal((n_samples, n_all))

    results = []
    for i in range(n_all):
        rec = {
            "smiles": all_data["smiles_all"][i],
            "intermediate_class": all_data["class_all"][i],
            "Ea_mean": np.mean(Ea_samples[:, i]),
            "Ea_std": np.std(Ea_samples[:, i]),
            "Ea_q025": np.percentile(Ea_samples[:, i], 2.5),
            "Ea_q975": np.percentile(Ea_samples[:, i], 97.5),
            "lnA_mean": np.mean(lnA_samples[:, i]),
            "lnA_std": np.std(lnA_samples[:, i]),
        }

        for t_c in temps_C:
            t_k = t_c + 273.15
            log_kd = lnA_samples[:, i] - Ea_samples[:, i] / (R_GAS * t_k)
            kd = np.exp(log_kd)
            thalf = np.log(2) / kd

            rec[f"t_half_{t_c}C_median_s"] = np.median(thalf)
            rec[f"t_half_{t_c}C_q025_s"] = np.percentile(thalf, 2.5)
            rec[f"t_half_{t_c}C_q975_s"] = np.percentile(thalf, 97.5)
            rec[f"log10_t_half_{t_c}C_mean"] = np.mean(np.log10(thalf))
            rec[f"log10_t_half_{t_c}C_std"] = np.std(np.log10(thalf))

        results.append(rec)

    return pd.DataFrame(results), Ea_samples, lnA_samples


# ── 4. Diagnostics ───────────────────────────────────────────────────

def print_diagnostics(trace, obs_data, features):
    """Print MCMC diagnostics and posterior summaries."""
    print("\n═══ MCMC Diagnostics ═══")

    # R-hat and ESS
    summary = az.summary(trace, var_names=["beta_Ea_0", "beta_Ea", "beta_lnA_0", "beta_lnA",
                                            "sigma_Ea", "sigma_lnA", "sigma_obs"])
    print(summary.to_string())

    # Check convergence
    rhat_max = summary["r_hat"].max()
    ess_min = summary["ess_bulk"].min()
    print(f"\n  Max R-hat: {rhat_max:.4f} (want < 1.05)")
    print(f"  Min ESS:   {ess_min:.0f} (want > 200)")

    # Posterior Ea for observed substrates
    print("\n═══ Posterior Ea/lnA for observed substrates ═══")
    Ea_post = trace.posterior["Ea"].values.reshape(-1, obs_data["n_obs_subs"])
    lnA_post = trace.posterior["lnA"].values.reshape(-1, obs_data["n_obs_subs"])

    # Compare with Arrhenius fits
    arr = pd.read_csv(DATA_DIR / "phase_b_arrhenius.csv")
    arr["can_smi"] = arr.intermediate_smiles.apply(
        lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s)) if Chem.MolFromSmiles(s) else s
    )

    for i, smi in enumerate(obs_data["obs_smiles"]):
        ea_mean = Ea_post[:, i].mean()
        ea_std = Ea_post[:, i].std()
        lna_mean = lnA_post[:, i].mean()
        lna_std = lnA_post[:, i].std()

        arr_match = arr[arr.can_smi == smi]
        if len(arr_match) > 0:
            ea_true = arr_match.iloc[0]["Ea_decomp_kJ_mol"]
            lna_true = arr_match.iloc[0]["ln_A"]
            print(f"  {smi:50s}")
            print(f"    Ea:  posterior={ea_mean:.1f}±{ea_std:.1f}, Arrhenius={ea_true:.1f}")
            print(f"    lnA: posterior={lna_mean:.1f}±{lna_std:.1f}, Arrhenius={lna_true:.1f}")
        else:
            print(f"  {smi:50s}")
            print(f"    Ea:  posterior={ea_mean:.1f}±{ea_std:.1f}  (no Arrhenius fit)")
            print(f"    lnA: posterior={lna_mean:.1f}±{lna_std:.1f}")

    # Regression coefficients
    print("\n═══ Regression coefficients (standardized features) ═══")
    for k, name in enumerate(features):
        bea = trace.posterior["beta_Ea"].values[:, :, k].flatten()
        bla = trace.posterior["beta_lnA"].values[:, :, k].flatten()
        print(f"  {name:20s}  β_Ea={bea.mean():+.2f}±{bea.std():.2f}  β_lnA={bla.mean():+.2f}±{bla.std():.2f}")


# ── 5. Visualization ─────────────────────────────────────────────────

def plot_arrhenius_posterior(trace, obs_data):
    """Plot Arrhenius lines (posterior) vs observed k_d points."""
    Ea_post = trace.posterior["Ea"].values.reshape(-1, obs_data["n_obs_subs"])
    lnA_post = trace.posterior["lnA"].values.reshape(-1, obs_data["n_obs_subs"])
    kd_valid = obs_data["kd_valid"]

    n_subs = obs_data["n_obs_subs"]
    ncols = 4
    nrows = (n_subs + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
    axes = axes.flatten()

    for i, smi in enumerate(obs_data["obs_smiles"]):
        ax = axes[i]
        sub_data = kd_valid[kd_valid.can_smi == smi]

        # Observed points
        inv_T = 1000.0 / sub_data["T_K"].values
        ax.scatter(inv_T, sub_data["log_kd"].values, c="k", s=30, zorder=5)

        # Posterior Arrhenius lines (sample 200 draws)
        T_grid = np.linspace(190, 310, 100)
        inv_T_grid = 1000.0 / T_grid
        draws = np.random.choice(len(Ea_post), 200, replace=False)
        for d in draws:
            log_kd_line = lnA_post[d, i] - Ea_post[d, i] / (R_GAS * T_grid)
            ax.plot(inv_T_grid, log_kd_line, c="steelblue", alpha=0.03, lw=0.5)

        # Median line
        ea_med = np.median(Ea_post[:, i])
        lna_med = np.median(lnA_post[:, i])
        log_kd_med = lna_med - ea_med / (R_GAS * T_grid)
        ax.plot(inv_T_grid, log_kd_med, c="darkblue", lw=1.5, zorder=4)

        ax.set_xlabel("1000/T (K⁻¹)")
        ax.set_ylabel("ln(k_d)")
        label = smi if len(smi) <= 30 else smi[:27] + "..."
        ax.set_title(label, fontsize=8)
        ax.invert_xaxis()

    # Hide unused axes
    for j in range(n_subs, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    out = FIG_DIR / "stage2_arrhenius_posterior.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


def plot_thalf_credible_intervals(predictions, temp_C=-78):
    """Forest plot of predicted t½ with 95% CrI."""
    col_med = f"t_half_{temp_C}C_median_s"
    col_lo = f"t_half_{temp_C}C_q025_s"
    col_hi = f"t_half_{temp_C}C_q975_s"

    df = predictions.dropna(subset=[col_med]).sort_values(col_med)

    class_colors = {
        "ArLi": "steelblue", "oxiranylLi": "darkorange", "carbenoid": "forestgreen",
        "benzylLi": "crimson", "vinylLi": "purple", "alkylLi": "brown",
        "aziridinylLi": "teal", "carbanion": "gray",
    }

    fig, ax = plt.subplots(figsize=(12, max(6, len(df) * 0.22)))

    y_pos = range(len(df))
    colors = [class_colors.get(c, "gray") for c in df["intermediate_class"]]

    # Log-scale credible intervals
    medians = np.log10(df[col_med].clip(lower=1e-15))
    lows = np.log10(df[col_lo].clip(lower=1e-15))
    highs = np.log10(df[col_hi].clip(lower=1e-15))

    for y, med, lo, hi, color in zip(y_pos, medians, lows, highs, colors):
        ax.plot([lo, hi], [y, y], color=color, lw=2, alpha=0.5)
        ax.scatter([med], [y], color=color, s=25, zorder=5, edgecolors="k", linewidth=0.3)

    ax.set_yticks(list(y_pos))
    labels = []
    for _, row in df.iterrows():
        smi = row["smiles"]
        if pd.isna(smi):
            smi = "?"
        elif len(smi) > 40:
            smi = smi[:37] + "..."
        labels.append(smi)
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel(f"log₁₀(t½ / s) at {temp_C}°C")
    ax.set_title(f"Hierarchical Bayesian: Predicted t½ at {temp_C}°C with 95% CrI")

    # Reference lines
    ax.axvline(0, color="k", lw=0.5, alpha=0.3, label="1 s")
    ax.axvline(np.log10(60), color="gray", lw=0.5, ls="--", alpha=0.3, label="1 min")

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=v, label=k) for k, v in class_colors.items()
                       if k in df["intermediate_class"].values]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=7)

    plt.tight_layout()
    out = FIG_DIR / f"stage2_thalf_credible_{temp_C}C.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


def plot_parity_Ea(trace, obs_data):
    """Parity plot: posterior Ea vs Arrhenius-fit Ea."""
    arr = pd.read_csv(DATA_DIR / "phase_b_arrhenius.csv")
    arr["can_smi"] = arr.intermediate_smiles.apply(
        lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s)) if Chem.MolFromSmiles(s) else s
    )

    Ea_post = trace.posterior["Ea"].values.reshape(-1, obs_data["n_obs_subs"])

    fig, ax = plt.subplots(figsize=(6, 6))
    for i, smi in enumerate(obs_data["obs_smiles"]):
        arr_match = arr[arr.can_smi == smi]
        if len(arr_match) == 0:
            continue
        ea_true = arr_match.iloc[0]["Ea_decomp_kJ_mol"]
        ea_mean = Ea_post[:, i].mean()
        ea_lo = np.percentile(Ea_post[:, i], 2.5)
        ea_hi = np.percentile(Ea_post[:, i], 97.5)

        ax.errorbar(ea_true, ea_mean, yerr=[[ea_mean - ea_lo], [ea_hi - ea_mean]],
                    fmt="o", color="steelblue", capsize=3, markersize=6)
        label = smi if len(smi) <= 25 else smi[:22] + "..."
        ax.annotate(label, (ea_true, ea_mean), fontsize=6, alpha=0.7,
                    xytext=(4, 4), textcoords="offset points")

    lim = ax.get_xlim()
    ax.plot(lim, lim, "k--", alpha=0.3)
    ax.set_xlabel("Arrhenius-fit Ea (kJ/mol)")
    ax.set_ylabel("Bayesian posterior Ea (kJ/mol)")
    ax.set_title("Hierarchical Bayesian: Ea Recovery")
    plt.tight_layout()
    out = FIG_DIR / "stage2_parity_Ea.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


# ── 6. Main ──────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Stage 2: Hierarchical Bayesian Model")
    print("=" * 60)
    print()

    # Load data
    kd_data = load_kd_data()
    desc_df = pd.read_csv(DATA_DIR / "clean_organolithium_unified_descriptors.csv")

    # Features: dipole + benzyne (best from Stage 1)
    features = ["dft_dipole_D", "delta_benzyne"]

    # Prepare
    obs_data, all_data = prepare_features(kd_data, desc_df, features)
    print(f"\nObserved: {obs_data['n_obs_subs']} substrates, {obs_data['n_obs']} observations")
    print(f"All substrates for prediction: {len(all_data['X_all'])}")

    # Build and sample
    model, trace = build_and_sample(obs_data, features,
                                     n_samples=2000, n_tune=2000, target_accept=0.9)

    # Diagnostics
    print_diagnostics(trace, obs_data, features)

    # Plots for observed substrates
    plot_arrhenius_posterior(trace, obs_data)
    plot_parity_Ea(trace, obs_data)

    # Predict for all substrates
    print("\n═══ Predicting for all substrates ═══")
    predictions, Ea_samples, lnA_samples = predict_new_substrates(
        trace, all_data, features, temps_C=[-78, -40, 0, 25]
    )

    # Print summary
    col = "t_half_-78C_median_s"
    ranked = predictions.sort_values(col)
    print(f"\n  {'SMILES':50s} {'class':15s} {'Ea':>10s} {'t½@-78°C':>14s} {'95% CrI':>24s}")
    print(f"  {'─'*50} {'─'*15} {'─'*10} {'─'*14} {'─'*24}")
    for _, row in ranked.head(30).iterrows():
        smi = str(row.smiles)[:48]
        cls = row.intermediate_class if pd.notna(row.intermediate_class) else "?"
        ea_str = f"{row.Ea_mean:.1f}±{row.Ea_std:.1f}"
        thalf = row[col]
        lo = row["t_half_-78C_q025_s"]
        hi = row["t_half_-78C_q975_s"]

        def fmt_time(t):
            if t < 0.001: return f"{t*1e6:.0f} μs"
            if t < 1: return f"{t*1000:.0f} ms"
            if t < 60: return f"{t:.1f} s"
            if t < 3600: return f"{t/60:.1f} min"
            return f"{t/3600:.1f} hr"

        print(f"  {smi:50s} {cls:15s} {ea_str:>10s} {fmt_time(thalf):>14s} [{fmt_time(lo)} – {fmt_time(hi)}]")

    if len(ranked) > 30:
        print(f"  ... ({len(ranked) - 30} more substrates)")

    # Save
    out_csv = DATA_DIR / "stage2_bayesian_predictions.csv"
    predictions.to_csv(out_csv, index=False)
    print(f"\nSaved predictions: {out_csv}")

    # Forest plot
    plot_thalf_credible_intervals(predictions, temp_C=-78)

    # Validation
    print("\n═══ Validation: Known experimental order ═══")
    known = [
        ("o-I-ArLi", "[Li]c1ccccc1I"),
        ("o-Br-ArLi", "[Li]c1ccccc1Br"),
        ("o-CO2Me-ArLi", "[Li]c1ccccc1C(=O)OC"),
        ("o-CO2tBu-ArLi", "[Li]c1ccccc1C(=O)OC(C)(C)C"),
        ("p-CN-ArLi", "[Li]c1ccc(C#N)cc1"),
    ]
    for name, smi in known:
        match = predictions[predictions.smiles == smi]
        if len(match) > 0:
            r = match.iloc[0]
            print(f"  {name:20s}: t½@-78°C = {r[col]:.2f} s  "
                  f"[{r['t_half_-78C_q025_s']:.2f} – {r['t_half_-78C_q975_s']:.2f}]  "
                  f"Ea = {r.Ea_mean:.1f}±{r.Ea_std:.1f}")
        else:
            print(f"  {name:20s}: not in predictions")

    # Save trace summary
    summary = az.summary(trace)
    summary.to_csv(DATA_DIR / "stage2_trace_summary.csv")
    print(f"\nSaved trace summary: {DATA_DIR / 'stage2_trace_summary.csv'}")


if __name__ == "__main__":
    main()
