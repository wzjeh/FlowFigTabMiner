"""
Stage 1: Augmented LFER Baseline — DFT descriptors → Ea/lnA → t½

Compares three models:
  (A) Empirical LFER: Ea ~ σ + Es + δ_ortho + δ_benzyne  (9 ArLi, from UNIFIED_MODEL_ANALYSIS)
  (B) DFT-LFER:       Ea ~ dipole + Gsolv + BDE           (up to 14 substrates)
  (C) Hybrid:         Ea ~ σ + dipole + BDE                (9 ArLi with both)

LOO cross-validation for each, then predict t½ at −78°C for all 72 substrates.

Usage:
  cd /Users/zhaowenyuan/Projects/FlowFigTabMiner
  source flowfigtabminer/bin/activate
  python data/ml_lifetime/model_dft_lfer.py
"""

import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from rdkit import Chem

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).resolve().parent
FIG_DIR = DATA_DIR / "analysis_figures"
FIG_DIR.mkdir(exist_ok=True)

R_GAS = 8.314e-3  # kJ/(mol·K)


# ── 1. Load data ──────────────────────────────────────────────────────

def load_arrhenius_with_descriptors():
    """Merge Arrhenius parameters with DFT descriptors.

    Returns DataFrame with columns: smiles, Ea, lnA, + descriptor columns.
    """
    arr = pd.read_csv(DATA_DIR / "phase_b_arrhenius.csv")

    # Filter to Tier 1 only (n_T≥3, r²≥0.85)
    if "quality_tier" in arr.columns:
        arr = arr[arr.quality_tier == "tier1"].copy()

    desc = pd.read_csv(DATA_DIR / "clean_organolithium_unified_descriptors.csv")
    cache_path = DATA_DIR / "xtb_cache.json"

    # Canonicalize Arrhenius SMILES to match descriptor CSV
    arr["can_smi"] = arr.intermediate_smiles.apply(
        lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s)) if Chem.MolFromSmiles(s) else None
    )

    # Get unique descriptors per canonical SMILES
    desc_unique = desc.drop_duplicates(subset="intermediate_smiles_canonical")
    desc_cols = [
        "intermediate_smiles_canonical", "intermediate_class",
        "sigma_hammett", "Es_taft", "delta_ortho", "delta_benzyne",
        "dft_charge_Li", "dft_charge_C_ipso", "dft_HOMO_eV", "dft_LUMO_eV",
        "dft_LiC_bond_A", "dft_LiC_BDE_kJ", "dft_dipole_D", "dft_Gsolv_kJ",
    ]
    desc_sub = desc_unique[desc_cols].copy()

    merged = arr.merge(desc_sub, left_on="can_smi",
                       right_on="intermediate_smiles_canonical", how="left")

    # For substrates missing from descriptor CSV, try xtb cache
    if cache_path.exists():
        cache = json.load(open(cache_path))
        for idx, row in merged.iterrows():
            if pd.isna(row.get("dft_HOMO_eV")):
                smi = row["intermediate_smiles"]
                if smi in cache:
                    d = cache[smi]
                    merged.at[idx, "dft_charge_Li"] = d.get("charge_Li")
                    merged.at[idx, "dft_charge_C_ipso"] = d.get("charge_C_ipso")
                    merged.at[idx, "dft_HOMO_eV"] = d.get("homo_eV")
                    merged.at[idx, "dft_LUMO_eV"] = d.get("lumo_eV")
                    merged.at[idx, "dft_LiC_bond_A"] = d.get("lic_bond_A")
                    merged.at[idx, "dft_LiC_BDE_kJ"] = d.get("bde_kJ")
                    merged.at[idx, "dft_dipole_D"] = d.get("dipole_D")
                    merged.at[idx, "dft_Gsolv_kJ"] = d.get("gsolv_kJ")

    # Fill empirical descriptors for known substrates not in desc CSV
    empirical_map = {
        "[Li]C1CO1":     {"sigma_hammett": None, "Es_taft": 0.0, "delta_ortho": 0, "delta_benzyne": 0, "intermediate_class": "oxiranylLi"},
        "[Li]c1ccccc1":  {"sigma_hammett": 0.0,  "Es_taft": 0.0, "delta_ortho": 0, "delta_benzyne": 0, "intermediate_class": "ArLi"},
        "[Li]C(Cl)I":    {"sigma_hammett": None, "Es_taft": 0.0, "delta_ortho": 0, "delta_benzyne": 0, "intermediate_class": "carbenoid"},
    }
    for idx, row in merged.iterrows():
        smi = row["intermediate_smiles"]
        if smi in empirical_map:
            for k, v in empirical_map[smi].items():
                if pd.isna(row.get(k)):
                    merged.at[idx, k] = v

    print(f"Arrhenius substrates: {len(merged)}")
    print(f"  with DFT descriptors: {merged.dft_HOMO_eV.notna().sum()}")
    print(f"  with sigma_hammett:   {merged.sigma_hammett.notna().sum()}")
    print()

    return merged


# ── 2. Model definitions ─────────────────────────────────────────────

def loo_cv(X, y, names=None):
    """Leave-one-out cross-validation for OLS. Returns predictions, residuals, metrics."""
    n = len(y)
    y_pred = np.zeros(n)

    for i in range(n):
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y, i)
        reg = LinearRegression().fit(X_train, y_train)
        y_pred[i] = reg.predict(X[i:i+1])[0]

    residuals = y - y_pred
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - y.mean())**2)
    r2_loo = 1 - ss_res / ss_tot
    mae = mean_absolute_error(y, y_pred)

    # Full model fit for coefficients
    reg_full = LinearRegression().fit(X, y)
    r2_full = r2_score(y, reg_full.predict(X))

    return {
        "y_pred_loo": y_pred,
        "residuals": residuals,
        "r2_full": r2_full,
        "r2_loo": r2_loo,
        "mae_loo": mae,
        "coefs": reg_full.coef_,
        "intercept": reg_full.intercept_,
        "feature_names": names or [f"x{i}" for i in range(X.shape[1])],
        "n": n,
    }


def run_model_a(data):
    """Model A: Empirical LFER (σ, Es, δ_ortho, δ_benzyne) → Ea."""
    mask = data.sigma_hammett.notna()
    df = data[mask].copy()

    features = ["sigma_hammett", "Es_taft", "delta_ortho", "delta_benzyne"]
    X = df[features].values.astype(float)
    y_ea = df["Ea_decomp_kJ_mol"].values
    y_lna = df["ln_A"].values

    ea_result = loo_cv(X, y_ea, features)
    lna_result = loo_cv(X, y_lna, features)

    print("═══ Model A: Empirical LFER ═══")
    print(f"  N = {ea_result['n']} ArLi substrates")
    print(f"  Features: {features}")
    print(f"  Ea:  R²={ea_result['r2_full']:.4f},  LOO-R²={ea_result['r2_loo']:.4f},  LOO-MAE={ea_result['mae_loo']:.2f} kJ/mol")
    print(f"  lnA: R²={lna_result['r2_full']:.4f},  LOO-R²={lna_result['r2_loo']:.4f}")
    print(f"  Ea coefficients: {dict(zip(features, ea_result['coefs'].round(3)))}")
    print(f"  Ea intercept: {ea_result['intercept']:.3f}")
    print()

    return {"ea": ea_result, "lna": lna_result, "data": df, "features": features}


def run_model_b(data):
    """Model B: DFT-LFER (dipole, Gsolv, BDE) → Ea."""
    mask = data.dft_dipole_D.notna()
    df = data[mask].copy()

    # Try several DFT feature combinations
    combos = {
        "dipole_only":          ["dft_dipole_D"],
        "dipole+Gsolv":         ["dft_dipole_D", "dft_Gsolv_kJ"],
        "dipole+BDE":           ["dft_dipole_D", "dft_LiC_BDE_kJ"],
        "dipole+Gsolv+BDE":    ["dft_dipole_D", "dft_Gsolv_kJ", "dft_LiC_BDE_kJ"],
        "dipole+benzyne":       ["dft_dipole_D", "delta_benzyne"],
        "dipole+benzyne+BDE":   ["dft_dipole_D", "delta_benzyne", "dft_LiC_BDE_kJ"],
    }

    print("═══ Model B: DFT-LFER ═══")
    print(f"  N = {len(df)} substrates (all classes)")
    print()

    best_combo = None
    best_loo = -np.inf

    for name, features in combos.items():
        X = df[features].values.astype(float)
        y_ea = df["Ea_decomp_kJ_mol"].values

        # Check for NaN in features
        valid = ~np.isnan(X).any(axis=1)
        if valid.sum() < 4:
            continue

        X_v = X[valid]
        y_v = y_ea[valid]

        result = loo_cv(X_v, y_v, features)
        flag = " ★" if result["r2_loo"] > best_loo else ""
        print(f"  [{name}] n={result['n']}, R²={result['r2_full']:.4f}, LOO-R²={result['r2_loo']:.4f}, MAE={result['mae_loo']:.2f}{flag}")

        if result["r2_loo"] > best_loo:
            best_loo = result["r2_loo"]
            best_combo = name
            best_result = result
            best_features = features
            best_df = df[valid].copy()

    print()
    print(f"  Best DFT combo: {best_combo}")
    print(f"    LOO-R² = {best_result['r2_loo']:.4f}, MAE = {best_result['mae_loo']:.2f} kJ/mol")
    print(f"    Coefficients: {dict(zip(best_features, best_result['coefs'].round(3)))}")
    print(f"    Intercept: {best_result['intercept']:.3f}")
    print()

    # Also fit lnA with best features
    X_best = best_df[best_features].values.astype(float)
    y_lna = best_df["ln_A"].values
    lna_result = loo_cv(X_best, y_lna, best_features)
    print(f"  lnA with {best_combo}: R²={lna_result['r2_full']:.4f}, LOO-R²={lna_result['r2_loo']:.4f}")
    print()

    return {"ea": best_result, "lna": lna_result, "data": best_df,
            "features": best_features, "combo_name": best_combo}


# ── 3. Predict t½ for all substrates ─────────────────────────────────

def predict_all_substrates(model_result, desc_df, model_name):
    """Predict Ea, lnA → t½ at key temperatures for all substrates with descriptors."""
    features = model_result["features"]
    ea_res = model_result["ea"]
    lna_res = model_result["lna"]

    # Full-data fit
    reg_ea = LinearRegression()
    reg_ea.coef_ = ea_res["coefs"]
    reg_ea.intercept_ = ea_res["intercept"]

    reg_lna = LinearRegression()
    reg_lna.coef_ = lna_res["coefs"]
    reg_lna.intercept_ = lna_res["intercept"]

    # Get unique substrates with required features
    desc_unique = desc_df.drop_duplicates(subset="intermediate_smiles_canonical")
    mask = desc_unique[features].notna().all(axis=1)
    subs = desc_unique[mask].copy()

    X_all = subs[features].values.astype(float)
    subs["pred_Ea"] = reg_ea.predict(X_all)
    subs["pred_lnA"] = reg_lna.predict(X_all)

    # t½ = ln(2) / k_d = ln(2) / (A * exp(-Ea/RT))
    temps_C = [-78, -40, 0, 25]
    for t_c in temps_C:
        t_k = t_c + 273.15
        kd = np.exp(subs["pred_lnA"] - subs["pred_Ea"] / (R_GAS * t_k))
        subs[f"pred_t_half_{t_c}C_s"] = np.log(2) / kd

    subs["model"] = model_name

    return subs


# ── 4. Visualization ─────────────────────────────────────────────────

def plot_parity(model_a, model_b, arr_data):
    """Parity plot: predicted vs actual Ea for both models."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Model A
    ax = axes[0]
    mask_a = arr_data.sigma_hammett.notna()
    df_a = arr_data[mask_a]
    y_a = df_a["Ea_decomp_kJ_mol"].values
    y_pred_a = model_a["ea"]["y_pred_loo"]
    ax.scatter(y_a, y_pred_a, c="steelblue", s=60, edgecolors="k", zorder=3)
    for i, row in enumerate(df_a.itertuples()):
        label = row.intermediate_smiles[:20]
        ax.annotate(label, (y_a[i], y_pred_a[i]), fontsize=6, alpha=0.7,
                    xytext=(3, 3), textcoords="offset points")
    lim = [min(y_a.min(), y_pred_a.min()) - 3, max(y_a.max(), y_pred_a.max()) + 3]
    ax.plot(lim, lim, "k--", alpha=0.3)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Actual Ea (kJ/mol)")
    ax.set_ylabel("LOO-Predicted Ea (kJ/mol)")
    ax.set_title(f"Model A: Empirical LFER (n={model_a['ea']['n']})\n"
                 f"LOO-R²={model_a['ea']['r2_loo']:.3f}, MAE={model_a['ea']['mae_loo']:.1f} kJ/mol")

    # Model B
    ax = axes[1]
    y_b = model_b["data"]["Ea_decomp_kJ_mol"].values
    y_pred_b = model_b["ea"]["y_pred_loo"]
    ax.scatter(y_b, y_pred_b, c="darkorange", s=60, edgecolors="k", zorder=3)
    for i, row in enumerate(model_b["data"].itertuples()):
        label = row.intermediate_smiles[:20]
        ax.annotate(label, (y_b[i], y_pred_b[i]), fontsize=6, alpha=0.7,
                    xytext=(3, 3), textcoords="offset points")
    lim = [min(y_b.min(), y_pred_b.min()) - 3, max(y_b.max(), y_pred_b.max()) + 3]
    ax.plot(lim, lim, "k--", alpha=0.3)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Actual Ea (kJ/mol)")
    ax.set_ylabel("LOO-Predicted Ea (kJ/mol)")
    ax.set_title(f"Model B: DFT-LFER [{model_b['combo_name']}] (n={model_b['ea']['n']})\n"
                 f"LOO-R²={model_b['ea']['r2_loo']:.3f}, MAE={model_b['ea']['mae_loo']:.1f} kJ/mol")

    plt.tight_layout()
    out = FIG_DIR / "stage1_parity_Ea.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


def plot_thalf_ranking(predictions, temps_C=[-78]):
    """Bar chart of predicted t½ at −78°C, color-coded by intermediate class."""
    class_colors = {
        "ArLi": "steelblue", "oxiranylLi": "darkorange", "carbenoid": "forestgreen",
        "benzylLi": "crimson", "vinylLi": "purple", "alkylLi": "brown",
        "aziridinylLi": "teal", "carbanion": "gray",
    }

    for t_c in temps_C:
        col = f"pred_t_half_{t_c}C_s"
        df = predictions.dropna(subset=[col]).sort_values(col)

        fig, ax = plt.subplots(figsize=(14, max(6, len(df) * 0.25)))
        colors = [class_colors.get(c, "gray") for c in df["intermediate_class"]]
        bars = ax.barh(range(len(df)), np.log10(df[col].clip(lower=1e-10)), color=colors,
                       edgecolor="k", linewidth=0.3)

        ax.set_yticks(range(len(df)))
        labels = []
        for _, row in df.iterrows():
            smi = row["intermediate_smiles_canonical"]
            if pd.isna(smi):
                smi = "?"
            elif len(smi) > 35:
                smi = smi[:32] + "..."
            labels.append(smi)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel(f"log₁₀(predicted t½ / s) at {t_c}°C")
        ax.set_title(f"Predicted Half-Lives at {t_c}°C — DFT-LFER Model")
        ax.axvline(0, color="k", lw=0.5, alpha=0.3)

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=v, label=k) for k, v in class_colors.items()
                           if k in df["intermediate_class"].values]
        ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

        plt.tight_layout()
        out = FIG_DIR / f"stage1_thalf_ranking_{t_c}C.png"
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"Saved: {out}")


# ── 5. Main ──────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Stage 1: Augmented LFER Baseline")
    print("=" * 60)
    print()

    # Load and merge
    arr_data = load_arrhenius_with_descriptors()

    # Print data summary
    print("─── Per-substrate data ───")
    for _, row in arr_data.iterrows():
        sigma = f"{row.sigma_hammett:.2f}" if pd.notna(row.sigma_hammett) else " NaN"
        dipole = f"{row.dft_dipole_D:.2f}" if pd.notna(row.dft_dipole_D) else " NaN"
        bde = f"{row.dft_LiC_BDE_kJ:.1f}" if pd.notna(row.dft_LiC_BDE_kJ) else "  NaN"
        gsolv = f"{row.dft_Gsolv_kJ:.1f}" if pd.notna(row.dft_Gsolv_kJ) else "  NaN"
        cls = row.get("intermediate_class", "?")
        if pd.isna(cls):
            cls = "?"
        print(f"  {row.intermediate_smiles[:35]:35s}  Ea={row.Ea_decomp_kJ_mol:6.1f}  "
              f"σ={sigma}  μ={dipole}  BDE={bde}  Gsolv={gsolv}  [{cls}]")
    print()

    # Run models
    model_a = run_model_a(arr_data)
    model_b = run_model_b(arr_data)

    # Parity plot
    plot_parity(model_a, model_b, arr_data)

    # Predict for all substrates using best DFT model
    desc_df = pd.read_csv(DATA_DIR / "clean_organolithium_unified_descriptors.csv")
    predictions = predict_all_substrates(model_b, desc_df, "DFT-LFER")

    print(f"\n═══ Predictions for all substrates ═══")
    print(f"  Substrates with predictions: {len(predictions)}")
    print(f"  Intermediate classes: {predictions.intermediate_class.value_counts().to_dict()}")
    print()

    # Print t½ ranking at -78°C
    col = "pred_t_half_-78C_s"
    ranked = predictions.sort_values(col)
    print(f"  {'SMILES':50s} {'class':15s} {'Ea':>8s} {'lnA':>8s} {'t½@-78°C':>12s}")
    print(f"  {'─'*50} {'─'*15} {'─'*8} {'─'*8} {'─'*12}")
    for _, row in ranked.iterrows():
        smi = row.intermediate_smiles_canonical
        if pd.isna(smi) or len(smi) > 48:
            smi = str(smi)[:48]
        cls = row.intermediate_class if pd.notna(row.intermediate_class) else "?"
        thalf = row[col]
        if thalf < 1:
            t_str = f"{thalf*1000:.1f} ms"
        elif thalf < 60:
            t_str = f"{thalf:.1f} s"
        elif thalf < 3600:
            t_str = f"{thalf/60:.1f} min"
        else:
            t_str = f"{thalf/3600:.1f} hr"
        print(f"  {smi:50s} {cls:15s} {row.pred_Ea:8.1f} {row.pred_lnA:8.2f} {t_str:>12s}")

    # Save predictions
    out_csv = DATA_DIR / "stage1_dft_lfer_predictions.csv"
    predictions.to_csv(out_csv, index=False)
    print(f"\nSaved predictions: {out_csv}")

    # Ranking plot
    plot_thalf_ranking(predictions, [-78])

    # Validation: check known experimental order
    print("\n═══ Validation: Known experimental order ═══")
    known_order = [
        ("o-I-ArLi", "[Li]c1ccccc1I"),
        ("o-Br-ArLi", "[Li]c1ccccc1Br"),
        ("PhLi", "[Li]c1ccccc1"),
    ]
    for name, smi in known_order:
        match = predictions[predictions.intermediate_smiles_canonical == smi]
        if len(match) > 0:
            t = match.iloc[0][col]
            print(f"  {name:20s}: t½@-78°C = {t:.2f} s")
        else:
            print(f"  {name:20s}: not in predictions")
    print("  Expected order: o-I-ArLi < o-Br-ArLi < PhLi (fast→slow)")


if __name__ == "__main__":
    main()
