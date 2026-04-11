"""
Method B: Kinetic Surrogate Model for ArLi Intermediate Stability
=================================================================
Instead of fitting t½ per substrate then modeling t½ ~ σ/Es,
directly model: yield = f(σ, Es, δ_ortho, δ_benzyne, T, tR)

Data source: organolithium_tr_subdataset_vlm_enriched.csv (heatmap tR1 curves)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern, WhiteKernel
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = Path(__file__).resolve().parent
OUT_DIR = DATA_DIR / "method_b_results"
OUT_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────
# 1. Feature mapping: SMILES → (σ, Es, δ_ortho, δ_benzyne, short_name)
# ─────────────────────────────────────────────
FEATURE_MAP = {
    # 10 ArLi with Hammett σ from UNIFIED_MODEL_ANALYSIS
    "Brc1ccccc1[Li]":                  (0.39,  0.00,  0, 1, "o-Br-ArLi"),
    "Ic1ccccc1[Li]":                   (0.35,  0.00,  0, 1, "o-I-ArLi"),
    "[Li]c1ccccc1":                    (0.00,  0.00,  0, 0, "PhLi"),
    "CC(C)(C)OC(=O)c1ccc([Li])cc1":   (0.45,  0.00,  0, 0, "p-CO₂ᵗBu-ArLi"),
    "CC(C)(C)OC(=O)c1ccccc1[Li]":     (0.45, -1.54,  1, 0, "o-CO₂ᵗBu-ArLi"),
    "CC(C)OC(=O)c1ccccc1[Li]":        (0.45, -0.47,  1, 0, "o-CO₂ⁱPr-ArLi"),
    "CCOC(=O)c1ccccc1[Li]":           (0.45, -0.07,  1, 0, "o-CO₂Et-ArLi"),
    "COC(=O)c1ccccc1[Li]":            (0.45,  0.00,  1, 0, "o-CO₂Me-ArLi"),
    "N#Cc1ccc([Li])cc1":              (0.66,  0.00,  0, 0, "p-CN-ArLi"),
    "N#Cc1cccc([Li])c1":              (0.56,  0.00,  0, 0, "m-CN-ArLi"),
    # Extra: o-CN (ortho cyano — Li···N chelation possible, δ_ortho=1 tentatively)
    "N#Cc1ccccc1[Li]":                (0.56,  0.00,  1, 0, "o-CN-ArLi"),
}


def load_and_prepare(quality_filter=True):
    """Load enriched heatmap data, filter to ArLi with σ, add features.

    If quality_filter=True, remove known-problematic curves:
    - p-CO₂ᵗBu-ArLi curves with R²<0.3 (noisy — 352/364 points)
    - PhLi borylation data at T>30°C (different reaction type)
    """
    df = pd.read_csv(ROOT / "data/final_output/organolithium_tr_subdataset_vlm_enriched.csv")

    # Load original curve fit quality
    halflives = pd.read_csv(DATA_DIR / "phase_a_halflives.csv")
    # Build (smiles, T) → fit_r2 lookup
    fit_quality = {}
    for _, row in halflives.iterrows():
        key = (row["intermediate_smiles"], round(row["T_C"]))
        fit_quality[key] = row["fit_r2"]

    records = []
    n_dropped_quality = 0
    n_dropped_phli_hot = 0
    for _, row in df.iterrows():
        smi = row["intermediate_smiles"]
        if pd.isna(smi) or smi not in FEATURE_MAP:
            continue
        sigma, es, d_ort, d_bz, short = FEATURE_MAP[smi]

        tR = row["tR1_s"]
        T = row["T1_C"]
        y = row["yield_pct"]

        if pd.isna(tR) or pd.isna(T) or pd.isna(y):
            continue

        if quality_filter:
            # Drop PhLi borylation at T>30°C (not stability data)
            if smi == "[Li]c1ccccc1" and T > 30:
                n_dropped_phli_hot += 1
                continue
            # Drop curves with known poor fit (R²<0.3)
            key = (smi, round(T))
            r2 = fit_quality.get(key, 1.0)
            if r2 < 0.3:
                n_dropped_quality += 1
                continue

        records.append({
            "short_name": short,
            "smiles": smi,
            "sigma": sigma,
            "Es": es,
            "d_ortho": d_ort,
            "d_benzyne": d_bz,
            "tR_s": tR,
            "log10_tR": np.log10(max(tR, 1e-4)),
            "T_C": T,
            "inv_T_K": 1.0 / (T + 273.15),
            "yield_pct": np.clip(y, 0, 100),
            "paper": row["paper"],
            "solvent": row.get("solvent", "THF"),
        })

    mdf = pd.DataFrame(records)
    if quality_filter:
        print(f"Quality filter: dropped {n_dropped_quality} poor-fit points, "
              f"{n_dropped_phli_hot} PhLi borylation (T>30°C)")
    print(f"Loaded {len(mdf)} data points across {mdf['short_name'].nunique()} substrates")
    print(f"  tR range: {mdf['tR_s'].min():.4f} – {mdf['tR_s'].max():.1f} s")
    print(f"  T  range: {mdf['T_C'].min():.0f} – {mdf['T_C'].max():.0f} °C")
    print(f"  yield range: {mdf['yield_pct'].min():.0f} – {mdf['yield_pct'].max():.0f}%")
    print()
    for name, grp in sorted(mdf.groupby("short_name"), key=lambda x: -len(x[1])):
        print(f"  {name:20s}  n={len(grp):>4d}  σ={grp['sigma'].iloc[0]:+.2f}  "
              f"Es={grp['Es'].iloc[0]:+.2f}  T=[{grp['T_C'].min():.0f}..{grp['T_C'].max():.0f}]")
    return mdf


# ─────────────────────────────────────────────
# 2. Feature matrix
# ─────────────────────────────────────────────
FEATURE_COLS_BASE = ["sigma", "Es", "d_ortho", "d_benzyne", "log10_tR", "inv_T_K"]
FEATURE_COLS_EXTENDED = FEATURE_COLS_BASE + [
    "sigma_x_invT",       # σ × 1/T interaction (Hammett slope varies with T)
    "sigma_x_logtR",      # σ × log(tR) (EWG effect on decay rate × time)
    "Es_x_invT",          # Es × 1/T interaction
]
FEATURE_COLS = FEATURE_COLS_EXTENDED  # default to extended


def add_interaction_features(mdf):
    """Add physics-motivated interaction features."""
    mdf = mdf.copy()
    mdf["sigma_x_invT"] = mdf["sigma"] * mdf["inv_T_K"]
    mdf["sigma_x_logtR"] = mdf["sigma"] * mdf["log10_tR"]
    mdf["Es_x_invT"] = mdf["Es"] * mdf["inv_T_K"]
    return mdf


def make_Xy(mdf):
    X = mdf[FEATURE_COLS].values.astype(np.float64)
    y = mdf["yield_pct"].values.astype(np.float64)
    return X, y


# ─────────────────────────────────────────────
# 3. Models
# ─────────────────────────────────────────────
def build_gp():
    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(length_scale=np.ones(len(FEATURE_COLS)), nu=2.5,
                 length_scale_bounds=(1e-2, 1e2))
        + WhiteKernel(noise_level=5.0, noise_level_bounds=(1e-1, 1e3))
    )
    return GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=10,
        alpha=1e-6,
        normalize_y=True,
    )


def build_gbr():
    return GradientBoostingRegressor(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=5,
        random_state=42,
    )


# ─────────────────────────────────────────────
# 4. Leave-One-Substrate-Out CV (LOSO)
# ─────────────────────────────────────────────
def loso_cv(mdf, model_fn, model_name):
    """Leave one substrate out, train on rest, predict held-out substrate."""
    substrates = sorted(mdf["short_name"].unique())
    results = []

    for held_out in substrates:
        train = mdf[mdf["short_name"] != held_out]
        test = mdf[mdf["short_name"] == held_out]

        X_train, y_train = make_Xy(train)
        X_test, y_test = make_Xy(test)

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = model_fn()
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        y_pred = np.clip(y_pred, 0, 100)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred) if len(y_test) > 1 else float("nan")

        results.append({
            "substrate": held_out,
            "n_test": len(test),
            "n_train": len(train),
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2,
            "y_test": y_test,
            "y_pred": y_pred,
            "T_test": test["T_C"].values,
            "tR_test": test["tR_s"].values,
        })
        print(f"  LOSO {held_out:20s}  n={len(test):>4d}  RMSE={rmse:6.1f}  MAE={mae:5.1f}  R²={r2:+.3f}")

    all_y = np.concatenate([r["y_test"] for r in results])
    all_pred = np.concatenate([r["y_pred"] for r in results])
    overall_rmse = np.sqrt(mean_squared_error(all_y, all_pred))
    overall_mae = mean_absolute_error(all_y, all_pred)
    overall_r2 = r2_score(all_y, all_pred)
    print(f"\n  {model_name} LOSO overall:  RMSE={overall_rmse:.1f}  MAE={overall_mae:.1f}  R²={overall_r2:.3f}")
    return results, (all_y, all_pred, overall_rmse, overall_mae, overall_r2)


# ─────────────────────────────────────────────
# 5. Full-data fit + train metrics
# ─────────────────────────────────────────────
def full_fit(mdf, model_fn, model_name):
    X, y = make_Xy(mdf)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    model = model_fn()
    model.fit(X_s, y)
    y_pred = np.clip(model.predict(X_s), 0, 100)

    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    print(f"  {model_name} full-fit:  RMSE={rmse:.1f}  R²={r2:.3f}")
    return model, scaler


# ─────────────────────────────────────────────
# 6. Plotting
# ─────────────────────────────────────────────
def plot_loso_parity(results, stats, model_name):
    """Parity plot: predicted vs observed yield, colored by substrate."""
    fig, ax = plt.subplots(figsize=(8, 8))
    cmap = plt.cm.tab20
    substrates = sorted(set(r["substrate"] for r in results))
    colors = {s: cmap(i / len(substrates)) for i, s in enumerate(substrates)}

    for r in results:
        ax.scatter(r["y_test"], r["y_pred"], c=[colors[r["substrate"]]] * len(r["y_test"]),
                   s=12, alpha=0.6, label=r["substrate"])

    ax.plot([0, 100], [0, 100], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("Observed yield (%)", fontsize=12)
    ax.set_ylabel("Predicted yield (%)", fontsize=12)
    ax.set_title(f"LOSO CV — {model_name}\n"
                 f"RMSE={stats[2]:.1f}  MAE={stats[3]:.1f}  R²={stats[4]:.3f}", fontsize=13)
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=8, loc="lower right", ncol=2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"loso_parity_{model_name.lower().replace(' ', '_')}.png", dpi=150)
    plt.close()


def plot_kinetic_curves(mdf, results, model_name):
    """For each substrate, plot observed vs predicted yield curves at each temperature."""
    substrates = sorted(mdf["short_name"].unique())
    n = len(substrates)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    result_map = {r["substrate"]: r for r in results}

    for idx, sub in enumerate(substrates):
        ax = axes[idx // ncols][idx % ncols]
        r = result_map[sub]
        sub_data = mdf[mdf["short_name"] == sub]

        temps = sorted(sub_data["T_C"].unique())
        cmap_t = plt.cm.coolwarm
        norm = plt.Normalize(min(temps), max(temps))

        for T in temps:
            mask = sub_data["T_C"] == T
            tR_obs = sub_data.loc[mask, "tR_s"].values
            y_obs = sub_data.loc[mask, "yield_pct"].values
            order = np.argsort(tR_obs)
            color = cmap_t(norm(T))

            ax.scatter(tR_obs[order], y_obs[order], c=[color], s=20, alpha=0.7, zorder=3)

            # Predicted (from LOSO)
            mask_pred = r["T_test"] == T
            if mask_pred.any():
                tR_p = r["tR_test"][mask_pred]
                y_p = r["y_pred"][mask_pred]
                order_p = np.argsort(tR_p)
                ax.plot(tR_p[order_p], y_p[order_p], c=color, alpha=0.5, lw=1.5, ls="--")

        ax.set_xscale("log")
        ax.set_xlabel("tR (s)")
        ax.set_ylabel("yield (%)")
        ax.set_title(f"{sub}\nσ={sub_data['sigma'].iloc[0]:+.2f}", fontsize=10)
        ax.set_ylim(-5, 105)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(f"Method B Kinetic Curves — {model_name} (LOSO)", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"kinetic_curves_{model_name.lower().replace(' ', '_')}.png",
                dpi=150, bbox_inches="tight")
    plt.close()


def plot_feature_importance(mdf, model_name="GBR"):
    """Train GBR on all data and show feature importance."""
    X, y = make_Xy(mdf)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    model = build_gbr()
    model.fit(X_s, y)

    importances = model.feature_importances_
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(FEATURE_COLS, importances, color="steelblue")
    ax.set_xlabel("Feature importance")
    ax.set_title(f"GBR Feature Importance (full data, n={len(mdf)})")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "feature_importance_gbr.png", dpi=150)
    plt.close()
    print(f"\n  Feature importances ({model_name}):")
    for name, imp in sorted(zip(FEATURE_COLS, importances), key=lambda x: -x[1]):
        print(f"    {name:15s}  {imp:.3f}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Method B: Kinetic Surrogate Model for ArLi Stability")
    print("=" * 60)
    print()

    # ── Round 1: Unfiltered (all 735 points) ──
    print("\n══════ Round 1: ALL data (no quality filter) ══════\n")
    mdf_raw = load_and_prepare(quality_filter=False)
    mdf_raw = add_interaction_features(mdf_raw)
    print()
    print("─── GBR (unfiltered) ───")
    full_fit(mdf_raw, build_gbr, "GBR-raw")
    gbr_raw_res, gbr_raw_stats = loso_cv(mdf_raw, build_gbr, "GBR-raw")

    # ── Round 2: Quality-filtered ──
    print("\n══════ Round 2: Quality-filtered data ══════\n")
    mdf = load_and_prepare(quality_filter=True)
    mdf = add_interaction_features(mdf)
    print()

    # --- GBR ---
    print("─── GradientBoosting (GBR) ───")
    full_fit(mdf, build_gbr, "GBR")
    print()
    print("LOSO CV:")
    gbr_results, gbr_stats = loso_cv(mdf, build_gbr, "GBR")
    plot_loso_parity(gbr_results, gbr_stats, "GBR")
    plot_kinetic_curves(mdf, gbr_results, "GBR")
    plot_feature_importance(mdf)

    # --- GP ---
    print()
    print("─── Gaussian Process (GP) ───")
    full_fit(mdf, build_gp, "GP")
    print()
    print("LOSO CV:")
    gp_results, gp_stats = loso_cv(mdf, build_gp, "GP")
    plot_loso_parity(gp_results, gp_stats, "GP")
    plot_kinetic_curves(mdf, gp_results, "GP")

    # --- Summary ---
    print()
    print("=" * 60)
    print("SUMMARY — Method B Kinetic Surrogate")
    print("=" * 60)
    print(f"\n  Round 1 (unfiltered): {len(mdf_raw)} pts, {mdf_raw['short_name'].nunique()} substrates")
    print(f"    GBR LOSO:  RMSE={gbr_raw_stats[2]:.1f}  R²={gbr_raw_stats[4]:.3f}")
    print(f"\n  Round 2 (quality-filtered): {len(mdf)} pts, {mdf['short_name'].nunique()} substrates")
    print(f"    GBR LOSO:  RMSE={gbr_stats[2]:.1f}  R²={gbr_stats[4]:.3f}")
    print(f"    GP  LOSO:  RMSE={gp_stats[2]:.1f}  R²={gp_stats[4]:.3f}")
    print(f"    Features: {FEATURE_COLS}")
    print(f"\n  Plots saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
