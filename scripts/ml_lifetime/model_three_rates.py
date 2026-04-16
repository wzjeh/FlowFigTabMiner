"""
Three-Rate Kinetic Modeling with DFT Descriptors.

Models the three reaction rates in organolithium flow chemistry:
  k_f  (formation):    ArBr + n-BuLi → ArLi          (tR1-controlled)
  k_d  (decomposition): ArLi → Decomposition products  (competes during tR1)
  k_2  (trapping):     ArLi + Electrophile → Product  (tR2-controlled)

Uses both empirical (σ, Es, δ_ortho, δ_benzyne) and DFT descriptors
(charge_Li, HOMO, LUMO, BDE, Wiberg, dipole, Gsolv) to build predictive models.

Outputs:
  data/ml_lifetime/three_rate_analysis/
    ├── model_comparison.png          — Bar chart: R² for each feature set
    ├── kd_dft_parity.png             — Ea/lnA parity (DFT model)
    ├── kd_feature_importance.png     — DFT descriptor importance for Ea, lnA
    ├── yield_loso.png                — LOSO yield prediction (direct model)
    ├── k2_trapping_analysis.png      — k2/trapping rate analysis
    ├── three_rate_summary.csv        — All rate parameters + descriptors
    └── model_report.txt              — Text summary of all model results

Usage:
  cd /Users/zhaowenyuan/Projects/FlowFigTabMiner
  source flowfigtabminer/bin/activate
  python scripts/ml_lifetime/model_three_rates.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression, Ridge, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import LeaveOneOut

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data/ml_lifetime")
ARRHENIUS_CSV = os.path.join(DATA_DIR, "phase_b_arrhenius.csv")
HALFLIFE_CSV = os.path.join(DATA_DIR, "phase_a_halflives.csv")
DESCRIPTORS_CSV = os.path.join(DATA_DIR, "clean_organolithium_unified_descriptors.csv")
OUT_DIR = os.path.join(DATA_DIR, "three_rate_analysis")
os.makedirs(OUT_DIR, exist_ok=True)

R_GAS = 8.314e-3  # kJ/(mol·K)

# ── DFT descriptor columns ──
DFT_COLS = [
    "dft_charge_Li", "dft_charge_C_ipso", "dft_HOMO_eV", "dft_LUMO_eV",
    "dft_LiC_bond_A", "dft_LiC_BDE_kJ", "dft_wiberg_LiC", "dft_dipole_D",
    "dft_Gsolv_kJ",
]
EMP_COLS = ["sigma_hammett", "Es_taft", "delta_ortho", "delta_benzyne"]


# ========================================================================
#  Part 1: Load and merge data
# ========================================================================

def load_data():
    """Load Arrhenius parameters, half-life fits, and descriptor dataset."""
    arr = pd.read_csv(ARRHENIUS_CSV)
    arr = arr.rename(columns={
        "Ea_decomp_kJ_mol": "Ea",
        "intermediate_smiles": "smiles",
    })

    hl = pd.read_csv(HALFLIFE_CSV)

    desc = pd.read_csv(DESCRIPTORS_CSV)

    return arr, hl, desc


def merge_arrhenius_with_descriptors(arr, desc):
    """Merge Arrhenius parameters with DFT/empirical descriptors.

    Links via intermediate_smiles matching to intermediate_smiles_canonical.
    """
    from rdkit import Chem

    # Manual overrides for SMILES that don't match automatically.
    # Arrhenius CSV uses simplified SMILES; descriptor CSV uses substituted forms.
    SMILES_OVERRIDES = {
        # oxiranyllithium: parent [Li]C1CO1 → use alpha-phenyloxiranyllithium as proxy
        "[Li]C1CO1": "[Li]C1(c2ccccc2)CO1",
        # PhLi: plain phenyllithium → use p-OMe-PhLi (closest in descriptor set)
        # Actually PhLi is unique — compute DFT directly instead
        "[Li]c1ccccc1": None,  # No match; will use DFT cache if available
        # CHLi(Cl)(I): chloroiodomethyllithium not in dataset
        "[Li]C(Cl)I": None,
    }

    # Build canonical SMILES lookup from descriptor dataset
    uniq = desc.drop_duplicates(subset=["intermediate_smiles_canonical"])
    smiles_to_desc = {}
    for _, row in uniq.iterrows():
        smi = row["intermediate_smiles_canonical"]
        if pd.notna(smi):
            smiles_to_desc[smi] = row

    # Also build a raw SMILES → canonical mapping
    raw_to_canon = {}
    for _, row in desc.drop_duplicates(subset=["intermediate_smiles"]).iterrows():
        raw = row.get("intermediate_smiles")
        canon = row.get("intermediate_smiles_canonical")
        if pd.notna(raw) and pd.notna(canon):
            raw_to_canon[raw] = canon

    # Also build RDKit canonical → dataset canonical mapping
    rdkit_to_dataset = {}
    for smi in smiles_to_desc:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            rdkit_to_dataset[Chem.MolToSmiles(mol)] = smi

    # Match Arrhenius intermediates to descriptors
    merged = []
    for _, arow in arr.iterrows():
        raw_smi = arow["smiles"]

        # Check manual overrides first
        if raw_smi in SMILES_OVERRIDES:
            override = SMILES_OVERRIDES[raw_smi]
            if override is not None:
                canon = override
            else:
                canon = None
        else:
            # Try canonical match
            canon = raw_to_canon.get(raw_smi)
            if canon is None:
                # Try RDKit canonicalization
                mol = Chem.MolFromSmiles(raw_smi)
                if mol:
                    rdkit_canon = Chem.MolToSmiles(mol)
                    canon = rdkit_to_dataset.get(rdkit_canon, rdkit_canon)

        drow = smiles_to_desc.get(canon)
        entry = {
            "intermediate": arow["intermediate"],
            "smiles": raw_smi,
            "smiles_canonical": canon,
            "Ea": arow["Ea"],
            "ln_A": arow["ln_A"],
            "t_half_m40": arow["t_half_m40C_s"],
            "n_T": arow["n_temperatures"],
            "arrhenius_r2": arow["arrhenius_r2"],
        }

        if drow is not None:
            for col in DFT_COLS + EMP_COLS:
                entry[col] = drow.get(col, np.nan)
            entry["intermediate_class"] = drow.get("intermediate_class", "")
        else:
            for col in DFT_COLS + EMP_COLS:
                entry[col] = np.nan
            entry["intermediate_class"] = ""
            print(f"  [WARN] No descriptor match for: {arow['intermediate'][:50]}")

        merged.append(entry)

    return pd.DataFrame(merged)


# ========================================================================
#  Part 2: kd modeling — Arrhenius parameters from descriptors
# ========================================================================

def loocv_regression(X, y, names=None, model_class=None, **model_kwargs):
    """Leave-one-out cross-validation for regression."""
    if model_class is None:
        model_class = LinearRegression
    loo = LeaveOneOut()
    y_pred = np.full_like(y, np.nan)
    for train_idx, test_idx in loo.split(X):
        reg = model_class(**model_kwargs).fit(X[train_idx], y[train_idx])
        y_pred[test_idx] = reg.predict(X[test_idx])
    q2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - y.mean()) ** 2)
    return y_pred, q2


def model_kd(merged_df, report_lines):
    """Model kd (decomposition) Arrhenius parameters with different feature sets."""
    report_lines.append("=" * 70)
    report_lines.append("PART 1: kd (DECOMPOSITION) MODELING")
    report_lines.append("=" * 70)

    # Feature sets to compare
    feature_sets = {}

    # A: Empirical only (σ + Es + δ_ortho + δ_benzyne) — only for ArLi with sigma
    emp_mask = merged_df["sigma_hammett"].notna()
    if emp_mask.sum() >= 5:
        feature_sets["Empirical\n(σ,Es,δ_o,δ_b)"] = {
            "cols": EMP_COLS,
            "mask": emp_mask,
            "label": "empirical",
            "regularize": False,
        }

    # B: DFT only (Ridge regularized) — for all intermediates with DFT
    dft_mask = merged_df["dft_charge_Li"].notna()
    if dft_mask.sum() >= 5:
        feature_sets["DFT-Ridge\n(9 desc, reg.)"] = {
            "cols": DFT_COLS,
            "mask": dft_mask,
            "label": "dft_ridge",
            "regularize": True,
        }

    # C: DFT best-2 (selected by univariate correlation with Ea)
    # Pre-select top 2 DFT features by |r| with Ea
    dft_sub = merged_df[dft_mask].dropna(subset=DFT_COLS + ["Ea"])
    if len(dft_sub) >= 5:
        corrs = {}
        for col in DFT_COLS:
            r, _ = pearsonr(dft_sub[col], dft_sub["Ea"])
            corrs[col] = abs(r)
        top2 = sorted(corrs, key=corrs.get, reverse=True)[:2]
        top3 = sorted(corrs, key=corrs.get, reverse=True)[:3]

        feature_sets[f"DFT-best2\n({','.join(c.replace('dft_','') for c in top2)})"] = {
            "cols": top2,
            "mask": dft_mask,
            "label": "dft2",
            "regularize": False,
        }
        feature_sets[f"DFT-best3\n({','.join(c.replace('dft_','') for c in top3)})"] = {
            "cols": top3,
            "mask": dft_mask,
            "label": "dft3",
            "regularize": False,
        }

    # D: Combined (empirical + top DFT) — only for ArLi with sigma
    combo_mask = emp_mask & dft_mask
    if combo_mask.sum() >= 5 and len(top2) >= 2:
        feature_sets["Emp+DFT-2\n(σ,Es,δ+best2)"] = {
            "cols": EMP_COLS + top2,
            "mask": combo_mask,
            "label": "combined",
            "regularize": False,
        }

    # Run all models
    results = {}
    for target_name, target_col in [("Ea", "Ea"), ("ln_A", "ln_A")]:
        report_lines.append(f"\n--- {target_name} Prediction ---")
        results[target_name] = {}

        # Univariate DFT correlations
        report_lines.append(f"\n  Univariate DFT correlations with {target_name}:")
        dft_sub_t = merged_df[dft_mask].dropna(subset=DFT_COLS + [target_col])
        for col in DFT_COLS:
            r, p = pearsonr(dft_sub_t[col], dft_sub_t[target_col])
            report_lines.append(f"    {col:25s}: r={r:+.3f}  (p={p:.3f})")

        for fs_name, fs_info in feature_sets.items():
            mask = fs_info["mask"]
            df_sub = merged_df[mask].copy()
            cols = fs_info["cols"]
            label = fs_info["label"]
            use_ridge = fs_info.get("regularize", False)

            X = df_sub[cols].values
            y = df_sub[target_col].values
            names = df_sub["intermediate"].values
            n = len(y)

            # Drop any NaN rows
            valid_rows = ~np.any(np.isnan(X), axis=1) & ~np.isnan(y)
            X = X[valid_rows]
            y = y[valid_rows]
            names = names[valid_rows]
            n = len(y)

            if X.shape[1] == 0 or n < 4:
                continue

            if use_ridge:
                # Standardize for Ridge
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                # Find optimal alpha via LOOCV
                best_q2 = -1e10
                best_alpha = 1.0
                for alpha in [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]:
                    _, q2_test = loocv_regression(X_scaled, y, model_class=Ridge, alpha=alpha)
                    if q2_test > best_q2:
                        best_q2 = q2_test
                        best_alpha = alpha

                reg = Ridge(alpha=best_alpha).fit(X_scaled, y)
                y_fit = reg.predict(X_scaled)
                r2 = r2_score(y, y_fit)
                y_loo, q2 = loocv_regression(X_scaled, y, model_class=Ridge, alpha=best_alpha)

                # Convert coefficients back to original scale for interpretation
                coef_original = reg.coef_ / scaler.scale_
                coef_dict = dict(zip(cols, coef_original))
            else:
                # Standard OLS
                reg = LinearRegression().fit(X, y)
                y_fit = reg.predict(X)
                r2 = r2_score(y, y_fit)
                y_loo, q2 = loocv_regression(X, y, names)
                coef_dict = dict(zip(cols, reg.coef_))

            results[target_name][fs_name] = {
                "r2": r2, "q2": q2, "n": n,
                "coefs": coef_dict,
                "intercept": reg.intercept_,
                "y_true": y, "y_fit": y_fit, "y_loo": y_loo,
                "names": names, "label": label,
            }

            report_lines.append(f"\n  Feature set: {fs_name.replace(chr(10), ' ')}")
            report_lines.append(f"  n={n}, features={len(cols)}")
            if use_ridge:
                report_lines.append(f"  Ridge alpha={best_alpha}")
            report_lines.append(f"  R² = {r2:.4f},  Q²_LOO = {q2:.4f}")
            report_lines.append(f"  Intercept = {reg.intercept_:.3f}")
            for c in cols:
                report_lines.append(f"    {c:25s}: {coef_dict[c]:+.4f}")

    return results, feature_sets


def plot_model_comparison(results, report_lines):
    """Bar chart comparing R² and Q² across feature sets for Ea and ln_A."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, target in zip(axes, ["Ea", "ln_A"]):
        if target not in results:
            continue
        res = results[target]
        names = list(res.keys())
        r2s = [res[n]["r2"] for n in names]
        q2s = [res[n]["q2"] for n in names]
        ns = [res[n]["n"] for n in names]

        x = np.arange(len(names))
        w = 0.35
        bars1 = ax.bar(x - w / 2, r2s, w, label="R² (train)", color="#2171B5", edgecolor="white")
        bars2 = ax.bar(x + w / 2, q2s, w, label="Q² (LOOCV)", color="#E6550D", edgecolor="white")

        for bar, val in zip(bars1, r2s):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", fontsize=8, fontweight="bold")
        for bar, val in zip(bars2, q2s):
            ax.text(bar.get_x() + bar.get_width() / 2, max(0, bar.get_height()) + 0.01,
                    f"{val:.3f}", ha="center", fontsize=8, fontweight="bold",
                    color="red" if val < 0 else "black")

        ax.set_xticks(x)
        ax.set_xticklabels([f"{n}\n(n={ns[i]})" for i, n in enumerate(names)], fontsize=9)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title(f"{target} Prediction", fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        ax.set_ylim(min(0, min(q2s) - 0.1), 1.1)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("kd Model Comparison: Empirical vs DFT Descriptors", fontsize=14, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "model_comparison.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    report_lines.append(f"\n  Saved: {path}")
    print(f"  Saved: {path}")


def plot_kd_parity(results, report_lines):
    """Parity plots for best DFT-based model (Ea and ln_A)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, target in zip(axes, ["Ea", "ln_A"]):
        res = results[target]
        # Pick best DFT-based model by Q² (prefer Ridge, then best3, then best2)
        dft_keys = [k for k in res if "DFT" in k]
        if not dft_keys:
            continue
        best_key = max(dft_keys, key=lambda k: res[k]["q2"])
        r = res[best_key]

        y_true = r["y_true"]
        y_fit = r["y_fit"]
        y_loo = r["y_loo"]
        names = r["names"]

        ax.scatter(y_true, y_fit, s=100, c="#2171B5", edgecolors="black",
                   zorder=5, label=f"Fit (R²={r['r2']:.3f})")
        ax.scatter(y_true, y_loo, s=80, c="#E6550D", edgecolors="black",
                   marker="^", zorder=4, alpha=0.7,
                   label=f"LOOCV (Q²={r['q2']:.3f})")

        lims = [min(y_true.min(), y_fit.min()) - 3,
                max(y_true.max(), y_fit.max()) + 3]
        ax.plot(lims, lims, "k--", alpha=0.4, label="1:1 line")
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        # Label points
        for i, name in enumerate(names):
            short = name[:20]
            ax.annotate(short, (y_true[i], y_fit[i]), xytext=(5, 5),
                        textcoords="offset points", fontsize=6)

        unit = "kJ/mol" if target == "Ea" else ""
        short_key = best_key.replace("\n", " ")
        ax.set_xlabel(f"Observed {target} {unit}", fontsize=12)
        ax.set_ylabel(f"Predicted {target} {unit}", fontsize=12)
        ax.set_title(f"{target} — {short_key} (n={r['n']})", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle("kd Prediction: Best DFT Model", fontsize=14, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "kd_dft_parity.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    report_lines.append(f"\n  Saved: {path}")
    print(f"  Saved: {path}")


def plot_kd_feature_importance(results, report_lines):
    """Horizontal bar chart of DFT descriptor coefficients for Ea and ln_A."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    for ax, target in zip(axes, ["Ea", "ln_A"]):
        res = results[target]
        # Pick best DFT-based model by Q²
        dft_keys = [k for k in res if "DFT" in k]
        if not dft_keys:
            continue
        best_key = max(dft_keys, key=lambda k: res[k]["q2"])
        r = res[best_key]

        coefs = r["coefs"]
        # Standardized importance: coef * std of feature
        names_sorted = sorted(coefs.keys(), key=lambda k: abs(coefs[k]), reverse=True)
        vals = [coefs[n] for n in names_sorted]
        colors = ["#2171B5" if v > 0 else "#E6550D" for v in vals]

        y_pos = np.arange(len(names_sorted))
        ax.barh(y_pos, vals, color=colors, edgecolor="white")
        ax.set_yticks(y_pos)
        ax.set_yticklabels([n.replace("dft_", "") for n in names_sorted], fontsize=10)
        ax.invert_yaxis()
        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_xlabel("Coefficient", fontsize=12)
        ax.set_title(f"{target} Model Coefficients", fontsize=12, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle("DFT Descriptor Importance for kd Prediction", fontsize=14, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "kd_feature_importance.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ========================================================================
#  Part 3: Direct yield prediction — f(tR1, T1, descriptors)
# ========================================================================

def model_direct_yield(desc, report_lines):
    """Direct yield prediction from tR1, T1, and molecular descriptors.

    Uses GBR with leave-one-substrate-out (LOSO) cross-validation.
    """
    report_lines.append("\n" + "=" * 70)
    report_lines.append("PART 2: DIRECT YIELD PREDICTION (kd_clean + kd_valid)")
    report_lines.append("=" * 70)

    # Filter to kd data with valid reaction conditions
    kd = desc[desc["analysis_subset"].isin(["kd_clean", "kd_valid"])].copy()
    kd = kd.dropna(subset=["tR1_s", "T1_C", "yield_pct"])
    kd = kd[kd["dft_charge_Li"].notna()]  # Need DFT descriptors

    # Features
    condition_cols = ["tR1_s", "T1_C"]
    # Add derived features
    kd["log_tR1"] = np.log10(kd["tR1_s"].clip(lower=0.01))
    kd["inv_T_K"] = 1000.0 / (kd["T1_C"] + 273.15)
    derived_cols = ["log_tR1", "inv_T_K"]

    feat_cols = derived_cols + DFT_COLS
    target_col = "yield_pct"

    # Drop rows with any NaN in features
    valid = kd.dropna(subset=feat_cols + [target_col])
    report_lines.append(f"\n  Data: {len(valid)} rows, {valid['intermediate'].nunique()} intermediates")

    X = valid[feat_cols].values
    y = valid[target_col].values
    substrates = valid["intermediate"].values

    # Unique substrates for LOSO
    unique_subs = np.unique(substrates)
    report_lines.append(f"  LOSO substrates: {len(unique_subs)}")

    # LOSO cross-validation with GBR
    y_pred_loso = np.full_like(y, np.nan, dtype=float)

    loso_results = []
    for sub in unique_subs:
        test_mask = substrates == sub
        train_mask = ~test_mask

        if train_mask.sum() < 10:
            continue

        model = GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=42
        )
        model.fit(X[train_mask], y[train_mask])
        pred = model.predict(X[test_mask])
        y_pred_loso[test_mask] = pred

        r2_sub = r2_score(y[test_mask], pred) if test_mask.sum() > 1 else np.nan
        mae_sub = mean_absolute_error(y[test_mask], pred)
        loso_results.append({
            "substrate": sub[:40],
            "n_test": test_mask.sum(),
            "r2": r2_sub,
            "mae": mae_sub,
        })

    # Overall LOSO metrics
    valid_mask = ~np.isnan(y_pred_loso)
    r2_loso = r2_score(y[valid_mask], y_pred_loso[valid_mask])
    mae_loso = mean_absolute_error(y[valid_mask], y_pred_loso[valid_mask])

    report_lines.append(f"\n  LOSO GBR Results:")
    report_lines.append(f"    R²_LOSO = {r2_loso:.4f}")
    report_lines.append(f"    MAE_LOSO = {mae_loso:.2f}%")
    report_lines.append(f"\n  Per-substrate LOSO:")
    for r in sorted(loso_results, key=lambda x: x["r2"] if not np.isnan(x["r2"]) else -999, reverse=True):
        report_lines.append(f"    {r['substrate']:40s} n={r['n_test']:>4d}  R²={r['r2']:>7.3f}  MAE={r['mae']:.1f}%")

    # Full model for feature importance
    model_full = GradientBoostingRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=42
    )
    model_full.fit(X, y)
    feat_imp = dict(zip(feat_cols, model_full.feature_importances_))

    # Plot LOSO parity
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.scatter(y[valid_mask], y_pred_loso[valid_mask], s=8, alpha=0.3, c="#2171B5")
    ax1.plot([0, 100], [0, 100], "k--", alpha=0.4)
    ax1.set_xlabel("Observed yield (%)", fontsize=12)
    ax1.set_ylabel("Predicted yield (LOSO) (%)", fontsize=12)
    ax1.set_title(f"LOSO Yield Prediction (GBR)\nR²={r2_loso:.3f}, MAE={mae_loso:.1f}%",
                  fontsize=12, fontweight="bold")
    ax1.set_xlim(-5, 105)
    ax1.set_ylim(-5, 105)
    ax1.grid(alpha=0.3)

    # Feature importance
    sorted_feats = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)
    feat_names = [f[0].replace("dft_", "").replace("log_tR1", "log₁₀(tR1)").replace("inv_T_K", "1000/T") for f in sorted_feats]
    feat_vals = [f[1] for f in sorted_feats]

    ax2.barh(range(len(feat_names)), feat_vals, color="#2171B5", edgecolor="white")
    ax2.set_yticks(range(len(feat_names)))
    ax2.set_yticklabels(feat_names, fontsize=10)
    ax2.invert_yaxis()
    ax2.set_xlabel("Feature Importance", fontsize=12)
    ax2.set_title("GBR Feature Importance", fontsize=12, fontweight="bold")
    ax2.grid(axis="x", alpha=0.3)

    fig.suptitle("Direct Yield Model: kd Data with DFT Descriptors", fontsize=14, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "yield_loso.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    report_lines.append(f"\n  Saved: {path}")
    print(f"  Saved: {path}")

    return r2_loso, mae_loso


# ========================================================================
#  Part 4: kf (formation) analysis
# ========================================================================

def analyze_kf(hl_df, merged_df, report_lines):
    """Analyze formation rate kf from phase_a competing model fits."""
    report_lines.append("\n" + "=" * 70)
    report_lines.append("PART 3: kf (FORMATION RATE) ANALYSIS")
    report_lines.append("=" * 70)

    # Load kf from phase_a_halflives (competing model rows have k_f)
    competing = hl_df[hl_df["model"] == "competing"].copy()
    competing = competing[competing["k_f"].notna() & (competing["k_f"] != "")]
    competing["k_f"] = pd.to_numeric(competing["k_f"], errors="coerce")
    competing = competing.dropna(subset=["k_f"])
    competing["T_K"] = competing["T_C"] + 273.15

    if len(competing) == 0:
        report_lines.append("  No kf data available from competing model fits.")
        return

    report_lines.append(f"\n  kf data points: {len(competing)}")
    report_lines.append(f"  Intermediates: {competing['intermediate'].nunique()}")

    # kf Arrhenius analysis per intermediate
    report_lines.append("\n  kf by intermediate:")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    cmap = plt.cm.tab10
    intermediates_with_kf = []
    for idx, (name, grp) in enumerate(competing.groupby("intermediate")):
        if len(grp) < 2:
            continue

        inv_T = 1.0 / grp["T_K"].values
        ln_kf = np.log(grp["k_f"].values)

        # Filter out unreasonable kf values
        valid = (ln_kf > -5) & (ln_kf < 15) & np.isfinite(ln_kf)
        if valid.sum() < 2:
            continue

        inv_T = inv_T[valid]
        ln_kf = ln_kf[valid]

        color = cmap(idx % 10)
        short = name[:25]

        ax1.scatter(inv_T * 1000, ln_kf, s=60, c=[color], edgecolors="black",
                    zorder=5, label=short)

        # Arrhenius fit: ln(k) = ln(A) - Ea/(R*T)
        if len(inv_T) >= 2:
            from scipy.stats import linregress
            slope, intercept, r, p, se = linregress(inv_T, ln_kf)
            Ea_kf = -slope * R_GAS * 1000  # Convert to kJ/mol (slope is -Ea/R)
            ln_A_kf = intercept

            x_line = np.linspace(inv_T.min() - 0.0002, inv_T.max() + 0.0002, 50)
            ax1.plot(x_line * 1000, slope * x_line + intercept, color=color, alpha=0.5, linewidth=1)

            intermediates_with_kf.append({
                "intermediate": name[:40],
                "Ea_kf_kJ": Ea_kf,
                "ln_A_kf": ln_A_kf,
                "n_T": len(inv_T),
                "r2_kf": r ** 2,
            })

            report_lines.append(f"    {name[:40]:40s} Ea_kf={Ea_kf:>7.1f} kJ/mol, ln_A_kf={ln_A_kf:>7.2f}, "
                                f"n_T={len(inv_T)}, r²={r**2:.3f}")

    ax1.set_xlabel("1000/T (K⁻¹)", fontsize=12)
    ax1.set_ylabel("ln(kf) (s⁻¹)", fontsize=12)
    ax1.set_title("kf Arrhenius Plot\n(Formation Rate)", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=7, loc="best")
    ax1.grid(alpha=0.3)

    # kf vs kd comparison
    if intermediates_with_kf:
        kf_df = pd.DataFrame(intermediates_with_kf)

        # Merge with kd Ea from Arrhenius
        kd_ea = merged_df.set_index("intermediate")["Ea"].to_dict()

        matched = []
        for _, row in kf_df.iterrows():
            name = row["intermediate"]
            ea_kd = kd_ea.get(name)
            if ea_kd is not None:
                matched.append({"name": name[:20], "Ea_kf": row["Ea_kf_kJ"], "Ea_kd": ea_kd})

        if matched:
            match_df = pd.DataFrame(matched)
            ax2.scatter(match_df["Ea_kd"], match_df["Ea_kf"], s=100, c="#2171B5",
                        edgecolors="black", zorder=5)
            for _, r in match_df.iterrows():
                ax2.annotate(r["name"], (r["Ea_kd"], r["Ea_kf"]),
                             xytext=(5, 5), textcoords="offset points", fontsize=7)

            ax2.set_xlabel("Ea_kd (decomposition, kJ/mol)", fontsize=12)
            ax2.set_ylabel("Ea_kf (formation, kJ/mol)", fontsize=12)
            ax2.set_title("Formation vs Decomposition\nActivation Energy", fontsize=12, fontweight="bold")
            ax2.grid(alpha=0.3)
        else:
            ax2.text(0.5, 0.5, "No matched intermediates\nfor kf vs kd comparison",
                     transform=ax2.transAxes, ha="center", fontsize=12)

    fig.suptitle("kf (Formation Rate) Analysis", fontsize=14, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "kf_formation_analysis.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    report_lines.append(f"\n  Saved: {path}")
    print(f"  Saved: {path}")


# ========================================================================
#  Part 5: k2 (trapping) analysis
# ========================================================================

def analyze_k2(desc, report_lines):
    """Analyze k2 trapping rate from tR2 data."""
    report_lines.append("\n" + "=" * 70)
    report_lines.append("PART 4: k2 (TRAPPING RATE) ANALYSIS")
    report_lines.append("=" * 70)

    # tR2 data
    k2_data = desc[desc["analysis_subset"].isin(["k2_trapping", "kd_and_trapping"])].copy()
    k2_data = k2_data.dropna(subset=["tR2_s"])

    report_lines.append(f"\n  tR2 data: {len(k2_data)} rows")
    report_lines.append(f"  Papers: {k2_data['paper_id'].nunique()}")
    report_lines.append(f"  Intermediates: {k2_data['intermediate'].nunique()}")
    report_lines.append(f"  Electrophiles: {k2_data['electrophile'].nunique()}")

    if len(k2_data) < 5:
        report_lines.append("  Insufficient tR2 data for detailed modeling.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: tR2 vs yield, colored by electrophile
    ax1 = axes[0]
    has_yield = k2_data.dropna(subset=["yield_pct"])
    if len(has_yield) > 0:
        electrophiles = has_yield["electrophile"].unique()
        cmap = plt.cm.Set2
        for i, elec in enumerate(electrophiles):
            sub = has_yield[has_yield["electrophile"] == elec]
            ax1.scatter(sub["tR2_s"], sub["yield_pct"], s=60,
                        c=[cmap(i % 8)], edgecolors="black",
                        label=f"{str(elec)[:20]} (n={len(sub)})", zorder=5)

    ax1.set_xlabel("tR2 (s)", fontsize=12)
    ax1.set_ylabel("Yield (%)", fontsize=12)
    ax1.set_title("tR2 vs Yield\n(Trapping Step)", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=7, loc="best")
    ax1.grid(alpha=0.3)

    # Panel 2: tR1 vs tR2 scatter (for dual-mixer experiments)
    ax2 = axes[1]
    dual = k2_data[k2_data["tR1_s"].notna() & k2_data["tR2_s"].notna()]
    if len(dual) > 0:
        sc = ax2.scatter(dual["tR1_s"], dual["tR2_s"], s=60,
                         c=dual["yield_pct"], cmap="RdYlGn",
                         edgecolors="black", vmin=0, vmax=100, zorder=5)
        plt.colorbar(sc, ax=ax2, label="Yield (%)")

    ax2.set_xlabel("tR1 (s) — formation", fontsize=12)
    ax2.set_ylabel("tR2 (s) — trapping", fontsize=12)
    ax2.set_title("tR1 vs tR2\n(Dual-Mixer Experiments)", fontsize=12, fontweight="bold")
    ax2.grid(alpha=0.3)

    # Summary stats
    report_lines.append(f"\n  tR2 range: {k2_data['tR2_s'].min():.2f} - {k2_data['tR2_s'].max():.2f} s")
    if "yield_pct" in k2_data.columns:
        report_lines.append(f"  Yield range: {k2_data['yield_pct'].min():.1f} - {k2_data['yield_pct'].max():.1f}%")

    # Per-paper breakdown
    for paper, grp in k2_data.groupby("paper_id"):
        report_lines.append(f"\n  Paper: {paper}")
        report_lines.append(f"    Rows: {len(grp)}")
        report_lines.append(f"    Intermediates: {grp['intermediate'].unique().tolist()}")
        report_lines.append(f"    Electrophiles: {grp['electrophile'].unique().tolist()}")
        if grp["T2_C"].notna().any():
            report_lines.append(f"    T2 range: {grp['T2_C'].min():.0f} to {grp['T2_C'].max():.0f} °C")

    fig.suptitle("k2 (Trapping Rate) Analysis", fontsize=14, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "k2_trapping_analysis.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    report_lines.append(f"\n  Saved: {path}")
    print(f"  Saved: {path}")


# ========================================================================
#  Part 6: Comprehensive summary figure
# ========================================================================

def plot_summary(kd_results, merged_df, r2_yield, report_lines):
    """4-panel summary figure."""
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    # Panel (a): Ea parity — empirical vs best DFT
    ax_a = fig.add_subplot(gs[0, 0])
    emp_key = "Empirical\n(σ,Es,δ_o,δ_b)"
    dft_keys_ea = [k for k in kd_results.get("Ea", {}) if "DFT" in k]
    best_dft_key = max(dft_keys_ea, key=lambda k: kd_results["Ea"][k]["q2"]) if dft_keys_ea else None

    for key, marker, color, label_prefix in [
        (emp_key, "o", "#2171B5", "Empirical"),
        (best_dft_key, "^", "#E6550D", "Best DFT"),
    ]:
        if key and key in kd_results["Ea"]:
            r = kd_results["Ea"][key]
            ax_a.scatter(r["y_true"], r["y_fit"], s=80, marker=marker, c=color,
                         edgecolors="black", label=f"{label_prefix} (R²={r['r2']:.3f}, n={r['n']})")

    lims = [0, 70]
    ax_a.plot(lims, lims, "k--", alpha=0.4)
    ax_a.set_xlim(lims)
    ax_a.set_ylim(lims)
    ax_a.set_xlabel("Observed Ea (kJ/mol)", fontsize=11)
    ax_a.set_ylabel("Predicted Ea (kJ/mol)", fontsize=11)
    ax_a.set_title("(a) Ea Prediction: Empirical vs DFT", fontsize=12, fontweight="bold")
    ax_a.legend(fontsize=9)
    ax_a.grid(alpha=0.3)

    # Panel (b): ln_A parity — empirical vs best DFT
    ax_b = fig.add_subplot(gs[0, 1])
    dft_keys_lnA = [k for k in kd_results.get("ln_A", {}) if "DFT" in k]
    best_dft_key_lnA = max(dft_keys_lnA, key=lambda k: kd_results["ln_A"][k]["q2"]) if dft_keys_lnA else None

    for key, marker, color, label_prefix in [
        (emp_key, "o", "#2171B5", "Empirical"),
        (best_dft_key_lnA, "^", "#E6550D", "Best DFT"),
    ]:
        if key and key in kd_results["ln_A"]:
            r = kd_results["ln_A"][key]
            ax_b.scatter(r["y_true"], r["y_fit"], s=80, marker=marker, c=color,
                         edgecolors="black", label=f"{label_prefix} (R²={r['r2']:.3f}, n={r['n']})")

    lims_b = [-5, 40]
    ax_b.plot(lims_b, lims_b, "k--", alpha=0.4)
    ax_b.set_xlim(lims_b)
    ax_b.set_ylim(lims_b)
    ax_b.set_xlabel("Observed ln(A)", fontsize=11)
    ax_b.set_ylabel("Predicted ln(A)", fontsize=11)
    ax_b.set_title("(b) ln(A) Prediction: Empirical vs DFT", fontsize=12, fontweight="bold")
    ax_b.legend(fontsize=9)
    ax_b.grid(alpha=0.3)

    # Panel (c): DFT descriptor correlation matrix with Ea
    ax_c = fig.add_subplot(gs[1, 0])
    dft_with_ea = merged_df.dropna(subset=["Ea"] + DFT_COLS)
    if len(dft_with_ea) >= 4:
        corrs = {}
        for col in DFT_COLS:
            r, p = pearsonr(dft_with_ea[col], dft_with_ea["Ea"])
            corrs[col.replace("dft_", "")] = r

        sorted_corrs = sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True)
        names_c = [c[0] for c in sorted_corrs]
        vals_c = [c[1] for c in sorted_corrs]
        colors_c = ["#2171B5" if v > 0 else "#E6550D" for v in vals_c]

        ax_c.barh(range(len(names_c)), vals_c, color=colors_c, edgecolor="white")
        ax_c.set_yticks(range(len(names_c)))
        ax_c.set_yticklabels(names_c, fontsize=10)
        ax_c.invert_yaxis()
        ax_c.axvline(0, color="black", linewidth=0.5)
        ax_c.set_xlabel("Pearson r with Ea", fontsize=11)
        ax_c.set_title("(c) DFT Descriptor Correlations with Ea", fontsize=12, fontweight="bold")
        ax_c.grid(axis="x", alpha=0.3)
        ax_c.set_xlim(-1, 1)

    # Panel (d): Model comparison bar chart
    ax_d = fig.add_subplot(gs[1, 1])
    model_names = ["Empirical\nEa R²", "Best DFT\nEa R²", "Empirical\nEa Q²", "Best DFT\nEa Q²",
                    "Direct Yield\nR²_LOSO"]
    model_vals = []
    for key in [emp_key, best_dft_key]:
        if key and key in kd_results["Ea"]:
            model_vals.append(kd_results["Ea"][key]["r2"])
        else:
            model_vals.append(0)
    for key in [emp_key, best_dft_key]:
        if key and key in kd_results["Ea"]:
            model_vals.append(kd_results["Ea"][key]["q2"])
        else:
            model_vals.append(0)
    model_vals.append(r2_yield)

    colors_d = ["#2171B5", "#E6550D", "#6BAED6", "#FC9272", "#31A354"]
    bars = ax_d.bar(range(len(model_names)), model_vals, color=colors_d, edgecolor="white")
    ax_d.set_xticks(range(len(model_names)))
    ax_d.set_xticklabels(model_names, fontsize=9)
    ax_d.set_ylabel("Score", fontsize=11)
    ax_d.set_title("(d) Model Performance Summary", fontsize=12, fontweight="bold")
    ax_d.grid(axis="y", alpha=0.3)
    ax_d.set_ylim(min(0, min(model_vals) - 0.1), 1.1)
    ax_d.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    for bar, val in zip(bars, model_vals):
        ax_d.text(bar.get_x() + bar.get_width() / 2, max(0, bar.get_height()) + 0.02,
                  f"{val:.3f}", ha="center", fontsize=9, fontweight="bold")

    fig.suptitle("Three-Rate Kinetic Analysis: Empirical vs DFT Descriptors",
                 fontsize=15, fontweight="bold")
    path = os.path.join(OUT_DIR, "three_rate_summary.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    report_lines.append(f"\n  Saved: {path}")
    print(f"  Saved: {path}")


# ========================================================================
#  Main
# ========================================================================

def main():
    report_lines = []
    report_lines.append("THREE-RATE KINETIC MODELING REPORT")
    report_lines.append("=" * 70)
    report_lines.append(f"Dataset: {DESCRIPTORS_CSV}")
    report_lines.append(f"Arrhenius: {ARRHENIUS_CSV}")

    print("Loading data...")
    arr, hl, desc = load_data()
    report_lines.append(f"Total rows: {len(desc)}")

    print("Merging Arrhenius with descriptors...")
    merged = merge_arrhenius_with_descriptors(arr, desc)
    report_lines.append(f"Arrhenius intermediates: {len(merged)}")
    report_lines.append(f"  with DFT: {merged['dft_charge_Li'].notna().sum()}")
    report_lines.append(f"  with sigma: {merged['sigma_hammett'].notna().sum()}")

    # Save merged summary
    merged.to_csv(os.path.join(OUT_DIR, "three_rate_summary.csv"), index=False)
    print(f"  Saved: {OUT_DIR}/three_rate_summary.csv")

    # Part 1: kd modeling
    print("\n--- Part 1: kd (decomposition) modeling ---")
    kd_results, feature_sets = model_kd(merged, report_lines)
    plot_model_comparison(kd_results, report_lines)
    plot_kd_parity(kd_results, report_lines)
    plot_kd_feature_importance(kd_results, report_lines)

    # Part 2: Direct yield prediction
    print("\n--- Part 2: Direct yield prediction ---")
    r2_yield, mae_yield = model_direct_yield(desc, report_lines)

    # Part 3: kf (formation) analysis
    print("\n--- Part 3: kf (formation rate) analysis ---")
    analyze_kf(hl, merged, report_lines)

    # Part 4: k2 (trapping) analysis
    print("\n--- Part 4: k2 (trapping rate) analysis ---")
    analyze_k2(desc, report_lines)

    # Part 5: Summary figure
    print("\n--- Part 5: Summary figure ---")
    plot_summary(kd_results, merged, r2_yield, report_lines)

    # Save report
    report_path = os.path.join(OUT_DIR, "model_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"\n  Report saved: {report_path}")

    # Print key results
    print("\n" + "=" * 70)
    print("KEY RESULTS")
    print("=" * 70)
    for target in ["Ea", "ln_A"]:
        if target in kd_results:
            print(f"\n{target}:")
            for name, r in kd_results[target].items():
                print(f"  {name.replace(chr(10), ' '):30s} R²={r['r2']:.4f}  Q²={r['q2']:.4f}  n={r['n']}")
    print(f"\nDirect yield LOSO: R²={r2_yield:.4f}, MAE={mae_yield:.1f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
