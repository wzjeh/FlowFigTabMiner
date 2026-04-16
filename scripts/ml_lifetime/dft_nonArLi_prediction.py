"""
DFT Descriptors for Non-ArLi Lifetime Prediction

The empirical LFER (σ, Es, δ_ortho, δ_benzyne) only covers ArLi.
This script evaluates whether DFT descriptors can:
  1. Predict Ea/t½ for the 3 non-ArLi intermediates with known Arrhenius data
  2. Extrapolate to ~50 intermediates without any kinetic measurements
  3. Provide a universal stability ranking across all organolithium classes

Usage:
  cd /Users/zhaowenyuan/Projects/FlowFigTabMiner
  source flowfigtabminer/bin/activate
  python scripts/ml_lifetime/dft_nonArLi_prediction.py
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data/ml_lifetime")
SUMMARY_CSV = os.path.join(DATA_DIR, "three_rate_analysis/three_rate_summary.csv")
DESCRIPTORS_CSV = os.path.join(DATA_DIR, "clean_organolithium_unified_descriptors.csv")
OUT_DIR = os.path.join(DATA_DIR, "three_rate_analysis")

R_GAS = 8.314e-3  # kJ/(mol·K)
T_REF = -40 + 273.15  # 233.15 K

DFT_COLS = [
    "dft_charge_Li", "dft_charge_C_ipso", "dft_HOMO_eV", "dft_LUMO_eV",
    "dft_LiC_bond_A", "dft_LiC_BDE_kJ", "dft_wiberg_LiC", "dft_dipole_D",
    "dft_Gsolv_kJ",
]


def load_data():
    merged = pd.read_csv(SUMMARY_CSV)
    merged = merged[merged["dft_charge_Li"].notna()].copy()
    desc = pd.read_csv(DESCRIPTORS_CSV)
    return merged, desc


def predict_ea_from_lna(Ea, ln_A, T=T_REF):
    """Compute t½ at temperature T from Arrhenius params."""
    kd = np.exp(ln_A - Ea / (R_GAS * T))
    return np.log(2) / kd


def main():
    report = []
    report.append("=" * 70)
    report.append("DFT DESCRIPTORS FOR NON-ArLi LIFETIME PREDICTION")
    report.append("=" * 70)

    merged, desc_full = load_data()
    n_total = len(merged)

    # Split: ArLi (has sigma) vs non-ArLi (no sigma)
    is_arli = merged["sigma_hammett"].notna()
    arli = merged[is_arli].copy()
    non_arli = merged[~is_arli].copy()

    report.append(f"\nArrhenius intermediates with DFT: {n_total}")
    report.append(f"  ArLi (has σ): {len(arli)}")
    report.append(f"  Non-ArLi (no σ): {len(non_arli)}")
    report.append(f"\nNon-ArLi intermediates:")
    for _, r in non_arli.iterrows():
        report.append(f"  {r['intermediate'][:50]:50s} Ea={r['Ea']:>5.1f}  "
                      f"t½(-40°C)={r['t_half_m40']:>8.3f}s")

    # ================================================================
    # Analysis 1: Train on 9 ArLi, predict 3 non-ArLi
    # ================================================================
    report.append("\n" + "=" * 70)
    report.append("ANALYSIS 1: Train on ArLi → Predict Non-ArLi")
    report.append("=" * 70)

    X_train = arli[DFT_COLS].values
    X_test = non_arli[DFT_COLS].values

    results_a1 = {}
    for target_name, target_col in [("Ea", "Ea"), ("ln_A", "ln_A")]:
        y_train = arli[target_col].values
        y_test = non_arli[target_col].values

        # Try multiple DFT feature subsets
        # a) All 9 DFT with Ridge
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_train)
        X_te_s = scaler.transform(X_test)

        # Find best alpha via LOOCV on ArLi
        best_alpha, best_q2 = 1.0, -1e10
        for alpha in [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]:
            loo = LeaveOneOut()
            y_loo = np.full_like(y_train, np.nan)
            for tr_idx, te_idx in loo.split(X_tr_s):
                reg = Ridge(alpha=alpha).fit(X_tr_s[tr_idx], y_train[tr_idx])
                y_loo[te_idx] = reg.predict(X_tr_s[te_idx])
            q2 = 1 - np.sum((y_train - y_loo)**2) / np.sum((y_train - y_train.mean())**2)
            if q2 > best_q2:
                best_q2, best_alpha = q2, alpha

        reg_ridge = Ridge(alpha=best_alpha).fit(X_tr_s, y_train)
        y_pred_ridge = reg_ridge.predict(X_te_s)
        y_fit_ridge = reg_ridge.predict(X_tr_s)

        # b) Best 2 DFT features (dipole_D, Gsolv_kJ)
        best2_idx = [DFT_COLS.index("dft_dipole_D"), DFT_COLS.index("dft_Gsolv_kJ")]
        X_tr_2 = X_train[:, best2_idx]
        X_te_2 = X_test[:, best2_idx]
        reg_2 = LinearRegression().fit(X_tr_2, y_train)
        y_pred_2 = reg_2.predict(X_te_2)
        y_fit_2 = reg_2.predict(X_tr_2)
        r2_2 = 1 - np.sum((y_train - y_fit_2)**2) / np.sum((y_train - y_train.mean())**2)

        # c) Best 3 DFT features (dipole_D, Gsolv_kJ, LiC_BDE_kJ)
        best3_idx = [DFT_COLS.index("dft_dipole_D"), DFT_COLS.index("dft_Gsolv_kJ"),
                     DFT_COLS.index("dft_LiC_BDE_kJ")]
        X_tr_3 = X_train[:, best3_idx]
        X_te_3 = X_test[:, best3_idx]
        reg_3 = LinearRegression().fit(X_tr_3, y_train)
        y_pred_3 = reg_3.predict(X_te_3)
        y_fit_3 = reg_3.predict(X_tr_3)
        r2_3 = 1 - np.sum((y_train - y_fit_3)**2) / np.sum((y_train - y_train.mean())**2)

        # d) Single best: dipole_D only
        dip_idx = DFT_COLS.index("dft_dipole_D")
        X_tr_1 = X_train[:, dip_idx:dip_idx+1]
        X_te_1 = X_test[:, dip_idx:dip_idx+1]
        reg_1 = LinearRegression().fit(X_tr_1, y_train)
        y_pred_1 = reg_1.predict(X_te_1)
        y_fit_1 = reg_1.predict(X_tr_1)
        r2_1 = 1 - np.sum((y_train - y_fit_1)**2) / np.sum((y_train - y_train.mean())**2)

        results_a1[target_name] = {
            "y_test": y_test,
            "ridge": {"pred": y_pred_ridge, "alpha": best_alpha, "q2_train": best_q2},
            "best2": {"pred": y_pred_2, "r2_train": r2_2},
            "best3": {"pred": y_pred_3, "r2_train": r2_3},
            "best1": {"pred": y_pred_1, "r2_train": r2_1},
        }

        report.append(f"\n  --- {target_name} ---")
        models = [
            ("dipole only", y_pred_1, r2_1),
            ("dipole+Gsolv", y_pred_2, r2_2),
            ("dipole+Gsolv+BDE", y_pred_3, r2_3),
            (f"Ridge-9 (α={best_alpha})", y_pred_ridge, best_q2),
        ]
        report.append(f"  {'Model':25s} {'R²_train':>8s}  " +
                      "  ".join(f"{n[:12]:>12s}" for n in non_arli["intermediate"].str[:12].values))
        report.append(f"  {'':25s} {'':>8s}  " +
                      "  ".join(f"{v:>12.2f}" for v in y_test))

        for mname, pred, r2_or_q2 in models:
            report.append(f"  {mname:25s} {r2_or_q2:>8.3f}  " +
                          "  ".join(f"{v:>12.2f}" for v in pred))

        # Prediction errors
        report.append(f"\n  Absolute errors ({target_name}):")
        for mname, pred, _ in models:
            errs = np.abs(pred - y_test)
            report.append(f"    {mname:25s} " +
                          "  ".join(f"{e:>8.2f}" for e in errs) +
                          f"  MAE={np.mean(errs):.2f}")

    # ================================================================
    # Analysis 2: LOOCV on all 12 — focus on non-ArLi LOO errors
    # ================================================================
    report.append("\n" + "=" * 70)
    report.append("ANALYSIS 2: LOOCV on All 12 — Non-ArLi LOO Errors")
    report.append("=" * 70)

    X_all = merged[DFT_COLS].values
    names_all = merged["intermediate"].values

    for target_name, target_col in [("Ea", "Ea"), ("ln_A", "ln_A")]:
        y_all = merged[target_col].values

        # DFT-best2 LOOCV
        best2_idx = [DFT_COLS.index("dft_dipole_D"), DFT_COLS.index("dft_Gsolv_kJ")]
        X_2 = X_all[:, best2_idx]
        y_loo = np.zeros(len(y_all))
        for i in range(len(y_all)):
            mask = np.ones(len(y_all), dtype=bool)
            mask[i] = False
            reg = LinearRegression().fit(X_2[mask], y_all[mask])
            y_loo[i] = reg.predict(X_2[i:i+1])[0]

        errors = y_all - y_loo
        report.append(f"\n  {target_name} LOO errors (DFT-best2, all 12):")
        report.append(f"  {'Intermediate':45s} {'Observed':>8s} {'LOO pred':>8s} {'Error':>8s} {'Type':>8s}")
        for i in range(len(y_all)):
            itype = "non-ArLi" if pd.isna(merged.iloc[i]["sigma_hammett"]) else "ArLi"
            report.append(f"  {names_all[i][:45]:45s} {y_all[i]:>8.2f} {y_loo[i]:>8.2f} "
                          f"{errors[i]:>+8.2f} {itype:>8s}")

        arli_errs = errors[is_arli.values]
        non_errs = errors[~is_arli.values]
        report.append(f"\n  ArLi MAE: {np.mean(np.abs(arli_errs)):.2f}")
        report.append(f"  Non-ArLi MAE: {np.mean(np.abs(non_errs)):.2f}")

    # ================================================================
    # Analysis 3: Predict t½ for all 53 unique non-ArLi intermediates
    # ================================================================
    report.append("\n" + "=" * 70)
    report.append("ANALYSIS 3: Predicted Stability Ranking (All Intermediates)")
    report.append("=" * 70)

    # Build model on all 12 known intermediates (DFT-best2: dipole + Gsolv)
    X_12 = merged[["dft_dipole_D", "dft_Gsolv_kJ"]].values
    Ea_12 = merged["Ea"].values
    lnA_12 = merged["ln_A"].values

    reg_Ea = LinearRegression().fit(X_12, Ea_12)
    reg_lnA = LinearRegression().fit(X_12, lnA_12)

    report.append(f"\n  Model trained on {len(merged)} intermediates:")
    report.append(f"  Ea = {reg_Ea.intercept_:.2f} {reg_Ea.coef_[0]:+.3f}·dipole {reg_Ea.coef_[1]:+.4f}·Gsolv")
    report.append(f"  ln_A = {reg_lnA.intercept_:.2f} {reg_lnA.coef_[0]:+.3f}·dipole {reg_lnA.coef_[1]:+.4f}·Gsolv")

    # Get all unique intermediates with DFT
    uniq = desc_full[desc_full["dft_charge_Li"].notna()].drop_duplicates(
        subset=["intermediate_smiles_canonical"]
    ).copy()

    X_pred = uniq[["dft_dipole_D", "dft_Gsolv_kJ"]].values
    Ea_pred = reg_Ea.predict(X_pred)
    lnA_pred = reg_lnA.predict(X_pred)

    # Compute predicted t½ at -40°C
    kd_pred = np.exp(lnA_pred - Ea_pred / (R_GAS * T_REF))
    t_half_pred = np.log(2) / np.maximum(kd_pred, 1e-30)
    # Clip unreasonable values
    t_half_pred = np.clip(t_half_pred, 1e-6, 1e12)

    uniq = uniq.copy()
    uniq["pred_Ea"] = Ea_pred
    uniq["pred_lnA"] = lnA_pred
    uniq["pred_t_half_m40"] = t_half_pred
    uniq["pred_log_t_half"] = np.log10(t_half_pred)

    # Check if we have observed values
    arr_lookup = dict(zip(merged["smiles_canonical"].dropna(), merged["t_half_m40"]))
    uniq["obs_t_half"] = uniq["intermediate_smiles_canonical"].map(arr_lookup)

    # Sort by predicted t½
    uniq_sorted = uniq.sort_values("pred_t_half_m40", ascending=False)

    report.append(f"\n  Predicted stability ranking (t½ at -40°C, {len(uniq_sorted)} intermediates):")
    report.append(f"  {'#':>3s} {'Intermediate':45s} {'Class':12s} {'dipole':>7s} {'Gsolv':>7s} "
                  f"{'Ea_pred':>7s} {'t½_pred':>10s} {'t½_obs':>10s}")

    for rank, (_, r) in enumerate(uniq_sorted.iterrows(), 1):
        obs_str = f"{r['obs_t_half']:.3f}" if pd.notna(r['obs_t_half']) else "--"
        t_str = f"{r['pred_t_half_m40']:.3g}"
        report.append(f"  {rank:>3d} {r['intermediate'][:45]:45s} {r['intermediate_class'][:12]:12s} "
                      f"{r['dft_dipole_D']:>7.2f} {r['dft_Gsolv_kJ']:>7.1f} "
                      f"{r['pred_Ea']:>7.1f} {t_str:>10s} {obs_str:>10s}")

    # ================================================================
    # FIGURE: 4-panel analysis
    # ================================================================
    fig = plt.figure(figsize=(18, 16))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    # Panel (a): DFT descriptor space — dipole vs Gsolv, all intermediates
    ax = fig.add_subplot(gs[0, 0])

    # Background: all unique intermediates colored by class
    class_colors = {
        "ArLi": "#2171B5", "oxiranylLi": "#E6550D", "vinylLi": "#31A354",
        "carbanion": "#756BB1", "benzylLi": "#D6616B", "alkylLi": "#8C6D31",
        "carbenoid": "#636363", "aziridinylLi": "#E7969C",
    }
    for cls, grp in uniq.groupby("intermediate_class"):
        color = class_colors.get(cls, "#999999")
        ax.scatter(grp["dft_dipole_D"], grp["dft_Gsolv_kJ"],
                   s=40, c=color, alpha=0.5, edgecolors="white", label=f"{cls} ({len(grp)})")

    # Overlay: 12 known intermediates with black edge
    for _, r in merged.iterrows():
        marker = "s" if pd.isna(r["sigma_hammett"]) else "o"
        edge = "red" if pd.isna(r["sigma_hammett"]) else "black"
        lw = 2 if pd.isna(r["sigma_hammett"]) else 1
        ax.scatter(r["dft_dipole_D"], r["dft_Gsolv_kJ"], s=120,
                   c="gold", marker=marker, edgecolors=edge, linewidths=lw, zorder=10)
        ax.annotate(r["intermediate"][:15], (r["dft_dipole_D"], r["dft_Gsolv_kJ"]),
                    fontsize=6, xytext=(4, 4), textcoords="offset points")

    ax.set_xlabel("Dipole Moment (D)", fontsize=12)
    ax.set_ylabel("Gsolv (kJ/mol)", fontsize=12)
    ax.set_title("(a) DFT Descriptor Space\n(gold = known Arrhenius; □ = non-ArLi)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=7, loc="lower left", ncol=2)
    ax.grid(alpha=0.3)

    # Panel (b): Train-on-ArLi, predict non-ArLi parity
    ax = fig.add_subplot(gs[0, 1])
    y_Ea_test = results_a1["Ea"]["y_test"]
    y_Ea_pred_models = {
        "dipole only": results_a1["Ea"]["best1"]["pred"],
        "dipole+Gsolv": results_a1["Ea"]["best2"]["pred"],
        "dipole+Gsolv+BDE": results_a1["Ea"]["best3"]["pred"],
    }
    markers = {"dipole only": "o", "dipole+Gsolv": "s", "dipole+Gsolv+BDE": "^"}
    colors_m = {"dipole only": "#AAAAAA", "dipole+Gsolv": "#E6550D", "dipole+Gsolv+BDE": "#31A354"}

    # Also show ArLi training fit
    best2_idx = [DFT_COLS.index("dft_dipole_D"), DFT_COLS.index("dft_Gsolv_kJ")]
    X_tr_2 = arli[DFT_COLS].values[:, best2_idx]
    reg_2_ea = LinearRegression().fit(X_tr_2, arli["Ea"].values)
    y_fit_arli = reg_2_ea.predict(X_tr_2)
    ax.scatter(arli["Ea"].values, y_fit_arli, s=60, c="#2171B5", alpha=0.5,
               edgecolors="black", label=f"ArLi train (R²={reg_2_ea.score(X_tr_2, arli['Ea'].values):.3f})", zorder=4)

    for mname in ["dipole+Gsolv"]:
        pred = y_Ea_pred_models[mname]
        ax.scatter(y_Ea_test, pred, s=150, c=colors_m[mname], marker=markers[mname],
                   edgecolors="black", linewidths=2, zorder=10,
                   label=f"Non-ArLi test ({mname})")
        for i in range(len(y_Ea_test)):
            ax.annotate(non_arli.iloc[i]["intermediate"][:18],
                        (y_Ea_test[i], pred[i]),
                        fontsize=7, xytext=(5, 5), textcoords="offset points", fontweight="bold")

    lims = [0, 75]
    ax.plot(lims, lims, "k--", alpha=0.4)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Observed Ea (kJ/mol)", fontsize=12)
    ax.set_ylabel("Predicted Ea (kJ/mol)", fontsize=12)
    ax.set_title("(b) Ea: Train on ArLi → Predict Non-ArLi\n(dipole + Gsolv model)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Panel (c): Predicted t½ ranking — horizontal bar
    ax = fig.add_subplot(gs[1, 0])
    # Show top-20 most stable and bottom-5 least stable
    top_n = 25
    ranking = uniq_sorted.head(top_n).copy()
    y_pos = np.arange(len(ranking))
    log_t = ranking["pred_log_t_half"].values
    has_obs = ranking["obs_t_half"].notna().values

    # Color by class
    bar_colors = [class_colors.get(cls, "#999999") for cls in ranking["intermediate_class"]]
    bars = ax.barh(y_pos, log_t, color=bar_colors, edgecolor="white", height=0.7)

    # Overlay observed values
    for i, (_, r) in enumerate(ranking.iterrows()):
        if pd.notna(r["obs_t_half"]):
            obs_log = np.log10(r["obs_t_half"])
            ax.scatter(obs_log, i, s=100, c="gold", edgecolors="black",
                       zorder=10, marker="*")

    ax.set_yticks(y_pos)
    labels = [f"{r['intermediate'][:35]} ({r['intermediate_class'][:8]})"
              for _, r in ranking.iterrows()]
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("log₁₀(t½ / s) at -40°C", fontsize=12)
    ax.set_title(f"(c) Predicted Stability Ranking (top {top_n})\n(★ = observed t½)",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    ax.axvline(0, color="black", linewidth=0.5)

    # Panel (d): dipole vs Ea — the key univariate relationship
    ax = fig.add_subplot(gs[1, 1])
    # All 12 known
    for _, r in merged.iterrows():
        is_non = pd.isna(r["sigma_hammett"])
        color = "#E6550D" if is_non else "#2171B5"
        marker = "s" if is_non else "o"
        size = 120 if is_non else 80
        ax.scatter(r["dft_dipole_D"], r["Ea"], s=size, c=color, marker=marker,
                   edgecolors="black", zorder=5)
        ax.annotate(r["intermediate"][:20], (r["dft_dipole_D"], r["Ea"]),
                    fontsize=6, xytext=(4, 4), textcoords="offset points")

    # Regression line through all 12
    x_dip = merged["dft_dipole_D"].values
    y_ea = merged["Ea"].values
    reg_dip = LinearRegression().fit(x_dip.reshape(-1, 1), y_ea)
    x_line = np.linspace(x_dip.min() - 0.5, x_dip.max() + 0.5, 100)
    ax.plot(x_line, reg_dip.predict(x_line.reshape(-1, 1)), "k--", alpha=0.5)
    r_val, p_val = pearsonr(x_dip, y_ea)
    ax.text(0.95, 0.95, f"r = {r_val:.3f}\np = {p_val:.3f}\nn = {len(x_dip)}",
            transform=ax.transAxes, fontsize=10, va="top", ha="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    ax.set_xlabel("Dipole Moment (D)", fontsize=12)
    ax.set_ylabel("Ea (kJ/mol)", fontsize=12)
    ax.set_title("(d) Dipole vs Ea (all 12 known intermediates)\n(● ArLi  ■ non-ArLi)",
                 fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3)

    fig.suptitle("DFT-Based Lifetime Prediction: Beyond Empirical LFER",
                 fontsize=15, fontweight="bold")
    path = os.path.join(OUT_DIR, "dft_nonArLi_prediction.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    report.append(f"\n  Saved: {path}")
    print(f"  Saved: {path}")

    # ================================================================
    # FIGURE 2: Train-on-ArLi extrapolation detail
    # ================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, (target, unit) in zip(axes[:2], [("Ea", "kJ/mol"), ("ln_A", "")]):
        y_test = results_a1[target]["y_test"]
        # Show all models
        models_to_show = [
            ("dipole", results_a1[target]["best1"]["pred"], "#AAAAAA", "o"),
            ("dipole+Gsolv", results_a1[target]["best2"]["pred"], "#E6550D", "s"),
            ("dip+Gsolv+BDE", results_a1[target]["best3"]["pred"], "#31A354", "^"),
            (f"Ridge-9", results_a1[target]["ridge"]["pred"], "#756BB1", "D"),
        ]
        x_jitter = np.arange(len(y_test))
        width = 0.18
        for mi, (mname, pred, color, marker) in enumerate(models_to_show):
            offsets = x_jitter + (mi - 1.5) * width
            ax.scatter(offsets, pred, s=80, c=color, marker=marker, edgecolors="black",
                       zorder=5, label=mname)

        # Observed values
        for i in range(len(y_test)):
            ax.axhline(y_test[i], xmin=(i - 0.3) / len(y_test),
                       xmax=(i + 1.3) / len(y_test),
                       color="red", linewidth=2, alpha=0.5)

        ax.set_xticks(x_jitter)
        ax.set_xticklabels([n[:15] for n in non_arli["intermediate"].values],
                           fontsize=8, rotation=15)
        ax.set_ylabel(f"Predicted {target} {unit}", fontsize=12)
        ax.set_title(f"{target}: Model Comparison\n(red line = observed)", fontsize=12, fontweight="bold")
        ax.legend(fontsize=8, loc="best")
        ax.grid(axis="y", alpha=0.3)

    # Panel 3: Predicted vs observed t½
    ax = axes[2]
    # Compute predicted t½ for non-ArLi from each model
    for mname, color in [("dipole+Gsolv", "#E6550D")]:
        Ea_p = results_a1["Ea"]["best2"]["pred"]
        lnA_p = results_a1["ln_A"]["best2"]["pred"]
        kd_p = np.exp(lnA_p - Ea_p / (R_GAS * T_REF))
        t_half_p = np.log(2) / np.maximum(kd_p, 1e-30)

        t_half_obs = non_arli["t_half_m40"].values
        ax.scatter(np.log10(t_half_obs), np.log10(t_half_p), s=120,
                   c=color, edgecolors="black", zorder=5, label=mname)
        for i in range(len(t_half_obs)):
            ax.annotate(non_arli.iloc[i]["intermediate"][:18],
                        (np.log10(t_half_obs[i]), np.log10(t_half_p[i])),
                        fontsize=7, xytext=(5, 5), textcoords="offset points")

    # Also show ArLi
    Ea_fit_arli = reg_2_ea.predict(X_tr_2)
    reg_2_lnA = LinearRegression().fit(X_tr_2, arli["ln_A"].values)
    lnA_fit_arli = reg_2_lnA.predict(X_tr_2)
    kd_fit_arli = np.exp(lnA_fit_arli - Ea_fit_arli / (R_GAS * T_REF))
    t_half_fit_arli = np.log(2) / np.maximum(kd_fit_arli, 1e-30)
    t_half_obs_arli = arli["t_half_m40"].values

    ax.scatter(np.log10(t_half_obs_arli), np.log10(t_half_fit_arli), s=60,
               c="#2171B5", alpha=0.5, edgecolors="black", zorder=4, label="ArLi (train)")

    all_log_obs = np.concatenate([np.log10(t_half_obs), np.log10(t_half_obs_arli)])
    all_log_pred = np.concatenate([np.log10(t_half_p), np.log10(t_half_fit_arli)])
    lims_t = [min(all_log_obs.min(), all_log_pred.min()) - 0.5,
              max(all_log_obs.max(), all_log_pred.max()) + 0.5]
    ax.plot(lims_t, lims_t, "k--", alpha=0.4)
    ax.set_xlim(lims_t)
    ax.set_ylim(lims_t)
    ax.set_xlabel("Observed log₁₀(t½ / s)", fontsize=12)
    ax.set_ylabel("Predicted log₁₀(t½ / s)", fontsize=12)
    ax.set_title("t½ at -40°C: Parity Plot\n(dipole + Gsolv model)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle("DFT Model Extrapolation: ArLi → Non-ArLi", fontsize=14, fontweight="bold")
    fig.tight_layout()
    path2 = os.path.join(OUT_DIR, "dft_nonArLi_extrapolation.png")
    fig.savefig(path2, dpi=200, bbox_inches="tight")
    plt.close(fig)
    report.append(f"  Saved: {path2}")
    print(f"  Saved: {path2}")

    # ================================================================
    # Analysis 4: Applicability domain — how far are non-ArLi from training?
    # ================================================================
    report.append("\n" + "=" * 70)
    report.append("ANALYSIS 4: Applicability Domain")
    report.append("=" * 70)

    # Leverage-based AD: hat values of new points relative to training set
    X_tr_full = arli[DFT_COLS].values
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr_full)

    # Hat matrix for training: H = X(X'X)^-1 X'
    XtX_inv = np.linalg.pinv(X_tr_s.T @ X_tr_s)
    avg_h = (len(DFT_COLS)) / len(arli)  # average leverage

    # Leverage of non-ArLi points
    X_te_s = scaler.transform(non_arli[DFT_COLS].values)
    h_new = np.array([x @ XtX_inv @ x.T for x in X_te_s])

    report.append(f"\n  Average training leverage: {avg_h:.3f}")
    report.append(f"  3× threshold: {3*avg_h:.3f}")
    report.append(f"\n  Non-ArLi leverage (9-DFT descriptor space):")
    for i, (_, r) in enumerate(non_arli.iterrows()):
        flag = " ⚠️ OUTSIDE AD" if h_new[i] > 3 * avg_h else " ✓ inside"
        report.append(f"    {r['intermediate'][:40]:40s} h={h_new[i]:.3f}{flag}")

    # Also for all unique intermediates (2-feature model)
    X_tr_2f = arli[["dft_dipole_D", "dft_Gsolv_kJ"]].values
    scaler_2f = StandardScaler()
    X_tr_2f_s = scaler_2f.fit_transform(X_tr_2f)
    XtX_inv_2f = np.linalg.pinv(X_tr_2f_s.T @ X_tr_2f_s)
    avg_h_2f = 2 / len(arli)

    X_all_2f = uniq[["dft_dipole_D", "dft_Gsolv_kJ"]].values
    X_all_2f_s = scaler_2f.transform(X_all_2f)
    h_all = np.array([x @ XtX_inv_2f @ x.T for x in X_all_2f_s])
    uniq_sorted["leverage_2f"] = uniq_sorted.index.map(
        dict(zip(uniq.index, h_all))
    )

    n_inside = np.sum(h_all <= 3 * avg_h_2f)
    report.append(f"\n  Applicability domain (dipole+Gsolv model):")
    report.append(f"    Inside AD (h ≤ {3*avg_h_2f:.3f}): {n_inside}/{len(h_all)} intermediates")
    report.append(f"    Outside AD: {len(h_all) - n_inside}/{len(h_all)}")

    # Save report
    report_path = os.path.join(OUT_DIR, "dft_nonArLi_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(report))
    print(f"\n  Report saved: {report_path}")

    # Save ranking CSV
    ranking_cols = ["intermediate", "intermediate_class", "intermediate_smiles_canonical",
                    "dft_dipole_D", "dft_Gsolv_kJ", "dft_LiC_BDE_kJ",
                    "pred_Ea", "pred_lnA", "pred_t_half_m40", "pred_log_t_half", "obs_t_half"]
    ranking_out = uniq_sorted[[c for c in ranking_cols if c in uniq_sorted.columns]].copy()
    ranking_path = os.path.join(OUT_DIR, "predicted_stability_ranking.csv")
    ranking_out.to_csv(ranking_path, index=False)
    print(f"  Ranking saved: {ranking_path}")

    # Summary
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)
    # Prediction errors for non-ArLi
    Ea_obs = non_arli["Ea"].values
    Ea_pred_2 = results_a1["Ea"]["best2"]["pred"]
    lnA_obs = non_arli["ln_A"].values
    lnA_pred_2 = results_a1["ln_A"]["best2"]["pred"]
    print(f"\nTrain-on-ArLi → Predict non-ArLi (dipole+Gsolv):")
    for i, (_, r) in enumerate(non_arli.iterrows()):
        print(f"  {r['intermediate'][:35]:35s} Ea: {Ea_obs[i]:.1f} → {Ea_pred_2[i]:.1f} "
              f"(Δ={Ea_pred_2[i]-Ea_obs[i]:+.1f})  "
              f"ln_A: {lnA_obs[i]:.1f} → {lnA_pred_2[i]:.1f} "
              f"(Δ={lnA_pred_2[i]-lnA_obs[i]:+.1f})")

    print(f"\nEa MAE (non-ArLi): {np.mean(np.abs(Ea_pred_2 - Ea_obs)):.1f} kJ/mol")
    print(f"ln_A MAE (non-ArLi): {np.mean(np.abs(lnA_pred_2 - lnA_obs)):.1f}")
    print(f"\nPredicted stability ranking saved: {ranking_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
