"""
Emp+DFT-2 组合模型诊断分析

对 Ea 和 ln_A 的 6 参数组合模型 (σ, Es, δ_ortho, δ_benzyne, dipole_D, Gsolv_kJ)
进行残差分析、共线性诊断和稳健性检验。

Usage:
  cd /Users/zhaowenyuan/Projects/FlowFigTabMiner
  source flowfigtabminer/bin/activate
  python scripts/ml_lifetime/diagnose_combined_model.py
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data/ml_lifetime")
SUMMARY_CSV = os.path.join(DATA_DIR, "three_rate_analysis/three_rate_summary.csv")
OUT_DIR = os.path.join(DATA_DIR, "three_rate_analysis")

FEAT_COLS = ["sigma_hammett", "Es_taft", "delta_ortho", "delta_benzyne",
             "dft_dipole_D", "dft_Gsolv_kJ"]
FEAT_SHORT = ["σ", "Es", "δ_ortho", "δ_benzyne", "dipole", "Gsolv"]


def load_model_data():
    """Load merged data and filter to Emp+DFT-2 subset (n=9)."""
    df = pd.read_csv(SUMMARY_CSV)
    mask = df["sigma_hammett"].notna() & df["dft_dipole_D"].notna()
    sub = df[mask].copy()
    return sub


def fit_ols(X, y):
    """Fit OLS and return full diagnostics."""
    n, p = X.shape
    reg = LinearRegression().fit(X, y)
    y_fit = reg.predict(X)
    resid = y - y_fit

    # Hat matrix H = X (X'X)^-1 X'
    X1 = np.column_stack([np.ones(n), X])  # add intercept column
    try:
        H = X1 @ np.linalg.inv(X1.T @ X1) @ X1.T
    except np.linalg.LinAlgError:
        H = X1 @ np.linalg.pinv(X1.T @ X1) @ X1.T
    leverage = np.diag(H)

    # Degrees of freedom
    df_resid = n - p - 1  # n - (p features + 1 intercept)

    # MSE
    SSE = np.sum(resid ** 2)
    MSE = SSE / df_resid if df_resid > 0 else np.inf

    # Studentized residuals (internally)
    sigma_hat = np.sqrt(MSE)
    int_student = resid / (sigma_hat * np.sqrt(np.maximum(1 - leverage, 1e-10)))

    # Externally studentized (leave-one-out)
    ext_student = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        reg_i = LinearRegression().fit(X[mask], y[mask])
        y_pred_i = reg_i.predict(X[i:i+1])[0]
        e_i = y[i] - y_pred_i
        # s_(i) from LOO
        y_fit_i = reg_i.predict(X[mask])
        SSE_i = np.sum((y[mask] - y_fit_i) ** 2)
        df_i = (n - 1) - p - 1
        MSE_i = SSE_i / df_i if df_i > 0 else np.inf
        s_i = np.sqrt(MSE_i)
        ext_student[i] = e_i / (s_i * np.sqrt(max(1 + leverage[i], 1e-10)))
        # Note: for externally studentized, denominator uses (1-h_ii) not (1+h_ii)
        # Correct formula: t_i = e_i / (s_(i) * sqrt(1 - h_ii))
        ext_student[i] = e_i / (s_i * np.sqrt(max(1 - leverage[i], 1e-10)))

    # Cook's distance
    cooks_d = (int_student ** 2 / (p + 1)) * (leverage / np.maximum(1 - leverage, 1e-10))

    # DFFITS
    dffits = ext_student * np.sqrt(leverage / np.maximum(1 - leverage, 1e-10))

    # LOO predictions
    y_loo = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        reg_i = LinearRegression().fit(X[mask], y[mask])
        y_loo[i] = reg_i.predict(X[i:i+1])[0]

    return {
        "reg": reg, "y_fit": y_fit, "resid": resid, "y_loo": y_loo,
        "leverage": leverage, "int_student": int_student,
        "ext_student": ext_student, "cooks_d": cooks_d, "dffits": dffits,
        "MSE": MSE, "df_resid": df_resid, "H": H,
    }


def compute_vif(X):
    """Compute Variance Inflation Factor for each feature."""
    n, p = X.shape
    vifs = np.zeros(p)
    for j in range(p):
        # Regress x_j on all other x's
        others = np.delete(X, j, axis=1)
        reg = LinearRegression().fit(others, X[:, j])
        r2 = reg.score(others, X[:, j])
        vifs[j] = 1.0 / (1.0 - r2) if r2 < 1.0 else np.inf
    return vifs


def condition_number(X):
    """Compute condition number of the design matrix (with intercept)."""
    X1 = np.column_stack([np.ones(X.shape[0]), X])
    # Standardize columns (except intercept) for condition number
    scaler = StandardScaler()
    X_std = np.column_stack([np.ones(X.shape[0]), scaler.fit_transform(X)])
    sv = np.linalg.svd(X_std, compute_uv=False)
    return sv.max() / sv.min(), sv


def bootstrap_coefficients(X, y, n_boot=2000, seed=42):
    """Bootstrap confidence intervals for OLS coefficients."""
    rng = np.random.RandomState(seed)
    n = len(y)
    p = X.shape[1]
    coefs_boot = np.zeros((n_boot, p + 1))  # +1 for intercept

    for b in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        reg = LinearRegression().fit(X[idx], y[idx])
        coefs_boot[b, 0] = reg.intercept_
        coefs_boot[b, 1:] = reg.coef_

    return coefs_boot


def plot_diagnostics(sub, diag_Ea, diag_lnA, report):
    """Generate comprehensive diagnostic plots."""

    names = sub["intermediate"].str[:25].values
    X = sub[FEAT_COLS].values

    fig = plt.figure(figsize=(20, 24))
    gs = GridSpec(4, 3, figure=fig, hspace=0.4, wspace=0.35)

    # ── Row 1: Residual plots (Ea) ──

    # (a) Residuals vs Fitted
    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(diag_Ea["y_fit"], diag_Ea["resid"], s=80, c="#2171B5",
               edgecolors="black", zorder=5)
    ax.axhline(0, color="red", linestyle="--", alpha=0.5)
    for i, nm in enumerate(names):
        ax.annotate(nm, (diag_Ea["y_fit"][i], diag_Ea["resid"][i]),
                    fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Fitted Ea (kJ/mol)", fontsize=11)
    ax.set_ylabel("Residual (kJ/mol)", fontsize=11)
    ax.set_title("(a) Ea: Residuals vs Fitted", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3)

    # (b) Q-Q plot
    ax = fig.add_subplot(gs[0, 1])
    sorted_resid = np.sort(diag_Ea["int_student"])
    n = len(sorted_resid)
    theoretical = stats.norm.ppf((np.arange(1, n + 1) - 0.375) / (n + 0.25))
    ax.scatter(theoretical, sorted_resid, s=80, c="#2171B5", edgecolors="black", zorder=5)
    lim = max(abs(theoretical).max(), abs(sorted_resid).max()) + 0.5
    ax.plot([-lim, lim], [-lim, lim], "r--", alpha=0.5)
    for i in range(n):
        idx_orig = np.argsort(diag_Ea["int_student"])[i]
        ax.annotate(names[idx_orig], (theoretical[i], sorted_resid[i]),
                    fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Theoretical Quantiles", fontsize=11)
    ax.set_ylabel("Studentized Residuals", fontsize=11)
    ax.set_title("(b) Ea: Normal Q-Q Plot", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3)

    # Shapiro-Wilk test
    if n >= 3:
        sw_stat, sw_p = stats.shapiro(diag_Ea["resid"])
        ax.text(0.05, 0.95, f"Shapiro-Wilk p={sw_p:.3f}",
                transform=ax.transAxes, fontsize=9, va="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
        report.append(f"  Ea Shapiro-Wilk: W={sw_stat:.4f}, p={sw_p:.3f}")

    # (c) Cook's distance
    ax = fig.add_subplot(gs[0, 2])
    x_pos = np.arange(n)
    colors_cook = ["#E6550D" if c > 4.0 / n else "#2171B5" for c in diag_Ea["cooks_d"]]
    ax.bar(x_pos, diag_Ea["cooks_d"], color=colors_cook, edgecolor="white")
    ax.axhline(4.0 / n, color="red", linestyle="--", alpha=0.7, label=f"4/n = {4.0/n:.2f}")
    ax.axhline(1.0, color="darkred", linestyle=":", alpha=0.5, label="Cook's D = 1")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Cook's Distance", fontsize=11)
    ax.set_title("(c) Ea: Cook's Distance", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # ── Row 2: Residual plots (ln_A) ──

    ax = fig.add_subplot(gs[1, 0])
    ax.scatter(diag_lnA["y_fit"], diag_lnA["resid"], s=80, c="#E6550D",
               edgecolors="black", zorder=5)
    ax.axhline(0, color="red", linestyle="--", alpha=0.5)
    for i, nm in enumerate(names):
        ax.annotate(nm, (diag_lnA["y_fit"][i], diag_lnA["resid"][i]),
                    fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Fitted ln(A)", fontsize=11)
    ax.set_ylabel("Residual", fontsize=11)
    ax.set_title("(d) ln_A: Residuals vs Fitted", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[1, 1])
    sorted_resid_lnA = np.sort(diag_lnA["int_student"])
    theoretical_lnA = stats.norm.ppf((np.arange(1, n + 1) - 0.375) / (n + 0.25))
    ax.scatter(theoretical_lnA, sorted_resid_lnA, s=80, c="#E6550D", edgecolors="black", zorder=5)
    ax.plot([-lim, lim], [-lim, lim], "r--", alpha=0.5)
    ax.set_xlabel("Theoretical Quantiles", fontsize=11)
    ax.set_ylabel("Studentized Residuals", fontsize=11)
    ax.set_title("(e) ln_A: Normal Q-Q Plot", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3)
    if n >= 3:
        sw_stat2, sw_p2 = stats.shapiro(diag_lnA["resid"])
        ax.text(0.05, 0.95, f"Shapiro-Wilk p={sw_p2:.3f}",
                transform=ax.transAxes, fontsize=9, va="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
        report.append(f"  ln_A Shapiro-Wilk: W={sw_stat2:.4f}, p={sw_p2:.3f}")

    ax = fig.add_subplot(gs[1, 2])
    colors_cook2 = ["#E6550D" if c > 4.0 / n else "#2171B5" for c in diag_lnA["cooks_d"]]
    ax.bar(x_pos, diag_lnA["cooks_d"], color=colors_cook2, edgecolor="white")
    ax.axhline(4.0 / n, color="red", linestyle="--", alpha=0.7, label=f"4/n = {4.0/n:.2f}")
    ax.axhline(1.0, color="darkred", linestyle=":", alpha=0.5, label="Cook's D = 1")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Cook's Distance", fontsize=11)
    ax.set_title("(f) ln_A: Cook's Distance", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # ── Row 3: Collinearity diagnostics ──

    # (g) VIF bar chart
    ax = fig.add_subplot(gs[2, 0])
    vifs = compute_vif(X)
    colors_vif = ["#E6550D" if v > 5 else "#FDD0A2" if v > 2.5 else "#2171B5" for v in vifs]
    ax.barh(range(len(FEAT_SHORT)), vifs, color=colors_vif, edgecolor="white")
    ax.set_yticks(range(len(FEAT_SHORT)))
    ax.set_yticklabels(FEAT_SHORT, fontsize=11)
    ax.invert_yaxis()
    ax.axvline(5, color="red", linestyle="--", alpha=0.7, label="VIF=5 (concern)")
    ax.axvline(10, color="darkred", linestyle=":", alpha=0.5, label="VIF=10 (severe)")
    ax.set_xlabel("VIF", fontsize=11)
    ax.set_title("(g) Variance Inflation Factors", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="x", alpha=0.3)

    report.append(f"\n  VIF values:")
    for name, vif in zip(FEAT_SHORT, vifs):
        flag = " ⚠️" if vif > 5 else ""
        report.append(f"    {name:12s}: {vif:.2f}{flag}")

    # (h) Correlation matrix heatmap
    ax = fig.add_subplot(gs[2, 1])
    corr_matrix = np.corrcoef(X.T)
    im = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(FEAT_SHORT)))
    ax.set_xticklabels(FEAT_SHORT, fontsize=9, rotation=45, ha="right")
    ax.set_yticks(range(len(FEAT_SHORT)))
    ax.set_yticklabels(FEAT_SHORT, fontsize=9)
    # Annotate values
    for i in range(len(FEAT_SHORT)):
        for j in range(len(FEAT_SHORT)):
            ax.text(j, i, f"{corr_matrix[i,j]:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if abs(corr_matrix[i,j]) > 0.6 else "black")
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("(h) Feature Correlation Matrix", fontsize=12, fontweight="bold")

    # (i) Leverage vs Studentized Residuals (influence plot)
    ax = fig.add_subplot(gs[2, 2])
    # Use Ea diagnostics
    ax.scatter(diag_Ea["leverage"], diag_Ea["ext_student"], s=80,
               c="#2171B5", edgecolors="black", zorder=5)
    # Threshold lines
    avg_h = (len(FEAT_COLS) + 1) / n
    ax.axvline(2 * avg_h, color="orange", linestyle="--", alpha=0.7,
               label=f"2(p+1)/n = {2*avg_h:.2f}")
    ax.axhline(2, color="red", linestyle="--", alpha=0.5)
    ax.axhline(-2, color="red", linestyle="--", alpha=0.5)
    for i, nm in enumerate(names):
        ax.annotate(nm, (diag_Ea["leverage"][i], diag_Ea["ext_student"][i]),
                    fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Leverage (hᵢᵢ)", fontsize=11)
    ax.set_ylabel("Ext. Studentized Residual", fontsize=11)
    ax.set_title("(i) Ea: Influence Plot", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    report.append(f"\n  Leverage (hᵢᵢ) and influence (Ea):")
    report.append(f"  {'Intermediate':25s} {'h_ii':>6s} {'Cook D':>8s} {'DFFITS':>8s} {'|t*|':>6s}")
    for i in range(n):
        flag = ""
        if diag_Ea["leverage"][i] > 2 * avg_h:
            flag += " [high-lev]"
        if diag_Ea["cooks_d"][i] > 4.0 / n:
            flag += " [high-Cook]"
        if abs(diag_Ea["dffits"][i]) > 2 * np.sqrt((len(FEAT_COLS) + 1) / n):
            flag += " [high-DFFIT]"
        report.append(f"  {names[i]:25s} {diag_Ea['leverage'][i]:6.3f} "
                      f"{diag_Ea['cooks_d'][i]:8.3f} {diag_Ea['dffits'][i]:+8.3f} "
                      f"{abs(diag_Ea['ext_student'][i]):6.3f}{flag}")

    # ── Row 4: Bootstrap CIs and LOO sensitivity ──

    # (j) Bootstrap coefficient CIs for Ea
    ax = fig.add_subplot(gs[3, 0])
    y_Ea = sub["Ea"].values
    boot_coefs = bootstrap_coefficients(X, y_Ea, n_boot=5000)
    # Standardize coefficients for comparison
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    boot_std = bootstrap_coefficients(X_std, y_Ea, n_boot=5000)

    # Plot standardized coefficient CIs (skip intercept)
    medians = np.median(boot_std[:, 1:], axis=0)
    ci_lo = np.percentile(boot_std[:, 1:], 2.5, axis=0)
    ci_hi = np.percentile(boot_std[:, 1:], 97.5, axis=0)

    colors_ci = ["#E6550D" if (lo > 0 or hi < 0) else "#AAAAAA"
                 for lo, hi in zip(ci_lo, ci_hi)]
    y_pos = np.arange(len(FEAT_SHORT))
    ax.barh(y_pos, medians, xerr=[medians - ci_lo, ci_hi - medians],
            color=colors_ci, edgecolor="white", capsize=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(FEAT_SHORT, fontsize=11)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Standardized Coefficient", fontsize=11)
    ax.set_title("(j) Ea: Bootstrap 95% CI\n(standardized coefs)", fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    report.append(f"\n  Ea Bootstrap 95% CI (standardized, n_boot=5000):")
    for i, name in enumerate(FEAT_SHORT):
        sig = "***" if (ci_lo[i] > 0 or ci_hi[i] < 0) else "n.s."
        report.append(f"    {name:12s}: median={medians[i]:+.3f}  "
                      f"95% CI=[{ci_lo[i]:+.3f}, {ci_hi[i]:+.3f}]  {sig}")

    # (k) LOO sensitivity — how much does each point move R² and Q²?
    ax = fig.add_subplot(gs[3, 1])
    r2_full = 1 - np.sum(diag_Ea["resid"]**2) / np.sum((y_Ea - y_Ea.mean())**2)
    loo_resid = y_Ea - diag_Ea["y_loo"]
    q2_full = 1 - np.sum(loo_resid**2) / np.sum((y_Ea - y_Ea.mean())**2)

    delta_q2 = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        X_sub = X[mask]
        y_sub = y_Ea[mask]
        # Recompute Q² without point i
        y_loo_sub = np.zeros(n - 1)
        for j in range(n - 1):
            m2 = np.ones(n - 1, dtype=bool)
            m2[j] = False
            reg_j = LinearRegression().fit(X_sub[m2], y_sub[m2])
            y_loo_sub[j] = reg_j.predict(X_sub[j:j+1])[0]
        q2_i = 1 - np.sum((y_sub - y_loo_sub)**2) / np.sum((y_sub - y_sub.mean())**2)
        delta_q2[i] = q2_i - q2_full

    colors_dq = ["#31A354" if dq > 0 else "#E6550D" for dq in delta_q2]
    ax.barh(x_pos, delta_q2, color=colors_dq, edgecolor="white")
    ax.set_yticks(x_pos)
    ax.set_yticklabels(names, fontsize=7)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("ΔQ² (drop point → Q² change)", fontsize=11)
    ax.set_title("(k) Ea: LOO Sensitivity\n(green=removing helps)", fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    report.append(f"\n  Ea LOO Sensitivity (full Q²={q2_full:.4f}):")
    for i in range(n):
        report.append(f"    Drop {names[i]:25s} → ΔQ²={delta_q2[i]:+.4f}  "
                      f"(new Q²={q2_full + delta_q2[i]:.4f})")

    # (l) Condition number & eigenvalue spectrum
    ax = fig.add_subplot(gs[3, 2])
    cond, sv = condition_number(X)
    ax.bar(range(len(sv)), sv / sv.max(), color="#2171B5", edgecolor="white")
    ax.set_xlabel("Singular Value Index", fontsize=11)
    ax.set_ylabel("Normalized Singular Value", fontsize=11)
    ax.set_title(f"(l) Singular Value Spectrum\nCondition # = {cond:.1f}",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    report.append(f"\n  Condition number (standardized X): {cond:.1f}")
    report.append(f"  Singular values: {', '.join(f'{s:.3f}' for s in sv)}")
    if cond > 30:
        report.append(f"  ⚠️ Condition number > 30 → moderate multicollinearity")
    if cond > 100:
        report.append(f"  ⚠️ Condition number > 100 → severe multicollinearity")

    fig.suptitle("Emp+DFT-2 Combined Model Diagnostics\n"
                 "(σ, Es, δ_ortho, δ_benzyne, dipole_D, Gsolv_kJ → Ea, ln_A; n=9)",
                 fontsize=15, fontweight="bold", y=1.01)
    path = os.path.join(OUT_DIR, "combined_model_diagnostics.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    report.append(f"\n  Saved: {path}")
    print(f"  Saved: {path}")

    return vifs, cond


def summarize(sub, diag_Ea, diag_lnA, vifs, cond, report):
    """Print structured summary."""
    n = len(sub)
    p = len(FEAT_COLS)
    df_resid = n - p - 1

    report.insert(0, "=" * 70)
    report.insert(0, "EMP+DFT-2 COMBINED MODEL DIAGNOSTICS")
    report.insert(1, "=" * 70)
    report.insert(2, f"Model: Ea, ln_A ~ σ + Es + δ_ortho + δ_benzyne + dipole_D + Gsolv_kJ")
    report.insert(3, f"n = {n}, p = {p}, df_residual = {df_resid}")
    report.insert(4, f"")
    report.insert(5, f"⚠️  df_resid = {df_resid}: extremely low → model is near-saturated")
    report.insert(6, f"   (7 parameters estimated from 9 observations)")
    report.insert(7, f"")

    # Ea model equation
    reg_Ea = diag_Ea["reg"]
    report.insert(8, f"Ea model equation:")
    eq = f"  Ea = {reg_Ea.intercept_:.2f}"
    for col, coef, short in zip(FEAT_COLS, reg_Ea.coef_, FEAT_SHORT):
        eq += f" {coef:+.3f}·{short}"
    report.insert(9, eq)
    report.insert(10, f"  R² = {1 - np.sum(diag_Ea['resid']**2)/np.sum((sub['Ea'].values - sub['Ea'].values.mean())**2):.4f}")
    loo_r = sub["Ea"].values - diag_Ea["y_loo"]
    q2 = 1 - np.sum(loo_r**2) / np.sum((sub["Ea"].values - sub["Ea"].values.mean())**2)
    report.insert(11, f"  Q²_LOO = {q2:.4f}")
    report.insert(12, f"  Max |residual| = {np.max(np.abs(diag_Ea['resid'])):.3f} kJ/mol")
    report.insert(13, f"  RMSE = {np.sqrt(np.mean(diag_Ea['resid']**2)):.3f} kJ/mol")
    report.insert(14, f"")


def main():
    report = []

    print("Loading Emp+DFT-2 model data...")
    sub = load_model_data()
    print(f"  n = {len(sub)} intermediates")

    X = sub[FEAT_COLS].values
    y_Ea = sub["Ea"].values
    y_lnA = sub["ln_A"].values

    print("Fitting OLS and computing diagnostics...")
    diag_Ea = fit_ols(X, y_Ea)
    diag_lnA = fit_ols(X, y_lnA)

    print("Generating diagnostic plots...")
    vifs, cond = plot_diagnostics(sub, diag_Ea, diag_lnA, report)

    print("Generating summary...")
    summarize(sub, diag_Ea, diag_lnA, vifs, cond, report)

    # Save report
    report_path = os.path.join(OUT_DIR, "combined_model_diagnostics.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(report))
    print(f"  Report saved: {report_path}")

    # Print key warnings
    print("\n" + "=" * 70)
    print("KEY DIAGNOSTICS")
    print("=" * 70)
    print(f"df_residual = {len(sub) - len(FEAT_COLS) - 1} (near-saturated)")
    print(f"Condition # = {cond:.1f}")
    max_vif = max(vifs)
    max_vif_name = FEAT_SHORT[np.argmax(vifs)]
    print(f"Max VIF = {max_vif:.1f} ({max_vif_name})")
    print(f"Max Cook's D (Ea) = {diag_Ea['cooks_d'].max():.3f}")
    print(f"Max |ext. studentized| (Ea) = {np.max(np.abs(diag_Ea['ext_student'])):.3f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
