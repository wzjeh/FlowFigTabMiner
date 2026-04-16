"""
DFT Lifetime Prediction — 诚实评估

核心问题：dipole 与 Ea 有强相关 (r=-0.83)，但 Ea 与 ln_A 之间存在
极强的动力学补偿效应 (r=0.989)。这意味着 Ea 的预测精度不能直接转化
为 t½ 的预测精度。

本脚本回答：DFT (dipole, Gsolv) 到底能把 t½ 预测到什么程度？

Usage:
  cd /Users/zhaowenyuan/Projects/FlowFigTabMiner
  source flowfigtabminer/bin/activate
  python scripts/ml_lifetime/dft_lifetime_honest.py
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data/ml_lifetime")
OUT_DIR = os.path.join(DATA_DIR, "three_rate_analysis")

R_GAS = 8.314e-3  # kJ/(mol·K)


def load_all():
    summary = pd.read_csv(os.path.join(OUT_DIR, "three_rate_summary.csv"))
    has_dft = summary[summary["dft_charge_Li"].notna()].copy()

    hl = pd.read_csv(os.path.join(DATA_DIR, "phase_a_halflives.csv"))
    comp = hl[hl["model"] == "competing"].copy()
    comp = comp[comp["t_half_s"].notna() & (comp["t_half_s"] > 0)]

    arr = pd.read_csv(os.path.join(DATA_DIR, "phase_b_arrhenius.csv"))
    return has_dft, comp, arr


def merge_halflife_with_dft(comp, has_dft, arr):
    """Merge per-temperature t½ with DFT descriptors."""
    # Build intermediate name → DFT lookup
    name_to_smi = dict(zip(arr["intermediate"], arr["intermediate_smiles"]))
    smi_to_dft = {}
    for _, r in has_dft.iterrows():
        smi_to_dft[r["smiles"]] = r

    rows = []
    for _, r in comp.iterrows():
        smi = name_to_smi.get(r["intermediate"])
        dft_row = smi_to_dft.get(smi)
        if dft_row is None:
            continue
        rows.append({
            "intermediate": r["intermediate"],
            "T_C": r["T_C"],
            "T_K": r["T_K"],
            "inv_T": 1000.0 / r["T_K"],
            "t_half_s": r["t_half_s"],
            "log_thalf": np.log10(r["t_half_s"]),
            "dipole": dft_row["dft_dipole_D"],
            "Gsolv": dft_row["dft_Gsolv_kJ"],
            "Ea": dft_row["Ea"],
            "ln_A": dft_row["ln_A"],
            "sigma": dft_row.get("sigma_hammett", np.nan),
            "is_arli": pd.notna(dft_row.get("sigma_hammett")),
        })
    return pd.DataFrame(rows)


def main():
    has_dft, comp, arr = load_all()
    df = merge_halflife_with_dft(comp, has_dft, arr)

    # ── Figure: 6-panel honest assessment ──
    fig = plt.figure(figsize=(18, 18))
    gs = GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)

    names_12 = has_dft["intermediate"].str[:25].values
    Ea_12 = has_dft["Ea"].values
    lnA_12 = has_dft["ln_A"].values
    dip_12 = has_dft["dft_dipole_D"].values
    thalf_12 = has_dft["t_half_m40"].values
    log_thalf_12 = np.log10(thalf_12)
    is_arli_12 = has_dft["sigma_hammett"].notna().values

    # ═══════════════════════════════════════════════════════
    # (a) The compensation effect: Ea vs ln_A
    # ═══════════════════════════════════════════════════════
    ax = fig.add_subplot(gs[0, 0])
    colors_a = ["#2171B5" if a else "#E6550D" for a in is_arli_12]
    ax.scatter(Ea_12, lnA_12, s=100, c=colors_a, edgecolors="black", zorder=5)
    for i, nm in enumerate(names_12):
        ax.annotate(nm, (Ea_12[i], lnA_12[i]), fontsize=6, xytext=(4, 4),
                    textcoords="offset points")

    # Regression line
    reg_comp = LinearRegression().fit(Ea_12.reshape(-1, 1), lnA_12)
    x_line = np.linspace(0, 70, 100)
    ax.plot(x_line, reg_comp.predict(x_line.reshape(-1, 1)), "k--", alpha=0.5)
    r_comp, p_comp = pearsonr(Ea_12, lnA_12)

    ax.text(0.05, 0.95, f"r = {r_comp:.3f}\np < 0.001",
            transform=ax.transAxes, fontsize=11, va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
    ax.set_xlabel("Ea (kJ/mol)", fontsize=12)
    ax.set_ylabel("ln(A)", fontsize=12)
    ax.set_title("(a) Kinetic Compensation Effect\nEa vs ln(A) — r = 0.989",
                 fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3)

    # ═══════════════════════════════════════════════════════
    # (b) dipole vs Ea (strong) and dipole vs log(t½) (weak)
    # ═══════════════════════════════════════════════════════
    ax = fig.add_subplot(gs[0, 1])
    ax2 = ax.twinx()

    # dipole vs Ea
    ax.scatter(dip_12, Ea_12, s=100, c="#2171B5", edgecolors="black",
               zorder=5, label="Ea (kJ/mol)")
    r_ea, p_ea = pearsonr(dip_12, Ea_12)
    reg_ea = LinearRegression().fit(dip_12.reshape(-1, 1), Ea_12)
    x_d = np.linspace(2, 12, 100)
    ax.plot(x_d, reg_ea.predict(x_d.reshape(-1, 1)), "#2171B5", alpha=0.4, linewidth=2)

    # dipole vs log(t½) on twin axis
    ax2.scatter(dip_12, log_thalf_12, s=80, c="#E6550D", marker="^",
                edgecolors="black", zorder=4, label="log₁₀(t½)")
    r_th, p_th = pearsonr(dip_12, log_thalf_12)
    reg_th = LinearRegression().fit(dip_12.reshape(-1, 1), log_thalf_12)
    ax2.plot(x_d, reg_th.predict(x_d.reshape(-1, 1)), "#E6550D", alpha=0.4, linewidth=2)

    ax.set_xlabel("Dipole Moment (D)", fontsize=12)
    ax.set_ylabel("Ea (kJ/mol)", fontsize=12, color="#2171B5")
    ax2.set_ylabel("log₁₀(t½ / s) at -40°C", fontsize=12, color="#E6550D")
    ax.set_title(f"(b) Dipole Predicts Ea (r={r_ea:.2f}***)\n"
                 f"but NOT t½ (r={r_th:+.2f}, p={p_th:.2f})",
                 fontsize=13, fontweight="bold")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper right")
    ax.grid(alpha=0.3)

    # ═══════════════════════════════════════════════════════
    # (c) Why compensation kills prediction:
    #     Predicted t½ via DFT(Ea) + DFT(lnA) vs observed
    # ═══════════════════════════════════════════════════════
    ax = fig.add_subplot(gs[1, 0])

    # Build dipole+Gsolv models for Ea and ln_A
    X_dft = np.column_stack([dip_12, has_dft["dft_Gsolv_kJ"].values])
    reg_Ea_dft = LinearRegression().fit(X_dft, Ea_12)
    reg_lnA_dft = LinearRegression().fit(X_dft, lnA_12)

    Ea_pred = reg_Ea_dft.predict(X_dft)
    lnA_pred = reg_lnA_dft.predict(X_dft)

    T_ref = 233.15  # -40°C
    kd_pred = np.exp(lnA_pred - Ea_pred / (R_GAS * T_ref))
    thalf_pred = np.log(2) / np.maximum(kd_pred, 1e-30)
    log_thalf_pred = np.log10(np.clip(thalf_pred, 1e-6, 1e12))

    # LOOCV for t½
    log_thalf_loo = np.zeros(12)
    for i in range(12):
        mask = np.ones(12, dtype=bool)
        mask[i] = False
        reg_Ea_i = LinearRegression().fit(X_dft[mask], Ea_12[mask])
        reg_lnA_i = LinearRegression().fit(X_dft[mask], lnA_12[mask])
        Ea_i = reg_Ea_i.predict(X_dft[i:i+1])[0]
        lnA_i = reg_lnA_i.predict(X_dft[i:i+1])[0]
        kd_i = np.exp(lnA_i - Ea_i / (R_GAS * T_ref))
        thalf_i = np.log(2) / max(kd_i, 1e-30)
        log_thalf_loo[i] = np.log10(np.clip(thalf_i, 1e-6, 1e12))

    r2_fit = r2_score(log_thalf_12, log_thalf_pred)
    r2_loo = r2_score(log_thalf_12, log_thalf_loo)
    mae_fit = mean_absolute_error(log_thalf_12, log_thalf_pred)
    mae_loo = mean_absolute_error(log_thalf_12, log_thalf_loo)

    colors_c = ["#2171B5" if a else "#E6550D" for a in is_arli_12]
    ax.scatter(log_thalf_12, log_thalf_pred, s=100, c=colors_c,
               edgecolors="black", zorder=5, label=f"Fit (R²={r2_fit:.3f})")
    ax.scatter(log_thalf_12, log_thalf_loo, s=80, c=colors_c, marker="^",
               edgecolors="black", alpha=0.6, zorder=4,
               label=f"LOOCV (R²={r2_loo:.3f})")

    for i, nm in enumerate(names_12):
        ax.annotate(nm, (log_thalf_12[i], log_thalf_pred[i]), fontsize=5.5,
                    xytext=(4, 4), textcoords="offset points")

    lims = [min(log_thalf_12.min(), log_thalf_pred.min()) - 0.5,
            max(log_thalf_12.max(), log_thalf_pred.max()) + 0.5]
    ax.plot(lims, lims, "k--", alpha=0.4)
    # ±1 decade bands
    ax.fill_between(lims, [l - 1 for l in lims], [l + 1 for l in lims],
                    alpha=0.08, color="green", label="±1 decade")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Observed log₁₀(t½ / s)", fontsize=12)
    ax.set_ylabel("Predicted log₁₀(t½ / s)", fontsize=12)
    ax.set_title(f"(c) DFT → t½ at -40°C (dipole+Gsolv)\n"
                 f"R²={r2_fit:.3f}, LOOCV R²={r2_loo:.3f}, MAE={mae_loo:.2f} decades",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # ═══════════════════════════════════════════════════════
    # (d) Per-temperature parity: predicted vs observed t½
    # ═══════════════════════════════════════════════════════
    ax = fig.add_subplot(gs[1, 1])

    # Use Arrhenius model: t½(T) = ln2 / exp(lnA_pred - Ea_pred/(R*T))
    # LOSO: train on 11 intermediates, predict t½ at all T for the 12th
    unique_int = df["intermediate"].unique()
    df["log_thalf_loso"] = np.nan

    for sub in unique_int:
        test_mask = df["intermediate"] == sub
        train_mask = ~test_mask

        # Get DFT data for training intermediates
        train_ints = df[train_mask]["intermediate"].unique()
        train_rows = has_dft[has_dft["intermediate"].isin(train_ints)]

        if len(train_rows) < 3:
            continue

        X_tr = train_rows[["dft_dipole_D", "dft_Gsolv_kJ"]].values
        Ea_tr = train_rows["Ea"].values
        lnA_tr = train_rows["ln_A"].values

        reg_Ea_i = LinearRegression().fit(X_tr, Ea_tr)
        reg_lnA_i = LinearRegression().fit(X_tr, lnA_tr)

        # Get DFT for test intermediate
        test_int_row = has_dft[has_dft["intermediate"] == sub]
        if len(test_int_row) == 0:
            continue
        X_te = test_int_row[["dft_dipole_D", "dft_Gsolv_kJ"]].values
        Ea_pred_i = reg_Ea_i.predict(X_te)[0]
        lnA_pred_i = reg_lnA_i.predict(X_te)[0]

        # Predict t½ at each temperature
        for idx in df.index[test_mask]:
            T_K = df.loc[idx, "T_K"]
            kd_p = np.exp(lnA_pred_i - Ea_pred_i / (R_GAS * T_K))
            thalf_p = np.log(2) / max(kd_p, 1e-30)
            df.loc[idx, "log_thalf_loso"] = np.log10(np.clip(thalf_p, 1e-6, 1e12))

    valid = df["log_thalf_loso"].notna()
    y_obs = df.loc[valid, "log_thalf"].values
    y_pred = df.loc[valid, "log_thalf_loso"].values

    r2_all = r2_score(y_obs, y_pred)
    mae_all = mean_absolute_error(y_obs, y_pred)
    rho, _ = spearmanr(y_obs, y_pred)

    # Color by intermediate class
    for sub in unique_int:
        mask = valid & (df["intermediate"] == sub)
        if mask.sum() == 0:
            continue
        is_a = df.loc[mask, "is_arli"].iloc[0]
        color = "#2171B5" if is_a else "#E6550D"
        label = sub[:20] if not is_a else None
        ax.scatter(df.loc[mask, "log_thalf"], df.loc[mask, "log_thalf_loso"],
                   s=50, c=color, edgecolors="black", alpha=0.7, zorder=5,
                   label=label)

    lims_d = [min(y_obs.min(), y_pred.min()) - 0.5,
              max(y_obs.max(), y_pred.max()) + 0.5]
    ax.plot(lims_d, lims_d, "k--", alpha=0.4)
    ax.fill_between(lims_d, [l - 1 for l in lims_d], [l + 1 for l in lims_d],
                    alpha=0.08, color="green")
    ax.set_xlim(lims_d)
    ax.set_ylim(lims_d)
    ax.set_xlabel("Observed log₁₀(t½ / s)", fontsize=12)
    ax.set_ylabel("LOSO Predicted log₁₀(t½ / s)", fontsize=12)
    ax.set_title(f"(d) LOSO t½ at All Temperatures\n"
                 f"n={valid.sum()}, R²={r2_all:.3f}, MAE={mae_all:.2f} dec, ρ={rho:.3f}",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=7, title="● ArLi  ■ non-ArLi", title_fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)

    # ═══════════════════════════════════════════════════════
    # (e) Rank correlation: does DFT get the ORDER right?
    # ═══════════════════════════════════════════════════════
    ax = fig.add_subplot(gs[2, 0])

    # At -40°C, rank all 12 by predicted vs observed t½
    rank_obs = np.argsort(np.argsort(-log_thalf_12)) + 1  # descending rank (1=most stable)
    rank_pred = np.argsort(np.argsort(-log_thalf_pred)) + 1

    rho_rank, p_rank = spearmanr(log_thalf_12, log_thalf_pred)

    ax.scatter(rank_obs, rank_pred, s=120, c=colors_c, edgecolors="black", zorder=5)
    for i, nm in enumerate(names_12):
        ax.annotate(nm, (rank_obs[i], rank_pred[i]), fontsize=6.5,
                    xytext=(4, 4), textcoords="offset points")

    ax.plot([0, 13], [0, 13], "k--", alpha=0.4)
    ax.fill_between([0, 13], [0 - 2, 13 - 2], [0 + 2, 13 + 2],
                    alpha=0.1, color="green", label="±2 ranks")
    ax.set_xlim(0.5, 12.5)
    ax.set_ylim(0.5, 12.5)
    ax.set_xlabel("Observed Stability Rank (1=most stable)", fontsize=12)
    ax.set_ylabel("Predicted Stability Rank", fontsize=12)
    ax.set_title(f"(e) Stability Ranking at -40°C\n"
                 f"Spearman ρ = {rho_rank:.3f} (p={p_rank:.3f})",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    within_1_dec = np.sum(np.abs(log_thalf_12 - log_thalf_loo) <= 1.0)
    within_1_dec_pct = within_1_dec / 12 * 100
    within_05_dec = np.sum(np.abs(log_thalf_12 - log_thalf_loo) <= 0.5)
    within_05_dec_pct = within_05_dec / 12 * 100

    # ═══════════════════════════════════════════════════════
    # (f) Per-intermediate error bar chart
    # ═══════════════════════════════════════════════════════
    ax = fig.add_subplot(gs[2, 1])
    errors = log_thalf_loo - log_thalf_12  # positive = overestimates stability
    sort_idx = np.argsort(errors)
    bar_colors = ["#2171B5" if is_arli_12[i] else "#E6550D" for i in sort_idx]

    y_pos = np.arange(12)
    ax.barh(y_pos, errors[sort_idx], color=bar_colors, edgecolor="white", height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([names_12[i] for i in sort_idx], fontsize=8)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.axvline(-1, color="red", linestyle="--", alpha=0.4)
    ax.axvline(+1, color="red", linestyle="--", alpha=0.4)
    ax.set_xlabel("LOOCV Error: log₁₀(t½_pred / t½_obs)", fontsize=11)
    ax.set_title(f"(f) Per-Intermediate LOOCV Error\n"
                 f"MAE={mae_loo:.2f} dec ({10**mae_loo:.0f}x), "
                 f"{within_1_dec}/12 within ±1 decade",
                 fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    fig.suptitle("Can DFT Descriptors Predict Organolithium Lifetime?",
                 fontsize=16, fontweight="bold")
    path = os.path.join(OUT_DIR, "dft_lifetime_honest.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # ── Print summary ──
    print("\n" + "=" * 70)
    print("DFT LIFETIME PREDICTION — HONEST ASSESSMENT")
    print("=" * 70)
    print(f"\nEa prediction (dipole+Gsolv, n=12):")
    print(f"  R² = {r2_score(Ea_12, Ea_pred):.3f}")
    print(f"  dipole vs Ea: r = {r_ea:.3f} (p = {p_ea:.4f})")
    print(f"\nCompensation effect:")
    print(f"  Ea vs ln_A: r = {r_comp:.3f}")
    print(f"  → Ea varies over {Ea_12.max()-Ea_12.min():.0f} kJ/mol range")
    print(f"  → But t½ only varies over {log_thalf_12.max()-log_thalf_12.min():.1f} decades")
    print(f"  → Because ln_A co-varies to partially cancel Ea effect")
    print(f"\nt½ prediction at -40°C (LOOCV):")
    print(f"  R² = {r2_loo:.3f}")
    print(f"  MAE = {mae_loo:.2f} decades ({10**mae_loo:.0f}× factor)")
    print(f"  Within ±1 decade: {within_1_dec}/12 ({within_1_dec_pct:.0f}%)")
    print(f"  Within ±0.5 decade: {within_05_dec}/12 ({within_05_dec_pct:.0f}%)")
    print(f"\nt½ prediction across all temperatures (LOSO, n={valid.sum()}):")
    print(f"  R² = {r2_all:.3f}")
    print(f"  MAE = {mae_all:.2f} decades")
    print(f"  Spearman ρ = {rho:.3f}")
    print(f"\nStability ranking (n=12):")
    print(f"  Spearman ρ = {rho_rank:.3f} (p = {p_rank:.3f})")

    print(f"\n{'Intermediate':35s} {'t½_obs':>10s} {'t½_pred':>10s} {'ratio':>8s} {'rank_Δ':>8s}")
    for i in range(12):
        ratio = thalf_pred[i] / thalf_12[i]
        rank_delta = rank_pred[i] - rank_obs[i]
        print(f"{names_12[i]:35s} {thalf_12[i]:>10.3f} {thalf_pred[i]:>10.3g} "
              f"{ratio:>8.1f}x {rank_delta:>+8d}")
    print("=" * 70)


if __name__ == "__main__":
    main()
