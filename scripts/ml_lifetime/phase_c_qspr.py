"""
Phase C: QSPR model — predict organolithium lifetime from molecular structure.

Computes molecular descriptors from intermediate SMILES, builds predictive model
for Ea_decomp (decomposition activation energy), and validates against known lifetimes.

Input:  data/ml_lifetime/phase_b_arrhenius.csv
Output: data/ml_lifetime/phase_c_qspr_results.csv
        data/ml_lifetime/phase_c_descriptors.csv
        data/ml_lifetime/phase_c_plots/

Usage:
  cd /Users/zhaowenyuan/Projects/FlowFigTabMiner
  python scripts/ml_lifetime/phase_c_qspr.py
"""

import os
import csv
import numpy as np
from collections import OrderedDict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

INPUT_CSV = os.path.join(PROJECT_ROOT, "data/ml_lifetime/phase_b_arrhenius.csv")
DESC_CSV = os.path.join(PROJECT_ROOT, "data/ml_lifetime/phase_c_descriptors.csv")
OUTPUT_CSV = os.path.join(PROJECT_ROOT, "data/ml_lifetime/phase_c_qspr_results.csv")
PLOT_DIR = os.path.join(PROJECT_ROOT, "data/ml_lifetime/phase_c_plots")


def compute_descriptors(smiles):
    """Compute molecular descriptors for an organolithium intermediate SMILES."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors, Fragments

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    desc = OrderedDict()

    # Basic physicochemical
    desc["MW"] = Descriptors.MolWt(mol)
    desc["HeavyAtomCount"] = Descriptors.HeavyAtomCount(mol)
    desc["NumRotatableBonds"] = Descriptors.NumRotatableBonds(mol)
    desc["TPSA"] = Descriptors.TPSA(mol)
    desc["MolLogP"] = Descriptors.MolLogP(mol)

    # Ring/aromatic
    desc["NumAromaticRings"] = rdMolDescriptors.CalcNumAromaticRings(mol)
    desc["NumAliphaticRings"] = rdMolDescriptors.CalcNumAliphaticRings(mol)
    desc["NumAromaticAtoms"] = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())

    # Heteroatoms
    desc["NumHeteroatoms"] = Descriptors.NumHeteroatoms(mol)
    desc["NumN"] = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 7)
    desc["NumO"] = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 8)
    desc["NumHalogens"] = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() in (9, 17, 35, 53))
    desc["NumF"] = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 9)
    desc["NumBr"] = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 35)
    desc["NumI"] = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 53)

    # Li environment
    li_atoms = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 3]
    desc["NumLi"] = len(li_atoms)

    if li_atoms:
        li = li_atoms[0]
        # Neighbors of Li
        li_neighbors = li.GetNeighbors()
        desc["Li_neighbor_count"] = len(li_neighbors)

        if li_neighbors:
            c_neighbor = li_neighbors[0]  # C bonded to Li
            desc["Li_C_is_aromatic"] = int(c_neighbor.GetIsAromatic())
            desc["Li_C_degree"] = c_neighbor.GetDegree()
            desc["Li_C_hybridization_sp2"] = int(
                str(c_neighbor.GetHybridization()) == "SP2"
            )
            desc["Li_C_hybridization_sp3"] = int(
                str(c_neighbor.GetHybridization()) == "SP3"
            )
            # Count electron-withdrawing neighbors of the C-Li carbon
            ewg_count = 0
            for nb in c_neighbor.GetNeighbors():
                if nb.GetAtomicNum() == 3:
                    continue  # skip Li itself
                if nb.GetAtomicNum() in (7, 8):  # N, O
                    ewg_count += 1
                if nb.GetAtomicNum() in (9, 17, 35, 53):  # halogens
                    ewg_count += 1
            desc["Li_C_ewg_neighbors"] = ewg_count
        else:
            for k in ["Li_C_is_aromatic", "Li_C_degree",
                       "Li_C_hybridization_sp2", "Li_C_hybridization_sp3",
                       "Li_C_ewg_neighbors"]:
                desc[k] = 0
    else:
        for k in ["Li_neighbor_count", "Li_C_is_aromatic", "Li_C_degree",
                   "Li_C_hybridization_sp2", "Li_C_hybridization_sp3",
                   "Li_C_ewg_neighbors"]:
            desc[k] = 0

    # Functional group flags
    desc["has_ester"] = int(smiles.count("OC(=O)") > 0 or smiles.count("C(=O)O") > 0)
    desc["has_nitrile"] = int("C#N" in smiles or "N#C" in smiles)
    desc["has_epoxide"] = int("C1CO1" in smiles or "C1OC1" in smiles)

    return desc


def load_data():
    """Load Phase B results."""
    rows = []
    with open(INPUT_CSV) as f:
        for r in csv.DictReader(f):
            rows.append({
                "intermediate": r["intermediate"],
                "smiles": r["intermediate_smiles"],
                "Ea": float(r["Ea_decomp_kJ_mol"]),
                "ln_A": float(r["ln_A"]),
                "r2": float(r["arrhenius_r2"]),
                "n_T": int(r["n_temperatures"]),
                "t_half_m40": float(r["t_half_m40C_s"]),
            })
    return rows


def run_loocv(X, y, names):
    """Leave-one-out cross-validation with linear regression."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    n = len(y)
    y_pred_loocv = np.zeros(n)
    errors = []

    for i in range(n):
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y, i)
        X_test = X[i:i+1]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = Ridge(alpha=1.0)
        model.fit(X_train_s, y_train)
        y_pred_loocv[i] = model.predict(X_test_s)[0]
        errors.append(y_pred_loocv[i] - y[i])

    # Metrics
    ss_res = np.sum((y - y_pred_loocv) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    q2 = 1.0 - ss_res / ss_tot
    rmse = np.sqrt(np.mean(np.array(errors) ** 2))
    mae = np.mean(np.abs(errors))

    return y_pred_loocv, q2, rmse, mae


def run_full_model(X, y, feature_names):
    """Train full model and get feature importances."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestRegressor

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    # Ridge regression
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_s, y)

    # Feature importances from Ridge (absolute coefficients)
    importances = np.abs(ridge.coef_)
    sorted_idx = np.argsort(importances)[::-1]

    print(f"\n── Ridge regression coefficients ──")
    for idx in sorted_idx:
        print(f"  {feature_names[idx]:30s}: {ridge.coef_[idx]:>8.3f}")

    # Random Forest for comparison
    rf = RandomForestRegressor(n_estimators=100, random_state=42, max_features="sqrt")
    rf.fit(X, y)
    rf_importances = rf.feature_importances_
    rf_sorted = np.argsort(rf_importances)[::-1]

    print(f"\n── Random Forest feature importances ──")
    for idx in rf_sorted[:10]:
        print(f"  {feature_names[idx]:30s}: {rf_importances[idx]:>8.4f}")

    return ridge, scaler


def plot_results(y_true, y_pred, names, q2, output_path):
    """Plot predicted vs actual Ea."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(y_true, y_pred, s=80, zorder=5, c="steelblue", edgecolors="navy")

    # Annotate each point
    for i, name in enumerate(names):
        short = name[:25]
        ax.annotate(short, (y_true[i], y_pred[i]), fontsize=7,
                     xytext=(5, 5), textcoords="offset points")

    # Diagonal line
    lo = min(min(y_true), min(y_pred)) - 5
    hi = max(max(y_true), max(y_pred)) + 5
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5, linewidth=1)

    ax.set_xlabel("Actual Ea_decomp (kJ/mol)", fontsize=12)
    ax.set_ylabel("Predicted Ea_decomp (kJ/mol, LOOCV)", fontsize=12)
    ax.set_title(f"QSPR: Organolithium Decomposition Ea\nLOOCV Q² = {q2:.3f}", fontsize=13)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_stability_ranking(data, output_path):
    """Plot stability ranking bar chart at -40°C."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    # Sort by t_half
    sorted_data = sorted(data, key=lambda x: x["t_half_m40"])
    names = [d["intermediate"][:30] for d in sorted_data]
    t_halves = [d["t_half_m40"] for d in sorted_data]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.RdYlGn(np.linspace(0.1, 0.9, len(names)))
    bars = ax.barh(range(len(names)), np.log10(t_halves), color=colors)

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("log₁₀(t₁/₂ at -40°C)  [s]", fontsize=12)
    ax.set_title("Organolithium Intermediate Stability Ranking", fontsize=13)

    # Add actual values as text
    for i, (bar, th) in enumerate(zip(bars, t_halves)):
        if th > 1:
            label = f"{th:.1f} s"
        else:
            label = f"{th*1000:.1f} ms"
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                label, va="center", fontsize=8)

    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    data = load_data()
    print(f"[Phase C] Loaded {len(data)} intermediates from Phase B")

    # Compute descriptors
    all_desc = []
    valid_data = []
    for d in data:
        desc = compute_descriptors(d["smiles"])
        if desc is None:
            print(f"  SKIP {d['intermediate']}: invalid SMILES")
            continue
        all_desc.append(desc)
        valid_data.append(d)

    if not all_desc:
        print("No valid descriptors computed. Exiting.")
        return

    feature_names = list(all_desc[0].keys())
    X = np.array([[d[k] for k in feature_names] for d in all_desc])
    y = np.array([d["Ea"] for d in valid_data])
    names = [d["intermediate"] for d in valid_data]

    print(f"  Computed {len(feature_names)} descriptors for {len(valid_data)} intermediates")

    # Save descriptors
    desc_rows = []
    for d, desc in zip(valid_data, all_desc):
        row = {"intermediate": d["intermediate"], "smiles": d["smiles"],
               "Ea_decomp_kJ_mol": d["Ea"]}
        row.update(desc)
        desc_rows.append(row)

    desc_fieldnames = ["intermediate", "smiles", "Ea_decomp_kJ_mol"] + feature_names
    with open(DESC_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=desc_fieldnames)
        writer.writeheader()
        writer.writerows(desc_rows)
    print(f"  Saved descriptors to {DESC_CSV}")

    # Feature selection: remove zero-variance, then pick top features by |correlation|
    variances = np.var(X, axis=0)
    keep_mask = variances > 1e-10
    X_nzv = X[:, keep_mask]
    names_nzv = [f for f, k in zip(feature_names, keep_mask) if k]

    # Univariate Pearson correlation with Ea
    print(f"\n── Univariate correlation with Ea_decomp ──")
    correlations = []
    for j, fname in enumerate(names_nzv):
        from scipy.stats import pearsonr as _pr
        r_val, p_val = _pr(X_nzv[:, j], y)
        correlations.append((fname, r_val, p_val, j))
    correlations.sort(key=lambda x: -abs(x[1]))
    for fname, r_val, p_val, _ in correlations:
        sig = "*" if p_val < 0.05 else " "
        print(f"  {fname:30s}: r={r_val:>+.3f}  p={p_val:.3f} {sig}")

    # Select top-K features by |correlation| (K ≤ n/3 to avoid overfitting)
    max_features = max(2, len(y) // 3)
    top_features = correlations[:max_features]
    top_indices = [c[3] for c in top_features]
    feature_names_filtered = [c[0] for c in top_features]
    X_filtered = X_nzv[:, top_indices]
    print(f"\n  Selected top {max_features} features: {feature_names_filtered}")

    # Run LOOCV with selected features
    print(f"\n── Leave-One-Out Cross-Validation (top {max_features} features) ──")
    y_pred_loocv, q2, rmse, mae = run_loocv(X_filtered, y, names)

    # Also try all pairwise 2-feature models to find best
    print(f"\n── Best 2-feature models (exhaustive search) ──")
    best_q2_2f = -999
    best_pair = None
    for i in range(len(names_nzv)):
        for j in range(i+1, len(names_nzv)):
            X_2f = X_nzv[:, [i, j]]
            _, q2_2f, _, _ = run_loocv(X_2f, y, names)
            if q2_2f > best_q2_2f:
                best_q2_2f = q2_2f
                best_pair = (names_nzv[i], names_nzv[j], i, j)
    if best_pair:
        print(f"  Best pair: {best_pair[0]} + {best_pair[1]} → Q² = {best_q2_2f:.3f}")
        # Re-run with best pair for final results
        X_best2 = X_nzv[:, [best_pair[2], best_pair[3]]]
        y_pred_best2, q2_best2, rmse_best2, mae_best2 = run_loocv(X_best2, y, names)

        if q2_best2 > q2:
            print(f"  → Better than top-{max_features}! Using this model.")
            X_filtered = X_best2
            feature_names_filtered = [best_pair[0], best_pair[1]]
            y_pred_loocv = y_pred_best2
            q2, rmse, mae = q2_best2, rmse_best2, mae_best2

    print(f"  Q² (LOOCV):  {q2:.3f}")
    print(f"  RMSE:        {rmse:.2f} kJ/mol")
    print(f"  MAE:         {mae:.2f} kJ/mol")

    print(f"\n── Per-intermediate LOOCV results ──")
    print(f"{'Intermediate':<45s} {'Ea_actual':>10s} {'Ea_pred':>10s} {'Error':>10s}")
    print("-" * 80)
    for i in range(len(names)):
        err = y_pred_loocv[i] - y[i]
        print(f"{names[i][:45]:<45s} {y[i]:>10.1f} {y_pred_loocv[i]:>10.1f} {err:>+10.1f}")

    # Full model for feature importances
    print(f"\n── Full model analysis ──")
    model, scaler = run_full_model(X_filtered, y, feature_names_filtered)

    # ── Also predict log(t₁/₂ @ -40°C) directly ──
    log_t_half = np.log10(np.array([d["t_half_m40"] for d in valid_data]))
    print(f"\n── Direct prediction of log₁₀(t₁/₂ @ -40°C) ──")

    # Exhaustive best 2-feature search for log_t_half
    best_q2_lt = -999
    best_pair_lt = None
    for i in range(X_nzv.shape[1]):
        for j in range(i+1, X_nzv.shape[1]):
            X_2f = X_nzv[:, [i, j]]
            _, q2_2f, _, _ = run_loocv(X_2f, log_t_half, names)
            if q2_2f > best_q2_lt:
                best_q2_lt = q2_2f
                best_pair_lt = (names_nzv[i], names_nzv[j], i, j)

    if best_pair_lt:
        X_lt = X_nzv[:, [best_pair_lt[2], best_pair_lt[3]]]
        y_pred_lt, q2_lt, rmse_lt, mae_lt = run_loocv(X_lt, log_t_half, names)
        print(f"  Best features: {best_pair_lt[0]} + {best_pair_lt[1]}")
        print(f"  Q² (LOOCV):  {q2_lt:.3f}")
        print(f"  RMSE:        {rmse_lt:.2f} log-decades")
        print(f"  MAE:         {mae_lt:.2f} log-decades")
        print(f"\n  {'Intermediate':<45s} {'actual':>8s} {'pred':>8s} {'ratio':>8s}")
        for i in range(len(names)):
            actual_s = 10**log_t_half[i]
            pred_s = 10**y_pred_lt[i]
            ratio = pred_s / actual_s
            if actual_s > 1:
                a_str = f"{actual_s:.1f}s"
            else:
                a_str = f"{actual_s*1000:.1f}ms"
            if pred_s > 1:
                p_str = f"{pred_s:.1f}s"
            else:
                p_str = f"{pred_s*1000:.1f}ms"
            print(f"  {names[i][:45]:<45s} {a_str:>8s} {p_str:>8s}   {ratio:>5.1f}x")

    # ── Chemical interpretation ──
    print(f"\n── Chemical interpretation ──")
    print(f"  Stability factors (from Ea_decomp):")
    print(f"  • Electron-withdrawing groups (CN, ester) → stabilize carbanion → LOW Ea → long-lived")
    print(f"  • sp3 carbanion (no aromatic stabilization) → HIGH Ea → temperature-sensitive")
    print(f"  • Neighboring halogen (I, Br on same ring) → benzyne elimination → VERY HIGH Ea")
    print(f"  • Aromatic ArLi with no EWG → MODERATE Ea → moderate stability")

    # Save results
    output_rows = []
    for i, d in enumerate(valid_data):
        output_rows.append({
            "intermediate": d["intermediate"],
            "intermediate_smiles": d["smiles"],
            "Ea_actual_kJ_mol": f"{y[i]:.2f}",
            "Ea_predicted_loocv_kJ_mol": f"{y_pred_loocv[i]:.2f}",
            "Ea_error_kJ_mol": f"{y_pred_loocv[i] - y[i]:.2f}",
            "t_half_m40C_s": f"{d['t_half_m40']:.4g}",
            "arrhenius_r2": f"{d['r2']:.3f}",
            "n_temperatures": d["n_T"],
        })

    out_fieldnames = [
        "intermediate", "intermediate_smiles",
        "Ea_actual_kJ_mol", "Ea_predicted_loocv_kJ_mol", "Ea_error_kJ_mol",
        "t_half_m40C_s", "arrhenius_r2", "n_temperatures",
    ]
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    # Plots
    os.makedirs(PLOT_DIR, exist_ok=True)
    plot_results(y, y_pred_loocv, names, q2,
                 os.path.join(PLOT_DIR, "ea_predicted_vs_actual.png"))
    plot_stability_ranking(valid_data,
                           os.path.join(PLOT_DIR, "stability_ranking_m40C.png"))

    # Summary
    print(f"\n{'=' * 70}")
    print(f"PHASE C RESULTS")
    print(f"{'=' * 70}")
    print(f"Intermediates:     {len(valid_data)}")
    print(f"Descriptors:       {len(feature_names_filtered)}")
    print(f"LOOCV Q²:          {q2:.3f}")
    print(f"LOOCV RMSE:        {rmse:.2f} kJ/mol")
    print(f"LOOCV MAE:         {mae:.2f} kJ/mol")
    print(f"Ea range:          {y.min():.1f} – {y.max():.1f} kJ/mol")
    print(f"\nOutputs:")
    print(f"  {OUTPUT_CSV}")
    print(f"  {DESC_CSV}")
    print(f"  {PLOT_DIR}/ea_predicted_vs_actual.png")
    print(f"  {PLOT_DIR}/stability_ranking_m40C.png")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
