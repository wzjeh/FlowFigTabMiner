"""
ArLi t½ prediction — v2: chemically improved descriptors.

Improvements over v1:
  1. σ → σ_I (Taft inductive): removes resonance contamination at meta/ortho
  2. δ_ortho → chelation_strength (continuous): 0=none, 0.3=OMe, 0.7=ester, 1.0=acyl
  3. Add δ_benzyne as explicit "reactivity pathway" term
  4. Test HOMO/LUMO as additional reactivity terms
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from rdkit import Chem
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import warnings, pathlib

warnings.filterwarnings("ignore")
plt.rcParams.update({"font.size": 9, "font.family": "Arial"})

OUT = pathlib.Path("data/ml_lifetime/arli_halflife")
OUT.mkdir(parents=True, exist_ok=True)

# ── 1. Load data ─────────────────────────────────────────────────────

def canon(s):
    if pd.isna(s):
        return None
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m) if m else s

arr = pd.read_csv("data/ml_lifetime/phase_b_arrhenius.csv")
clean = pd.read_csv("data/ml_lifetime/clean_organolithium_unified_descriptors.csv")
arr["canon_smi"] = arr["intermediate_smiles"].apply(canon)
clean["canon_smi"] = clean["intermediate_smiles_canonical"].apply(canon)

arli_arr = arr[arr["canon_smi"].apply(lambda s: s is not None and "c1" in str(s))].drop_duplicates("canon_smi")
clean_uniq = clean[clean["intermediate_class"] == "ArLi"].drop_duplicates("canon_smi")

dft_cols = [
    "sigma_hammett", "Es_taft", "delta_ortho", "delta_benzyne",
    "dft_dipole_D", "dft_Gsolv_kJ", "dft_HOMO_eV", "dft_LUMO_eV",
    "dft_charge_Li", "dft_LiC_BDE_kJ", "dft_wiberg_LiC",
]
merged = arli_arr.merge(
    clean_uniq[["canon_smi", "intermediate"] + dft_cols].rename(columns={"intermediate": "clean_name"}),
    on="canon_smi", how="inner"
)
merged["log_t_half"] = np.log10(merged["t_half_m40C_s"].clip(lower=1e-3))

# ── 2. Build improved descriptors ────────────────────────────────────

# σ_I (Taft inductive) lookup — derived from σ_I ≈ σ_meta for most groups
# Source: Hansch, Leo & Taft, Chem Rev 1991
SIGMA_I = {  # Substituent → σ_I
    "I": 0.39, "Br": 0.44, "Cl": 0.47, "F": 0.50,
    "CN": 0.56, "NO2": 0.67,
    "COOR": 0.30, "COR": 0.30, "COOH": 0.30,
    "OMe": 0.27, "OH": 0.29, "NH2": 0.12,
    "H": 0.00, "CH3": -0.04, "CF3": 0.40,
    "phenyl": 0.12, "vinyl": 0.11,
    "NMe2": 0.06, "SMe": 0.23,
    "SO2Me": 0.59, "CHO": 0.31,
}

SIGMA_R = {  # Substituent → σ_R (= σ_p - σ_I, resonance at para)
    "I": -0.04, "Br": -0.21, "Cl": -0.24, "F": -0.39,
    "CN": +0.10, "NO2": +0.11,
    "COOR": +0.15, "COR": +0.17, "COOH": +0.15,
    "OMe": -0.56, "OH": -0.64, "NH2": -0.74,
    "H": 0.00, "CH3": -0.13, "CF3": +0.07,
    "phenyl": -0.08, "vinyl": -0.05,
    "NMe2": -0.88, "SMe": -0.17,
    "SO2Me": +0.03, "CHO": +0.13,
}

# Chelation strength: depends on ortho group's Lewis basicity toward Li
CHELATION = {
    "H": 0.0, "CH3": 0.0, "CF3": 0.0,
    "F": 0.05, "Cl": 0.05, "Br": 0.05, "I": 0.05,  # very weak
    "OH": 0.3, "OMe": 0.3, "SMe": 0.2,
    "CN": 0.5,   # N lone pair can coordinate Li
    "COOR": 0.7, # C=O oxygen chelates Li (5-membered ring)
    "COR": 0.8,  # ketone C=O, slightly stronger
    "CHO": 0.8,
    "NO2": 0.6,  # O lone pair
    "NH2": 0.4, "NMe2": 0.5,
    "COOH": 0.7, "SO2Me": 0.4,
    "phenyl": 0.0, "vinyl": 0.0,
}

def identify_substituent(smiles):
    """Identify the main substituent on the aromatic ring (other than Li)."""
    if pd.isna(smiles):
        return "H", "unknown"
    s = str(smiles)
    # Determine position relative to Li
    # Simple heuristic based on SMILES patterns
    position = "unknown"
    if "c1ccccc1" in s:
        # Li and substituent both on bare benzene
        pass

    # Identify functional group
    if "C(=O)O" in s or "OC(=O)" in s:
        sub = "COOR"
    elif "C(=O)" in s:
        sub = "COR"
    elif "C#N" in s or "N#C" in s:
        sub = "CN"
    elif "[N+](=O)[O-]" in s:
        sub = "NO2"
    elif "OC" in s and "OC(=O)" not in s:
        sub = "OMe"
    elif "Br" in s:
        sub = "Br"
    elif "I" in s and "[Li]" in s:
        sub = "I"
    elif "Cl" in s:
        sub = "Cl"
    elif "F" in s:
        sub = "F"
    elif "N" in s and "C#N" not in s:
        sub = "NH2"
    else:
        sub = "H"

    return sub, position

# For training data: manually assign based on known chemistry
training_assignments = {
    # SMILES → (substituent, position)
    "[Li]c1ccccc1I": ("I", "ortho"),
    "[Li]c1ccccc1Br": ("Br", "ortho"),
    "COC(=O)c1ccccc1[Li]": ("COOR", "ortho"),
    "CCOC(=O)c1ccccc1[Li]": ("COOR", "ortho"),
    "CC(C)OC(=O)c1ccccc1[Li]": ("COOR", "ortho"),
    "CC(C)(C)OC(=O)c1ccccc1[Li]": ("COOR", "ortho"),
    "CC(C)(C)OC(=O)c1ccc([Li])cc1": ("COOR", "para"),
    "N#Cc1cccc([Li])c1": ("CN", "meta"),
    "N#Cc1ccc([Li])cc1": ("CN", "para"),
}

# Build improved descriptors for training set
for idx, row in merged.iterrows():
    smi = row["intermediate_smiles"]
    # Try exact match first
    if smi in training_assignments:
        sub, pos = training_assignments[smi]
    else:
        sub, pos = identify_substituent(smi)

    merged.loc[idx, "substituent"] = sub
    merged.loc[idx, "position"] = pos
    merged.loc[idx, "sigma_I"] = SIGMA_I.get(sub, 0.3)
    merged.loc[idx, "sigma_R"] = SIGMA_R.get(sub, 0.0)

    # σ_eff: σ_I for meta/ortho, σ_I + σ_R for para
    if pos == "para":
        merged.loc[idx, "sigma_eff"] = SIGMA_I.get(sub, 0.3) + SIGMA_R.get(sub, 0.0)
    else:
        merged.loc[idx, "sigma_eff"] = SIGMA_I.get(sub, 0.3)

    # Chelation: only at ortho
    if pos == "ortho":
        merged.loc[idx, "chelation"] = CHELATION.get(sub, 0.0)
    else:
        merged.loc[idx, "chelation"] = 0.0

# Derived DFT descriptors
merged["hardness"] = (merged["dft_LUMO_eV"] - merged["dft_HOMO_eV"]) / 2
merged["chem_pot"] = (merged["dft_HOMO_eV"] + merged["dft_LUMO_eV"]) / 2
merged["electrophilicity"] = merged["chem_pot"]**2 / (2 * merged["hardness"])

print("=== Improved descriptors for training set ===")
print(f"{'Name':<40s} {'Sub':>5s} {'Pos':>6s} {'σ':>5s} {'σ_I':>5s} {'σ_eff':>6s} {'δo':>3s} {'chel':>5s} {'δb':>3s} {'μ':>5s} {'HOMO':>6s} {'η':>5s} {'log_t½':>7s}")
print("-"*110)
for _, r in merged.sort_values("log_t_half").iterrows():
    print(f"  {r['intermediate'][:38]:<38s} {r['substituent']:>5s} {r['position']:>6s} "
          f"{r['sigma_hammett']:>5.2f} {r['sigma_I']:>5.2f} {r['sigma_eff']:>6.2f} "
          f"{int(r['delta_ortho']):>3d} {r['chelation']:>5.1f} {int(r['delta_benzyne']):>3d} "
          f"{r['dft_dipole_D']:>5.2f} {r['dft_HOMO_eV']:>6.2f} {r['hardness']:>5.2f} {r['log_t_half']:>+7.2f}")

# ── 3. LOO cross-validation ──────────────────────────────────────────

def loo_cv(X, y, model_cls=LinearRegression, **kw):
    n = len(y)
    y_pred = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool); mask[i] = False
        m = model_cls(**kw)
        m.fit(X[mask], y[mask])
        y_pred[i] = m.predict(X[i:i+1])[0]
    return y_pred

def eval_model(X, y, name, model_cls=LinearRegression, **kw):
    m = model_cls(**kw)
    m.fit(X, y)
    y_fit = m.predict(X)
    y_loo = loo_cv(X, y, model_cls, **kw)
    r2 = r2_score(y, y_fit)
    q2 = 1 - np.sum((y - y_loo)**2) / np.sum((y - y.mean())**2)
    return {
        "name": name, "model": m, "r2": r2, "q2": q2,
        "mae_fit": mean_absolute_error(y, y_fit),
        "mae_loo": mean_absolute_error(y, y_loo),
        "y_fit": y_fit, "y_loo": y_loo,
        "coef": m.coef_, "intercept": m.intercept_,
    }

y = merged["log_t_half"].values

# ── 4. Systematic model comparison ───────────────────────────────────

# V1 models (from previous analysis)
v1_models = {
    "v1: σ+Es+δo+μ":         ["sigma_hammett", "Es_taft", "delta_ortho", "dft_dipole_D"],
    "v1: σ+Es+δo+δb":        ["sigma_hammett", "Es_taft", "delta_ortho", "delta_benzyne"],
}

# V2 models: improved descriptors
v2_models = {
    # Core: σ_I replaces σ
    "v2: σ_I+Es+chel+μ":     ["sigma_I", "Es_taft", "chelation", "dft_dipole_D"],
    "v2: σ_eff+Es+chel+μ":   ["sigma_eff", "Es_taft", "chelation", "dft_dipole_D"],
    # With benzyne pathway
    "v2: σ_I+Es+chel+δb+μ":  ["sigma_I", "Es_taft", "chelation", "delta_benzyne", "dft_dipole_D"],
    "v2: σ_eff+Es+chel+δb+μ": ["sigma_eff", "Es_taft", "chelation", "delta_benzyne", "dft_dipole_D"],
    # With HOMO (reactivity)
    "v2: σ_I+Es+chel+μ+HOMO": ["sigma_I", "Es_taft", "chelation", "dft_dipole_D", "dft_HOMO_eV"],
    # With LUMO (electrophilicity)
    "v2: σ_I+Es+chel+μ+LUMO": ["sigma_I", "Es_taft", "chelation", "dft_dipole_D", "dft_LUMO_eV"],
    # With hardness (HOMO-LUMO gap)
    "v2: σ_I+Es+chel+μ+η":   ["sigma_I", "Es_taft", "chelation", "dft_dipole_D", "hardness"],
    # Minimal (3 params)
    "v2: σ_I+chel+μ":        ["sigma_I", "chelation", "dft_dipole_D"],
    # σ_I + δ_benzyne + μ (no chelation)
    "v2: σ_I+δb+μ":          ["sigma_I", "delta_benzyne", "dft_dipole_D"],
}

all_models = {**v1_models, **v2_models}

print("\n" + "="*90)
print(f"MODEL COMPARISON: predicting log₁₀(t½) at -40°C for ArLi (n={len(y)})")
print("="*90)
print(f"  {'Model':<30s} {'p':>2s} {'R²':>6s} {'Q²_LOO':>7s} {'MAE_LOO':>8s} {'df_res':>6s} {'Saturation':>10s}")
print("  "+"-"*75)

results = {}
for name, cols in all_models.items():
    X = merged[cols].values
    p = X.shape[1]
    df_res = len(y) - p - 1
    sat = (p + 1) / len(y) * 100

    # Choose method based on p
    if p <= 3:
        res = eval_model(X, y, name)
    else:
        # Ridge with LOO-optimized alpha
        best_q2, best_alpha = -999, 1.0
        for alpha in [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
            Xs = StandardScaler().fit_transform(X)
            q2_test = eval_model(Xs, y, name, Ridge, alpha=alpha)["q2"]
            if q2_test > best_q2:
                best_q2, best_alpha = q2_test, alpha
        Xs = StandardScaler().fit_transform(X)
        res = eval_model(Xs, y, name, Ridge, alpha=best_alpha)
        res["alpha"] = best_alpha
        res["scaler"] = StandardScaler().fit(X)

    results[name] = res
    flag = " ★★" if res["q2"] > 0.9 else " ★" if res["q2"] > 0.7 else ""
    warn = " ⚠" if sat > 70 else ""
    print(f"  {name:<30s} {p:>2d} {res['r2']:>6.3f} {res['q2']:>7.3f} {res['mae_loo']:>8.3f} {df_res:>6d} {sat:>8.0f}%{warn}{flag}")

# ── 5. Best model analysis ───────────────────────────────────────────

best_name = max(results, key=lambda k: results[k]["q2"])
best = results[best_name]
best_cols = all_models[best_name]

print(f"\n★ Best model: {best_name}")
print(f"  Q²_LOO = {best['q2']:.4f}, R² = {best['r2']:.4f}")
print(f"  MAE_LOO = {best['mae_loo']:.3f} decades (≈ {10**best['mae_loo']:.1f}× factor)")

# Print equation in raw (unstandardized) space
if "alpha" in best:
    scaler = best["scaler"]
    X_train_raw = merged[best_cols].values
    X_train_std = scaler.transform(X_train_raw)
    model_std = Ridge(alpha=best["alpha"]).fit(X_train_std, y)
    coef_raw = model_std.coef_ / scaler.scale_
    intercept_raw = model_std.intercept_ - np.sum(coef_raw * scaler.mean_)
    print(f"\n  Raw equation:")
    eq = f"  log₁₀(t½) = {intercept_raw:.3f}"
    for fn, c in zip(best_cols, coef_raw):
        eq += f" {c:+.3f}·{fn}"
    print(eq)
else:
    coef_raw = best["coef"]
    intercept_raw = best["intercept"]
    print(f"\n  Equation:")
    eq = f"  log₁₀(t½) = {intercept_raw:.3f}"
    for fn, c in zip(best_cols, coef_raw):
        eq += f" {c:+.3f}·{fn}"
    print(eq)

# ── 6. Per-intermediate LOO analysis ─────────────────────────────────

y_loo = best["y_loo"]
print(f"\n{'Name':<40s} {'Observed':>9s} {'LOO_pred':>9s} {'Residual':>9s} {'|error|':>8s}")
print("-"*80)
for j, (_, r) in enumerate(merged.sort_values("log_t_half").iterrows()):
    name = r["intermediate"][:38]
    obs = y[j] if j < len(y) else r["log_t_half"]
    # Need to match index properly
    idx_in_y = merged.sort_values("log_t_half").index.get_loc(r.name)
    obs = y[idx_in_y]
    pred = y_loo[idx_in_y]
    res = pred - obs
    print(f"  {name:<38s} {obs:>+9.2f} {pred:>+9.2f} {res:>+9.3f} {abs(res):>8.3f}")

# ── 7. Chemical interpretation ────────────────────────────────────────

print("\n" + "="*70)
print("CHEMICAL INTERPRETATION")
print("="*70)

# Sort coefficients by importance
coef_importance = sorted(zip(best_cols, coef_raw), key=lambda x: abs(x[1]), reverse=True)
for fn, c in coef_importance:
    direction = "longer t½ (more stable)" if c > 0 else "shorter t½ (less stable)"
    # Physical meaning
    meanings = {
        "sigma_I": "Inductive electron withdrawal (σ_I)",
        "sigma_eff": "Effective electronic effect (σ_I + resonance at para)",
        "Es_taft": "Taft steric parameter (more negative = bulkier)",
        "chelation": "Li···heteroatom chelation at ortho (0=none, 0.7=ester)",
        "delta_ortho": "Ortho substitution (binary)",
        "delta_benzyne": "Benzyne elimination pathway available",
        "dft_dipole_D": "Molecular dipole moment (DFT)",
        "dft_HOMO_eV": "HOMO energy (reactivity)",
        "dft_LUMO_eV": "LUMO energy (electrophilicity)",
        "dft_Gsolv_kJ": "Solvation free energy (DFT)",
        "hardness": "Chemical hardness η = (LUMO-HOMO)/2",
    }
    meaning = meanings.get(fn, fn)
    print(f"\n  {fn}: coefficient = {c:+.3f}")
    print(f"    Physical: {meaning}")
    print(f"    Effect: higher value → {direction}")

# ── 8. Predict for all ArLi ──────────────────────────────────────────

# Need to assign improved descriptors to all ArLi
all_arli = clean_uniq.copy()

# Remove generic entries
all_arli = all_arli[~all_arli["intermediate"].str.contains("aryllithium / heteroaryl", na=False)].copy()

# Identify substituent and position for each
for idx, row in all_arli.iterrows():
    smi = row.get("intermediate_smiles_canonical", "")
    sub, pos = identify_substituent(smi)

    # Override position from δ_ortho/δ_benzyne
    if row.get("delta_ortho", 0) == 1:
        pos = "ortho"
    elif row.get("delta_benzyne", 0) == 1:
        pos = "ortho"
    # Try to infer meta/para from name
    name_lower = str(row.get("intermediate", "")).lower()
    if "m-" in name_lower or "meta" in name_lower:
        pos = "meta"
    elif "p-" in name_lower or "para" in name_lower:
        pos = "para"
    elif "o-" in name_lower or "ortho" in name_lower:
        pos = "ortho"

    all_arli.loc[idx, "substituent"] = sub
    all_arli.loc[idx, "position"] = pos
    all_arli.loc[idx, "sigma_I"] = SIGMA_I.get(sub, 0.3)
    all_arli.loc[idx, "sigma_R"] = SIGMA_R.get(sub, 0.0)
    if pos == "para":
        all_arli.loc[idx, "sigma_eff"] = SIGMA_I.get(sub, 0.3) + SIGMA_R.get(sub, 0.0)
    else:
        all_arli.loc[idx, "sigma_eff"] = SIGMA_I.get(sub, 0.3)
    if pos == "ortho":
        all_arli.loc[idx, "chelation"] = CHELATION.get(sub, 0.0)
    else:
        all_arli.loc[idx, "chelation"] = 0.0
    all_arli.loc[idx, "hardness"] = (row.get("dft_LUMO_eV", -5) - row.get("dft_HOMO_eV", -9)) / 2

# Predict
avail = all_arli.dropna(subset=best_cols).copy()
X_train_raw = merged[best_cols].values

if "alpha" in best:
    scaler = StandardScaler().fit(X_train_raw)
    final_model = Ridge(alpha=best["alpha"]).fit(scaler.transform(X_train_raw), y)
    X_pred = scaler.transform(avail[best_cols].values)
else:
    final_model = LinearRegression().fit(X_train_raw, y)
    X_pred = avail[best_cols].values

avail["log_t_half_pred"] = final_model.predict(X_pred)
avail["t_half_pred_s"] = 10**avail["log_t_half_pred"]

# Applicability domain
X_std_train = scaler.transform(X_train_raw) if "alpha" in best else X_train_raw
XtX_inv = np.linalg.pinv(X_std_train.T @ X_std_train)
X_std_pred = scaler.transform(avail[best_cols].values) if "alpha" in best else avail[best_cols].values
h_pred = np.array([x @ XtX_inv @ x for x in X_std_pred])
h_star = 3 * len(best_cols) / len(y)
avail["leverage"] = h_pred
train_smi = set(merged["canon_smi"])
avail["is_training"] = avail["canon_smi"].isin(train_smi)

# Reliability tiers
for col in best_cols:
    lo = merged[col].min()
    hi = merged[col].max()
    margin = 0.15 * (hi - lo) if hi > lo else 0.15
    avail[f"_inr_{col}"] = avail[col].between(lo - margin, hi + margin)
range_cols = [f"_inr_{col}" for col in best_cols]
avail["in_desc_range"] = avail[range_cols].all(axis=1)
train_y_range = [y.min() - 1.5, y.max() + 1.5]
avail["in_y_range"] = avail["log_t_half_pred"].between(*train_y_range)

avail["reliability"] = "extrapolation"
avail.loc[avail["in_desc_range"], "reliability"] = "moderate"
avail.loc[(avail["leverage"] <= h_star) & avail["in_y_range"], "reliability"] = "high"

avail = avail.sort_values("t_half_pred_s", ascending=False)

print("\n" + "="*90)
print(f"PREDICTED ArLi STABILITY RANKING (model: {best_name})")
print("="*90)
print(f"  {'#':>3s} {'Intermediate':<38s} {'Sub':>5s} {'Pos':>6s} {'t½_pred':>10s} {'log₁₀':>6s} {'h':>5s} {'Status':>8s}")
print("  "+"-"*83)
for i, (_, r) in enumerate(avail.iterrows(), 1):
    name = r["intermediate"][:36]
    t = r["t_half_pred_s"]
    logt = r["log_t_half_pred"]
    hii = r["leverage"]
    sub = str(r.get("substituent","?"))[:5]
    pos = str(r.get("position","?"))[:5]
    if r["is_training"]:
        status = "TRAIN"
    elif r["reliability"] == "high":
        status = "HIGH"
    elif r["reliability"] == "moderate":
        status = "~OK"
    else:
        status = "extrap"
    if t >= 3600: tstr = f"{t/3600:.1f}h"
    elif t >= 60: tstr = f"{t/60:.1f}min"
    elif t >= 1: tstr = f"{t:.1f}s"
    elif t >= 0.001: tstr = f"{t*1000:.0f}ms"
    else: tstr = f"{t*1e6:.0f}μs"
    print(f"  {i:>3d}. {name:<36s} {sub:>5s} {pos:>6s} {tstr:>10s} {logt:>+6.2f} {hii:>5.2f} {status:>8s}")

# ── 9. Figure ─────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("ArLi Half-Life Prediction — Improved Descriptors (v2)", fontsize=13, fontweight="bold")

# (a) Parity plot
ax = axes[0, 0]
y_loo_sorted = best["y_loo"]
# Recompute in original order
y_orig = merged["log_t_half"].values
ax.scatter(y_orig, y_loo_sorted, c="royalblue", s=70, edgecolors="k", linewidths=0.5, zorder=3)
for j, (_, r) in enumerate(merged.iterrows()):
    short = r["intermediate"].split("(")[0].strip()[:22]
    ax.annotate(short, (y_orig[j], y_loo_sorted[j]), fontsize=6.5, ha="left", va="bottom",
                xytext=(4, 4), textcoords="offset points")
lims = [min(y_orig.min(), y_loo_sorted.min()) - 0.4, max(y_orig.max(), y_loo_sorted.max()) + 0.4]
ax.plot(lims, lims, "k--", alpha=0.4, lw=1)
ax.fill_between(lims, [l-0.5 for l in lims], [l+0.5 for l in lims], alpha=0.08, color="gray")
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel("Observed log₁₀(t½ / s)")
ax.set_ylabel("LOO-Predicted log₁₀(t½ / s)")
ax.set_title(f"(a) LOO Cross-Validation\nQ² = {best['q2']:.3f}, MAE = {best['mae_loo']:.2f} dec", fontsize=10)
ax.set_aspect("equal")

# (b) v1 vs v2 comparison
ax = axes[0, 1]
compare_models = [k for k in results if k.startswith("v1") or results[k]["q2"] > 0.3]
# Always include v2 models
compare_models = list(results.keys())
q2_vals = [results[n]["q2"] for n in compare_models]
colors = ["#EF5350" if "v1" in n else "#2196F3" for n in compare_models]
bars = ax.barh(range(len(compare_models)), q2_vals, color=colors, edgecolor="k", linewidth=0.5)
ax.set_yticks(range(len(compare_models)))
ax.set_yticklabels([n.replace("v1: ","").replace("v2: ","") for n in compare_models], fontsize=7)
ax.set_xlabel("Q²_LOO")
ax.axvline(0, color="k", lw=0.5)
ax.axvline(0.5, color="gray", lw=0.5, ls=":")
ax.set_title("(b) v1 vs v2 Model Comparison", fontsize=10)
best_idx = compare_models.index(best_name)
ax.barh(best_idx, q2_vals[best_idx], color="gold", edgecolor="k", linewidth=1.5)
ax.legend([Patch(fc="#EF5350"), Patch(fc="#2196F3"), Patch(fc="gold")],
          ["v1 (original)", "v2 (improved)", "Best"], loc="lower right", fontsize=8)

# (c) Stability ranking (reliable only)
ax = axes[1, 0]
show = avail[avail["reliability"].isin(["high", "moderate"]) | avail["is_training"]].copy()
top_n = min(25, len(show))
top = show.head(top_n)
names = [r["intermediate"][:28] for _, r in top.iterrows()]
log_vals = top["log_t_half_pred"].values
bar_colors = []
for _, r in top.iterrows():
    if r["is_training"]: bar_colors.append("#4CAF50")
    elif r["reliability"] == "high": bar_colors.append("#42A5F5")
    else: bar_colors.append("#FFB74D")
ax.barh(range(top_n), log_vals, color=bar_colors, edgecolor="k", linewidth=0.5)
ax.set_yticks(range(top_n))
ax.set_yticklabels(names, fontsize=6.5)
ax.invert_yaxis()
ax.set_xlabel("Predicted log₁₀(t½ / s) at −40°C")
ax.set_title(f"(c) Stability Ranking", fontsize=10)
ax.legend([Patch(fc="#4CAF50"), Patch(fc="#42A5F5"), Patch(fc="#FFB74D")],
          ["Training", "High conf.", "Moderate"], loc="lower right", fontsize=7)

# (d) Descriptor importance (coefficient magnitude)
ax = axes[1, 1]
feat_names = best_cols
coef_abs = np.abs(coef_raw)
order = np.argsort(coef_abs)[::-1]
feat_sorted = [feat_names[i] for i in order]
coef_sorted = [coef_raw[i] for i in order]
colors_coef = ["#4CAF50" if c > 0 else "#F44336" for c in coef_sorted]
ax.barh(range(len(feat_sorted)), coef_sorted, color=colors_coef, edgecolor="k", linewidth=0.5)
ax.set_yticks(range(len(feat_sorted)))
labels_nice = {
    "sigma_I": "σ_I (inductive)",
    "sigma_eff": "σ_eff (electronic)",
    "Es_taft": "Es (steric)",
    "chelation": "Chelation strength",
    "delta_ortho": "δ_ortho",
    "delta_benzyne": "δ_benzyne (pathway)",
    "dft_dipole_D": "μ (dipole, DFT)",
    "dft_HOMO_eV": "HOMO (DFT)",
    "dft_LUMO_eV": "LUMO (DFT)",
    "hardness": "η (hardness)",
}
ax.set_yticklabels([labels_nice.get(f, f) for f in feat_sorted], fontsize=8)
ax.axvline(0, color="k", lw=0.5)
ax.set_xlabel("Coefficient (raw, unstandardized)")
ax.set_title("(d) Descriptor Importance", fontsize=10)
ax.legend([Patch(fc="#4CAF50"), Patch(fc="#F44336")],
          ["Stabilizing (+t½)", "Destabilizing (−t½)"], loc="lower right", fontsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT / "arli_halflife_v2.png", dpi=200, bbox_inches="tight")
print(f"\nFigure saved: {OUT / 'arli_halflife_v2.png'}")

# Save predictions
avail.to_csv(OUT / "arli_t_half_predictions_v2.csv", index=False)
print(f"Predictions saved: {OUT / 'arli_t_half_predictions_v2.csv'}")
