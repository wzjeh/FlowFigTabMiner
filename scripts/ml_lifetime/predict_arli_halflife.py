"""
ArLi t½ prediction — focused, clean analysis.

Goal: predict half-life of ArLi intermediates from molecular descriptors.
Approach: directly regress log10(t½) on descriptors (no Ea/lnA decomposition).
Training set: 9 ArLi with Arrhenius-derived t½ at -40 °C.
Prediction set: all ArLi with required descriptors.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from rdkit import Chem
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from itertools import combinations
import warnings, pathlib

warnings.filterwarnings("ignore")
plt.rcParams.update({"font.size": 9, "font.family": "Arial"})

OUT = pathlib.Path("data/ml_lifetime/arli_halflife")
OUT.mkdir(parents=True, exist_ok=True)

# ── 1. Load & merge by canonical SMILES ──────────────────────────────

def canon(s):
    if pd.isna(s):
        return None
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m) if m else s

arr = pd.read_csv("data/ml_lifetime/phase_b_arrhenius.csv")
clean = pd.read_csv("data/ml_lifetime/clean_organolithium_unified_descriptors.csv")

arr["canon_smi"] = arr["intermediate_smiles"].apply(canon)
clean["canon_smi"] = clean["intermediate_smiles_canonical"].apply(canon)

# Keep only ArLi in Arrhenius (those with aromatic ring)
arli_arr = arr[arr["canon_smi"].apply(lambda s: s is not None and "c1" in str(s))].copy()

# Get unique DFT descriptors per SMILES from clean
dft_cols = [
    "sigma_hammett", "Es_taft", "delta_ortho", "delta_benzyne",
    "dft_dipole_D", "dft_Gsolv_kJ", "dft_charge_Li", "dft_HOMO_eV",
    "dft_LiC_BDE_kJ", "dft_wiberg_LiC",
]
clean_uniq = (
    clean[clean["intermediate_class"] == "ArLi"]
    .drop_duplicates("canon_smi")
    [["canon_smi", "intermediate"] + dft_cols]
    .rename(columns={"intermediate": "clean_name"})
)

# Merge: Arrhenius t½ ← DFT descriptors
merged = arli_arr.drop_duplicates("canon_smi").merge(clean_uniq, on="canon_smi", how="inner")
print(f"ArLi with Arrhenius + descriptors: {len(merged)}")

# ── 2. Prepare target variable ───────────────────────────────────────

# Use t½ at -40°C (standard comparison temperature for ArLi)
merged["log_t_half"] = np.log10(merged["t_half_m40C_s"].clip(lower=1e-3))

# Display the training set
print("\n" + "="*80)
print("TRAINING DATA: ArLi intermediates with measured t½ at -40°C")
print("="*80)
for _, r in merged.sort_values("log_t_half").iterrows():
    name = r["intermediate"][:50]
    t = r["t_half_m40C_s"]
    logt = r["log_t_half"]
    sig = r["sigma_hammett"]
    es = r["Es_taft"]
    dip = r["dft_dipole_D"]
    print(f"  {name:50s}  t½={t:>8.1f}s  log₁₀={logt:+5.2f}  σ={sig:5.2f}  Es={es:5.2f}  μ={dip:5.2f}D")

# ── 3. LOO cross-validation function ─────────────────────────────────

def loo_cv(X, y, model_cls=LinearRegression, **kw):
    """Leave-one-out cross-validation. Returns predicted values."""
    n = len(y)
    y_pred = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        m = model_cls(**kw)
        m.fit(X[mask], y[mask])
        y_pred[i] = m.predict(X[i:i+1])[0]
    return y_pred

def eval_model(X, y, name, model_cls=LinearRegression, **kw):
    """Fit model, compute metrics."""
    m = model_cls(**kw)
    m.fit(X, y)
    y_fit = m.predict(X)
    y_loo = loo_cv(X, y, model_cls, **kw)
    r2 = r2_score(y, y_fit)
    q2 = 1 - np.sum((y - y_loo)**2) / np.sum((y - y.mean())**2)
    mae_fit = mean_absolute_error(y, y_fit)
    mae_loo = mean_absolute_error(y, y_loo)
    return {
        "name": name, "model": m, "r2": r2, "q2": q2,
        "mae_fit": mae_fit, "mae_loo": mae_loo,
        "y_fit": y_fit, "y_loo": y_loo,
        "coef": m.coef_ if hasattr(m, "coef_") else None,
        "intercept": m.intercept_ if hasattr(m, "intercept_") else None,
    }

# ── 4. Test multiple models ──────────────────────────────────────────

y = merged["log_t_half"].values
results = {}

# Empirical descriptors
emp_features = {
    "σ": ["sigma_hammett"],
    "σ+Es": ["sigma_hammett", "Es_taft"],
    "σ+Es+δ_ortho": ["sigma_hammett", "Es_taft", "delta_ortho"],
    "σ+Es+δ_ortho+δ_benzyne": ["sigma_hammett", "Es_taft", "delta_ortho", "delta_benzyne"],
}

# DFT descriptors
dft_features = {
    "μ": ["dft_dipole_D"],
    "μ+Gsolv": ["dft_dipole_D", "dft_Gsolv_kJ"],
    "μ+Gsolv+BDE": ["dft_dipole_D", "dft_Gsolv_kJ", "dft_LiC_BDE_kJ"],
}

# Combined
combo_features = {
    "σ+Es+δ_ortho+μ": ["sigma_hammett", "Es_taft", "delta_ortho", "dft_dipole_D"],
    "σ+Es+δ_ortho+Gsolv": ["sigma_hammett", "Es_taft", "delta_ortho", "dft_Gsolv_kJ"],
}

all_features = {**emp_features, **dft_features, **combo_features}

print("\n" + "="*80)
print("MODEL COMPARISON: predicting log₁₀(t½) at -40°C for ArLi (n=9)")
print("="*80)
print(f"{'Model':<30s} {'p':>2s} {'R²':>6s} {'Q²_LOO':>7s} {'MAE_fit':>8s} {'MAE_LOO':>8s}")
print("-"*65)

for name, cols in all_features.items():
    X = merged[cols].values
    p = X.shape[1]
    # OLS for p<=3, Ridge for p>=4 (only 9 samples)
    if p <= 3:
        res = eval_model(X, y, name)
    else:
        # Find best alpha by LOO
        best_q2, best_alpha = -999, 1.0
        for alpha in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
            Xs = StandardScaler().fit_transform(X)
            q2_test = eval_model(Xs, y, name, Ridge, alpha=alpha)["q2"]
            if q2_test > best_q2:
                best_q2, best_alpha = q2_test, alpha
        Xs = StandardScaler().fit_transform(X)
        res = eval_model(Xs, y, name, Ridge, alpha=best_alpha)
        res["alpha"] = best_alpha
    results[name] = res
    flag = " ★" if res["q2"] > 0.8 else ""
    print(f"  {name:<28s} {p:>2d} {res['r2']:>6.3f} {res['q2']:>7.3f} {res['mae_fit']:>8.3f} {res['mae_loo']:>8.3f}{flag}")

# ── 5. Best model details ────────────────────────────────────────────

best_name = max(results, key=lambda k: results[k]["q2"])
best = results[best_name]
print(f"\n★ Best model: {best_name}  (Q²_LOO = {best['q2']:.3f})")
print(f"  R² = {best['r2']:.4f}")
print(f"  MAE_fit = {best['mae_fit']:.3f} decades")
print(f"  MAE_LOO = {best['mae_loo']:.3f} decades")
if best["coef"] is not None:
    feat_names = all_features[best_name]
    print(f"  Equation: log₁₀(t½) = {best['intercept']:.3f}", end="")
    for fn, c in zip(feat_names, best["coef"]):
        print(f" {c:+.3f}·{fn}", end="")
    print()

# ── 6. Predict t½ for ALL ArLi (with applicability domain) ───────────

best_cols = all_features[best_name]
all_arli = clean_uniq.dropna(subset=best_cols).copy()

# Remove generic entries (no specific SMILES or "aryllithium" catch-all)
all_arli = all_arli[~all_arli["clean_name"].str.contains("aryllithium / heteroaryl", na=False)].copy()

# Fit final model on all training data
X_train = merged[best_cols].values
if best.get("alpha"):
    scaler = StandardScaler().fit(X_train)
    final_model = Ridge(alpha=best["alpha"]).fit(scaler.transform(X_train), y)
    X_pred_raw = all_arli[best_cols].values
    X_pred = scaler.transform(X_pred_raw)
else:
    final_model = LinearRegression().fit(X_train, y)
    X_pred_raw = all_arli[best_cols].values
    X_pred = X_pred_raw

all_arli["log_t_half_pred"] = final_model.predict(X_pred)
all_arli["t_half_pred_s"] = 10**all_arli["log_t_half_pred"]

# Mark training points
train_smi = set(merged["canon_smi"])
all_arli["is_training"] = all_arli["canon_smi"].isin(train_smi)

# Applicability domain: leverage-based (Williams plot)
# Use standardized X for leverage computation (consistent with Ridge)
if best.get("alpha"):
    X_std_train = scaler.transform(X_train)
    X_std_pred = scaler.transform(X_pred_raw)
else:
    X_std_train = X_train
    X_std_pred = X_pred_raw

XtX_inv = np.linalg.pinv(X_std_train.T @ X_std_train)
h_star = 3 * X_train.shape[1] / len(y)  # leverage threshold
h_pred = np.array([x @ XtX_inv @ x for x in X_std_pred])
h_train = np.diag(X_std_train @ XtX_inv @ X_std_train.T)
all_arli["leverage"] = h_pred
all_arli["in_domain"] = h_pred <= h_star

# Descriptor-range check: each descriptor within [min-10%, max+10%] of training
for col in best_cols:
    lo = merged[col].min()
    hi = merged[col].max()
    margin = 0.1 * (hi - lo) if hi > lo else 0.1
    all_arli[f"in_range_{col}"] = all_arli[col].between(lo - margin, hi + margin)
range_cols = [f"in_range_{c}" for c in best_cols]
all_arli["in_desc_range"] = all_arli[range_cols].all(axis=1)

# Flag if prediction is far outside training range
train_range = [y.min() - 1.5, y.max() + 1.5]  # ±1.5 decade margin
all_arli["in_y_range"] = all_arli["log_t_half_pred"].between(*train_range)

# Three-tier reliability
all_arli["reliability"] = "extrapolation"
all_arli.loc[all_arli["in_desc_range"], "reliability"] = "moderate"
all_arli.loc[all_arli["in_domain"] & all_arli["in_y_range"], "reliability"] = "high"
all_arli["reliable"] = all_arli["reliability"].isin(["high", "moderate"])

# Sort by predicted stability
all_arli = all_arli.sort_values("t_half_pred_s", ascending=False)

n_high = (all_arli["reliability"] == "high").sum()
n_moderate = (all_arli["reliability"] == "moderate").sum()
n_extrap = (all_arli["reliability"] == "extrapolation").sum()
n_non_train = (~all_arli["is_training"]).sum()
print(f"\nApplicability domain: h* = {h_star:.2f}")
print(f"  Predictions (excl. training): {n_non_train}")
print(f"    High confidence (leverage OK): {n_high}")
print(f"    Moderate (descriptors in range): {n_moderate}")
print(f"    Extrapolation: {n_extrap}")

print("\n" + "="*80)
print(f"PREDICTED ArLi STABILITY RANKING (model: {best_name})")
print("="*80)
print(f"{'#':>3s} {'Intermediate':<45s} {'t½_pred':>10s} {'log₁₀':>6s} {'h_ii':>5s} {'Status':>10s}")
print("-"*85)
for i, (_, r) in enumerate(all_arli.iterrows(), 1):
    name = r["clean_name"][:43]
    t = r["t_half_pred_s"]
    logt = r["log_t_half_pred"]
    hii = r["leverage"]
    # Status
    if r["is_training"]:
        status = "TRAIN"
    elif r["reliability"] == "high":
        status = "HIGH"
    elif r["reliability"] == "moderate":
        status = "~OK"
    else:
        status = "⚠ extrap"
    # Format time nicely
    if t >= 3600:
        tstr = f"{t/3600:.1f} h"
    elif t >= 60:
        tstr = f"{t/60:.1f} min"
    elif t >= 1:
        tstr = f"{t:.1f} s"
    elif t >= 0.001:
        tstr = f"{t*1000:.0f} ms"
    else:
        tstr = f"{t*1e6:.0f} μs"
    print(f"  {i:>2d}. {name:<43s} {tstr:>10s} {logt:>+6.2f} {hii:>5.2f} {status:>10s}")

# Save predictions
all_arli.to_csv(OUT / "arli_t_half_predictions.csv", index=False)

# ── 7. Figure: 4-panel summary ───────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(11, 10))
fig.suptitle("ArLi Half-Life Prediction from Molecular Descriptors", fontsize=13, fontweight="bold")

# ── Panel (a): Parity plot (LOO) ──
ax = axes[0, 0]
y_loo = best["y_loo"]
ax.scatter(y, y_loo, c="royalblue", s=60, edgecolors="k", linewidths=0.5, zorder=3)
for j, (_, r) in enumerate(merged.iterrows()):
    short = r["intermediate"].split("(")[0].strip()[:25]
    ax.annotate(short, (y[j], y_loo[j]), fontsize=6, ha="left", va="bottom",
                xytext=(3, 3), textcoords="offset points")
lims = [min(y.min(), y_loo.min()) - 0.3, max(y.max(), y_loo.max()) + 0.3]
ax.plot(lims, lims, "k--", alpha=0.4, lw=1)
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel("Observed log₁₀(t½ / s)")
ax.set_ylabel("LOO-Predicted log₁₀(t½ / s)")
ax.set_title(f"(a) LOO Cross-Validation\nQ² = {best['q2']:.3f}, MAE = {best['mae_loo']:.2f} dec", fontsize=10)
ax.set_aspect("equal")

# ── Panel (b): Model comparison bar chart ──
ax = axes[0, 1]
model_names = list(results.keys())
q2_vals = [results[n]["q2"] for n in model_names]
colors = ["#2196F3" if n in emp_features else "#FF9800" if n in dft_features else "#4CAF50"
          for n in model_names]
bars = ax.barh(range(len(model_names)), q2_vals, color=colors, edgecolor="k", linewidth=0.5)
ax.set_yticks(range(len(model_names)))
ax.set_yticklabels(model_names, fontsize=8)
ax.set_xlabel("Q²_LOO")
ax.axvline(0, color="k", lw=0.5)
ax.set_title("(b) Model Comparison", fontsize=10)
# Legend
from matplotlib.patches import Patch
ax.legend(
    [Patch(fc="#2196F3"), Patch(fc="#FF9800"), Patch(fc="#4CAF50")],
    ["Empirical", "DFT", "Combined"], loc="lower right", fontsize=8
)
# Mark best
best_idx = model_names.index(best_name)
ax.barh(best_idx, q2_vals[best_idx], color="gold", edgecolor="k", linewidth=1.5)

# ── Panel (c): Stability ranking (reliable only) ──
ax = axes[1, 0]
# Show reliable predictions + training data
show = all_arli[all_arli["reliable"] | all_arli["is_training"]].copy()
top_n = min(25, len(show))
top = show.head(top_n)
names = [r["clean_name"][:30] for _, r in top.iterrows()]
log_vals = top["log_t_half_pred"].values
bar_colors = []
for _, r in top.iterrows():
    if r["is_training"]:
        bar_colors.append("#4CAF50")
    elif r["reliable"]:
        bar_colors.append("#90CAF9")
    else:
        bar_colors.append("#FFCDD2")
ax.barh(range(top_n), log_vals, color=bar_colors, edgecolor="k", linewidth=0.5)
ax.set_yticks(range(top_n))
ax.set_yticklabels(names, fontsize=7)
ax.invert_yaxis()
ax.set_xlabel("Predicted log₁₀(t½ / s) at -40°C")
ax.set_title(f"(c) Stability Ranking (in-domain)", fontsize=10)
ax.legend(
    [Patch(fc="#4CAF50"), Patch(fc="#90CAF9"), Patch(fc="#FFCDD2")],
    ["Training", "In-domain prediction", "Extrapolation"], loc="lower right", fontsize=8
)

# ── Panel (d): LOO residuals per intermediate ──
ax = axes[1, 1]
residuals = y_loo - y
abs_res = np.abs(residuals)
order = np.argsort(abs_res)[::-1]
short_names = [merged.iloc[j]["intermediate"].split("(")[0].strip()[:25] for j in order]
colors_res = ["#F44336" if abs_res[j] > 0.5 else "#2196F3" for j in order]
ax.barh(range(len(order)), residuals[order], color=colors_res, edgecolor="k", linewidth=0.5)
ax.set_yticks(range(len(order)))
ax.set_yticklabels(short_names, fontsize=7)
ax.axvline(0, color="k", lw=0.5)
ax.set_xlabel("LOO Residual (decades)")
ax.set_title("(d) LOO Prediction Errors", fontsize=10)

plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT / "arli_halflife_prediction.png", dpi=200, bbox_inches="tight")
print(f"\nFigure saved: {OUT / 'arli_halflife_prediction.png'}")
print(f"Predictions saved: {OUT / 'arli_t_half_predictions.csv'}")

# ── 8. Summary report ────────────────────────────────────────────────

report = f"""
{'='*70}
ArLi HALF-LIFE PREDICTION — SUMMARY REPORT
{'='*70}

DATA:
  Training set: {len(merged)} ArLi intermediates with Arrhenius-derived t½
  Prediction set: {len(all_arli)} ArLi with required descriptors
  Target: log₁₀(t½) at -40°C
  t½ range: {merged['t_half_m40C_s'].min():.2f} s to {merged['t_half_m40C_s'].max():.1f} s
            ({merged['log_t_half'].min():.2f} to {merged['log_t_half'].max():.2f} decades)

BEST MODEL: {best_name}
  Q²_LOO = {best['q2']:.4f}
  R²     = {best['r2']:.4f}
  MAE_LOO = {best['mae_loo']:.3f} decades (= {10**best['mae_loo']:.1f}× factor)
  MAE_fit = {best['mae_fit']:.3f} decades

EQUATION:
  log₁₀(t½) = {best['intercept']:.4f}"""

if best["coef"] is not None:
    for fn, c in zip(all_features[best_name], best["coef"]):
        report += f" {c:+.4f}·{fn}"

report += f"""

MODEL COMPARISON (all on n={len(merged)} ArLi):
{'  ':<2s}{'Model':<30s} {'p':>2s} {'R²':>6s} {'Q²':>7s} {'MAE_LOO':>8s}
{'  '+'-'*56}"""
for name in all_features:
    r = results[name]
    p = len(all_features[name])
    report += f"\n  {name:<30s} {p:>2d} {r['r2']:>6.3f} {r['q2']:>7.3f} {r['mae_loo']:>8.3f}"

report += f"""

APPLICABILITY DOMAIN:
  Leverage threshold h* = {h_star:.2f} (= 3p/n)
  High confidence: {n_high}/{n_non_train} predictions
  Moderate (in descriptor range): {n_moderate}/{n_non_train} predictions
  Extrapolation: {n_extrap}/{n_non_train} predictions

INTERPRETATION:
  - MAE_LOO = {best['mae_loo']:.2f} decades means predictions are accurate
    to within ~{10**best['mae_loo']:.1f}x of true t1/2 on average.
  - This is {"excellent" if best['q2'] > 0.9 else "good" if best['q2'] > 0.7 else "moderate" if best['q2'] > 0.5 else "poor"} predictive power for a QSPR model.

CHEMICAL INTERPRETATION:
  log10(t1/2) = {best['intercept']:.2f}"""

if best["coef"] is not None:
    feat_names = all_features[best_name]
    for fn, c in zip(feat_names, best["coef"]):
        report += f" {c:+.2f}*{fn}"
    report += """

  Key drivers (ranked by |coefficient|):"""
    coef_list = sorted(zip(feat_names, best["coef"]), key=lambda x: abs(x[1]), reverse=True)
    for fn, c in coef_list:
        direction = "increases" if c > 0 else "decreases"
        report += f"\n    {fn}: {c:+.2f} → higher {fn} {direction} t1/2"

report += """
"""

with open(OUT / "arli_halflife_report.txt", "w") as f:
    f.write(report)
print(report)
