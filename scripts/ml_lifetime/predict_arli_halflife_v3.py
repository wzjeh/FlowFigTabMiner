"""
ArLi t½ prediction — v3: expanded training set (n=17-18 vs n=9).

After re-running Phase A on clean_organolithium_unified_descriptors.csv,
we now have 18 ArLi with Arrhenius + σ + DFT descriptors.

Also tests improved v2 descriptors (σ_I, chelation) which might now work
with larger n.
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

# ── 1. Load & merge ──────────────────────────────────────────────────

def canon(s):
    if pd.isna(s): return None
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m) if m else s

arr = pd.read_csv("data/ml_lifetime/phase_b_arrhenius.csv")
clean = pd.read_csv("data/ml_lifetime/clean_organolithium_unified_descriptors.csv")
arr["canon_smi"] = arr["intermediate_smiles"].apply(canon)
clean["canon_smi"] = clean["intermediate_smiles_canonical"].apply(canon)

clean_uniq = clean.drop_duplicates("canon_smi")
merged_all = arr.merge(
    clean_uniq[["canon_smi", "intermediate", "intermediate_class",
                "sigma_hammett", "Es_taft", "delta_ortho", "delta_benzyne",
                "dft_dipole_D", "dft_Gsolv_kJ", "dft_HOMO_eV", "dft_LUMO_eV",
                "dft_charge_Li", "dft_LiC_BDE_kJ", "dft_wiberg_LiC"]].rename(
                    columns={"intermediate": "clean_name"}),
    on="canon_smi", how="inner"
)

# ── Fill σ for heteroaryllithiums (literature values) ──
# Source: Hansch-Leo-Taft 1991 (Chem Rev); Blanch 1966 (J Chem Soc B)
# σ_I additive approach for heteroaryl: sum of σ_I contributions at C-Li site
HETEROARYL_SIGMA = {
    # 2-methylbenzofuran-3-yllithium: O(ortho,σ_I=0.29) + CH3(ortho,σ_I=-0.04)
    "[Li]c1c(C)oc2ccccc12": 0.25,
    # 3-lithio-2-methylbenzothiophene: S(ortho,σ_I=0.23) + CH3(ortho,σ_I=-0.04)
    "[Li]c1c(C)sc2ccccc12": 0.19,
    # 2,3-dibromo-4-lithiopyridine: N(meta,σ=0.55) + Br(ortho,σ_I=0.44) + Br(meta,σ_m=0.39)
    "[Li]c1ccnc(Br)c1Br": 1.38,
}
for smi, sig in HETEROARYL_SIGMA.items():
    mask = merged_all["intermediate_smiles"] == smi
    if mask.any():
        merged_all.loc[mask, "sigma_hammett"] = sig
        print(f"  Assigned σ={sig:.2f} to {merged_all.loc[mask, 'intermediate'].iloc[0]}")

# Filter: ArLi only, has σ + DFT
arli = merged_all[
    (merged_all["intermediate_class"] == "ArLi") &
    merged_all["sigma_hammett"].notna() &
    merged_all["dft_dipole_D"].notna()
].copy()
arli["log_t_half"] = np.log10(arli["t_half_m40C_s"].clip(lower=1e-6))

# Quality tiers
arli["quality"] = "low"
arli.loc[arli["arrhenius_r2"] >= 0.9, "quality"] = "good"
arli.loc[(arli["arrhenius_r2"] >= 0.9) & (arli["Ea_decomp_kJ_mol"] > 5), "quality"] = "high"
# p-CN-PhLi: Ea=4.5 is low but fit is R²=1.0, include as "good"
# (low Ea may reflect true near-zero barrier, not bad fit)

print(f"ArLi with Arrhenius + σ + DFT: {len(arli)}")
print(f"  High quality (R²≥0.9, Ea>5): {(arli['quality']=='high').sum()}")
print(f"  Good quality (R²≥0.9):       {(arli['quality'].isin(['high','good'])).sum()}")
print(f"  Low quality:                  {(arli['quality']=='low').sum()}")

# ── 2. Build improved descriptors ────────────────────────────────────

SIGMA_I = {"I": 0.39, "Br": 0.44, "Cl": 0.47, "F": 0.50,
           "CN": 0.56, "NO2": 0.67, "COOR": 0.30, "COR": 0.30,
           "OMe": 0.27, "H": 0.00, "CH3": -0.04,
           "furanO": 0.25, "thioS": 0.19, "Br+N": 1.38}

SIGMA_R = {"I": -0.04, "Br": -0.21, "Cl": -0.24, "F": -0.39,
           "CN": +0.10, "NO2": +0.11, "COOR": +0.15, "COR": +0.17,
           "OMe": -0.56, "H": 0.00, "CH3": -0.13,
           "furanO": -0.04, "thioS": -0.03, "Br+N": 0.00}

CHELATION = {"H": 0.0, "Br": 0.05, "I": 0.05, "Cl": 0.05,
             "OMe": 0.3, "CN": 0.5, "COOR": 0.7, "COR": 0.8, "NO2": 0.6,
             "furanO": 0.3, "thioS": 0.2, "Br+N": 0.05}

# Es_alkyl: Taft steric parameter of the ester -OR group
# Applied to ALL positions (not just ortho), because ester alkyl affects t½
# universally: Me(0.21s) → Et(0.49s) → iPr(1.6s) → tBu(63s) at meta
ES_ALKYL = {"Me": 0.00, "Et": -0.07, "iPr": -0.47, "tBu": -1.54}

# SMILES → (substituent, position, ester_alkyl)
SMILES_TO_SUB_POS_ALKYL = {
    # ortho haloPhLi
    "[Li]c1ccccc1Br": ("Br", "ortho", None),
    "[Li]c1ccccc1I": ("I", "ortho", None),
    "Brc1ccccc1[Li]": ("Br", "ortho", None),
    "Ic1ccccc1[Li]": ("I", "ortho", None),
    # ortho esters
    "COC(=O)c1ccccc1[Li]": ("COOR", "ortho", "Me"),
    "CCOC(=O)c1ccccc1[Li]": ("COOR", "ortho", "Et"),
    "CC(C)OC(=O)c1ccccc1[Li]": ("COOR", "ortho", "iPr"),
    "CC(C)(C)OC(=O)c1ccccc1[Li]": ("COOR", "ortho", "tBu"),
    # meta esters
    "COC(=O)c1cccc([Li])c1": ("COOR", "meta", "Me"),
    "CCOC(=O)c1cccc([Li])c1": ("COOR", "meta", "Et"),
    "CC(C)OC(=O)c1cccc([Li])c1": ("COOR", "meta", "iPr"),
    "CC(C)(C)OC(=O)c1cccc([Li])c1": ("COOR", "meta", "tBu"),
    # para esters
    "COC(=O)c1ccc([Li])cc1": ("COOR", "para", "Me"),
    "CCOC(=O)c1ccc([Li])cc1": ("COOR", "para", "Et"),
    "CC(C)OC(=O)c1ccc([Li])cc1": ("COOR", "para", "iPr"),
    "CC(C)(C)OC(=O)c1ccc([Li])cc1": ("COOR", "para", "tBu"),
    # CN
    "N#Cc1cccc([Li])c1": ("CN", "meta", None),
    "N#Cc1ccc([Li])cc1": ("CN", "para", None),
    "N#Cc1ccccc1[Li]": ("CN", "ortho", None),
    # NO2
    "[O-][N+](=O)c1cccc([Li])c1": ("NO2", "meta", None),
    "[O-][N+](=O)c1ccc([Li])cc1": ("NO2", "para", None),
    "[Li]c1ccccc1[N+](=O)[O-]": ("NO2", "ortho", None),
    # Heteroaryl
    "[Li]c1c(C)oc2ccccc12": ("furanO", "ortho", None),   # benzofuranyl, O ortho to Li
    "[Li]c1c(C)sc2ccccc12": ("thioS", "ortho", None),    # benzothienyl, S ortho to Li
    "[Li]c1ccnc(Br)c1Br": ("Br+N", "ortho", None),       # dibromo pyridyl, Br ortho to Li
}

SMILES_TO_SUB_POS = {
    # ortho haloPhLi
    "[Li]c1ccccc1Br": ("Br", "ortho"),
    "[Li]c1ccccc1I": ("I", "ortho"),
    "Brc1ccccc1[Li]": ("Br", "ortho"),
    "Ic1ccccc1[Li]": ("I", "ortho"),
    # ortho esters
    "COC(=O)c1ccccc1[Li]": ("COOR", "ortho"),
    "CCOC(=O)c1ccccc1[Li]": ("COOR", "ortho"),
    "CC(C)OC(=O)c1ccccc1[Li]": ("COOR", "ortho"),
    "CC(C)(C)OC(=O)c1ccccc1[Li]": ("COOR", "ortho"),
    # meta esters
    "COC(=O)c1cccc([Li])c1": ("COOR", "meta"),
    "CCOC(=O)c1cccc([Li])c1": ("COOR", "meta"),
    "CC(C)OC(=O)c1cccc([Li])c1": ("COOR", "meta"),
    "CC(C)(C)OC(=O)c1cccc([Li])c1": ("COOR", "meta"),
    # para esters
    "CC(C)(C)OC(=O)c1ccc([Li])cc1": ("COOR", "para"),
    "COC(=O)c1ccc([Li])cc1": ("COOR", "para"),
    "CCOC(=O)c1ccc([Li])cc1": ("COOR", "para"),
    "CC(C)OC(=O)c1ccc([Li])cc1": ("COOR", "para"),
    # CN
    "N#Cc1cccc([Li])c1": ("CN", "meta"),
    "N#Cc1ccc([Li])cc1": ("CN", "para"),
    "N#Cc1ccccc1[Li]": ("CN", "ortho"),
    # NO2
    "[O-][N+](=O)c1cccc([Li])c1": ("NO2", "meta"),
    "[O-][N+](=O)c1ccc([Li])cc1": ("NO2", "para"),
    "[Li]c1ccccc1[N+](=O)[O-]": ("NO2", "ortho"),
}

def assign_sub_pos(row):
    smi = row.get("intermediate_smiles", "")
    csmi = row.get("canon_smi", "")
    name = str(row.get("intermediate", "")).lower()

    # Try SMILES_TO_SUB_POS_ALKYL first (has ester alkyl info)
    alkyl = None
    for s in [smi, csmi]:
        if s in SMILES_TO_SUB_POS_ALKYL:
            sub, pos, alkyl = SMILES_TO_SUB_POS_ALKYL[s]
            return (sub, pos, alkyl)

    # Try SMILES_TO_SUB_POS
    for s in [smi, csmi]:
        if s in SMILES_TO_SUB_POS:
            sub, pos = SMILES_TO_SUB_POS[s]
            return (sub, pos, None)

    # Infer from name
    pos = "unknown"
    if "m-" in name or "meta" in name: pos = "meta"
    elif "p-" in name or "para" in name: pos = "para"
    elif "o-" in name or "ortho" in name: pos = "ortho"

    # Infer substituent from SMILES
    sub = "H"
    if "C(=O)O" in str(smi) or "OC(=O)" in str(smi): sub = "COOR"
    elif "C(=O)" in str(smi): sub = "COR"
    elif "C#N" in str(smi) or "N#C" in str(smi): sub = "CN"
    elif "[N+](=O)" in str(smi): sub = "NO2"
    elif "Br" in str(smi): sub = "Br"
    elif "I" in str(smi): sub = "I"
    elif "OC" in str(smi): sub = "OMe"

    # Infer alkyl from name for esters
    if sub == "COOR":
        if "methyl" in name: alkyl = "Me"
        elif "ethyl" in name and "meth" not in name: alkyl = "Et"
        elif "isopropyl" in name: alkyl = "iPr"
        elif "tert-butyl" in name or "t-butyl" in name: alkyl = "tBu"

    return (sub, pos, alkyl)

for idx, row in arli.iterrows():
    sub, pos, alkyl = assign_sub_pos(row)
    arli.loc[idx, "substituent"] = sub
    arli.loc[idx, "position"] = pos
    arli.loc[idx, "ester_alkyl"] = alkyl if alkyl else ""
    arli.loc[idx, "sigma_I"] = SIGMA_I.get(sub, 0.3)
    arli.loc[idx, "sigma_R"] = SIGMA_R.get(sub, 0.0)
    arli.loc[idx, "sigma_eff"] = SIGMA_I.get(sub, 0.3) + (SIGMA_R.get(sub, 0.0) if pos == "para" else 0.0)
    arli.loc[idx, "chelation"] = CHELATION.get(sub, 0.0) if pos == "ortho" else 0.0
    # Es_alkyl: ester alkyl steric effect at ALL positions
    arli.loc[idx, "Es_alkyl"] = ES_ALKYL.get(alkyl, 0.0) if alkyl else 0.0

arli["hardness"] = (arli["dft_LUMO_eV"] - arli["dft_HOMO_eV"]) / 2

# Print training data
print(f"\n{'='*100}")
print(f"TRAINING DATA: {len(arli)} ArLi with Arrhenius + descriptors")
print(f"{'='*100}")
print(f"{'Name':<42s} {'Sub':>5s} {'Pos':>6s} {'Alk':>4s} {'σ':>5s} {'Es':>5s} {'Es_a':>5s} {'δo':>3s} {'δb':>3s} {'chel':>5s} {'μ':>5s} {'Ea':>5s} {'R²':>5s} {'log_t½':>7s}")
print("-"*115)
for _, r in arli.sort_values("log_t_half").iterrows():
    alk = r.get("ester_alkyl", "")[:4]
    print(f"  {r['intermediate'][:40]:<40s} {r['substituent']:>5s} {r['position']:>6s} {alk:>4s} "
          f"{r['sigma_hammett']:>5.2f} {r['Es_taft']:>5.2f} {r['Es_alkyl']:>5.2f} "
          f"{int(r['delta_ortho']):>3d} {int(r['delta_benzyne']):>3d} {r['chelation']:>5.1f} "
          f"{r['dft_dipole_D']:>5.1f} {r['Ea_decomp_kJ_mol']:>5.1f} {r['arrhenius_r2']:>5.3f} "
          f"{r['log_t_half']:>+7.2f}")

# ── 3. Model fitting ─────────────────────────────────────────────────

def loo_cv(X, y, model_cls=LinearRegression, **kw):
    n = len(y)
    y_pred = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool); mask[i] = False
        m = model_cls(**kw); m.fit(X[mask], y[mask])
        y_pred[i] = m.predict(X[i:i+1])[0]
    return y_pred

def eval_model(X, y, name, model_cls=LinearRegression, **kw):
    m = model_cls(**kw); m.fit(X, y)
    y_fit = m.predict(X)
    y_loo = loo_cv(X, y, model_cls, **kw)
    r2 = r2_score(y, y_fit)
    q2 = 1 - np.sum((y - y_loo)**2) / np.sum((y - y.mean())**2)
    return {"name": name, "model": m, "r2": r2, "q2": q2,
            "mae_fit": mean_absolute_error(y, y_fit),
            "mae_loo": mean_absolute_error(y, y_loo),
            "y_fit": y_fit, "y_loo": y_loo,
            "coef": m.coef_, "intercept": m.intercept_}

# Use high-quality subset for primary analysis
# Exclude heteroaryl: their stabilization mechanism is fundamentally different
# (negative hyperconjugation from ring O/S, five-membered ring constraints)
heteroaryl_subs = {"furanO", "thioS", "Br+N"}
n_before = len(arli)
arli = arli[~arli["substituent"].isin(heteroaryl_subs)].copy()
print(f"After excluding heteroaryl: {len(arli)} (removed {n_before - len(arli)})")

for quality_filter, label in [("all", "ALL DATA"), ("high", "HIGH QUALITY (R²≥0.9, Ea>5)")]:
    if quality_filter == "high":
        df = arli[arli["quality"] == "high"].copy()
    else:
        df = arli.copy()

    if len(df) < 5:
        print(f"\n[{label}] n={len(df)}, too few, skipping")
        continue

    y = df["log_t_half"].values
    n = len(y)

    models = {
        # v1 models (Es = ortho only)
        "σ+Es+δo+μ":         ["sigma_hammett", "Es_taft", "delta_ortho", "dft_dipole_D"],
        "σ+Es+δo+δb":        ["sigma_hammett", "Es_taft", "delta_ortho", "delta_benzyne"],
        # v3: Es_alkyl = ester alkyl steric for ALL positions
        "σ+Es_a+δo+μ":       ["sigma_hammett", "Es_alkyl", "delta_ortho", "dft_dipole_D"],
        "σ+Es_a+δo+δb":      ["sigma_hammett", "Es_alkyl", "delta_ortho", "delta_benzyne"],
        "σ+Es_a+δo+δb+μ":    ["sigma_hammett", "Es_alkyl", "delta_ortho", "delta_benzyne", "dft_dipole_D"],
        "σ+Es_a+chel+δb+μ":  ["sigma_hammett", "Es_alkyl", "chelation", "delta_benzyne", "dft_dipole_D"],
        "σ_I+Es_a+chel+δb+μ":["sigma_I", "Es_alkyl", "chelation", "delta_benzyne", "dft_dipole_D"],
        # Simpler
        "σ+Es_a":             ["sigma_hammett", "Es_alkyl"],
        "σ+Es_a+μ":           ["sigma_hammett", "Es_alkyl", "dft_dipole_D"],
        "σ+Es_a+δo":          ["sigma_hammett", "Es_alkyl", "delta_ortho"],
        # With Gsolv
        "σ+Es_a+δo+μ+Gsolv": ["sigma_hammett", "Es_alkyl", "delta_ortho", "dft_dipole_D", "dft_Gsolv_kJ"],
    }

    print(f"\n{'='*90}")
    print(f"MODEL COMPARISON [{label}] (n={n})")
    print(f"{'='*90}")
    print(f"  {'Model':<25s} {'p':>2s} {'R²':>6s} {'Q²_LOO':>7s} {'MAE_LOO':>8s} {'sat%':>5s}")
    print("  "+"-"*55)

    results = {}
    for mname, cols in models.items():
        X = df[cols].values
        p = X.shape[1]
        sat = (p+1)/n*100
        if sat > 80:
            continue  # skip over-saturated

        # OLS for p ≤ n/3, Ridge otherwise
        if p <= n // 3:
            res = eval_model(X, y, mname)
        else:
            best_q2, best_alpha = -999, 1.0
            for alpha in [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
                Xs = StandardScaler().fit_transform(X)
                q2_t = eval_model(Xs, y, mname, Ridge, alpha=alpha)["q2"]
                if q2_t > best_q2:
                    best_q2, best_alpha = q2_t, alpha
            Xs = StandardScaler().fit_transform(X)
            res = eval_model(Xs, y, mname, Ridge, alpha=best_alpha)
            res["alpha"] = best_alpha

        results[mname] = res
        flag = " ★★" if res["q2"] > 0.7 else " ★" if res["q2"] > 0.5 else ""
        print(f"  {mname:<25s} {p:>2d} {res['r2']:>6.3f} {res['q2']:>7.3f} {res['mae_loo']:>8.3f} {sat:>4.0f}%{flag}")

    # Best model
    best_name = max(results, key=lambda k: results[k]["q2"])
    best = results[best_name]
    best_cols = models[best_name]
    print(f"\n  ★ Best: {best_name} → Q²={best['q2']:.3f}, MAE={best['mae_loo']:.3f}")

    # Raw equation
    X_raw = df[best_cols].values
    if "alpha" in best:
        sc = StandardScaler().fit(X_raw)
        m_std = Ridge(alpha=best["alpha"]).fit(sc.transform(X_raw), y)
        coef_raw = m_std.coef_ / sc.scale_
        intercept_raw = m_std.intercept_ - np.sum(coef_raw * sc.mean_)
    else:
        coef_raw = best["coef"]
        intercept_raw = best["intercept"]

    eq = f"  log₁₀(t½) = {intercept_raw:.3f}"
    for fn, c in zip(best_cols, coef_raw):
        eq += f" {c:+.3f}·{fn}"
    print(eq)

    # Per-intermediate LOO
    y_loo = best["y_loo"]
    print(f"\n  {'Name':<42s} {'Obs':>7s} {'LOO':>7s} {'Err':>7s}")
    print("  "+"-"*66)
    sorted_idx = np.argsort(y)
    for j in sorted_idx:
        name = df.iloc[j]["intermediate"][:40]
        print(f"  {name:<42s} {y[j]:>+7.2f} {y_loo[j]:>+7.2f} {y_loo[j]-y[j]:>+7.3f}")

    # ── Save best results for high-quality set ──
    if quality_filter == "high":
        best_results = results
        best_y = y
        best_y_loo = y_loo
        best_df = df
        best_coef_raw = coef_raw
        best_intercept_raw = intercept_raw

# ── 4. Figure ─────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle(f"ArLi Half-Life Prediction — Expanded Dataset (n={len(best_df)})", fontsize=13, fontweight="bold")

# (a) Parity plot
ax = axes[0, 0]
y = best_y
y_loo = best_y_loo
ax.scatter(y, y_loo, c="royalblue", s=60, edgecolors="k", linewidths=0.5, zorder=3)
for j, (_, r) in enumerate(best_df.iterrows()):
    short = r["intermediate"].split("(")[0].strip()[:20]
    ax.annotate(short, (y[j], y_loo[j]), fontsize=5.5, ha="left", va="bottom",
                xytext=(3, 3), textcoords="offset points")
lims = [min(y.min(), y_loo.min()) - 0.5, max(y.max(), y_loo.max()) + 0.5]
ax.plot(lims, lims, "k--", alpha=0.4, lw=1)
ax.fill_between(lims, [l-0.5 for l in lims], [l+0.5 for l in lims], alpha=0.08, color="gray")
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel("Observed log₁₀(t½ / s)")
ax.set_ylabel("LOO-Predicted log₁₀(t½ / s)")
best_res = best_results[max(best_results, key=lambda k: best_results[k]["q2"])]
ax.set_title(f"(a) LOO Cross-Validation (n={len(best_df)})\n"
             f"Q² = {best_res['q2']:.3f}, MAE = {best_res['mae_loo']:.2f} dec", fontsize=10)
ax.set_aspect("equal")

# (b) Model comparison
ax = axes[0, 1]
model_names = list(best_results.keys())
q2_vals = [best_results[n]["q2"] for n in model_names]
colors_bar = ["#2196F3" if "σ_I" not in n and "σ_eff" not in n and "chel" not in n
              else "#FF9800" for n in model_names]
ax.barh(range(len(model_names)), q2_vals, color=colors_bar, edgecolor="k", linewidth=0.5)
ax.set_yticks(range(len(model_names)))
ax.set_yticklabels(model_names, fontsize=7)
ax.set_xlabel("Q²_LOO")
ax.axvline(0, color="k", lw=0.5)
ax.set_title("(b) Model Comparison", fontsize=10)
best_idx = model_names.index(max(best_results, key=lambda k: best_results[k]["q2"]))
ax.barh(best_idx, q2_vals[best_idx], color="gold", edgecolor="k", linewidth=1.5)
ax.legend([Patch(fc="#2196F3"), Patch(fc="#FF9800"), Patch(fc="gold")],
          ["v1 descriptors", "v2 descriptors", "Best"], fontsize=7, loc="lower right")

# (c) Descriptor importance
ax = axes[1, 0]
best_name_final = max(best_results, key=lambda k: best_results[k]["q2"])
feat_names = models[best_name_final]
coef_abs = np.abs(best_coef_raw)
order = np.argsort(coef_abs)[::-1]
feat_sorted = [feat_names[i] for i in order]
coef_sorted = [best_coef_raw[i] for i in order]
colors_coef = ["#4CAF50" if c > 0 else "#F44336" for c in coef_sorted]
ax.barh(range(len(feat_sorted)), coef_sorted, color=colors_coef, edgecolor="k", linewidth=0.5)
ax.set_yticks(range(len(feat_sorted)))
labels_nice = {"sigma_hammett": "σ (Hammett)", "sigma_I": "σ_I (inductive)",
               "sigma_eff": "σ_eff", "Es_taft": "Es (ortho)", "Es_alkyl": "Es_alkyl (ester)",
               "chelation": "Chelation",
               "delta_ortho": "δ_ortho", "delta_benzyne": "δ_benzyne",
               "dft_dipole_D": "μ (dipole)", "dft_HOMO_eV": "HOMO",
               "dft_Gsolv_kJ": "Gsolv", "hardness": "η (hardness)"}
ax.set_yticklabels([labels_nice.get(f, f) for f in feat_sorted], fontsize=8)
ax.axvline(0, color="k", lw=0.5)
ax.set_xlabel("Coefficient (raw)")
ax.set_title("(c) Descriptor Importance", fontsize=10)
ax.legend([Patch(fc="#4CAF50"), Patch(fc="#F44336")],
          ["Stabilizing (+t½)", "Destabilizing (−t½)"], fontsize=7, loc="lower right")

# (d) Residuals
ax = axes[1, 1]
residuals = best_y_loo - best_y
abs_res = np.abs(residuals)
order_res = np.argsort(abs_res)[::-1]
short_names = [best_df.iloc[j]["intermediate"].split("(")[0].strip()[:22] for j in order_res]
colors_res = ["#F44336" if abs_res[j] > 0.5 else "#2196F3" for j in order_res]
ax.barh(range(len(order_res)), residuals[order_res], color=colors_res, edgecolor="k", linewidth=0.5)
ax.set_yticks(range(len(order_res)))
ax.set_yticklabels(short_names, fontsize=6.5)
ax.axvline(0, color="k", lw=0.5)
ax.set_xlabel("LOO Residual (decades)")
ax.set_title("(d) Per-Intermediate LOO Errors", fontsize=10)

plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT / "arli_halflife_v3.png", dpi=200, bbox_inches="tight")
print(f"\nFigure saved: {OUT / 'arli_halflife_v3.png'}")

# Save
best_df.to_csv(OUT / "arli_training_v3.csv", index=False)
print(f"Training data saved: {OUT / 'arli_training_v3.csv'}")
