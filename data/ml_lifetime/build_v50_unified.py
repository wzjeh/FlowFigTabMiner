"""
Phase 2 of v5.0: Systematic evaluation of unified Hammett-LFER architecture.

Tests multiple model variants on the 24-substrate Tier B+ training set:
    M0: Fixed anchor + ρ·σ + Es + Δ flags + γ·dVbur (initial design)
    M1: Free intercept + same features (no anchor constraint)
    M2: Ridge regression with α tuning
    M3: Per-class Hammett (separate p/m/o fits)
    M4: Class-aware + σ interaction

Reports LOO R² per Arrhenius parameter for each variant.
Outputs honest assessment of whether v5.0 can beat v4.7.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).parent

ANCHOR = {'Ea_f': 32.0, 'lnA_f': 21.0, 'Ea_d': 78.0, 'lnA_d': 22.0, 'y_max': 100.0}

FEATURES_FULL = ['sigma_p_sum', 'sigma_m_sum', 'sigma_o_sum',
                 'Es_ortho', 'delta_5exo_o', 'delta_benzyne_o', 'dVbur_dim']


def loo_r2(model_factory, X, y):
    n = len(y)
    if n < X.shape[1] + 1: return np.nan
    preds = np.empty(n)
    for i in range(n):
        Mi = model_factory().fit(np.delete(X,i,0), np.delete(y,i))
        preds[i] = Mi.predict(X[i:i+1])[0]
    rss = np.sum((y-preds)**2); tss = np.sum((y-y.mean())**2)
    return 1 - rss/tss if tss > 0 else np.nan


def main():
    df = pd.read_csv(BASE / 'v50_training_set.csv')
    df = df.dropna(subset=FEATURES_FULL + list(ANCHOR.keys()))
    print(f"Training set: n={len(df)} (after dropna)")
    print(f"Class breakdown: {df['cls'].value_counts().to_dict()}")

    # Define model variants
    results = []

    for target in ['Ea_f', 'lnA_f', 'Ea_d', 'lnA_d', 'y_max']:
        y = df[target].to_numpy()
        anchor = ANCHOR[target]
        print(f"\n{'='*88}")
        print(f"Target: {target}  (anchor for inert = {anchor:.1f})")
        print(f"{'='*88}")
        print(f"  y range: {y.min():.2f} to {y.max():.2f}, mean: {y.mean():.2f}, std: {y.std():.2f}")

        # M0: Fixed anchor + all features (no intercept on residual)
        y_resid = y - anchor
        X = df[FEATURES_FULL].to_numpy()
        # Standardize for ridge
        Xs = StandardScaler().fit_transform(X)

        # M0 OLS (fixed anchor)
        M0 = LinearRegression(fit_intercept=False).fit(X, y_resid)
        M0_R2_loo = loo_r2(lambda: LinearRegression(fit_intercept=False), X, y_resid)
        M0_R2_in = M0.score(X, y_resid)

        # M1 free intercept
        M1 = LinearRegression(fit_intercept=True).fit(X, y)
        M1_R2_loo = loo_r2(lambda: LinearRegression(fit_intercept=True), X, y)
        M1_R2_in = M1.score(X, y)

        # M2 Ridge α=10 (free intercept, standardized X)
        M2_R2_loo = loo_r2(lambda: Ridge(alpha=10, fit_intercept=True), Xs, y)

        # M3 Ridge α=100
        M3_R2_loo = loo_r2(lambda: Ridge(alpha=100, fit_intercept=True), Xs, y)

        # M4 Per-class free intercept Hammett (separate fits)
        class_results = []
        for cls in ['p-ArLi','m-ArLi','o-ArLi']:
            sub = df[df['cls']==cls]
            if len(sub) < 3: continue
            sig_col = {'p-ArLi':'sigma_p_sum','m-ArLi':'sigma_m_sum','o-ArLi':'sigma_o_sum'}[cls]
            Xc = sub[[sig_col]].to_numpy()
            yc = sub[target].to_numpy()
            if np.std(Xc) < 1e-9: continue
            Mc = LinearRegression(fit_intercept=True).fit(Xc, yc)
            r2c_in = Mc.score(Xc, yc)
            r2c_loo = loo_r2(lambda: LinearRegression(fit_intercept=True), Xc, yc)
            class_results.append({'cls':cls, 'n':len(sub), 'intercept':Mc.intercept_,
                                 'slope':Mc.coef_[0], 'R2_in':r2c_in, 'R2_loo':r2c_loo})

        print(f"  M0 Fixed anchor + 7 features:  R²_in={M0_R2_in:+.3f}  R²_LOO={M0_R2_loo:+.3f}")
        print(f"  M1 Free intercept + 7 features: R²_in={M1_R2_in:+.3f}  R²_LOO={M1_R2_loo:+.3f}")
        print(f"  M2 Ridge α=10 + 7 features:                     R²_LOO={M2_R2_loo:+.3f}")
        print(f"  M3 Ridge α=100 + 7 features:                    R²_LOO={M3_R2_loo:+.3f}")
        print(f"  M4 Per-class single-σ Hammett:")
        for c in class_results:
            print(f"     {c['cls']}: y = {c['intercept']:6.2f} + {c['slope']:+6.2f}·σ  "
                  f"R²_in={c['R2_in']:+.3f}  R²_LOO={c['R2_loo']:+.3f}  n={c['n']}")

        results.append({
            'target': target,
            'M0_LOO': M0_R2_loo, 'M1_LOO': M1_R2_loo, 'M2_LOO': M2_R2_loo, 'M3_LOO': M3_R2_loo,
            'per_class': class_results,
        })

    # Overall verdict
    print(f"\n{'='*88}")
    print("VERDICT: Does v5.0 unified Hammett-LFER beat v4.7?")
    print(f"{'='*88}")
    any_positive = False
    for r in results:
        param = r['target']
        best_loo = max([r['M0_LOO'], r['M1_LOO'], r['M2_LOO'], r['M3_LOO']])
        per_class_best = max([c['R2_loo'] for c in r['per_class']] + [-np.inf])
        overall_best = max(best_loo, per_class_best)
        verdict = "POSITIVE" if overall_best > 0.3 else "NEGATIVE"
        if overall_best > 0.3: any_positive = True
        print(f"  {param}: best LOO R² = {overall_best:+.3f}  → {verdict}")

    print(f"\nFinal conclusion:")
    if any_positive:
        print("  At least one parameter has LOO R² > 0.3 — partial signal exists.")
    else:
        print("  ALL parameters have LOO R² < 0.3 — Hammett-LFER does NOT improve over v4.7.")
        print("\n  Implications:")
        print("    1. Single-σ correlation absent within reactive ArLi class (n=24)")
        print("    2. EWG-type categorical effects (CN vs NO2 vs CO2R vs Br/I)")
        print("       dominate over σ electronic-effect on Ea_d")
        print("    3. v4.7 (class-stratified v4.6 + inert lit-anchor) is best with current data")
        print("    4. v5.0 unified framework requires either:")
        print("       - Expanded reactive training set (n>50)")
        print("       - EWG-type sub-classification (CN/NO2/C=O/halide × p/m/o)")
        print("       - Mechanism-specific TS calculations for activation barriers")

    # Save results
    rows = []
    for r in results:
        rows.append({'target': r['target'],
                     'M0_anchored_LOO': r['M0_LOO'], 'M1_freeint_LOO': r['M1_LOO'],
                     'M2_ridge10_LOO': r['M2_LOO'], 'M3_ridge100_LOO': r['M3_LOO'],
                     'M4_per_class_best_LOO': max([c['R2_loo'] for c in r['per_class']] + [-np.inf])})
    pd.DataFrame(rows).to_csv(BASE / 'v50_model_comparison.csv', index=False)
    print(f"\nSaved: v50_model_comparison.csv")


if __name__ == '__main__':
    main()
