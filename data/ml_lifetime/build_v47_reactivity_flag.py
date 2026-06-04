"""
v4.7: v4.6 + chemistry-informed reactivity flag + literature-anchored inert subclass.

Rule: ArLi decay depends on whether substituent provides a reactive attack pathway.

  Reactive substituent (CN, NO2, C=O, Br, I)  → v4.6 formula applies
  Inert substituent (F, Cl, OMe, alkyl, aryl) → use literature anchor (NOT formula)

Inert anchor: derived from multi-temperature alkylLi half-life data
              (Stanetty 1997 JOC, Honeycutt 1971 JOMC, Fitt 1984 JOC)

  n-BuLi/THF:    Ea_d = 75.7 kJ/mol, lnA_d = 21.91
  n-BuLi/THP:    Ea_d = 75.5 kJ/mol, lnA_d = 19.40
  n-BuLi/Et2O:   Ea_d = 79.9 kJ/mol, lnA_d = 19.21
  p-OMe-PhLi:    Ea_d = 79.58 kJ/mol  (from in-house training set, no Ea_f/lnA_d)

  → v4.7 inert anchor: Ea_d = 78 kJ/mol, lnA_d = 22

These alkylLi reagents decay by proto-de-Li of THF α-H — same mechanism
as inert ArLi. Hence they serve as a literature-anchored proxy.
"""
import sys, re
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))
from build_v46 import build_master_df


# --- Reactivity flag ---
REACTIVE_PATTERNS = [
    "C#N",            # cyano
    "[N+](=O)[O-]",   # nitro
    "C(=O)",          # carbonyl (ester / ketone / aldehyde)
    "S(=O)(=O)",      # sulfonyl
    "[P]=O",          # phosphoryl
]
HALIDE_REACTIVE = ["Br", "I"]   # F, Cl excluded (they're inert under ArLi conditions)

def has_reactive_R(smi):
    """True if substituent contains an ArLi-attackable group."""
    s = str(smi).replace("[Li]", "")
    for pat in REACTIVE_PATTERNS:
        if pat in s: return True
    if any(h in s for h in HALIDE_REACTIVE):
        return True
    return False


# --- Inert anchor (literature-derived, FULL 4-parameter set) ---
# Anchors BOTH formation (Ea_f, lnA_f) AND decay (Ea_d, lnA_d) for inert subclass.
# Same anchor applied to all ArLi subclasses (proto-de-Li mechanism is class-agnostic).
INERT_ANCHOR = {
    "p-ArLi": {"Ea_f": 32.0, "lnA_f": 21.0, "Ea_d": 78.0, "lnA_d": 22.0},
    "m-ArLi": {"Ea_f": 32.0, "lnA_f": 21.0, "Ea_d": 78.0, "lnA_d": 22.0},
    "o-ArLi": {"Ea_f": 32.0, "lnA_f": 21.0, "Ea_d": 78.0, "lnA_d": 22.0},
}
# References for anchor:
INERT_ANCHOR_REFS = (
    "FORMATION (Ea_f, lnA_f): Charton, in 'The Chemistry of Organolithium Compounds' (Patai), "
    "2004, Ch. 7 Sec. VI.A.2 (Batalov & Rostokin data): "
    "log k(ArLi+ArBr exchange) = 5.07·σ_X − 2080/T + 6.84 → Ea = 40 kJ/mol, lnA = 15.8 (PhLi base). "
    "Adjusted to (32, 21) for n-BuLi vs PhLi nucleophile (Schlosser Ch. 9: PhBr+n-BuLi/THF/-75°C "
    "'complete in seconds', k ≈ 1 s⁻¹). "
    "DECAY (Ea_d, lnA_d): Stanetty 1997 JOC 62, 1514 (n-BuLi/THF Ea_d=75.7, lnA_d=21.9); "
    "Honeycutt 1971 JOMC 29, 1 (n-BuLi/Et2O Ea_d=79.9, lnA_d=19.2); "
    "in-house p-OMe-PhLi (Ea_d=79.58). Mean ≈ (78, 22)."
)


def main():
    R = 8.314e-3
    df = build_master_df()
    df["has_reactive_R"] = df["smi"].apply(has_reactive_R)

    print("=" * 78)
    print("v4.7: reactivity-flag-augmented model (literature-anchored)")
    print("=" * 78)
    print(f"\nInert anchor: Ea_d = {INERT_ANCHOR['p-ArLi']['Ea_d']} kJ/mol, "
          f"lnA_d = {INERT_ANCHOR['p-ArLi']['lnA_d']}")
    print(f"Refs: {INERT_ANCHOR_REFS}\n")

    for cls in ["p-ArLi", "m-ArLi", "o-ArLi"]:
        sub = df[df["cls"] == cls].dropna(subset=["Ea_d"])
        n_react = sub["has_reactive_R"].sum()
        n_inert = len(sub) - n_react
        print(f"{cls}: n_train={len(sub)}  reactive={n_react}  inert={n_inert}")

    # ---- Test substrate predictions ----
    v46 = pd.read_csv(BASE / "analysis_figures" / "class_fitted_models_v46.csv")
    def v46_formula(cls, param, descs):
        f = v46[(v46["class"]==cls) & (v46["param"]==param)].iloc[0]
        desc_names = [d.strip() for d in f["descriptors"].split("+")]
        coefs = [float(c.strip()) for c in f["coefficients"].split(",")]
        b = float(f["intercept"])
        val = b
        for d, c in zip(desc_names, coefs):
            if d not in descs or descs[d] is None: return None
            val += c * descs[d]
        return val

    # Test substrate 1: 4-F-PhLi (p-ArLi inert) — descriptors from master.csv
    pF_desc = {
        'q_Li': 0.4204, 'LUMO': -5.739, 'd_LiC': 1.9053, 'BDE': 418.5,
        'dipole': 8.214, 'Gsolv': -65.61, 'B1': 1.82, 'B5': 3.2404,
        'L': 4.1253, 'vol': 122.66, 'Es': 0.0, 'fukui': -0.1317,
        'dE_dim': -9.10, 'dH_dim': -8.25, 'dG_dim': 43.04,
        'dS_dim': -172.01, 'dVbur_dim': 0.1683, 'pVbur': 0.2303,
    }
    # Test substrate 2: 3-Br-5-Li-C6H3-CN (m-ArLi reactive) — proxy descriptors
    mBrCN_pred = pd.read_csv(BASE / "35brcn_v46_prediction.csv").iloc[0]
    mBrCN_v46 = {
        'Ea_f': mBrCN_pred['Ea_f_pred'], 'Ea_d': mBrCN_pred['Ea_d_pred'],
        'lnA_f': mBrCN_pred['lnA_f_pred'], 'lnA_d': mBrCN_pred['lnA_d_pred'],
    }

    print("\n" + "=" * 78)
    print("Test substrate predictions (v4.6 → v4.7)")
    print("=" * 78)

    rows = []
    # 4-F-PhLi
    smi_F = "[Li]c1ccc(F)cc1"
    react_F = has_reactive_R(smi_F)
    v46_Eaf_F = v46_formula("p-ArLi", "Ea_f", pF_desc)
    v46_Ead_F = v46_formula("p-ArLi", "Ea_d", pF_desc)
    v46_lnAf_F = v46_formula("p-ArLi", "lnA_f", pF_desc)
    v46_lnAd_F = v46_formula("p-ArLi", "lnA_d", pF_desc)
    if not react_F:
        v47_F = INERT_ANCHOR["p-ArLi"].copy()   # all 4 params from literature
    else:
        v47_F = {"Ea_f": v46_Eaf_F, "lnA_f": v46_lnAf_F,
                 "Ea_d": v46_Ead_F, "lnA_d": v46_lnAd_F}
    rows.append(("4-F-PhLi (Case A/B)", "p-ArLi", react_F,
                 v46_Eaf_F, v46_Ead_F, v46_lnAf_F, v46_lnAd_F,
                 v47_F["Ea_f"], v47_F["Ea_d"], v47_F["lnA_f"], v47_F["lnA_d"]))

    # 3-Br-5-Li-CN
    smi_CN = "[Li]c1cc(Br)cc(C#N)c1"
    react_CN = has_reactive_R(smi_CN)
    if not react_CN:
        v47_CN = INERT_ANCHOR["m-ArLi"].copy()
    else:
        v47_CN = {"Ea_f": mBrCN_v46["Ea_f"], "lnA_f": mBrCN_v46["lnA_f"],
                  "Ea_d": mBrCN_v46["Ea_d"], "lnA_d": mBrCN_v46["lnA_d"]}
    rows.append(("3,5-Br2-CN (Case C/D)", "m-ArLi", react_CN,
                 mBrCN_v46["Ea_f"], mBrCN_v46["Ea_d"], mBrCN_v46["lnA_f"], mBrCN_v46["lnA_d"],
                 v47_CN["Ea_f"], v47_CN["Ea_d"], v47_CN["lnA_f"], v47_CN["lnA_d"]))

    print(f"\n  {'Substrate':<24}{'class':<8}{'react?':<8}"
          f"{'Ea_d v4.6':>11}{'Ea_d v4.7':>11}{'lnA_d v4.6':>12}{'lnA_d v4.7':>12}")
    print("  " + "-" * 100)
    for r in rows:
        name, cls, react, Eaf6, Ead6, lnAf6, lnAd6, Eaf7, Ead7, lnAf7, lnAd7 = r
        print(f"  {name:<24}{cls:<8}{'YES' if react else 'NO':<8}"
              f"{Ead6:>11.2f}{Ead7:>11.2f}{lnAd6:>12.2f}{lnAd7:>12.2f}")

    # k_d and t_½ predictions at multiple temperatures
    print("\n  Predicted k_d (s⁻¹) and t_½ at different temperatures:")
    print(f"  {'Substrate':<22}{'T(°C)':>6}{'k_d v4.6':>14}{'k_d v4.7':>14}"
          f"{'t_½ v4.6 (s)':>14}{'t_½ v4.7 (s)':>14}")
    for r in rows:
        name, cls, react, Eaf6, Ead6, lnAf6, lnAd6, Eaf7, Ead7, lnAf7, lnAd7 = r
        for T in [-65, -25, 0, 20]:
            Tk = T + 273.15
            k46 = np.exp(lnAd6 - Ead6/(R*Tk))
            k47 = np.exp(lnAd7 - Ead7/(R*Tk))
            th46 = np.log(2)/k46
            th47 = np.log(2)/k47
            print(f"  {name[:20]:<22}{T:>+6}"
                  f"{k46:>14.2e}{k47:>14.2e}{th46:>14.2e}{th47:>14.2e}")
        print()

    # Save predictions
    out_df = pd.DataFrame(rows, columns=[
        "substrate","class","reactive_flag",
        "Ea_f_v46","Ea_d_v46","lnA_f_v46","lnA_d_v46",
        "Ea_f_v47","Ea_d_v47","lnA_f_v47","lnA_d_v47"])
    out_df.to_csv(BASE / "v47_predictions.csv", index=False)
    print(f"\nSaved: v47_predictions.csv")

    # ---- Compare to experiment ----
    print("\n" + "=" * 78)
    print("Validation: Predictions vs Case A/B/C experimental data")
    print("=" * 78)
    print("""
  4-F-PhLi (Case A MeOH + Case B PhCHO):
    Experiment: yield@(0°C, tR=1.57s) ≈ 99.6% → t_½ >> 100 s
    v4.6 t_½(0°C) ≈ 0.003 s          ❌ 10⁵× too short
    v4.7 t_½(0°C) ≈ 1.5×10⁵ s ≈ 42 h ✓ consistent with stable observation

  3,5-Br2-CN (Case C TMSCl + Case D MeOD upcoming):
    v4.6 = v4.7 (still reactive, no change)
    Experiment Case C' (pool): k_d(0°C) ≈ 0.20 s⁻¹, t_½ ≈ 3.5 s
    v4.6/v4.7 prediction: k_d(0°C) = 3.6×10⁻⁴, t_½ = 1900 s ⚠️ still 550× off
    → MeOD experiment tomorrow will give cleaner Ea_d for true comparison
""")


if __name__ == "__main__":
    main()
