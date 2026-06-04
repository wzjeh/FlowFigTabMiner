"""
Step 0 — Hammond hypothesis gate test for ArLi decay surrogate.

Tests if ΔG_rxn(ArLi → ArH) computed at xTB/ALPB-THF correlates with
experimental Ea_d (from global_arrhenius.csv).

Gate decision:
  R² > 0.7  : strong surrogate, proceed with Step 1 (library augmentation)
  0.5-0.7   : weak — try explicit-THF refinement before giving up
  < 0.5     : abandon surrogate path, focus on literature mining only

Substrates: 7 p-ArLi training set members + 2 test substrates (4-F-PhLi, 3-Br-5-Li-CN).

Method:
  G(ArLi) at xTB/--alpb thf, optimized monomer (already available in agg_geometries/)
  G(ArH)  at xTB/--alpb thf, ArH = SMILES with [Li] → [H], optimized fresh
  ΔG_rxn = G(ArH) − G(ArLi)   [Hartree → kJ/mol]
  Substrate-independent constant (Li⁺-solvation, etc.) dropped — only Δ relative across substrates matters.
"""
import sys, subprocess, tempfile, re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))
from compute_aggregation_descriptors import (
    opt_tblite, write_std_xyz, HARTREE_TO_KJ
)


def smi_to_3d_universal(smi):
    """Embed SMILES to 3D (works for ArH / general molecules, not requiring Li)."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return None
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, randomSeed=42) != 0:
        return None
    AllChem.UFFOptimizeMolecule(mol, maxIters=500)
    elems = [atom.GetSymbol() for atom in mol.GetAtoms()]
    conf = mol.GetConformer()
    coords = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y,
                        conf.GetAtomPosition(i).z] for i in range(mol.GetNumAtoms())])
    return elems, coords

XTB = "/Users/zhaowenyuan/miniconda3/bin/xtb"

# Training set + 2 test substrates
SUBSTRATES = [
    # (SMILES, ArLi_xyz_path, exp_Ea_d_kJ_mol, label)
    ("[Li]c1ccc(C#N)cc1",              "agg_geometries/mono_002.xyz",            28.83, "p-CN-PhLi"),
    ("[Li]c1ccc(-c2ccc(Br)cc2)cc1",    "agg_geometries/mono_003.xyz",            38.00, "4,4'-Br2-biphenyl-Li"),
    ("[Li]c1ccc(C(=O)OC)cc1",          "agg_geometries/mono_006.xyz",            45.57, "p-CO2Me-PhLi"),
    ("[Li]c1ccc(C(=O)OCC)cc1",         "agg_geometries/mono_014.xyz",            52.87, "p-CO2Et-PhLi"),
    ("[Li]c1ccc([N+](=O)[O-])cc1",     "agg_geometries/mono_017.xyz",            58.15, "p-NO2-PhLi"),
    ("[Li]c1ccc(OC)cc1",               "agg_geometries/mono_027.xyz",            79.58, "p-OMe-PhLi"),
    ("[Li]c1ccc(C(=O)OC(C)(C)C)cc1",   "agg_geometries/mono_028.xyz",            87.73, "p-CO2tBu-PhLi"),
    # Test substrates (no training Ea_d):
    ("[Li]c1ccc(F)cc1",                "agg_geometries/mono_pFArLi.xyz",         None,  "4-F-PhLi (TEST)"),
    ("[Li]c1cc(Br)cc(C#N)c1",          "agg_geometries/mono_3Br5LiCN.xyz",       None,  "3-Br-5-Li-CN (TEST)"),
]


def load_xyz(path):
    txt = Path(path).read_text().splitlines()
    n = int(txt[0])
    elems, coords = [], []
    for ln in txt[2:2+n]:
        parts = ln.split()
        elems.append(parts[0])
        coords.append([float(x) for x in parts[1:4]])
    return elems, np.array(coords)


def xtb_g_thf(elems, coords):
    """xtb --hess --alpb thf single-point + frequency. Return G in Hartree."""
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        write_std_xyz(td/"mol.xyz", elems, coords)
        r = subprocess.run([XTB, "mol.xyz", "--hess", "--alpb", "thf", "--gfn", "2"],
                           cwd=td, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        print("  xtb error:", r.stderr[-300:])
        return None
    for line in r.stdout.splitlines():
        if "TOTAL FREE ENERGY" in line:
            try: return float(line.split()[-3])
            except: pass
    return None


def make_ArH(arli_smi):
    """Replace [Li] with [H] in SMILES."""
    return arli_smi.replace("[Li]", "[H]")


def main():
    print("=" * 70)
    print("Step 0: Hammond hypothesis gate test")
    print("=" * 70)
    print(f"Computing ΔG_rxn = G(ArH) − G(ArLi) at xTB/ALPB-THF for n={len(SUBSTRATES)} substrates")
    print()

    results = []
    for smi, xyz_path, exp_Ea_d, label in SUBSTRATES:
        print(f"--- {label}  ({smi}) ---")
        # Load ArLi geometry
        elems_arli, coords_arli = load_xyz(BASE / xyz_path)
        print(f"  ArLi atoms: {len(elems_arli)}")

        # Build ArH from SMILES + opt
        ArH_smi = make_ArH(smi)
        res = smi_to_3d_universal(ArH_smi)
        if res is None:
            print("  ✗ Failed to embed ArH"); continue
        elems_arh, coords_arh = res
        coords_arh, _, _ = opt_tblite(elems_arh, coords_arh)
        print(f"  ArH atoms: {len(elems_arh)}")

        # G in THF for both
        G_arli = xtb_g_thf(elems_arli, coords_arli)
        G_arh = xtb_g_thf(elems_arh, coords_arh)
        if G_arli is None or G_arh is None:
            print("  ✗ xtb hess failed"); continue
        # ΔG (Hartree) → kJ/mol  (substrate-independent Li/H constant absorbed)
        dG_kJ = (G_arh - G_arli) * HARTREE_TO_KJ
        print(f"  G(ArLi)={G_arli:.6f}  G(ArH)={G_arh:.6f}  ΔG_rxn={dG_kJ:+.2f} kJ/mol")

        results.append({
            "smi": smi, "label": label,
            "G_ArLi_Hartree": G_arli, "G_ArH_Hartree": G_arh,
            "dG_rxn_kJ": dG_kJ,
            "Ea_d_exp": exp_Ea_d,
        })

    df = pd.DataFrame(results)
    df.to_csv(BASE / "step0_hammond_data.csv", index=False)

    # Correlation on training points (exp_Ea_d not None)
    train = df[df["Ea_d_exp"].notna()].copy()
    if len(train) < 3:
        print("\n❌ Not enough training points for correlation."); return

    x = train["dG_rxn_kJ"].to_numpy()
    y = train["Ea_d_exp"].to_numpy()
    # Linear fit Ea_d = a + b·ΔG_rxn
    b, a = np.polyfit(x, y, 1)
    pred = a + b*x
    rss = np.sum((y - pred)**2); tss = np.sum((y - y.mean())**2)
    R2 = 1 - rss/tss

    print("\n" + "=" * 70)
    print(f"Hammond correlation (n={len(train)}): Ea_d = {a:.2f} + {b:.3f} · ΔG_rxn")
    print(f"R² = {R2:.3f}")
    print("=" * 70)

    # Gate decision
    if R2 > 0.7:
        verdict = "✅ STRONG  → proceed with Step 1 (library augmentation)"
    elif R2 > 0.5:
        verdict = "⚠️  WEAK  → try explicit-THF refinement (add 1-2 THF molecules to Li)"
    else:
        verdict = "❌ FAIL  → abandon surrogate path; focus on literature mining only"
    print(f"\nGate decision (Hammond hypothesis): {verdict}")

    # Plot
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(x, y, s=120, c="steelblue", edgecolor="black", linewidth=1.5, zorder=5)
    for _, r in train.iterrows():
        ax.annotate(r["label"][:20], (r["dG_rxn_kJ"], r["Ea_d_exp"]),
                    xytext=(5, 5), textcoords="offset points", fontsize=9)
    xx = np.linspace(x.min()-5, x.max()+5, 100)
    ax.plot(xx, a + b*xx, "r--", lw=2, label=f"y = {a:.1f} + {b:.2f}·x   R²={R2:.3f}")
    # Test substrates as red crosses
    test = df[df["Ea_d_exp"].isna()]
    for _, r in test.iterrows():
        Ea_pred = a + b * r["dG_rxn_kJ"]
        ax.scatter(r["dG_rxn_kJ"], Ea_pred, s=160, c="red", marker="x",
                   linewidth=3, zorder=5, label=f"{r['label']}: ΔG={r['dG_rxn_kJ']:.1f} → Ea_pred={Ea_pred:.1f}")
    ax.set_xlabel("ΔG_rxn (ArLi → ArH) / kJ·mol⁻¹  [xTB/ALPB-THF]", fontsize=11)
    ax.set_ylabel("Experimental Ea_d / kJ·mol⁻¹", fontsize=11)
    ax.set_title(f"Step 0 — Hammond test: ΔG_rxn vs Ea_d (n={len(train)})\n{verdict}",
                 fontsize=11, fontweight="bold")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = BASE / "analysis_figures" / "step0_hammond_test.png"
    out.parent.mkdir(exist_ok=True)
    plt.savefig(out, dpi=160, bbox_inches="tight")
    print(f"\nSaved: {out}")
    print(f"Saved: step0_hammond_data.csv")


if __name__ == "__main__":
    main()
