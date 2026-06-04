"""
Virtual ArLi library for physics-informed model VETO + ranking validation.

Generates realistic aryllithiums obtainable by Br/Li exchange of (di)bromoarenes,
then computes FAST GFN2-xTB descriptors (single-points only; no hess/dimer/BDE):
  HOMO, LUMO, gap, dipole, Mulliken q_Li / q_Cipso, Gsolv(THF), d_LiC, mol_volume, %Vbur(Li)
plus tabulated Hammett σ_p / σ_m so empirical-σ models can also be applied.

Usage:
  python compute_virtual_arli.py --test 3      # quick pipeline check
  python compute_virtual_arli.py               # full library -> virtual_arli_descriptors.csv
"""
import sys, subprocess, tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

from compute_aggregation_descriptors import (
    smiles_to_3d, opt_tblite, compute_vbur, find_li_indices, write_std_xyz, XTB,
)

BASE = Path(__file__).parent
HARTREE_KJ = 2625.499

# --- Substituent fragments + Hammett σ_p, σ_m (Hansch-Leo-Taft 1991) ---
SUBS = {
    "H":   ("",            0.00,  0.00),
    "F":   ("F",           0.06,  0.34),
    "Cl":  ("Cl",          0.23,  0.37),
    "Br":  ("Br",          0.23,  0.39),
    "CN":  ("C#N",         0.66,  0.56),
    "NO2": ("[N+](=O)[O-]",0.78,  0.71),
    "CF3": ("C(F)(F)F",    0.54,  0.43),
    "CO2Me":("C(=O)OC",    0.45,  0.37),
    "CHO": ("C=O",         0.42,  0.35),
    "COMe":("C(C)=O",      0.50,  0.38),
    "Me":  ("C",          -0.17, -0.07),
    "OMe": ("OC",         -0.27,  0.12),
    "NMe2":("N(C)C",      -0.83, -0.16),
    "tBu": ("C(C)(C)C",   -0.20, -0.10),
    "OMOM":("OCOC",       -0.27,  0.12),
}

def build_library():
    """Return list of dicts: name, smi, sigma_p_sum, sigma_m_sum."""
    rows = []
    seen = set()
    def add(name, smi, sp, sm):
        m = Chem.MolFromSmiles(smi)
        if m is None: return
        cs = Chem.MolToSmiles(m)
        if cs in seen: return
        seen.add(cs)
        rows.append(dict(name=name, smi=cs, sigma_p_sum=sp, sigma_m_sum=sm))
    # mono-substituted at ortho / meta / para (Li at position 1)
    for k,(frag,sp,sm) in SUBS.items():
        if k == "H":
            add("PhLi", "[Li]c1ccccc1", 0.0, 0.0); continue
        add(f"o-{k}",  f"[Li]c1ccccc1{frag}",      0.0, sm)   # ortho carries σ_m-like
        add(f"m-{k}",  f"[Li]c1cccc({frag})c1",    0.0, sm)   # meta
        add(f"p-{k}",  f"[Li]c1ccc({frag})cc1",    sp,  0.0)  # para
    # di-substituted realistic Br/Li-exchange products (incl. residual Br)
    di = [
        ("5-Br-2-F-CN",  "[Li]c1cc(C#N)c(F)cc1", 0.06, 0.56),
        ("o-Br(resid)",  "[Li]c1ccccc1Br",       0.0,  0.39),
        ("p-Br(resid)",  "[Li]c1ccc(Br)cc1",     0.23, 0.0),
        ("4,4-diBr-bph", "[Li]c1ccc(-c2ccc(Br)cc2)cc1", 0.0, 0.0),
        ("2-Br-biph",    "[Li]c1ccccc1-c1ccccc1Br", 0.0, 0.0),
        ("3-CN-4-F",     "[Li]c1ccc(F)c(C#N)c1", 0.0, 0.56+0.34),
        ("2-CO2Me-4-Cl", "[Li]c1ccccc1C(=O)OC",  0.23, 0.37),
        ("3,5-diCN",     "[Li]c1cc(C#N)cc(C#N)c1", 0.0, 1.12),
        ("2-F-4-CF3",    "[Li]c1ccc(C(F)(F)F)cc1F", 0.54, 0.34),
        ("3-OMe-5-CN",   "[Li]c1cc(OC)cc(C#N)c1", 0.0, 0.12+0.56),
        ("2-OMe",        "[Li]c1ccccc1OC",        0.0, 0.12),
        ("3-NO2-4-Me",   "[Li]c1cc([N+](=O)[O-])c(C)cc1", 0.0, 0.71),
        ("2-naphthyl",   "[Li]c1ccc2ccccc2c1",    0.0, 0.0),
        ("1-naphthyl",   "[Li]c1cccc2ccccc12",    0.0, 0.0),
        ("2-Me-4-CN",    "[Li]c1ccc(C#N)cc1C",    0.66, 0.0),
    ]
    for name,smi,sp,sm in di: add(name, smi, sp, sm)
    return rows


def xtb_sp(elems, coords, li_idx, alpb=None):
    """GFN2-xTB single point. Returns E(Hartree), HOMO, LUMO, dipole, q_Li, q_Cipso."""
    with tempfile.TemporaryDirectory() as td:
        td = Path(td); write_std_xyz(td/"mol.xyz", elems, coords)
        cmd = [XTB, "mol.xyz", "--gfn", "2", "--sp"]
        if alpb: cmd += ["--alpb", alpb]
        r = subprocess.run(cmd, cwd=td, capture_output=True, text=True, timeout=600)
        if r.returncode != 0: return None
        E = HOMO = LUMO = dipole = None
        lines = r.stdout.splitlines()
        for i, ln in enumerate(lines):
            if "TOTAL ENERGY" in ln:
                try: E = float(ln.split()[-3])
                except: pass
            if "(HOMO)" in ln:
                t = ln.split()
                try: HOMO = float(t[t.index("(HOMO)")-1])
                except: pass
            if "(LUMO)" in ln:
                t = ln.split()
                try: LUMO = float(t[t.index("(LUMO)")-1])
                except: pass
            if "molecular dipole" in ln:
                for j in range(i, min(i+4, len(lines))):
                    if "full:" in lines[j]:
                        try: dipole = float(lines[j].split()[-1])
                        except: pass
        # Mulliken charges from 'charges' file
        q_Li = q_C = None
        cf = td/"charges"
        if cf.exists():
            q = [float(x) for x in cf.read_text().split()]
            if li_idx < len(q): q_Li = q[li_idx]
            # ipso C = nearest C to Li
            ci = nearest_c(elems, coords, li_idx)
            if ci is not None and ci < len(q): q_C = q[ci]
        return dict(E=E, HOMO=HOMO, LUMO=LUMO, dipole=dipole, q_Li=q_Li, q_Cipso=q_C)


def nearest_c(elems, coords, li_idx):
    best, bd = None, 1e9
    for i,e in enumerate(elems):
        if e == "C":
            d = np.linalg.norm(coords[i]-coords[li_idx])
            if d < bd: bd, best = d, i
    return best


def mol_volume(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    m = Chem.AddHs(m); AllChem.EmbedMolecule(m, randomSeed=42); AllChem.UFFOptimizeMolecule(m)
    try: return AllChem.ComputeMolVolume(m)
    except: return None


def compute_one(name, smi):
    g = smiles_to_3d(smi)
    if g is None: return None
    elems, coords, li_idx = g
    opt, _, _ = opt_tblite(elems, coords)
    vac = xtb_sp(elems, opt, li_idx, alpb=None)
    thf = xtb_sp(elems, opt, li_idx, alpb="thf")
    if vac is None or thf is None: return None
    ci = nearest_c(elems, opt, li_idx)
    d_LiC = float(np.linalg.norm(opt[li_idx]-opt[ci])) if ci is not None else None
    return dict(
        HOMO_eV=vac["HOMO"], LUMO_eV=vac["LUMO"],
        gap_eV=(vac["LUMO"]-vac["HOMO"]) if vac["HOMO"] and vac["LUMO"] else None,
        dipole_D=vac["dipole"], q_Li=vac["q_Li"], q_Cipso=vac["q_Cipso"],
        Gsolv_kJ=(thf["E"]-vac["E"])*HARTREE_KJ if (thf["E"] and vac["E"]) else None,
        d_LiC_A=d_LiC, vol_A3=mol_volume(smi),
        Vbur=compute_vbur(elems, opt, li_idx),
    )


def main():
    lib = build_library()
    n_test = None
    if "--test" in sys.argv:
        n_test = int(sys.argv[sys.argv.index("--test")+1])
        lib = lib[:n_test]
    print(f"Library: {len(lib)} ArLi" + (f" (TEST mode, first {n_test})" if n_test else ""))
    out = []
    for i, r in enumerate(lib):
        print(f"[{i+1}/{len(lib)}] {r['name']:14s} {r['smi']}", flush=True)
        d = compute_one(r["name"], r["smi"])
        if d is None:
            print("    FAILED"); continue
        out.append({**r, **d})
        print(f"    HOMO={d['HOMO_eV']}  Gsolv={d['Gsolv_kJ']:.1f}  d_LiC={d['d_LiC_A']:.3f}  "
              f"q_C={d['q_Cipso']}  Vbur={d['Vbur']}")
    df = pd.DataFrame(out)
    suffix = f"_test{n_test}" if n_test else ""
    p = BASE / f"virtual_arli_descriptors{suffix}.csv"
    df.to_csv(p, index=False)
    print(f"\nSaved {len(df)} rows -> {p}")


if __name__ == "__main__":
    main()
