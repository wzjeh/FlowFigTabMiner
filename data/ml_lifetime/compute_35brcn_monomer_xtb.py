"""
Plan A: GFN2-xTB-level monomer descriptors for 3-Br-5-Li-C6H3-CN
        replacing the m-Br-PhLi + m-CN-PhLi proxy averaging.

Computes:
  - LUMO (eV)              : from xtb -P 1 --alpb thf single-point
  - d_LiC (Å)              : from optimized geometry (Li to ortho-aromatic C)
  - vol (Å³)               : Connolly / vdW volume from morfeus or rdkit
  - Gsolv (kJ/mol)         : E_alpb_thf − E_vacuum (single points)
  - dipole (D)             : from xtb output (vacuum)
  - BDE (kJ/mol, optional) : homolytic Li-C via --uhf single points
                             (ArLi → Ar• + Li•)

Then plug into v4.6 m-ArLi formulas, compare with proxy-based prediction.
"""
import sys, re, subprocess, tempfile
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))
from compute_aggregation_descriptors import (
    smiles_to_3d, opt_tblite, write_std_xyz, find_li_indices, HARTREE_TO_KJ
)

XTB = "/Users/zhaowenyuan/miniconda3/bin/xtb"

SMI = "[Li]c1cc(Br)cc(C#N)c1"
NAME = "3Br5LiCN"


# ---------- xtb wrappers ----------

def xtb_singlepoint(elems, coords, alpb=None, uhf=0):
    """xtb single-point; return dict {E, LUMO_eV, HOMO_eV, dipole_D}."""
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        write_std_xyz(td / "mol.xyz", elems, coords)
        cmd = [XTB, "mol.xyz", "--gfn", "2", "--sp"]
        if alpb: cmd += ["--alpb", alpb]
        if uhf:  cmd += ["--uhf", str(uhf)]
        r = subprocess.run(cmd, cwd=td, capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        print("xtb error:", r.stderr[-500:])
        return None
    out = r.stdout
    E = None; HOMO = LUMO = None; dipole = None
    for line in out.splitlines():
        if "TOTAL ENERGY" in line:
            try: E = float(line.split()[-3])
            except: pass
        if "(HOMO)" in line:
            # Format: ".. -5.7390 (HOMO)"
            toks = line.split()
            try:
                idx = toks.index("(HOMO)")
                HOMO = float(toks[idx-1])   # in eV
            except: pass
        if "(LUMO)" in line:
            toks = line.split()
            try:
                idx = toks.index("(LUMO)")
                LUMO = float(toks[idx-1])
            except: pass
        # dipole — appears under "molecular dipole:" block; |total| line
        if "molecular dipole" in line:
            pass  # placeholder
    # Parse dipole more robustly
    m = re.search(r"full:.*?\n.*?\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)$", out, re.MULTILINE)
    if m:
        try: dipole = float(m.group(4))  # Debye (last field)
        except: dipole = None
    return {"E": E, "HOMO_eV": HOMO, "LUMO_eV": LUMO, "dipole_D": dipole}


def vdw_volume(elems, coords):
    """Compute van der Waals volume (Å³) using rdkit's compute_vdw or morfeus."""
    try:
        from morfeus import BuriedVolume
    except Exception:
        pass
    # Simple Bondi vdW radii sum and Connolly-like volume via rdkit:
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        # build mol from SMILES + 3D coords
        mol = Chem.MolFromSmiles(SMI)
        mol = Chem.AddHs(mol)
        # embed with coords
        conf = Chem.Conformer(mol.GetNumAtoms())
        # Match atoms by element ordering (approximate — assume same ordering as smiles_to_3d)
        if mol.GetNumAtoms() == len(elems):
            for i, (e, xyz) in enumerate(zip(elems, coords)):
                conf.SetAtomPosition(i, tuple(map(float, xyz)))
            mol.AddConformer(conf)
            try:
                vol = AllChem.ComputeMolVolume(mol)
                return vol
            except Exception as e:
                print("rdkit ComputeMolVolume failed:", e)
    except Exception as e:
        print("rdkit fallback:", e)
    # Last resort: Bondi sum (rough)
    BONDI = {"H":1.20,"C":1.70,"N":1.55,"O":1.52,"F":1.47,"Si":2.10,
             "P":1.80,"S":1.80,"Cl":1.75,"Br":1.85,"I":1.98,"Li":1.82}
    v = sum(4/3*np.pi*BONDI.get(e, 1.5)**3 for e in elems)
    return v * 0.7   # rough overlap correction


def li_c_bond_length(elems, coords):
    li_idx = elems.index("Li")
    li_pos = coords[li_idx]
    best = None
    for i, e in enumerate(elems):
        if e != "C": continue
        d = np.linalg.norm(coords[i] - li_pos)
        if best is None or d < best: best = d
    return best


# ---------- main ----------

def main():
    print(f"=== Plan A: xTB-level monomer descriptors for {SMI} ===\n")

    # 1) Generate / load optimized monomer geometry
    geom_file = BASE / "agg_geometries" / f"mono_{NAME}.xyz"
    if geom_file.exists():
        print(f"Loading existing geometry: {geom_file.name}")
        lines = geom_file.read_text().splitlines()
        n_atoms = int(lines[0])
        elems, coords = [], []
        for ln in lines[2:2+n_atoms]:
            parts = ln.split()
            elems.append(parts[0])
            coords.append([float(x) for x in parts[1:4]])
        elems = elems
        coords = np.array(coords)
    else:
        print("Computing fresh monomer geometry...")
        elems, coords, li_idx = smiles_to_3d(SMI)
        coords, _, _ = opt_tblite(elems, coords)

    print(f"  Atoms: {len(elems)}   Li at index {elems.index('Li')}")

    # 2) Single point in THF (alpb)
    print("\n[1/3] xtb single-point --alpb thf ...")
    res_thf = xtb_singlepoint(elems, coords, alpb="thf")
    print(f"   E_thf  = {res_thf['E']:.6f}  Hartree")
    print(f"   LUMO   = {res_thf['LUMO_eV']:.3f}  eV")
    print(f"   HOMO   = {res_thf['HOMO_eV']:.3f}  eV")
    print(f"   dipole = {res_thf['dipole_D']}")

    # 3) Single point in vacuum
    print("\n[2/3] xtb single-point (vacuum) ...")
    res_vac = xtb_singlepoint(elems, coords, alpb=None)
    print(f"   E_vac  = {res_vac['E']:.6f}  Hartree")

    # Gsolv (kJ/mol)
    Gsolv_kJ = (res_thf['E'] - res_vac['E']) * HARTREE_TO_KJ
    print(f"\n   Gsolv = {Gsolv_kJ:.2f} kJ/mol   (= ΔE_thf-vac, ~free energy of solvation)")

    # 4) Geometric descriptors
    d_LiC = li_c_bond_length(elems, coords)
    vol_A3 = vdw_volume(elems, coords)
    print(f"\n[3/3] Geometric:")
    print(f"   d_LiC = {d_LiC:.4f} Å")
    print(f"   vol   = {vol_A3:.2f} Å³")

    # ---------- compare to proxy and v4.6 prediction ----------
    print("\n" + "="*70)
    print("Compare: xTB-actual vs proxy (m-Br-PhLi + m-CN-PhLi avg)")
    print("="*70)

    proxy = {"BDE": 419.5, "vol": 136.34, "d_LiC": 1.9105,
             "LUMO": -6.2275, "Gsolv": -71.69}
    xtb = {"BDE": None,    # not computed yet
           "vol": vol_A3, "d_LiC": d_LiC,
           "LUMO": res_thf['LUMO_eV'], "Gsolv": Gsolv_kJ}

    print(f"\n  {'descriptor':<12} {'proxy avg':>12} {'xTB actual':>12} {'Δ (xtb-proxy)':>16}")
    for k in ["LUMO", "d_LiC", "vol", "Gsolv"]:
        d = xtb[k] - proxy[k]
        print(f"  {k:<12} {proxy[k]:>12.3f} {xtb[k]:>12.3f} {d:>+16.3f}")

    # Aggregation descriptors (unchanged, from previous xtb run)
    dVbur_dim = 0.3059   # from compute_35brcn_aggregation.py
    dH_dim    = -100.94
    dG_dim    = -38.75
    BDE       = proxy["BDE"]  # use proxy for BDE (we haven't computed yet)

    # ---------- v4.6 m-ArLi predictions ----------
    def predict(d_set):
        Ea_f = 0.4821*d_set["BDE"] + 0.0853*d_set["vol"] + 0.1776*dH_dim - 165.5877
        Ea_d = 4.1346*d_set["Gsolv"] + 1.8525*d_set["vol"] - 429.595*dVbur_dim + 204.2715
        lnA_f = -428.999*d_set["d_LiC"] + 0.0494*d_set["vol"] + 0.0601*dG_dim + 837.3001
        lnA_d = -4.7318*d_set["LUMO"] - 1.0510*d_set["Gsolv"] + 8.9372*dVbur_dim - 102.7021
        return Ea_f, lnA_f, Ea_d, lnA_d

    proxy_pred = predict(proxy)
    xtb_full = dict(xtb); xtb_full["BDE"] = proxy["BDE"]   # plug proxy BDE
    xtb_pred = predict(xtb_full)

    print("\n" + "="*70)
    print("v4.6 m-ArLi predictions")
    print("="*70)
    print(f"  {'param':<8} {'proxy-based':>14} {'xTB-based':>14} {'Δ':>10}")
    for name, p, x in zip(["Ea_f","lnA_f","Ea_d","lnA_d"], proxy_pred, xtb_pred):
        print(f"  {name:<8} {p:>14.2f} {x:>14.2f} {x-p:>+10.2f}")

    # rate constants
    R = 8.314e-3
    print(f"\n  k_d at 0°C (s⁻¹):")
    print(f"    proxy-based: {np.exp(proxy_pred[3] - proxy_pred[2]/(R*273.15)):.2e}")
    print(f"    xTB-based:   {np.exp(xtb_pred[3]   - xtb_pred[2]/(R*273.15)):.2e}")
    print(f"  Experiment (Case C'): ~ 2.0e-1 s⁻¹")

    # ---------- save ----------
    out = pd.DataFrame([{
        "smi": SMI, "name": NAME,
        # xtb-level monomer
        "LUMO_eV_xtb":  xtb["LUMO"],
        "d_LiC_A_xtb":  xtb["d_LiC"],
        "vol_A3_xtb":   xtb["vol"],
        "Gsolv_kJ_xtb": xtb["Gsolv"],
        "dipole_D_xtb": res_thf['dipole_D'],
        # aggregation (from prior xtb run)
        "dH_dim_kJ": dH_dim, "dG_dim_kJ": dG_dim, "dVbur_dim": dVbur_dim,
        # v4.6 predictions (proxy + xtb)
        "Ea_f_proxy": proxy_pred[0], "lnA_f_proxy": proxy_pred[1],
        "Ea_d_proxy": proxy_pred[2], "lnA_d_proxy": proxy_pred[3],
        "Ea_f_xtb":   xtb_pred[0],   "lnA_f_xtb":   xtb_pred[1],
        "Ea_d_xtb":   xtb_pred[2],   "lnA_d_xtb":   xtb_pred[3],
    }])
    out_csv = BASE / "35brcn_v46_xtb_prediction.csv"
    out.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv.name}")


if __name__ == "__main__":
    main()
