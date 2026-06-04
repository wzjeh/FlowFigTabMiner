"""
Compute GFN2-xTB level descriptors for 5-Li-2-F-3-CN-phenyllithium
(i.e. the ArLi from 5-bromo-2-fluorobenzonitrile + n-BuLi Li/Br exchange),
plus v4.6 m-ArLi class predictions.

Substrate (in flask): 5-bromo-2-fluorobenzonitrile
ArLi (after exchange): 5-Li-2-F-3-CN-C6H3   →  m-CN-ArLi with p-F (relative to Li)

SMILES of ArLi:  [Li]c1cc(C#N)c(F)cc1

Pipeline:
  1) monomer 3D + tblite opt
  2) monomer xtb --hess --alpb thf  → H_mono, G_mono
  3) monomer xtb --sp --alpb thf  → LUMO, dipole, E_thf
  4) monomer xtb --sp (vacuum)   → E_vac  → Gsolv = E_thf - E_vac (kJ/mol)
  5) BDE: ArLi → Ar• + Li•  via xtb --uhf 1 fragment calcs
  6) d_LiC: from optimized geometry; vol: from RDKit ComputeMolVolume
  7) dimer construct + tblite opt
  8) dimer xtb --hess --alpb thf → ΔH_dim, ΔG_dim, ΔS_dim
  9) Vbur(monomer/dimer Li centers) → ΔVbur_dim
 10) Apply v4.6 m-ArLi class formulas → Ea_f, lnA_f, Ea_d, lnA_d
 11) Predict yield over 25 experimental (T, tR) points and compare.

Output:
  - agg_geometries/mono_fbrcn.xyz, dim_fbrcn.xyz
  - fbrcn_descriptors.csv
  - fbrcn_v46_prediction.csv
"""
import sys, re, time, subprocess, tempfile
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))
from compute_aggregation_descriptors import (
    smiles_to_3d, opt_tblite, xtb_hess_alpb, write_std_xyz,
    construct_dimer, find_li_indices, compute_vbur,
    HARTREE_TO_KJ, T_REF, GEOM_DIR
)

XTB = "/Users/zhaowenyuan/miniconda3/bin/xtb"

SMI  = "[Li]c1cc(C#N)c(F)cc1"
NAME = "fbrcn"   # 5-Li-2-F-3-CN-PhLi (from 5-Br-2-F-benzonitrile)
DISPLAY = "5-Li-2-F-3-CN-C6H3 (m-CN-p-F-ArLi from 5-Br-2-F-CN)"

# v4.6 m-ArLi class formulas (analysis_figures/class_fitted_models_v46.csv)
V46 = dict(
    Ea_f  = lambda BDE, vol, dH: 0.4821*BDE + 0.0853*vol + 0.1776*dH - 165.5877,
    Ea_d  = lambda Gsolv, vol, dVbur: 4.1346*Gsolv + 1.8525*vol - 429.5950*dVbur + 204.2715,
    lnA_f = lambda d_LiC, vol, dG: -428.9989*d_LiC + 0.0494*vol + 0.0601*dG + 837.3001,
    lnA_d = lambda LUMO, Gsolv, dVbur: -4.7318*LUMO - 1.0510*Gsolv + 8.9372*dVbur - 102.7021,
)


def xtb_sp(elems, coords, alpb=None, uhf=0, chrg=0):
    """xtb single-point. Returns dict {E, HOMO, LUMO, dipole}."""
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        write_std_xyz(td / "mol.xyz", elems, coords)
        cmd = [XTB, "mol.xyz", "--gfn", "2", "--sp",
               "--chrg", str(chrg), "--uhf", str(uhf)]
        if alpb: cmd += ["--alpb", alpb]
        r = subprocess.run(cmd, cwd=td, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        print("xtb error:", r.stderr[-400:])
        return None
    out = r.stdout
    E = HOMO = LUMO = dipole = None
    for line in out.splitlines():
        if "TOTAL ENERGY" in line:
            try: E = float(line.split()[-3])
            except: pass
        if "(HOMO)" in line:
            try:
                toks = line.split()
                idx = toks.index("(HOMO)")
                HOMO = float(toks[idx-1])
            except: pass
        if "(LUMO)" in line:
            try:
                toks = line.split()
                idx = toks.index("(LUMO)")
                LUMO = float(toks[idx-1])
            except: pass
    m = re.search(r"full:.*?\n.*?\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)$",
                  out, re.MULTILINE)
    if m:
        try: dipole = float(m.group(4))
        except: pass
    return {"E": E, "HOMO_eV": HOMO, "LUMO_eV": LUMO, "dipole_D": dipole}


def li_c_bond_length(elems, coords):
    li_idx = elems.index("Li")
    li_pos = coords[li_idx]
    best = None
    for i, e in enumerate(elems):
        if e != "C": continue
        d = np.linalg.norm(coords[i] - li_pos)
        if best is None or d < best: best = d
    return float(best)


def rdkit_mol_volume(smi, elems, coords):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    mol = Chem.MolFromSmiles(smi)
    mol = Chem.AddHs(mol)
    if mol.GetNumAtoms() != len(elems):
        return None
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i, xyz in enumerate(coords):
        conf.SetAtomPosition(i, tuple(map(float, xyz)))
    mol.AddConformer(conf)
    try:
        return float(AllChem.ComputeMolVolume(mol))
    except Exception as e:
        print("rdkit ComputeMolVolume failed:", e)
        return None


def compute_bde(elems, coords):
    """ArLi → Ar• + Li•  (homolytic, both uhf=1)."""
    e_arLi = xtb_sp(elems, coords, alpb=None, uhf=0, chrg=0)
    if e_arLi is None: return None
    e_li = xtb_sp(["Li"], np.array([[0.0, 0.0, 0.0]]), alpb=None, uhf=1, chrg=0)
    if e_li is None: return None
    li_idx = elems.index("Li")
    frag_elems = [s for i, s in enumerate(elems) if i != li_idx]
    frag_coords = np.array([p for i, p in enumerate(coords) if i != li_idx])
    e_frag = xtb_sp(frag_elems, frag_coords, alpb=None, uhf=1, chrg=0)
    if e_frag is None: return None
    bde = (e_frag["E"] + e_li["E"] - e_arLi["E"]) * HARTREE_TO_KJ
    return float(bde)


def main():
    print(f"=== Compute descriptors for {DISPLAY} ===")
    print(f"SMILES: {SMI}\n")
    t0 = time.time()

    # ---------- monomer ----------
    print("[1/9] monomer 3D + tblite opt ...")
    res = smiles_to_3d(SMI)
    if res is None: raise RuntimeError("smiles_to_3d failed")
    elems_m, coords_m, li_idx = res
    print(f"  atoms: {len(elems_m)}  Li_idx={li_idx}")
    opt_m, _, _ = opt_tblite(elems_m, coords_m)
    GEOM_DIR.mkdir(exist_ok=True, parents=True)
    write_std_xyz(GEOM_DIR / f"mono_{NAME}.xyz", elems_m, opt_m, comment=SMI)

    print("[2/9] monomer xtb --hess --alpb thf ...")
    mono_hess = xtb_hess_alpb(elems_m, opt_m)
    if mono_hess is None: raise RuntimeError("monomer hess failed")
    H_m, G_m = mono_hess["H"], mono_hess["G"]
    print(f"  E={mono_hess['E']:.6f}  H={H_m:.6f}  G={G_m:.6f} Hartree")

    print("[3/9] monomer xtb --sp --alpb thf  (LUMO, dipole, E_thf) ...")
    sp_thf = xtb_sp(elems_m, opt_m, alpb="thf")
    LUMO = sp_thf["LUMO_eV"]
    dipole = sp_thf["dipole_D"]
    E_thf = sp_thf["E"]
    print(f"  LUMO={LUMO:.3f} eV  dipole={dipole}  E_thf={E_thf:.6f}")

    print("[4/9] monomer xtb --sp (vacuum) for Gsolv ...")
    sp_vac = xtb_sp(elems_m, opt_m, alpb=None)
    E_vac = sp_vac["E"]
    Gsolv = (E_thf - E_vac) * HARTREE_TO_KJ
    print(f"  E_vac={E_vac:.6f}  Gsolv={Gsolv:+.2f} kJ/mol")

    print("[5/9] BDE: ArLi -> Ar. + Li.  (uhf=1 fragments) ...")
    BDE = compute_bde(elems_m, opt_m)
    print(f"  BDE_LiC = {BDE:+.2f} kJ/mol" if BDE else "  BDE: FAILED")

    print("[6/9] geometry: d_LiC, vol ...")
    d_LiC = li_c_bond_length(elems_m, opt_m)
    vol = rdkit_mol_volume(SMI, elems_m, opt_m)
    print(f"  d_LiC = {d_LiC:.4f} Å    vol = {vol:.2f} Å³")

    # ---------- dimer ----------
    print("\n[7/9] dimer construct + tblite opt ...")
    d_elems, d_init = construct_dimer(elems_m, opt_m, li_idx)
    opt_d, _, _ = opt_tblite(d_elems, d_init, maxiter=500)
    write_std_xyz(GEOM_DIR / f"dim_{NAME}.xyz", d_elems, opt_d, comment=f"dimer of {SMI}")
    print(f"  dimer atoms: {len(d_elems)}")

    print("[8/9] dimer xtb --hess --alpb thf ...")
    dim_hess = xtb_hess_alpb(d_elems, opt_d)
    if dim_hess is None: raise RuntimeError("dimer hess failed")
    print(f"  E={dim_hess['E']:.6f}  H={dim_hess['H']:.6f}  G={dim_hess['G']:.6f}")

    dE_dim = (dim_hess['E'] - 2*mono_hess['E']) * HARTREE_TO_KJ
    dH_dim = (dim_hess['H'] - 2*H_m) * HARTREE_TO_KJ
    dG_dim = (dim_hess['G'] - 2*G_m) * HARTREE_TO_KJ
    dS_dim = (dH_dim - dG_dim) / T_REF * 1000.0
    print(f"  ΔE_dim={dE_dim:+.2f}  ΔH_dim={dH_dim:+.2f}  ΔG_dim={dG_dim:+.2f} kJ/mol  ΔS_dim={dS_dim:+.2f} J/(mol·K)")

    print("[9/9] Vbur(monomer/dimer Li) ...")
    vbur_m = compute_vbur(elems_m, opt_m, li_idx)
    li_d = find_li_indices(d_elems)
    vbur_d_vals = [v for v in (compute_vbur(d_elems, opt_d, i) for i in li_d) if v is not None]
    vbur_d = float(np.mean(vbur_d_vals)) if vbur_d_vals else None
    dVbur = (vbur_d - vbur_m) if (vbur_m is not None and vbur_d is not None) else None
    print(f"  Vbur_mono={vbur_m}   Vbur_dim={vbur_d}   ΔVbur={dVbur:+.4f}")

    elapsed = time.time() - t0
    print(f"\n=== Wall time: {elapsed:.0f}s ===\n")

    # ---------- v4.6 m-ArLi prediction ----------
    Ea_f  = V46["Ea_f"](BDE, vol, dH_dim)
    Ea_d  = V46["Ea_d"](Gsolv, vol, dVbur)
    lnA_f = V46["lnA_f"](d_LiC, vol, dG_dim)
    lnA_d = V46["lnA_d"](LUMO, Gsolv, dVbur)
    y_max_assume = 83.1   # v4.6 doesn't predict y_max; use 3-CN-PhLi value
    print("=" * 70)
    print("v4.6 m-ArLi class prediction for 5-Li-2-F-3-CN-ArLi")
    print("=" * 70)
    print(f"  Ea_f  = {Ea_f:7.2f} kJ/mol")
    print(f"  lnA_f = {lnA_f:7.2f}")
    print(f"  Ea_d  = {Ea_d:7.2f} kJ/mol")
    print(f"  lnA_d = {lnA_d:7.2f}")

    # ---------- save descriptors ----------
    descs = dict(
        smi=SMI, name=NAME,
        # monomer (xtb-level)
        LUMO_eV=LUMO, d_LiC_A=d_LiC, vol_A3=vol,
        Gsolv_kJ=Gsolv, BDE_kJ=BDE, dipole_D=dipole,
        # aggregation
        dE_dim_kJ=dE_dim, dH_dim_kJ=dH_dim, dG_dim_kJ=dG_dim, dS_dim_J=dS_dim,
        Vbur_mono=vbur_m, Vbur_dim=vbur_d, dVbur_dim=dVbur,
        # v4.6 prediction
        v46_Ea_f=Ea_f, v46_lnA_f=lnA_f, v46_Ea_d=Ea_d, v46_lnA_d=lnA_d,
        y_max_assumed=y_max_assume,
    )
    pd.DataFrame([descs]).to_csv(BASE / f"{NAME}_descriptors.csv", index=False)
    print(f"\nSaved: {NAME}_descriptors.csv")

    # ---------- predict yield over experimental grid ----------
    R = 8.314e-3
    exp = pd.read_csv(BASE / "experiment_fbrcn_summary.csv")
    def yld_v46(tR, T_C):
        Tk = T_C + 273.15
        kf = np.exp(lnA_f - Ea_f/(R*Tk))
        kd = np.exp(lnA_d - Ea_d/(R*Tk))
        return y_max_assume * (1 - np.exp(-kf*tR)) * np.exp(-kd*tR)
    exp["v46_pred"] = exp.apply(lambda r: yld_v46(r['tR_s'], r['T_C']), axis=1)
    exp["resid_v46"] = (exp['yield_pct'] - exp['v46_pred']).round(2)
    mae = (exp['yield_pct'] - exp['v46_pred']).abs().mean()
    mbe = (exp['yield_pct'] - exp['v46_pred']).mean()
    print(f"\nv4.6 (xTB descriptors) vs experiment: MAE={mae:.2f} pp, mean(exp-pred)={mbe:+.2f} pp")
    out_csv = BASE / f"{NAME}_v46_prediction.csv"
    exp[["sample","T_C","L_cm","tR_s","yield_pct","v46_pred","resid_v46"]].to_csv(out_csv, index=False)
    print(f"Saved: {out_csv.name}")

    # ---------- compare to 3-CN-PhLi analog (previously used) ----------
    analog = dict(Ea_f=34.69, lnA_f=22.11, Ea_d=27.3, lnA_d=9.08, y_max=83.1)
    def yld_analog(tR, T_C):
        Tk = T_C + 273.15
        kf = np.exp(analog['lnA_f'] - analog['Ea_f']/(R*Tk))
        kd = np.exp(analog['lnA_d'] - analog['Ea_d']/(R*Tk))
        return analog['y_max'] * (1 - np.exp(-kf*tR)) * np.exp(-kd*tR)
    exp["analog_pred"] = exp.apply(lambda r: yld_analog(r['tR_s'], r['T_C']), axis=1)
    mae_a = (exp['yield_pct'] - exp['analog_pred']).abs().mean()
    mbe_a = (exp['yield_pct'] - exp['analog_pred']).mean()
    print(f"\n3-CN-PhLi analog:                     MAE={mae_a:.2f} pp, mean(exp-pred)={mbe_a:+.2f} pp")
    print(f"v4.6 (xTB) improvement vs analog: Δ(MAE) = {mae - mae_a:+.2f} pp")


if __name__ == "__main__":
    main()
