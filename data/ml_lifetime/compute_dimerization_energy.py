"""
Compute dimerization energy ΔE_dim for organolithium monomers.

Method (绕过 xtb opt bug):
  - tblite + scipy L-BFGS-B for gas-phase geometry optimization
  - xtb CLI single-point at optimized geometry for ALPB(THF) energy
  - ΔE_dim = E(dimer) - 2 × E(monomer)  in both gas and ALPB(THF)

Outputs:
  dimerization_energies.csv with columns:
    smi, n_atoms, E_mono_gas, E_dim_gas, dE_dim_gas_kJ,
                  E_mono_alpb, E_dim_alpb, dE_dim_alpb_kJ, status
"""
import os
import json
import shutil
import tempfile
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from tblite.interface import Calculator

# ----------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent
XYZ_DIR = DATA_DIR / "hf_xyz"
MANIFEST = XYZ_DIR / "manifest.json"
XTB_BIN = "/Users/zhaowenyuan/miniconda3/bin/xtb"
OUT_CSV = DATA_DIR / "dimerization_energies.csv"

ANG_TO_BOHR = 1.8897259886
HARTREE_TO_KJ = 2625.5

ELEM2Z = {'H':1,'C':6,'N':7,'O':8,'F':9,'Si':14,'P':15,'S':16,
          'Cl':17,'Br':35,'I':53,'Li':3,'Na':11,'K':19}

# ----------------------------------------------------------------------------
def read_orca_xyz(path):
    with open(path) as f:
        lines = [l for l in f.read().splitlines() if l.strip()]
    elems, coords = [], []
    for line in lines[1:]:
        p = line.split()
        if len(p) >= 4:
            elems.append(p[0])
            coords.append([float(p[1]), float(p[2]), float(p[3])])
    return elems, np.array(coords)

def write_std_xyz(path, elems, coords):
    with open(path, 'w') as f:
        f.write(f'{len(elems)}\nmol\n')
        for e, (x,y,z) in zip(elems, coords):
            f.write(f'{e:<3s} {x:>14.8f} {y:>14.8f} {z:>14.8f}\n')

def energy_grad_gas(elems, coords_ang, charge=0):
    nums = np.array([ELEM2Z[e] for e in elems])
    pos_bohr = coords_ang * ANG_TO_BOHR
    calc = Calculator('GFN2-xTB', nums, pos_bohr)
    calc.set('verbosity', 0)
    if charge != 0:
        calc.set('charge', float(charge))
    res = calc.singlepoint()
    e = float(res.get('energy'))
    g = np.array(res.get('gradient')).reshape(-1, 3) * ANG_TO_BOHR
    return e, g.flatten()

def optimize_tblite(elems, coords_ang, maxiter=300, gtol=1e-3, charge=0):
    x0 = coords_ang.flatten()
    def fg(x):
        return energy_grad_gas(elems, x.reshape(-1, 3), charge)
    res = minimize(fg, x0, jac=True, method='L-BFGS-B',
                   options={'maxiter': maxiter, 'gtol': gtol})
    return res.x.reshape(-1, 3), res.fun, res.success

def alpb_singlepoint(elems, coords_ang, alpb='thf'):
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        write_std_xyz(td/'sp.xyz', elems, coords_ang)
        try:
            r = subprocess.run([XTB_BIN, 'sp.xyz', '--alpb', alpb,
                                '--gfn', '2', '--sp'],
                               cwd=td, capture_output=True, text=True, timeout=300)
        except subprocess.TimeoutExpired:
            return None
        for line in r.stdout.splitlines():
            if 'TOTAL ENERGY' in line:
                try: return float(line.split()[-3])
                except: pass
        return None

def construct_dimer(elems, coords, li_idx):
    """4-center [Li2R2] dimer: mirror about Li-Li axis."""
    coords = coords - coords[li_idx]
    mirrored = coords.copy()
    mirrored[:, 0] *= -1
    mirrored[:, 1] *= -1
    mirrored[:, 0] += 2.4   # Li-Li distance ≈ 2.4 Å
    return elems + elems, np.vstack([coords, mirrored])

# ----------------------------------------------------------------------------
def main():
    with open(MANIFEST) as f:
        manifest = json.load(f)
    print(f"[info] {len(manifest)} monomers in manifest")
    print(f"[info] writing to {OUT_CSV}")

    # 已有数据则恢复 (支持中断重跑)
    existing = {}
    if OUT_CSV.exists():
        df_old = pd.read_csv(OUT_CSV)
        for _, r in df_old.iterrows():
            if r.get('status') == 'ok':
                existing[r['smi']] = r.to_dict()
        print(f"[info] resuming: {len(existing)} compounds already done")

    rows = list(existing.values())
    t0 = time.time()
    for k, (smi, info) in enumerate(manifest.items(), start=1):
        if smi in existing:
            print(f"[{k:2d}/{len(manifest)}] {smi[:60]} (skip - already ok)")
            continue

        xyz_path = XYZ_DIR / info["file"]
        if not xyz_path.exists():
            rows.append({"smi": smi, "status": "no_xyz"}); continue

        elems, coords = read_orca_xyz(xyz_path)
        li_idx = info["li_idx"]
        natoms = len(elems)

        elapsed = time.time() - t0
        print(f"[{k:2d}/{len(manifest)}] {smi[:50]:<50s} ({natoms}→{2*natoms} atoms) [{elapsed:.0f}s]", flush=True)

        try:
            # 1. 单体 opt + ALPB 单点
            opt_m, E_m_gas, ok_m = optimize_tblite(elems, coords)
            if not ok_m and abs(E_m_gas) < 1e-3:
                rows.append({"smi": smi, "status": "mono_opt_fail"})
                continue
            E_m_alpb = alpb_singlepoint(elems, opt_m)
            if E_m_alpb is None:
                rows.append({"smi": smi, "E_mono_gas": E_m_gas,
                             "status": "mono_alpb_fail"})
                continue

            # 2. 二聚体构造 + opt + ALPB 单点
            d_elems, d_coords = construct_dimer(elems, opt_m, li_idx)
            opt_d, E_d_gas, ok_d = optimize_tblite(d_elems, d_coords, maxiter=400)
            if not ok_d and abs(E_d_gas) < 1e-3:
                rows.append({"smi": smi, "E_mono_gas": E_m_gas,
                             "E_mono_alpb": E_m_alpb,
                             "status": "dim_opt_fail"})
                continue
            E_d_alpb = alpb_singlepoint(d_elems, opt_d)
            if E_d_alpb is None:
                rows.append({"smi": smi, "E_mono_gas": E_m_gas,
                             "E_mono_alpb": E_m_alpb,
                             "E_dim_gas": E_d_gas,
                             "status": "dim_alpb_fail"})
                continue

            dE_gas_kj = (E_d_gas - 2*E_m_gas) * HARTREE_TO_KJ
            dE_alpb_kj = (E_d_alpb - 2*E_m_alpb) * HARTREE_TO_KJ
            print(f"     ΔE_dim(gas)={dE_gas_kj:+7.1f} kJ/mol   ΔE_dim(ALPB-THF)={dE_alpb_kj:+7.1f} kJ/mol")

            rows.append({
                "smi": smi, "n_atoms": natoms,
                "E_mono_gas":  round(E_m_gas, 6),
                "E_dim_gas":   round(E_d_gas, 6),
                "dE_dim_gas_kJ":  round(dE_gas_kj, 2),
                "E_mono_alpb": round(E_m_alpb, 6),
                "E_dim_alpb":  round(E_d_alpb, 6),
                "dE_dim_alpb_kJ": round(dE_alpb_kj, 2),
                "status": "ok",
            })
        except Exception as ex:
            rows.append({"smi": smi, "status": f"error:{type(ex).__name__}:{str(ex)[:80]}"})
            print(f"     ERROR: {ex}")

        # 中途保存
        if k % 3 == 0:
            pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    n_ok = (df["status"] == "ok").sum()
    print(f"\n[done] {n_ok}/{len(df)} successful in {time.time()-t0:.0f}s")
    if n_ok > 0:
        ok = df[df["status"] == "ok"]
        print(f"\nΔE_dim(gas):     {ok['dE_dim_gas_kJ'].min():+7.1f}  ~ {ok['dE_dim_gas_kJ'].max():+7.1f}  median {ok['dE_dim_gas_kJ'].median():+7.1f} kJ/mol")
        print(f"ΔE_dim(ALPB):    {ok['dE_dim_alpb_kJ'].min():+7.1f}  ~ {ok['dE_dim_alpb_kJ'].max():+7.1f}  median {ok['dE_dim_alpb_kJ'].median():+7.1f} kJ/mol")


if __name__ == "__main__":
    main()
