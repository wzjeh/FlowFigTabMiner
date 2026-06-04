"""
方案 B - 全套聚集态描述符:
  ΔE_dim, ΔH_dim, ΔG_dim, ΔS_dim, Δ%V_bur(Li)

工作流 (per compound):
  1. 单体: tblite L-BFGS-B opt → xtb --hess --alpb thf → E, H, G, %V_bur(Li)
  2. 二聚体 4-中心: 镜像构造 → tblite opt → xtb --hess --alpb thf → E, H, G, %V_bur(Li_avg)
  3. ΔX_dim = X(dim) - 2 * X(mono) for X ∈ {E, H, G}
  4. ΔS_dim = (ΔH_dim - ΔG_dim) / 298.15
  5. Δ%V_bur = %V_bur(Li in dim) - %V_bur(Li in mono)

Outputs: aggregation_descriptors.csv
  smi, n_atoms, E_mono_alpb, H_mono_alpb, G_mono_alpb,
  E_dim_alpb, H_dim_alpb, G_dim_alpb,
  dE_dim_kJ, dH_dim_kJ, dG_dim_kJ, dS_dim_J_per_K,
  Vbur_mono, Vbur_dim, dVbur, status
"""
import json
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.optimize import minimize
from tblite.interface import Calculator
from morfeus import BuriedVolume

DATA_DIR = Path(__file__).resolve().parent
XYZ_DIR = DATA_DIR / "hf_xyz"
GEOM_DIR = DATA_DIR / "agg_geometries"   # save optimized dimer/monomer xyz here
GEOM_DIR.mkdir(exist_ok=True)
MANIFEST = XYZ_DIR / "manifest.json"
XTB = "/Users/zhaowenyuan/miniconda3/bin/xtb"
OUT_CSV = DATA_DIR / "aggregation_descriptors.csv"

ANG_TO_BOHR = 1.8897259886
HARTREE_TO_KJ = 2625.5
T_REF = 298.15

ELEM2Z = {'H':1,'C':6,'N':7,'O':8,'F':9,'Si':14,'P':15,'S':16,
          'Cl':17,'Br':35,'I':53,'Li':3}


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


def write_std_xyz(path, elems, coords, comment="mol"):
    with open(path, 'w') as f:
        f.write(f'{len(elems)}\n{comment}\n')
        for e, (x, y, z) in zip(elems, coords):
            f.write(f'{e:<3s} {x:>14.8f} {y:>14.8f} {z:>14.8f}\n')


def smiles_to_3d(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return None
    mol = Chem.AddHs(mol)
    p = AllChem.ETKDGv3(); p.randomSeed = 42
    if AllChem.EmbedMolecule(mol, p) != 0: return None
    AllChem.UFFOptimizeMolecule(mol, maxIters=2000)
    elems = [a.GetSymbol() for a in mol.GetAtoms()]
    coords = mol.GetConformer().GetPositions()
    li_idx = next((i for i, e in enumerate(elems) if e == 'Li'), None)
    if li_idx is None: return None
    return elems, np.array(coords), li_idx


def energy_grad(elems, coords, charge=0):
    nums = np.array([ELEM2Z[e] for e in elems])
    calc = Calculator('GFN2-xTB', nums, coords * ANG_TO_BOHR)
    calc.set('verbosity', 0)
    if charge != 0:
        calc.set('charge', float(charge))
    res = calc.singlepoint()
    e = float(res.get('energy'))
    g = np.array(res.get('gradient')).reshape(-1, 3) * ANG_TO_BOHR
    return e, g.flatten()


def opt_tblite(elems, coords, maxiter=400):
    def fg(x):
        return energy_grad(elems, x.reshape(-1, 3))
    res = minimize(fg, coords.flatten(), jac=True, method='L-BFGS-B',
                   options={'maxiter': maxiter, 'gtol': 1e-3})
    return res.x.reshape(-1, 3), res.fun, res.success


def xtb_hess_alpb(elems, coords, alpb='thf'):
    """xtb --hess --alpb thf single-point + frequency.
    Returns dict with E, H, G (Hartree). None on failure.
    """
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        write_std_xyz(td/'mol.xyz', elems, coords)
        try:
            r = subprocess.run([XTB, 'mol.xyz', '--hess', '--alpb', alpb,
                                '--gfn', '2'],
                               cwd=td, capture_output=True, text=True, timeout=900)
        except subprocess.TimeoutExpired:
            return None
        if r.returncode != 0:
            return None
        E = H = G = None
        for line in r.stdout.splitlines():
            if 'TOTAL ENERGY' in line:
                try: E = float(line.split()[-3])
                except: pass
            elif 'TOTAL ENTHALPY' in line:
                try: H = float(line.split()[-3])
                except: pass
            elif 'TOTAL FREE ENERGY' in line:
                try: G = float(line.split()[-3])
                except: pass
        if E is None or H is None or G is None:
            return None
        return {'E': E, 'H': H, 'G': G}


def construct_dimer(elems, coords, li_idx):
    coords = coords - coords[li_idx]
    mirrored = coords.copy()
    mirrored[:, 0] *= -1
    mirrored[:, 1] *= -1
    mirrored[:, 0] += 2.4
    return elems + elems, np.vstack([coords, mirrored])


def compute_vbur(elems, coords, li_idx, radius=3.5):
    """%V_bur around Li atom."""
    try:
        # morfeus uses 1-indexed atoms
        bv = BuriedVolume(elems, coords, li_idx + 1, radius=radius)
        return float(bv.fraction_buried_volume)
    except Exception:
        return None


def find_li_indices(elems):
    return [i for i, e in enumerate(elems) if e == 'Li']


def get_geom(smi, info):
    """Load monomer geometry: prefer hf_xyz, fall back to RDKit."""
    if info is not None:
        xyz_path = XYZ_DIR / info['file']
        if xyz_path.exists():
            elems, coords = read_orca_xyz(xyz_path)
            return elems, coords, info['li_idx']
    # fallback
    return smiles_to_3d(smi)


def main():
    # Load Tier-C compounds (the 30 we model on)
    arr = pd.read_csv(DATA_DIR / 'global_arrhenius.csv')
    tier = pd.read_csv(DATA_DIR / 'model_comparison_L1_L2.csv')
    tier['tier_clean'] = tier['tier'].apply(
        lambda x: 'Tier A' if 'Tier A' in x else ('Tier B' if 'Tier B' in x else 'Tier C'))
    excl = set(tier[tier['tier_clean'].isin(['Tier A','Tier B'])]['smi'])
    target_smis = list(arr[~arr['smi'].isin(excl)]['smi'])
    print(f"[info] {len(target_smis)} Tier-C compounds")

    with open(MANIFEST) as f:
        manifest = json.load(f)

    # resume support
    existing = {}
    if OUT_CSV.exists():
        df_old = pd.read_csv(OUT_CSV)
        for _, r in df_old.iterrows():
            if r.get('status') == 'ok':
                existing[r['smi']] = r.to_dict()
        print(f"[info] resuming: {len(existing)} done")

    rows = list(existing.values())
    t0 = time.time()
    for k, smi in enumerate(target_smis, start=1):
        if smi in existing:
            continue
        info = manifest.get(smi)
        elapsed = time.time() - t0
        print(f"[{k:2d}/{len(target_smis)}] {smi[:55]} [{elapsed:.0f}s]", flush=True)

        try:
            res = get_geom(smi, info)
            if res is None:
                rows.append({'smi': smi, 'status': 'no_geom'}); continue
            elems_m, coords_m, li_idx = res

            # 1. monomer opt
            opt_m, _, _ = opt_tblite(elems_m, coords_m)
            write_std_xyz(GEOM_DIR / f"mono_{k:03d}.xyz", elems_m, opt_m, comment=smi)

            # 2. monomer hess
            mono_thermo = xtb_hess_alpb(elems_m, opt_m)
            if mono_thermo is None:
                rows.append({'smi': smi, 'status': 'mono_hess_fail'}); continue

            # 3. dimer opt
            d_elems, d_coords_init = construct_dimer(elems_m, opt_m, li_idx)
            opt_d, _, _ = opt_tblite(d_elems, d_coords_init, maxiter=500)
            write_std_xyz(GEOM_DIR / f"dim_{k:03d}.xyz", d_elems, opt_d, comment=f"dimer of {smi}")

            # 4. dimer hess
            dim_thermo = xtb_hess_alpb(d_elems, opt_d)
            if dim_thermo is None:
                rows.append({'smi': smi, 'status': 'dim_hess_fail'}); continue

            # 5. %V_bur on Li (dimer has 2 Li, take average)
            vbur_m = compute_vbur(elems_m, opt_m, li_idx)
            li_indices_d = find_li_indices(d_elems)
            vbur_d_list = [compute_vbur(d_elems, opt_d, i) for i in li_indices_d]
            vbur_d_list = [v for v in vbur_d_list if v is not None]
            vbur_d = np.mean(vbur_d_list) if vbur_d_list else None

            # 6. compute deltas
            dE = (dim_thermo['E'] - 2*mono_thermo['E']) * HARTREE_TO_KJ
            dH = (dim_thermo['H'] - 2*mono_thermo['H']) * HARTREE_TO_KJ
            dG = (dim_thermo['G'] - 2*mono_thermo['G']) * HARTREE_TO_KJ
            dS = (dH - dG) / T_REF * 1000  # J/(mol·K)
            dVbur = (vbur_d - vbur_m) if (vbur_m is not None and vbur_d is not None) else None

            print(f"     ΔE={dE:+7.1f}  ΔH={dH:+7.1f}  ΔG={dG:+7.1f} kJ/mol  ΔS={dS:+7.1f} J/K  Δ%Vbur={dVbur:+.3f}" if dVbur is not None else
                  f"     ΔE={dE:+7.1f}  ΔH={dH:+7.1f}  ΔG={dG:+7.1f} kJ/mol  ΔS={dS:+7.1f} J/K  Δ%Vbur=NaN")

            rows.append({
                'smi': smi, 'n_atoms': len(elems_m),
                'E_mono_alpb': round(mono_thermo['E'], 6),
                'H_mono_alpb': round(mono_thermo['H'], 6),
                'G_mono_alpb': round(mono_thermo['G'], 6),
                'E_dim_alpb': round(dim_thermo['E'], 6),
                'H_dim_alpb': round(dim_thermo['H'], 6),
                'G_dim_alpb': round(dim_thermo['G'], 6),
                'dE_dim_kJ': round(dE, 2),
                'dH_dim_kJ': round(dH, 2),
                'dG_dim_kJ': round(dG, 2),
                'dS_dim_J_K': round(dS, 2),
                'Vbur_mono': round(vbur_m, 4) if vbur_m else None,
                'Vbur_dim_avg': round(vbur_d, 4) if vbur_d else None,
                'dVbur': round(dVbur, 4) if dVbur else None,
                'status': 'ok',
            })
        except Exception as ex:
            rows.append({'smi': smi, 'status': f'err:{type(ex).__name__}'})
            print(f"     ERROR: {ex}")

        if k % 3 == 0:
            pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    n_ok = (df['status'] == 'ok').sum()
    print(f"\n[done] {n_ok}/{len(df)} successful in {time.time()-t0:.0f}s")
    if n_ok > 0:
        ok = df[df['status'] == 'ok']
        for col, name in [('dE_dim_kJ', 'ΔE_dim'),
                           ('dH_dim_kJ', 'ΔH_dim'),
                           ('dG_dim_kJ', 'ΔG_dim'),
                           ('dS_dim_J_K', 'ΔS_dim (J/K)'),
                           ('dVbur', 'Δ%V_bur')]:
            if col in ok.columns:
                vals = ok[col].dropna()
                if len(vals) > 0:
                    print(f"  {name}: range {vals.min():+.1f} ~ {vals.max():+.1f}, median {vals.median():+.1f}")


if __name__ == "__main__":
    main()
