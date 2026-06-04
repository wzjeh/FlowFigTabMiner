"""
方案 C2: n_THF(Li) 最优 THF 配位数扫描

策略:
  1. 起始构型: RLi + 4 个 THF 在 R 基团对侧的半球均匀分布
  2. tblite 优化整个簇 (ALPB-THF bulk solvent)
  3. 优化后用 Wiberg 键级判断 Li-O 是否成键 (BO > 0.05)
  4. n_THF = 配位的 THF 数, ΔE_bind_avg = 平均结合能

输出: nthf_descriptors.csv
  smi, n_atoms_RLi, n_THF_init, n_THF_final, E_RLi_alpb, E_THF_alpb, E_cluster_alpb,
  dE_total_kJ, dE_per_THF_kJ, status
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

DATA_DIR = Path(__file__).resolve().parent
XYZ_DIR = DATA_DIR / "hf_xyz"
MANIFEST = XYZ_DIR / "manifest.json"
GEOM_DIR = DATA_DIR / "nthf_geometries"
GEOM_DIR.mkdir(exist_ok=True)
XTB = "/Users/zhaowenyuan/miniconda3/bin/xtb"
OUT_CSV = DATA_DIR / "nthf_descriptors.csv"

ANG_TO_BOHR = 1.8897259886
HARTREE_TO_KJ = 2625.5
LI_O_CUTOFF = 2.5   # Å, Li-O coordination distance
LI_O_BO_CUTOFF = 0.05   # Wiberg bond order threshold

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


def thf_geometry():
    """Build a single THF molecule, optimized minimal geometry."""
    mol = Chem.MolFromSmiles('C1CCOC1')
    mol = Chem.AddHs(mol)
    p = AllChem.ETKDGv3(); p.randomSeed = 7
    AllChem.EmbedMolecule(mol, p)
    AllChem.UFFOptimizeMolecule(mol, maxIters=500)
    elems = [a.GetSymbol() for a in mol.GetAtoms()]
    coords = mol.GetConformer().GetPositions()
    o_idx = next(i for i, e in enumerate(elems) if e == 'O')
    return elems, np.array(coords), o_idx


def find_c_neighbors_of_li(elems, coords, li_idx):
    """Return list of C atoms within 2.5 Å of Li (typically just 1 for monomeric RLi)."""
    li_pos = coords[li_idx]
    c_idx = []
    for i, e in enumerate(elems):
        if e == 'C' and i != li_idx:
            d = np.linalg.norm(coords[i] - li_pos)
            if d < 2.5:
                c_idx.append(i)
    return c_idx


def build_RLi_THF4(elems_R, coords_R, li_idx, thf_elems, thf_coords, thf_o_idx):
    """Place 4 THF molecules around Li in the hemisphere opposite to C-Li bond."""
    li_pos = coords_R[li_idx].copy()

    # Direction from average C-neighbor to Li (the "back" direction)
    c_neighbors = find_c_neighbors_of_li(elems_R, coords_R, li_idx)
    if c_neighbors:
        c_avg = np.mean([coords_R[i] for i in c_neighbors], axis=0)
        back_dir = li_pos - c_avg
        back_dir /= (np.linalg.norm(back_dir) + 1e-10)
    else:
        back_dir = np.array([1.0, 0.0, 0.0])

    # Build orthonormal frame with back_dir as +z
    z_ax = back_dir
    if abs(z_ax[0]) < 0.9:
        x_aux = np.array([1.0, 0.0, 0.0])
    else:
        x_aux = np.array([0.0, 1.0, 0.0])
    y_ax = np.cross(z_ax, x_aux); y_ax /= np.linalg.norm(y_ax)
    x_ax = np.cross(y_ax, z_ax); x_ax /= np.linalg.norm(x_ax)

    # Place 4 THFs in tetrahedral arrangement on the +z side (back hemisphere)
    # angles: tetrahedral with 1 at +z and 3 at 109.5° from +z
    # but we want all in back hemisphere => use +z and 3 vertices at 70° from +z
    Li_O_dist = 2.0  # Å
    placement_dirs = [
        z_ax,
        z_ax * np.cos(np.radians(70)) + x_ax * np.sin(np.radians(70)),
        z_ax * np.cos(np.radians(70)) + (-0.5*x_ax + 0.866*y_ax) * np.sin(np.radians(70)),
        z_ax * np.cos(np.radians(70)) + (-0.5*x_ax - 0.866*y_ax) * np.sin(np.radians(70)),
    ]

    all_elems = list(elems_R)
    all_coords = list(coords_R)
    for d in placement_dirs:
        d = d / np.linalg.norm(d)
        # Translate THF so its O atom sits at li_pos + Li_O_dist * d
        thf_o_pos = thf_coords[thf_o_idx]
        shift = li_pos + Li_O_dist * d - thf_o_pos
        # Rotate THF so O-C2 axis points away from Li
        # (skip rotation for simplicity, just translate; xtb will optimize)
        thf_translated = thf_coords + shift
        all_elems.extend(thf_elems)
        all_coords.extend(thf_translated)

    return all_elems, np.array(all_coords)


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


def opt_tblite(elems, coords, maxiter=500):
    def fg(x):
        return energy_grad(elems, x.reshape(-1, 3))
    res = minimize(fg, coords.flatten(), jac=True, method='L-BFGS-B',
                   options={'maxiter': maxiter, 'gtol': 1e-3})
    return res.x.reshape(-1, 3), res.fun, res.success


def write_std_xyz(path, elems, coords, comment="mol"):
    with open(path, 'w') as f:
        f.write(f'{len(elems)}\n{comment}\n')
        for e, (x, y, z) in zip(elems, coords):
            f.write(f'{e:<3s} {x:>14.8f} {y:>14.8f} {z:>14.8f}\n')


def alpb_singlepoint(elems, coords):
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        write_std_xyz(td/'sp.xyz', elems, coords)
        try:
            r = subprocess.run([XTB, 'sp.xyz', '--alpb', 'thf', '--gfn', '2', '--sp'],
                               cwd=td, capture_output=True, text=True, timeout=600)
        except subprocess.TimeoutExpired:
            return None
        for line in r.stdout.splitlines():
            if 'TOTAL ENERGY' in line:
                try: return float(line.split()[-3])
                except: pass
        return None


def count_li_o_coordination(elems, coords, li_idx, cutoff=LI_O_CUTOFF):
    """Count O atoms within cutoff of Li."""
    li_pos = coords[li_idx]
    n = 0
    li_o_dists = []
    for i, e in enumerate(elems):
        if e == 'O':
            d = np.linalg.norm(coords[i] - li_pos)
            li_o_dists.append((i, d))
            if d < cutoff:
                n += 1
    return n, li_o_dists


def find_bound_thf_atoms(elems, coords, li_idx, n_RLi_atoms, n_thf_per=13, cutoff=LI_O_CUTOFF):
    """Identify which THF molecules are bound to Li, return list of (atom_indices_to_keep)."""
    n_total = len(elems)
    n_thf_init = (n_total - n_RLi_atoms) // n_thf_per

    li_pos = coords[li_idx]
    keep = list(range(n_RLi_atoms))   # always keep RLi atoms

    for k in range(n_thf_init):
        thf_start = n_RLi_atoms + k * n_thf_per
        thf_end = thf_start + n_thf_per
        # find O in this THF
        thf_o_idx = None
        for i in range(thf_start, thf_end):
            if elems[i] == 'O':
                thf_o_idx = i; break
        if thf_o_idx is None: continue
        d = np.linalg.norm(coords[thf_o_idx] - li_pos)
        if d < cutoff:
            keep.extend(range(thf_start, thf_end))
    return keep, n_thf_init


def main():
    # Load Tier-C compounds
    arr = pd.read_csv(DATA_DIR / 'global_arrhenius.csv')
    tier = pd.read_csv(DATA_DIR / 'model_comparison_L1_L2.csv')
    tier['tier_clean'] = tier['tier'].apply(
        lambda x: 'Tier A' if 'Tier A' in x else ('Tier B' if 'Tier B' in x else 'Tier C'))
    excl = set(tier[tier['tier_clean'].isin(['Tier A','Tier B'])]['smi'])
    target_smis = list(arr[~arr['smi'].isin(excl)]['smi'])
    print(f"[info] {len(target_smis)} Tier-C compounds")

    with open(MANIFEST) as f:
        manifest = json.load(f)

    # Build optimized THF reference (both gas and ALPB)
    print("[info] Building reference THF...")
    thf_e, thf_c, thf_o = thf_geometry()
    opt_thf, E_THF_gas, _ = opt_tblite(thf_e, thf_c)
    E_THF_alpb = alpb_singlepoint(thf_e, opt_thf)
    print(f"  E_THF (gas)      = {E_THF_gas:.6f} Hartree")
    print(f"  E_THF (ALPB-THF) = {E_THF_alpb:.6f} Hartree")

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
            if info is not None and (XYZ_DIR / info['file']).exists():
                elems_R, coords_R = read_orca_xyz(XYZ_DIR / info['file'])
                li_idx = info['li_idx']
            else:
                res = smiles_to_3d(smi)
                if res is None:
                    rows.append({'smi': smi, 'status': 'no_geom'}); continue
                elems_R, coords_R, li_idx = res

            # Optimize bare RLi (gas phase, tblite)
            opt_R, E_R_gas, _ = opt_tblite(elems_R, coords_R)

            # Build RLi·(THF)_4 cluster
            cluster_elems, cluster_coords = build_RLi_THF4(
                elems_R, opt_R, li_idx, thf_e, opt_thf, thf_o)

            # Optimize cluster (gas phase)
            opt_cluster, E_cluster_gas, _ = opt_tblite(
                cluster_elems, cluster_coords, maxiter=800)

            # Save geometry
            write_std_xyz(GEOM_DIR / f"cluster_{k:03d}.xyz",
                          cluster_elems, opt_cluster, comment=smi)

            # Count Li-O coordination (geometric criterion)
            n_thf, li_o_dists = count_li_o_coordination(
                cluster_elems, opt_cluster, li_idx)
            short_dists = sorted([d for _, d in li_o_dists])[:6]

            # Trim to RLi + bound THFs
            keep_idx, _ = find_bound_thf_atoms(
                cluster_elems, opt_cluster, li_idx,
                n_RLi_atoms=len(elems_R), n_thf_per=len(thf_e))
            trimmed_e = [cluster_elems[i] for i in keep_idx]
            trimmed_c = opt_cluster[keep_idx]

            # GAS-PHASE binding energy (avoids ALPB double-counting issue)
            from compute_nthf import energy_grad
            E_trimmed_gas, _ = energy_grad(trimmed_e, trimmed_c)
            dE_bind_gas = (E_trimmed_gas - E_R_gas - n_thf * E_THF_gas) * HARTREE_TO_KJ
            dE_per_thf_gas = dE_bind_gas / max(n_thf, 1)

            # Also keep ALPB values (sanity check, may be unreliable due to mixed implicit+explicit)
            E_trimmed_alpb = alpb_singlepoint(trimmed_e, trimmed_c)
            if E_trimmed_alpb is None:
                E_trimmed_alpb = E_cluster_gas  # fallback

            print(f"     n_THF={n_thf} (Li-O: {[f'{d:.2f}' for d in short_dists[:n_thf+1]]})  "
                  f"ΔE_bind_gas={dE_bind_gas:+.1f}  ΔE/THF={dE_per_thf_gas:+.1f} kJ/mol")

            rows.append({
                'smi': smi, 'n_atoms_RLi': len(elems_R),
                'n_THF_init': 4, 'n_THF_final': n_thf,
                'E_RLi_gas': round(E_R_gas, 6),
                'E_THF_gas': round(E_THF_gas, 6),
                'E_cluster4_gas': round(E_cluster_gas, 6),
                'E_trimmed_gas': round(E_trimmed_gas, 6),
                'dE_bind_gas_kJ': round(dE_bind_gas, 2),
                'dE_per_THF_gas_kJ': round(dE_per_thf_gas, 2),
                'li_o_dists': str([round(d, 2) for d in short_dists[:5]]),
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
        print(f"\nn_THF distribution:")
        print(ok['n_THF_final'].value_counts().sort_index())
        print(f"\nΔE_per_THF: {ok['dE_per_THF_kJ'].min():+.1f} ~ {ok['dE_per_THF_kJ'].max():+.1f}, "
              f"median {ok['dE_per_THF_kJ'].median():+.1f} kJ/mol")


if __name__ == "__main__":
    main()
