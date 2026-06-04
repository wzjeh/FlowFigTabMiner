"""
补充计算 hf_xyz 中没有的 modeling_set 化合物的 ΔE_dim
直接从 SMILES → RDKit 3D → tblite opt → dimer construction → ΔE_dim
"""
import json, tempfile, subprocess, os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from rdkit import Chem
from rdkit.Chem import AllChem
from tblite.interface import Calculator

XTB = '/Users/zhaowenyuan/miniconda3/bin/xtb'
ANG_TO_BOHR = 1.8897259886
HARTREE_TO_KJ = 2625.5
ELEM2Z = {'H':1,'C':6,'N':7,'O':8,'F':9,'Si':14,'P':15,'S':16,
          'Cl':17,'Br':35,'I':53,'Li':3}

def smiles_to_3d(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return None, None
    mol = Chem.AddHs(mol)
    p = AllChem.ETKDGv3(); p.randomSeed = 42
    if AllChem.EmbedMolecule(mol, p) != 0: return None, None
    AllChem.UFFOptimizeMolecule(mol, maxIters=2000)
    elems = [a.GetSymbol() for a in mol.GetAtoms()]
    coords = mol.GetConformer().GetPositions()
    li_idx = next((i for i, e in enumerate(elems) if e == 'Li'), None)
    return (elems, coords, li_idx) if li_idx is not None else (None, None, None)

def energy_grad(elems, coords):
    nums = np.array([ELEM2Z[e] for e in elems])
    calc = Calculator('GFN2-xTB', nums, coords * ANG_TO_BOHR)
    calc.set('verbosity', 0)
    res = calc.singlepoint()
    e = float(res.get('energy'))
    g = np.array(res.get('gradient')).reshape(-1,3) * ANG_TO_BOHR
    return e, g.flatten()

def opt_tblite(elems, coords, maxiter=400):
    def fg(x):
        return energy_grad(elems, x.reshape(-1,3))
    res = minimize(fg, coords.flatten(), jac=True, method='L-BFGS-B',
                   options={'maxiter': maxiter, 'gtol': 1e-3})
    return res.x.reshape(-1,3), res.fun, res.success

def alpb_sp(elems, coords):
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        with open(td/'sp.xyz','w') as f:
            f.write(f'{len(elems)}\nmol\n')
            for e,(x,y,z) in zip(elems, coords):
                f.write(f'{e:<3s} {x:>14.8f} {y:>14.8f} {z:>14.8f}\n')
        try:
            r = subprocess.run([XTB,'sp.xyz','--alpb','thf','--gfn','2','--sp'],
                               cwd=td, capture_output=True, text=True, timeout=300)
        except: return None
        for line in r.stdout.splitlines():
            if 'TOTAL ENERGY' in line:
                try: return float(line.split()[-3])
                except: pass
        return None

def main():
    df = pd.read_csv('dimerization_energies.csv')
    have = set(df[df['status']=='ok']['smi'])

    # 找缺失的
    arr = pd.read_csv('global_arrhenius.csv')
    tier = pd.read_csv('model_comparison_L1_L2.csv')
    tier['tier_clean'] = tier['tier'].apply(lambda x: 'Tier A' if 'Tier A' in x else ('Tier B' if 'Tier B' in x else 'Tier C'))
    excl = set(tier[tier['tier_clean'].isin(['Tier A','Tier B'])]['smi'])
    modeling = arr[~arr['smi'].isin(excl)]
    missing = [s for s in modeling['smi'] if s not in have]
    print(f'需要补充 {len(missing)} 个化合物')

    rows = df.to_dict('records')
    for k, smi in enumerate(missing, 1):
        print(f'[{k}/{len(missing)}] {smi[:55]}', flush=True)
        elems, coords, li_idx = smiles_to_3d(smi)
        if elems is None:
            rows.append({'smi': smi, 'status': 'rdkit_fail'}); continue

        try:
            opt_m, E_m_gas, ok = opt_tblite(elems, coords)
            E_m_alpb = alpb_sp(elems, opt_m)
            if E_m_alpb is None:
                rows.append({'smi': smi, 'status': 'mono_alpb_fail'}); continue

            # build dimer
            opt_m -= opt_m[li_idx]
            mir = opt_m.copy(); mir[:,0]*=-1; mir[:,1]*=-1; mir[:,0]+=2.4
            d_e, d_c = elems+elems, np.vstack([opt_m, mir])

            opt_d, E_d_gas, ok = opt_tblite(d_e, d_c, maxiter=500)
            E_d_alpb = alpb_sp(d_e, opt_d)
            if E_d_alpb is None:
                rows.append({'smi': smi, 'status': 'dim_alpb_fail'}); continue

            dE_g = (E_d_gas - 2*E_m_gas) * HARTREE_TO_KJ
            dE_a = (E_d_alpb - 2*E_m_alpb) * HARTREE_TO_KJ
            print(f'  ΔE_dim(gas)={dE_g:+.1f}  ΔE_dim(ALPB)={dE_a:+.1f} kJ/mol')
            rows.append({'smi': smi, 'n_atoms': len(elems),
                         'E_mono_gas': round(E_m_gas,6), 'E_dim_gas': round(E_d_gas,6),
                         'dE_dim_gas_kJ': round(dE_g,2),
                         'E_mono_alpb': round(E_m_alpb,6), 'E_dim_alpb': round(E_d_alpb,6),
                         'dE_dim_alpb_kJ': round(dE_a,2),
                         'status': 'ok'})
        except Exception as ex:
            rows.append({'smi': smi, 'status': f'err:{ex}'})

    pd.DataFrame(rows).to_csv('dimerization_energies.csv', index=False)
    print('✓ 已写入 dimerization_energies.csv')

if __name__=='__main__':
    main()
