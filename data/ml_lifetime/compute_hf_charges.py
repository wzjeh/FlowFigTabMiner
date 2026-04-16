"""
Compute HF/6-31+G* NPA charges for key organolithium intermediates using Psi4.

Run with conda base env (where psi4 is installed):
  conda run -n base python data/ml_lifetime/compute_hf_charges.py
"""

import json, sys, os, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).resolve().parent
CACHE_PATH = DATA_DIR / "hf_charge_cache.json"

# Need RDKit for 3D generation (available in both envs)
from rdkit import Chem
from rdkit.Chem import AllChem

ELEM_TO_NUM = {
    "H": 1, "Li": 3, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9,
    "Si": 14, "P": 15, "S": 16, "Cl": 17, "Br": 35, "I": 53,
}


def smiles_to_xyz(smiles):
    """SMILES → 3D xyz string for Psi4."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None, None
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    if AllChem.EmbedMolecule(mol, params) != 0:
        return None, None, None
    AllChem.UFFOptimizeMolecule(mol, maxIters=1000)

    conf = mol.GetConformer()
    symbols = [mol.GetAtomWithIdx(i).GetSymbol() for i in range(mol.GetNumAtoms())]
    coords = conf.GetPositions()

    # Find Li and C_ipso
    li_idx, c_idx = None, None
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == "Li":
            li_idx = atom.GetIdx()
            for nbr in atom.GetNeighbors():
                if nbr.GetSymbol() == "C":
                    c_idx = nbr.GetIdx()
                    break
            break

    # Build Psi4 geometry string
    lines = ["0 1"]  # charge=0, singlet
    for sym, (x, y, z) in zip(symbols, coords):
        lines.append(f"{sym}  {x:.6f}  {y:.6f}  {z:.6f}")
    xyz_str = "\n".join(lines)

    return xyz_str, li_idx, c_idx


def compute_hf_charges(smiles):
    """Compute HF/6-31+G* charges (Mulliken + Lowdin) for one molecule."""
    import psi4

    xyz_str, li_idx, c_idx = smiles_to_xyz(smiles)
    if xyz_str is None:
        return None

    psi4.core.set_output_file("/dev/null")
    psi4.set_memory("2 GB")
    psi4.set_num_threads(4)

    try:
        mol = psi4.geometry(xyz_str)
        e, wfn = psi4.energy("hf/6-31+g*", molecule=mol, return_wfn=True)

        # Mulliken charges
        psi4.oeprop(wfn, "MULLIKEN_CHARGES")
        mulliken = np.array([wfn.atomic_point_charges().np[i] for i in range(mol.natom())])

        # Lowdin charges
        psi4.oeprop(wfn, "LOWDIN_CHARGES")
        lowdin = np.array([wfn.atomic_point_charges().np[i] for i in range(mol.natom())])

        # Orbital energies
        eps = wfn.epsilon_a().np
        occ = wfn.occupation_a().np
        homo_idx = np.where(occ > 0.5)[0][-1]
        homo = eps[homo_idx] * 27.211  # Eh → eV
        lumo = eps[homo_idx + 1] * 27.211

        # Dipole
        psi4.oeprop(wfn, "DIPOLE")
        dipole = wfn.variable("SCF DIPOLE")  # Debye

        result = {
            "smiles": smiles,
            "hf_energy_Eh": e,
            "hf_mulliken_Li": mulliken[li_idx] if li_idx is not None else None,
            "hf_mulliken_C_ipso": mulliken[c_idx] if c_idx is not None else None,
            "hf_lowdin_Li": lowdin[li_idx] if li_idx is not None else None,
            "hf_lowdin_C_ipso": lowdin[c_idx] if c_idx is not None else None,
            "hf_HOMO_eV": homo,
            "hf_LUMO_eV": lumo,
            "hf_gap_eV": lumo - homo,
            "li_idx": li_idx,
            "c_idx": c_idx,
        }

        psi4.core.clean()
        return result

    except Exception as ex:
        psi4.core.clean()
        return {"smiles": smiles, "error": str(ex)}


def main():
    master = pd.read_csv(DATA_DIR / "intermediates_master.csv")

    # Key intermediates: all with Arrhenius + sufficient data
    key = master[
        (master.has_arrhenius == True) &
        (master.sufficient_for_arrhenius == True)
    ].sort_values("Ea_kJ_mol")

    smiles_list = key["intermediate_smiles_canonical"].tolist()
    print(f"Computing HF/6-31+G* for {len(smiles_list)} key intermediates...")

    # Load cache
    cache = {}
    if CACHE_PATH.exists():
        cache = json.load(open(CACHE_PATH))
        print(f"  Cache: {len(cache)} entries")

    results = []
    for i, smi in enumerate(smiles_list):
        if smi in cache and "error" not in cache[smi]:
            print(f"  [{i+1}/{len(smiles_list)}] {smi[:45]:45s} CACHED")
            results.append(cache[smi])
            continue

        t0 = time.time()
        r = compute_hf_charges(smi)
        dt = time.time() - t0

        if r is None:
            print(f"  [{i+1}/{len(smiles_list)}] {smi[:45]:45s} FAILED (3D)")
            continue

        if "error" in r:
            print(f"  [{i+1}/{len(smiles_list)}] {smi[:45]:45s} ERROR: {r['error'][:50]}")
            cache[smi] = r
        else:
            qc = r["hf_mulliken_C_ipso"]
            ql = r["hf_mulliken_Li"]
            homo = r["hf_HOMO_eV"]
            print(f"  [{i+1}/{len(smiles_list)}] {smi[:45]:45s} q_C={qc:+.4f} q_Li={ql:+.4f} HOMO={homo:.2f} ({dt:.1f}s)")
            cache[smi] = r
            results.append(r)

        # Save cache after each molecule
        with open(CACHE_PATH, "w") as f:
            json.dump(cache, f, indent=2, default=str)

    # Summary
    print(f"\n{'='*60}")
    print(f"Computed: {len(results)}/{len(smiles_list)}")
    print(f"Cache saved: {CACHE_PATH}")

    # Compare xTB vs HF charges for key molecules
    if results:
        print(f"\n═══ xTB vs HF charge comparison ═══")
        print(f"  {'SMILES':40s} {'q_C(xTB)':>10s} {'q_C(HF)':>10s} {'Δ':>8s} {'Ea':>6s}")
        for r in results:
            smi = r["smiles"]
            row = key[key.intermediate_smiles_canonical == smi]
            if len(row) == 0:
                continue
            q_xtb = row.iloc[0].dft_charge_C_ipso
            q_hf = r["hf_mulliken_C_ipso"]
            ea = row.iloc[0].Ea_kJ_mol
            if q_hf is not None and pd.notna(q_xtb):
                delta = q_hf - q_xtb
                print(f"  {smi[:40]:40s} {q_xtb:+10.4f} {q_hf:+10.4f} {delta:+8.4f} {ea:6.1f}")


if __name__ == "__main__":
    main()
