"""
Fill new descriptor columns in clean_organolithium_unified_descriptors.csv:
  - HOMO_LUMO_gap_eV (trivial, from existing DFT)
  - sterimol_B1, sterimol_B5, sterimol_L (morfeus, at C_ipso→Li)
  - buried_vol_Li (morfeus, around Li, r=3.5 Å)
  - mol_volume (RDKit)
  - fukui_f_minus_C (xtb CLI, electrophilic Fukui at C_ipso)

Usage:
  cd /Users/zhaowenyuan/Projects/FlowFigTabMiner
  source flowfigtabminer/bin/activate
  python data/ml_lifetime/fill_new_descriptors.py
"""

import os, json, warnings, subprocess, tempfile, re
import numpy as np
import pandas as pd
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
from morfeus import Sterimol, BuriedVolume

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).resolve().parent
CSV_PATH = DATA_DIR / "clean_organolithium_unified_descriptors.csv"
XTB_BIN = "/Users/zhaowenyuan/miniconda3/bin/xtb"


def get_3d_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    if AllChem.EmbedMolecule(mol, params) != 0:
        return None
    AllChem.UFFOptimizeMolecule(mol, maxIters=1000)
    return mol


def find_li_c(mol):
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == "Li":
            for nbr in atom.GetNeighbors():
                if nbr.GetSymbol() == "C":
                    return atom.GetIdx(), nbr.GetIdx()
    return None, None


def compute_sterimol_bv_vol(mol, li_idx, c_idx):
    """Compute Sterimol B1/B5/L, BuriedVolume, MolVolume."""
    coords = mol.GetConformer().GetPositions()
    elems = [mol.GetAtomWithIdx(i).GetSymbol() for i in range(mol.GetNumAtoms())]
    result = {}

    try:
        s = Sterimol(elems, coords, c_idx + 1, li_idx + 1)  # 1-indexed
        result["sterimol_B1"] = round(s.B_1_value, 4)
        result["sterimol_B5"] = round(s.B_5_value, 4)
        result["sterimol_L"] = round(s.L_value, 4)
    except Exception:
        pass

    try:
        bv = BuriedVolume(elems, coords, li_idx + 1, radius=3.5)
        result["buried_vol_Li"] = round(bv.fraction_buried_volume, 4)
    except Exception:
        pass

    try:
        result["mol_volume"] = round(AllChem.ComputeMolVolume(mol), 2)
    except Exception:
        pass

    return result


def compute_fukui(smiles, mol, c_idx):
    """Compute electrophilic Fukui f⁻ at C_ipso using xtb.

    f⁻(C) = q(C, N electrons) - q(C, N-1 electrons)
    where N = total electrons in neutral molecule.
    """
    if not os.path.isfile(XTB_BIN):
        return None

    conf = mol.GetConformer()
    coords = conf.GetPositions()
    symbols = [mol.GetAtomWithIdx(i).GetSymbol() for i in range(mol.GetNumAtoms())]
    n_atoms = mol.GetNumAtoms()

    with tempfile.TemporaryDirectory() as tmpdir:
        xyz_path = os.path.join(tmpdir, "mol.xyz")
        with open(xyz_path, "w") as f:
            f.write(f"{n_atoms}\n\n")
            for sym, (x, y, z) in zip(symbols, coords):
                f.write(f"{sym} {x:.6f} {y:.6f} {z:.6f}\n")

        def run_xtb_charges(charge, uhf):
            """Run xtb and read charges from the 'charges' file it writes."""
            # Remove old charges file
            charges_file = os.path.join(tmpdir, "charges")
            if os.path.exists(charges_file):
                os.remove(charges_file)

            cmd = [XTB_BIN, xyz_path, "--sp", "--gfn", "2",
                   "--chrg", str(charge), "--uhf", str(uhf)]
            try:
                subprocess.run(cmd, capture_output=True, text=True,
                              timeout=60, cwd=tmpdir)
            except Exception:
                return None

            if not os.path.exists(charges_file):
                return None

            with open(charges_file) as f:
                charges = [float(line.strip()) for line in f if line.strip()]
            return charges

        # Neutral (N electrons)
        q_neutral_all = run_xtb_charges(charge=0, uhf=0)
        # Cation (N-1 electrons)
        q_cation_all = run_xtb_charges(charge=1, uhf=1)

        if (q_neutral_all is not None and q_cation_all is not None
                and c_idx < len(q_neutral_all) and c_idx < len(q_cation_all)):
            return round(q_neutral_all[c_idx] - q_cation_all[c_idx], 4)

    return None


def main():
    df = pd.read_csv(CSV_PATH)
    smiles_list = df["intermediate_smiles_canonical"].dropna().unique()
    print(f"Computing descriptors for {len(smiles_list)} unique intermediates...")

    computed = {}
    for i, smi in enumerate(smiles_list):
        mol = get_3d_mol(smi)
        if mol is None:
            print(f"  [{i+1}/{len(smiles_list)}] SKIP {smi[:40]} — 3D failed")
            continue

        li_idx, c_idx = find_li_c(mol)
        if li_idx is None:
            print(f"  [{i+1}/{len(smiles_list)}] SKIP {smi[:40]} — no Li-C bond")
            continue

        rec = {}

        # Sterimol + BuriedVolume + MolVolume
        rec.update(compute_sterimol_bv_vol(mol, li_idx, c_idx))

        # Fukui f⁻
        fukui = compute_fukui(smi, mol, c_idx)
        if fukui is not None:
            rec["fukui_f_minus_C"] = fukui

        computed[smi] = rec

        b1 = rec.get("sterimol_B1", "—")
        fk = rec.get("fukui_f_minus_C", "—")
        print(f"  [{i+1}/{len(smiles_list)}] {smi[:40]:40s} B1={b1} fukui={fk}")

    # HOMO-LUMO gap (trivial)
    df["HOMO_LUMO_gap_eV"] = df["dft_LUMO_eV"] - df["dft_HOMO_eV"]

    # Broadcast per-SMILES values to all rows
    new_cols = ["sterimol_B1", "sterimol_B5", "sterimol_L",
                "buried_vol_Li", "mol_volume", "fukui_f_minus_C"]
    for col in new_cols:
        df[col] = df["intermediate_smiles_canonical"].map(
            lambda s: computed.get(s, {}).get(col, np.nan)
        )

    # Save
    df.to_csv(CSV_PATH, index=False)

    # Summary
    print(f"\n{'='*60}")
    print(f"Saved: {CSV_PATH}")
    print(f"  Rows: {len(df)}, Columns: {len(df.columns)}")
    print(f"\n  New descriptor fill rates:")
    for col in ["HOMO_LUMO_gap_eV"] + new_cols:
        n = df[col].notna().sum()
        pct = 100 * n / len(df)
        print(f"    {col:25s}: {n}/{len(df)} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
