"""
Fill DFT/xtb descriptors for organolithium intermediates using GFN2-xTB.

Pipeline per unique SMILES:
  1. SMILES → RDKit 3D → UFF pre-optimization
  2. GFN2-xTB geometry optimization (scipy L-BFGS-B + tblite gradients)
  3. tblite single-point at optimized geometry → charges, HOMO/LUMO, dipole, bond order
  4. xtb CLI single-point with ALPB(THF) → ΔG_solv  (tblite lacks solvation)
  5. xtb CLI fragment calculations → approximate BDE (tblite lacks open-shell)

Reads/Writes: clean_organolithium_unified_descriptors.csv

Usage:
  cd /Users/zhaowenyuan/Projects/FlowFigTabMiner
  source flowfigtabminer/bin/activate
  python data/ml_lifetime/fill_dft_descriptors.py
"""

import os
import sys
import io
import json
import warnings
import subprocess
import tempfile
import traceback
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.optimize import minimize
from tblite.interface import Calculator

# Suppress tblite C library stdout (prints bond-order matrices)
class _SuppressStdout:
    def __enter__(self):
        self._fd = os.dup(1)
        self._devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(self._devnull, 1)
        return self
    def __exit__(self, *a):
        os.dup2(self._fd, 1)
        os.close(self._fd)
        os.close(self._devnull)

warnings.filterwarnings("ignore")

BOHR_TO_ANG = 0.5291772109
ANG_TO_BOHR = 1.8897259886
HARTREE_TO_EV = 27.211386245988
HARTREE_TO_KJ = 2625.4996395

CSV_PATH = "data/ml_lifetime/clean_organolithium_unified_descriptors.csv"
CACHE_PATH = "data/ml_lifetime/xtb_cache.json"
XTB_BIN = "/Users/zhaowenyuan/miniconda3/bin/xtb"

ELEM_TO_NUM = {
    "H": 1, "He": 2, "Li": 3, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9,
    "Si": 14, "P": 15, "S": 16, "Cl": 17, "Br": 35, "I": 53,
}


def smiles_to_3d(smiles, max_attempts=5):
    """Convert SMILES to 3D coordinates via RDKit (single conformer, legacy)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None, None

    mol = Chem.AddHs(mol)

    for attempt in range(max_attempts):
        params = AllChem.ETKDGv3()
        params.randomSeed = 42 + attempt
        result = AllChem.EmbedMolecule(mol, params)
        if result == 0:
            break
    else:
        return None, None, None

    try:
        AllChem.UFFOptimizeMolecule(mol, maxIters=1000)
    except Exception:
        pass

    conf = mol.GetConformer()
    symbols = [mol.GetAtomWithIdx(i).GetSymbol() for i in range(mol.GetNumAtoms())]
    positions = np.array([
        [conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
        for i in range(mol.GetNumAtoms())
    ])

    return mol, symbols, positions


def smiles_to_3d_multiconf(smiles, n_confs=30):
    """Generate multiple 3D conformers via RDKit, UFF-optimize each.

    Returns list of (mol, symbols, positions) for each conformer,
    sorted by UFF energy (lowest first).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    params.pruneRmsThresh = 0.5
    cids = AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params)

    if len(cids) == 0:
        # Fallback: try single embed
        params2 = AllChem.ETKDGv3()
        params2.randomSeed = 42
        if AllChem.EmbedMolecule(mol, params2) != 0:
            return []
        cids = [0]

    # UFF optimize each conformer and collect energies
    uff_results = AllChem.UFFOptimizeMoleculeConfs(mol, maxIters=1000)

    symbols = [mol.GetAtomWithIdx(i).GetSymbol() for i in range(mol.GetNumAtoms())]

    conformers = []
    for idx, cid in enumerate(cids):
        conf = mol.GetConformer(cid)
        positions = np.array([
            [conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
            for i in range(mol.GetNumAtoms())
        ])
        # uff_results[idx] = (converged, energy) or may fail
        uff_energy = uff_results[idx][1] if idx < len(uff_results) else 1e10
        conformers.append((uff_energy, symbols, positions))

    # Sort by UFF energy
    conformers.sort(key=lambda x: x[0])

    return [(symbols, pos) for _, symbols, pos in conformers]


class CachedEnergyGradient:
    """Avoids duplicate tblite calls when scipy asks for energy and gradient separately."""

    def __init__(self, numbers, charge=0):
        self.numbers = numbers
        self.charge = charge
        self._last_x = None
        self._last_e = None
        self._last_g = None
        self.n_calls = 0

    def _compute(self, flat_pos):
        pos = flat_pos.reshape(-1, 3)
        with _SuppressStdout():
            calc = Calculator("GFN2-xTB", self.numbers, pos)
            calc.set("verbosity", 0)
            if self.charge != 0:
                calc.set("charge", float(self.charge))
            res = calc.singlepoint()
        self._last_x = flat_pos.copy()
        self._last_e = res.get("energy")
        self._last_g = res.get("gradient").flatten()
        self.n_calls += 1

    def energy(self, flat_pos):
        if self._last_x is None or not np.array_equal(flat_pos, self._last_x):
            self._compute(flat_pos)
        return self._last_e

    def gradient(self, flat_pos):
        if self._last_x is None or not np.array_equal(flat_pos, self._last_x):
            self._compute(flat_pos)
        return self._last_g


def xtb_optimize(numbers, positions_ang, charge=0, max_iter=200, gtol=5e-4):
    """Optimize geometry using GFN2-xTB via tblite + scipy L-BFGS-B."""
    pos_bohr = positions_ang * ANG_TO_BOHR
    x0 = pos_bohr.flatten()

    engine = CachedEnergyGradient(numbers, charge)

    result = minimize(
        fun=engine.energy,
        x0=x0,
        jac=engine.gradient,
        method="L-BFGS-B",
        options={"maxiter": max_iter, "gtol": gtol, "ftol": 1e-10},
    )

    opt_pos_ang = result.x.reshape(-1, 3) * BOHR_TO_ANG
    return opt_pos_ang, result.fun, result.success


def tblite_singlepoint(numbers, positions_ang, charge=0):
    """Run GFN2-xTB closed-shell single-point via tblite.
    Note: tblite 0.4 does not support open-shell (spin-polarization).
    """
    pos_bohr = positions_ang * ANG_TO_BOHR

    with _SuppressStdout():
        calc = Calculator("GFN2-xTB", numbers, pos_bohr)
        calc.set("verbosity", 0)
        if charge != 0:
            calc.set("charge", float(charge))
        res = calc.singlepoint()
        res_dict = res.dict()

    result = {
        "energy": res_dict.get("energy"),
        "charges": res_dict.get("charges"),
        "orbital_energies": res_dict.get("orbital-energies"),
        "orbital_occupations": res_dict.get("orbital-occupations"),
        "gradient": res_dict.get("gradient"),
        "dipole": res_dict.get("dipole"),
        "bond_orders": res_dict.get("bond-orders"),
    }

    return result


def xtb_solvation_energy(symbols, positions_ang, charge=0):
    """Run xtb CLI single-point with ALPB(THF) solvation.

    Returns solvation energy in Hartree, or None on failure.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        xyz_path = os.path.join(tmpdir, "mol.xyz")
        natoms = len(symbols)

        lines = [str(natoms), "molecule"]
        for sym, pos in zip(symbols, positions_ang):
            lines.append(f"{sym:2s} {pos[0]:12.6f} {pos[1]:12.6f} {pos[2]:12.6f}")

        with open(xyz_path, "w") as f:
            f.write("\n".join(lines) + "\n")

        cmd = [XTB_BIN, "mol.xyz", "--sp", "--gfn", "2", "--alpb", "thf",
               "--chrg", str(charge), "--uhf", "0"]

        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, cwd=tmpdir, timeout=120
            )
        except subprocess.TimeoutExpired:
            return None

        # Parse total energy from output
        for line in proc.stdout.split("\n"):
            if "TOTAL ENERGY" in line:
                parts = line.split()
                for j, p in enumerate(parts):
                    if p == "TOTAL" and j + 2 < len(parts):
                        try:
                            return float(parts[j + 2])
                        except ValueError:
                            pass

    return None


def extract_descriptors(smiles, symbols, numbers, opt_positions, sp_gas, e_solv):
    """Extract all DFT descriptors from calculation results."""
    desc = {}

    # Find Li and C_ipso indices
    li_idx = None
    c_ipso_idx = None
    for i, sym in enumerate(symbols):
        if sym == "Li":
            li_idx = i
            break

    if li_idx is not None:
        li_pos = opt_positions[li_idx]
        min_dist = 999.0
        for i, sym in enumerate(symbols):
            if sym == "C":
                d = np.linalg.norm(opt_positions[i] - li_pos)
                if d < min_dist:
                    min_dist = d
                    c_ipso_idx = i

    # Charges
    charges = sp_gas["charges"]
    if charges is not None and li_idx is not None:
        desc["dft_charge_Li"] = round(float(charges[li_idx]), 4)
    if charges is not None and c_ipso_idx is not None:
        desc["dft_charge_C_ipso"] = round(float(charges[c_ipso_idx]), 4)

    # HOMO / LUMO
    orb_e = sp_gas["orbital_energies"]
    orb_occ = sp_gas["orbital_occupations"]
    if orb_e is not None and orb_occ is not None:
        occupied = np.where(orb_occ > 0.5)[0]
        unoccupied = np.where(orb_occ < 0.5)[0]
        if len(occupied) > 0:
            homo_idx = occupied[-1]
            desc["dft_HOMO_eV"] = round(float(orb_e[homo_idx]) * HARTREE_TO_EV, 3)
        if len(unoccupied) > 0:
            lumo_idx = unoccupied[0]
            desc["dft_LUMO_eV"] = round(float(orb_e[lumo_idx]) * HARTREE_TO_EV, 3)

    # Li-C bond length
    if li_idx is not None and c_ipso_idx is not None:
        dist = np.linalg.norm(opt_positions[li_idx] - opt_positions[c_ipso_idx])
        desc["dft_LiC_bond_A"] = round(float(dist), 4)

    # Wiberg bond index (from GFN2-xTB bond orders)
    bond_orders = sp_gas.get("bond_orders")
    if bond_orders is not None and li_idx is not None and c_ipso_idx is not None:
        wbi = float(bond_orders[li_idx, c_ipso_idx])
        desc["dft_wiberg_LiC"] = round(wbi, 4)

    # Dipole moment
    dipole = sp_gas.get("dipole")
    if dipole is not None:
        dipole_debye = np.linalg.norm(dipole) * 2.5417464519
        desc["dft_dipole_D"] = round(float(dipole_debye), 3)

    # Solvation free energy
    e_gas = sp_gas["energy"]
    if e_solv is not None and e_gas is not None:
        dg_solv_kj = (e_solv - e_gas) * HARTREE_TO_KJ
        desc["dft_Gsolv_kJ"] = round(float(dg_solv_kj), 2)

    desc["dft_method"] = "GFN2-xTB"

    return desc


def xtb_sp_energy(symbols, positions_ang, charge=0, uhf=0):
    """Run xtb CLI single-point, return energy in Hartree or None."""
    with tempfile.TemporaryDirectory() as tmpdir:
        xyz_path = os.path.join(tmpdir, "mol.xyz")
        natoms = len(symbols)
        lines = [str(natoms), "fragment"]
        for sym, pos in zip(symbols, positions_ang):
            lines.append(f"{sym:2s} {pos[0]:12.6f} {pos[1]:12.6f} {pos[2]:12.6f}")
        with open(xyz_path, "w") as f:
            f.write("\n".join(lines) + "\n")

        cmd = [XTB_BIN, "mol.xyz", "--sp", "--gfn", "2",
               "--chrg", str(charge), "--uhf", str(uhf)]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, cwd=tmpdir, timeout=60
            )
        except subprocess.TimeoutExpired:
            return None

        for line in proc.stdout.split("\n"):
            if "TOTAL ENERGY" in line:
                parts = line.split()
                for j, p in enumerate(parts):
                    if p == "TOTAL" and j + 2 < len(parts):
                        try:
                            return float(parts[j + 2])
                        except ValueError:
                            pass
    return None


def compute_bde(symbols, numbers, opt_positions, gas_energy):
    """Compute approximate homolytic BDE via xtb CLI: E(ArLi) → E(Ar·) + E(Li·)"""
    li_idx = None
    for i, sym in enumerate(symbols):
        if sym == "Li":
            li_idx = i
            break

    if li_idx is None:
        return None

    # Li atom single-point (doublet, uhf=1)
    e_li = xtb_sp_energy(["Li"], np.array([[0.0, 0.0, 0.0]]), charge=0, uhf=1)
    if e_li is None:
        return None

    # Organic radical: remove Li (doublet, uhf=1)
    frag_symbols = [s for i, s in enumerate(symbols) if i != li_idx]
    frag_positions = np.array([p for i, p in enumerate(opt_positions) if i != li_idx])

    e_frag = xtb_sp_energy(frag_symbols, frag_positions, charge=0, uhf=1)
    if e_frag is None:
        return None

    bde = (e_frag + e_li - gas_energy) * HARTREE_TO_KJ
    return round(float(bde), 1)


def process_smiles(smiles, n_confs=30):
    """Full pipeline for one SMILES → dict of descriptors.

    Uses multi-conformer search: generates n_confs RDKit conformers,
    xTB-optimizes each, then takes the lowest-energy conformer for
    property extraction. This avoids conformational trapping (e.g.,
    ortho Li···O=C chelation found in some but not all single-seed runs).
    """
    # Step 1: Generate multiple 3D conformers
    conformers = smiles_to_3d_multiconf(smiles, n_confs=n_confs)
    if not conformers:
        return None, "3D embedding failed"

    symbols = conformers[0][0]
    numbers = np.array([ELEM_TO_NUM.get(s) for s in symbols])
    if any(n is None for n in numbers):
        unknown = [s for s in symbols if s not in ELEM_TO_NUM]
        return None, f"Unknown element(s): {unknown}"

    # Step 2: xTB-optimize each conformer, keep the lowest-energy one
    best_energy = 1e10
    best_pos = None
    n_ok = 0

    # Limit xTB optimizations to top-10 UFF conformers (already sorted)
    max_xtb = min(10, len(conformers))
    for ci, (_, positions) in enumerate(conformers[:max_xtb]):
        try:
            opt_pos, opt_energy, converged = xtb_optimize(numbers, positions)
            n_ok += 1
            if opt_energy < best_energy:
                best_energy = opt_energy
                best_pos = opt_pos
        except Exception:
            continue

    if best_pos is None:
        return None, f"All {max_xtb} conformer optimizations failed"

    if n_ok > 1:
        print(f"    Conformer search: {n_ok}/{max_xtb} optimized, best E={best_energy:.6f} Ha")

    opt_pos = best_pos
    opt_energy = best_energy

    # Step 3: Gas-phase single-point at best geometry
    try:
        sp_gas = tblite_singlepoint(numbers, opt_pos)
    except Exception as e:
        return None, f"Gas SP failed: {e}"

    # Step 4: Solvation single-point via xtb CLI
    e_solv = None
    try:
        e_solv = xtb_solvation_energy(symbols, opt_pos)
    except Exception as e:
        print(f"    Warning: solvation failed: {e}")

    # Step 5: Extract descriptors
    desc = extract_descriptors(smiles, symbols, numbers, opt_pos, sp_gas, e_solv)

    # Step 6: BDE
    try:
        bde = compute_bde(symbols, numbers, opt_pos, opt_energy)
        if bde is not None:
            desc["dft_LiC_BDE_kJ"] = bde
    except Exception as e:
        print(f"    Warning: BDE failed: {e}")

    return desc, None


def load_cache():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH) as f:
            return json.load(f)
    return {}


def save_cache(cache):
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)


def main():
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} rows")

    unique_smiles = df["intermediate_smiles_canonical"].dropna().unique()
    print(f"Unique non-null SMILES: {len(unique_smiles)}")

    cache = load_cache()
    results = {}
    failures = []

    for i, smi in enumerate(unique_smiles):
        print(f"\n[{i+1}/{len(unique_smiles)}] {smi[:60]}")

        if smi in cache and cache[smi].get("status") == "ok":
            print(f"  -> cached")
            results[smi] = cache[smi]["descriptors"]
            continue

        desc, error = process_smiles(smi)

        if error:
            print(f"  -> FAILED: {error}")
            failures.append((smi, error))
            cache[smi] = {"status": "failed", "error": error}
        else:
            print(f"  -> OK: Li={desc.get('dft_charge_Li', '?')}, "
                  f"HOMO={desc.get('dft_HOMO_eV', '?')} eV, "
                  f"LiC={desc.get('dft_LiC_bond_A', '?')} A, "
                  f"BDE={desc.get('dft_LiC_BDE_kJ', '?')} kJ/mol")
            results[smi] = desc
            cache[smi] = {"status": "ok", "descriptors": desc}

        if (i + 1) % 10 == 0:
            save_cache(cache)

    save_cache(cache)

    # Map descriptors to dataframe
    dft_cols = ["dft_charge_Li", "dft_charge_C_ipso", "dft_HOMO_eV", "dft_LUMO_eV",
                "dft_LiC_bond_A", "dft_method", "dft_LiC_BDE_kJ", "dft_wiberg_LiC",
                "dft_dipole_D", "dft_Gsolv_kJ"]

    for col in dft_cols:
        df[col] = df["intermediate_smiles_canonical"].map(
            lambda s, c=col: results.get(s, {}).get(c, np.nan) if isinstance(s, str) else np.nan
        )

    # Print summary
    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Successfully computed: {len(results)}/{len(unique_smiles)}")
    print(f"Failed: {len(failures)}/{len(unique_smiles)}")

    if failures:
        print(f"\nFailed SMILES:")
        for smi, err in failures:
            print(f"  {smi[:50]:50s} -- {err}")

    print(f"\n=== DFT descriptor fill rates ===")
    for col in dft_cols:
        filled = df[col].notna().sum()
        print(f"  {col:25s}: {filled}/{len(df)} ({100*filled/len(df):.1f}%)")

    # Sanity checks
    print(f"\n=== Sanity checks ===")
    for col, label, lo, hi in [
        ("dft_LiC_bond_A", "Li-C bond (A)", 1.7, 2.5),
        ("dft_HOMO_eV", "HOMO (eV)", -12, -1),
        ("dft_LUMO_eV", "LUMO (eV)", -8, 2),
        ("dft_LiC_BDE_kJ", "BDE (kJ/mol)", 50, 600),
        ("dft_charge_Li", "Li charge (e)", 0.1, 0.8),
        ("dft_dipole_D", "Dipole (D)", 0, 15),
        ("dft_Gsolv_kJ", "G_solv (kJ/mol)", -200, 0),
    ]:
        vals = df[col].dropna()
        if len(vals) > 0:
            ok = "OK" if vals.min() >= lo and vals.max() <= hi else "WARN"
            print(f"  {label:20s}: {vals.min():.2f} to {vals.max():.2f}  [{ok}]")

    df.to_csv(CSV_PATH, index=False)
    print(f"\nSaved to {CSV_PATH}")


if __name__ == "__main__":
    main()
