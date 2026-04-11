"""
Fill empirical LFER descriptors (sigma_hammett, Es_taft, delta_ortho, delta_benzyne)
for organolithium intermediates based on SMILES parsing.

Reads: clean_organolithium_unified_descriptors.csv
Writes: same file (in-place update of descriptor columns)

Usage:
  cd /Users/zhaowenyuan/Projects/FlowFigTabMiner
  source flowfigtabminer/bin/activate
  python data/ml_lifetime/fill_empirical_descriptors.py
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdmolops
from collections import defaultdict

CSV_PATH = "data/ml_lifetime/clean_organolithium_unified_descriptors.csv"

# ── Hammett sigma lookup (Hansch, Leo & Taft 1991) ──
# Keys are SMARTS patterns for the substituent atom bonded to the ring.
# We match the ring-attached atom and classify the functional group.

SIGMA_TABLE = {
    # (sigma_para, sigma_meta)
    "H":    (0.00, 0.00),
    "F":    (0.06, 0.34),
    "Cl":   (0.23, 0.37),
    "Br":   (0.23, 0.37),
    "I":    (0.35, 0.35),
    "NH2":  (-0.66, -0.16),
    "NMe2": (-0.83, -0.15),
    "OH":   (-0.37, 0.12),
    "OMe":  (-0.27, 0.12),
    "OR":   (-0.27, 0.12),    # generic ether
    "SMe":  (0.15, 0.15),
    "SR":   (0.15, 0.15),
    "CH3":  (-0.17, -0.07),
    "alkyl": (-0.17, -0.07),  # generic alkyl (Et, iPr, tBu, etc.)
    "phenyl": (-0.01, 0.06),
    "CO2R": (0.45, 0.37),     # ester
    "COOH": (0.45, 0.37),     # carboxylic acid
    "COR":  (0.50, 0.38),     # ketone/acyl
    "CHO":  (0.42, 0.35),     # aldehyde
    "CN":   (0.66, 0.56),
    "NO2":  (0.78, 0.71),
    "CF3":  (0.54, 0.43),
    "SO2R": (0.72, 0.56),
    "vinyl": (0.04, 0.06),    # -CH=CH2
    "CCH":  (0.23, 0.21),     # -C≡CH
    "Li":   (None, None),      # skip Li itself
    "SiR3": (-0.07, -0.04),   # silyl
}

# Taft Es values for ortho ester R groups
ES_TABLE = {
    "Me":  0.00,
    "Et":  -0.07,
    "iPr": -0.47,
    "tBu": -1.54,
}


def find_li_and_ring(mol):
    """Find the Li atom and the aromatic ring it's attached to.

    Returns: (li_idx, c_ipso_idx, ring_atom_indices) or (None, None, None)
    """
    li_atoms = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "Li"]
    if not li_atoms:
        return None, None, None

    li_idx = li_atoms[0]
    li_atom = mol.GetAtomWithIdx(li_idx)

    # Find carbon bonded to Li
    neighbors = li_atom.GetNeighbors()
    if not neighbors:
        return None, None, None

    c_ipso = neighbors[0]
    c_ipso_idx = c_ipso.GetIdx()

    # Check if c_ipso is aromatic
    if not c_ipso.GetIsAromatic():
        return li_idx, c_ipso_idx, None

    # Find the aromatic ring containing c_ipso
    ring_info = mol.GetRingInfo()
    for ring in ring_info.AtomRings():
        if c_ipso_idx in ring and len(ring) == 6:
            # Check if it's a carbocyclic ring (all C atoms, all aromatic)
            all_carbon = all(mol.GetAtomWithIdx(i).GetSymbol() == "C" for i in ring)
            all_aromatic = all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring)
            if all_carbon and all_aromatic:
                return li_idx, c_ipso_idx, ring

    return li_idx, c_ipso_idx, None


def get_ring_position(ring, c_ipso_idx, subst_ring_atom_idx):
    """Determine if a substituent is ortho, meta, or para relative to Li's carbon.

    Returns: 'ortho', 'meta', 'para', or None
    """
    ring_list = list(ring)
    try:
        pos_ipso = ring_list.index(c_ipso_idx)
        pos_subst = ring_list.index(subst_ring_atom_idx)
    except ValueError:
        return None

    # Ring distance (shortest path around ring)
    n = len(ring_list)
    dist = min(abs(pos_ipso - pos_subst), n - abs(pos_ipso - pos_subst))

    if dist == 1:
        return "ortho"
    elif dist == 2:
        return "meta"
    elif dist == 3:
        return "para"
    return None


def classify_substituent(mol, ring_atom_idx, ring_atoms_set):
    """Classify the substituent on a ring atom (excluding ring bonds and Li).

    Returns: substituent type string (key into SIGMA_TABLE), or None if just H.
    """
    atom = mol.GetAtomWithIdx(ring_atom_idx)

    # Get non-ring, non-Li neighbors
    ext_neighbors = []
    for nbr in atom.GetNeighbors():
        nbr_idx = nbr.GetIdx()
        if nbr_idx not in ring_atoms_set and nbr.GetSymbol() != "Li":
            ext_neighbors.append(nbr)

    if not ext_neighbors:
        return "H"  # no substituent, just H

    # Should be exactly one substituent atom for simple cases
    subst_atom = ext_neighbors[0]
    sym = subst_atom.GetSymbol()

    # Halogen
    if sym in ("F", "Cl", "Br", "I") and subst_atom.GetDegree() == 1:
        return sym

    # Nitrogen-based
    if sym == "N":
        # Check for NO2: N bonded to 2 O (one double, one single or both double)
        o_count = sum(1 for n in subst_atom.GetNeighbors()
                      if n.GetSymbol() == "O" and n.GetIdx() not in ring_atoms_set)
        if o_count >= 2:
            return "NO2"
        # Check for NR2 vs NH2
        non_ring_non_H = [n for n in subst_atom.GetNeighbors()
                          if n.GetIdx() not in ring_atoms_set and n.GetIdx() != ring_atom_idx]
        h_count = subst_atom.GetTotalNumHs()
        if h_count >= 2:
            return "NH2"
        return "NMe2"  # NR2 approximation

    # Oxygen-based
    if sym == "O":
        # -OH or -OR
        o_neighbors = [n for n in subst_atom.GetNeighbors() if n.GetIdx() != ring_atom_idx]
        if not o_neighbors or subst_atom.GetTotalNumHs() > 0:
            return "OH"
        return "OMe"  # -OR approximation

    # Sulfur-based
    if sym == "S":
        # Check for SO2
        o_count = sum(1 for n in subst_atom.GetNeighbors() if n.GetSymbol() == "O")
        if o_count >= 2:
            return "SO2R"
        return "SMe"  # -SR approximation

    # Carbon-based substituents
    if sym == "C":
        # Check for CN: C triple-bonded to N
        for nbr2 in subst_atom.GetNeighbors():
            if nbr2.GetIdx() == ring_atom_idx:
                continue
            bond = mol.GetBondBetweenAtoms(subst_atom.GetIdx(), nbr2.GetIdx())
            if nbr2.GetSymbol() == "N" and bond and bond.GetBondTypeAsDouble() == 3:
                return "CN"

        # Check for CF3: C bonded to 3 F
        f_count = sum(1 for n in subst_atom.GetNeighbors()
                      if n.GetSymbol() == "F" and n.GetIdx() != ring_atom_idx)
        if f_count == 3:
            return "CF3"

        # Check for C=O containing groups (ester, ketone, aldehyde, acid)
        has_double_O = False
        has_single_O = False
        double_O_idx = None
        single_O_atoms = []

        for nbr2 in subst_atom.GetNeighbors():
            if nbr2.GetIdx() == ring_atom_idx:
                continue
            bond = mol.GetBondBetweenAtoms(subst_atom.GetIdx(), nbr2.GetIdx())
            if nbr2.GetSymbol() == "O":
                if bond and bond.GetBondTypeAsDouble() == 2:
                    has_double_O = True
                    double_O_idx = nbr2.GetIdx()
                else:
                    has_single_O = True
                    single_O_atoms.append(nbr2)

        if has_double_O and has_single_O:
            return "CO2R"  # ester or acid
        if has_double_O and not has_single_O:
            # Check if it's an aldehyde (C=O with H) or ketone (C=O with C/R)
            h_on_carbonyl = subst_atom.GetTotalNumHs()
            if h_on_carbonyl > 0:
                return "CHO"
            return "COR"  # ketone

        # Check for aromatic neighbor (phenyl substituent)
        for nbr2 in subst_atom.GetNeighbors():
            if nbr2.GetIdx() == ring_atom_idx:
                continue
            if nbr2.GetIsAromatic():
                # This carbon is part of an aromatic system
                if subst_atom.GetIsAromatic():
                    return "phenyl"

        # Check for vinyl
        for nbr2 in subst_atom.GetNeighbors():
            if nbr2.GetIdx() == ring_atom_idx:
                continue
            bond = mol.GetBondBetweenAtoms(subst_atom.GetIdx(), nbr2.GetIdx())
            if bond and bond.GetBondTypeAsDouble() == 2 and nbr2.GetSymbol() == "C":
                return "vinyl"

        # Check for C≡C
        for nbr2 in subst_atom.GetNeighbors():
            if nbr2.GetIdx() == ring_atom_idx:
                continue
            bond = mol.GetBondBetweenAtoms(subst_atom.GetIdx(), nbr2.GetIdx())
            if bond and bond.GetBondTypeAsDouble() == 3 and nbr2.GetSymbol() == "C":
                return "CCH"

        # Generic alkyl (CH3, CH2R, CHR2, CR3)
        if subst_atom.GetTotalNumHs() >= 2:
            return "CH3"
        return "alkyl"

    # Silicon
    if sym == "Si":
        return "SiR3"

    return None  # unknown


def identify_ester_R_group(mol, ester_O_single, ring_atoms_set):
    """For an ester -CO2R, identify R (Me, Et, iPr, tBu) from the single-bonded O.

    Returns: Es key string or None
    """
    # Walk from the single-bonded O away from carbonyl
    for nbr in ester_O_single.GetNeighbors():
        if nbr.GetSymbol() == "C" and nbr.GetIdx() not in ring_atoms_set:
            # This is the R group carbon
            r_carbon = nbr
            # Count carbon neighbors of R (excluding O)
            c_neighbors = [n for n in r_carbon.GetNeighbors()
                          if n.GetSymbol() == "C" and n.GetIdx() != ester_O_single.GetIdx()]
            h_count = r_carbon.GetTotalNumHs()

            if h_count == 3 and len(c_neighbors) == 0:
                return "Me"
            elif h_count == 2 and len(c_neighbors) == 1:
                return "Et"
            elif h_count == 1 and len(c_neighbors) == 2:
                return "iPr"
            elif h_count == 0 and len(c_neighbors) == 3:
                return "tBu"
    return None


def compute_empirical_descriptors(smiles):
    """Compute sigma_hammett, Es_taft, delta_ortho, delta_benzyne from SMILES.

    Returns: dict with keys sigma_hammett, Es_taft, delta_ortho, delta_benzyne
             Values are float or NaN.
    """
    result = {
        "sigma_hammett": np.nan,
        "Es_taft": 0.0,
        "delta_ortho": 0,
        "delta_benzyne": 0,
    }

    if not isinstance(smiles, str) or not smiles.strip():
        return {k: np.nan for k in result}

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {k: np.nan for k in result}

    li_idx, c_ipso_idx, ring = find_li_and_ring(mol)

    if ring is None:
        # Not ArLi on a 6-membered carbocyclic ring → skip sigma
        return result

    ring_set = set(ring)

    # ── Analyze each ring position ──
    sigma_total = 0.0
    has_sigma = False
    has_ortho_ester = False
    ortho_ester_R = None
    has_ortho_benzyne_halogen = False

    for ring_atom_idx in ring:
        if ring_atom_idx == c_ipso_idx:
            continue

        position = get_ring_position(ring, c_ipso_idx, ring_atom_idx)
        if position is None:
            continue

        subst_type = classify_substituent(mol, ring_atom_idx, ring_set)

        if subst_type is None or subst_type == "H":
            continue

        if subst_type == "Li":
            continue

        # ── sigma ──
        sigma_vals = SIGMA_TABLE.get(subst_type)
        if sigma_vals:
            s_para, s_meta = sigma_vals
            if s_para is not None:
                if position == "para":
                    sigma_total += s_para
                    has_sigma = True
                elif position == "meta":
                    if s_meta is not None:
                        sigma_total += s_meta
                    else:
                        sigma_total += s_para
                    has_sigma = True
                elif position == "ortho":
                    # Use sigma_para as proxy for ortho (standard practice)
                    sigma_total += s_para
                    has_sigma = True

        # ── delta_ortho: ortho ester only ──
        if position == "ortho" and subst_type == "CO2R":
            has_ortho_ester = True
            # Find the ester's R group for Es
            ring_atom = mol.GetAtomWithIdx(ring_atom_idx)
            for nbr in ring_atom.GetNeighbors():
                if nbr.GetIdx() in ring_set or nbr.GetSymbol() == "Li":
                    continue
                # This is the carbonyl C
                if nbr.GetSymbol() == "C":
                    for nbr2 in nbr.GetNeighbors():
                        if nbr2.GetIdx() == ring_atom_idx:
                            continue
                        bond = mol.GetBondBetweenAtoms(nbr.GetIdx(), nbr2.GetIdx())
                        if nbr2.GetSymbol() == "O" and bond and bond.GetBondTypeAsDouble() == 1:
                            r_key = identify_ester_R_group(mol, nbr2, ring_set)
                            if r_key:
                                ortho_ester_R = r_key

        # ── delta_benzyne: ortho Br or I ──
        if position == "ortho" and subst_type in ("Br", "I"):
            has_ortho_benzyne_halogen = True

    # Assign sigma
    if has_sigma:
        result["sigma_hammett"] = round(sigma_total, 2)
    else:
        # Pure PhLi (no substituents other than H) → sigma = 0.00
        result["sigma_hammett"] = 0.00

    # Assign delta_ortho and delta_benzyne (mutually exclusive)
    if has_ortho_benzyne_halogen:
        result["delta_benzyne"] = 1
        result["delta_ortho"] = 0
    elif has_ortho_ester:
        result["delta_ortho"] = 1
        result["delta_benzyne"] = 0
        if ortho_ester_R and ortho_ester_R in ES_TABLE:
            result["Es_taft"] = ES_TABLE[ortho_ester_R]

    return result


def main():
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} rows")

    # Get unique SMILES
    unique_smiles = df["intermediate_smiles_canonical"].dropna().unique()
    print(f"Unique non-null SMILES: {len(unique_smiles)}")

    # Compute descriptors for each unique SMILES
    smiles_to_desc = {}
    for smi in unique_smiles:
        desc = compute_empirical_descriptors(smi)
        smiles_to_desc[smi] = desc

    # Map back to dataframe
    for col in ["sigma_hammett", "Es_taft", "delta_ortho", "delta_benzyne"]:
        df[col] = df["intermediate_smiles_canonical"].map(
            lambda s: smiles_to_desc.get(s, {}).get(col, np.nan) if isinstance(s, str) else np.nan
        )

    # ── Validation: check known ArLi from LFER model ──
    known_checks = {
        # canonical SMILES: (expected_sigma, expected_Es, expected_d_ortho, expected_d_benzyne)
        "[Li]c1ccc(C#N)cc1":              (0.66, 0.0, 0, 0),   # p-CN-ArLi
        "[Li]c1cccc(C#N)c1":              (0.56, 0.0, 0, 0),   # m-CN-ArLi
        "[Li]c1ccc(C(=O)OC(C)(C)C)cc1":  (0.45, 0.0, 0, 0),   # p-CO2tBu-ArLi
        "[Li]c1ccccc1C(=O)OC(C)(C)C":    (0.45, -1.54, 1, 0),  # o-CO2tBu-ArLi
        "[Li]c1ccccc1C(=O)OC":            (0.45, 0.0, 1, 0),   # o-CO2Me-ArLi
        "[Li]c1ccccc1C(=O)OCC":           (0.45, -0.07, 1, 0),  # o-CO2Et-ArLi
        "[Li]c1ccccc1C(=O)OC(C)C":       (0.45, -0.47, 1, 0),  # o-CO2iPr-ArLi
        "[Li]c1ccccc1Br":                 (0.23, 0.0, 0, 1),   # o-Br-ArLi
        "[Li]c1ccccc1I":                  (0.35, 0.0, 0, 1),   # o-I-ArLi
    }

    print("\n=== Validation against known LFER values ===")
    all_pass = True
    for smi, (exp_s, exp_es, exp_do, exp_db) in known_checks.items():
        desc = smiles_to_desc.get(smi, {})
        s = desc.get("sigma_hammett", np.nan)
        es = desc.get("Es_taft", 0.0)
        do = desc.get("delta_ortho", 0)
        db = desc.get("delta_benzyne", 0)

        ok = True
        issues = []
        if not np.isnan(s) and abs(s - exp_s) > 0.02:
            issues.append(f"σ={s} (exp {exp_s})")
            ok = False
        if abs(es - exp_es) > 0.02:
            issues.append(f"Es={es} (exp {exp_es})")
            ok = False
        if do != exp_do:
            issues.append(f"δ_o={do} (exp {exp_do})")
            ok = False
        if db != exp_db:
            issues.append(f"δ_b={db} (exp {exp_db})")
            ok = False

        status = "✅" if ok else "❌"
        detail = f" — {', '.join(issues)}" if issues else ""
        print(f"  {status} {smi[:50]:50s}{detail}")
        if not ok:
            all_pass = False

    if all_pass:
        print("  All known values match!")

    # ── Print fill rates ──
    print("\n=== Descriptor fill rates ===")
    for col in ["sigma_hammett", "Es_taft", "delta_ortho", "delta_benzyne"]:
        if col in ("Es_taft", "delta_ortho", "delta_benzyne"):
            # These have default 0, count non-NaN
            filled = df[col].notna().sum()
        else:
            filled = df[col].notna().sum()
        print(f"  {col:20s}: {filled}/{len(df)} ({100*filled/len(df):.1f}%)")

    # Show sigma distribution for assigned values
    sigma_vals = df["sigma_hammett"].dropna()
    if len(sigma_vals) > 0:
        print(f"\n=== sigma_hammett distribution (n={len(sigma_vals)}) ===")
        print(f"  min={sigma_vals.min():.2f}, max={sigma_vals.max():.2f}, "
              f"mean={sigma_vals.mean():.2f}")
        # Show unique sigma values and their intermediates
        unique_sigma = df[df["sigma_hammett"].notna()].drop_duplicates(
            "intermediate_smiles_canonical"
        )[["intermediate", "intermediate_smiles_canonical", "sigma_hammett",
           "Es_taft", "delta_ortho", "delta_benzyne"]].sort_values("sigma_hammett")
        print(f"\n=== Unique intermediates with sigma assigned ({len(unique_sigma)}) ===")
        for _, r in unique_sigma.iterrows():
            print(f"  σ={r.sigma_hammett:+.2f}  Es={r.Es_taft:+.2f}  "
                  f"δo={int(r.delta_ortho)}  δb={int(r.delta_benzyne)}  "
                  f"{r.intermediate[:50]}")

    # Save
    df.to_csv(CSV_PATH, index=False)
    print(f"\nSaved to {CSV_PATH}")


if __name__ == "__main__":
    main()
