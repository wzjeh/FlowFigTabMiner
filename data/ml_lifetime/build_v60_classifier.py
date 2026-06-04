"""
v6.0 Phase 1: Mechanism-based classifier for organolithium intermediates.

Classes (per Zhao 2026-05-24):
  C1  inert / proto-de-Li     no EWG, no ortho special group
  C2  remote-EWG              p/m-EWG (CN, NO2, CO2R, CF3), no ortho special
  C3  ortho-chelation         o-CO2R, o-OR, o-NR2 (Li...X coordination)
  C4  ortho-5-exo             o-CN, o-NO2, o-CHO, o-C(=O)R (direct attack)
  C5  ortho-benzyne           o-Br, o-I (1,2-elimination)
  OXI oxiranylLi              3-membered ring containing C-Li
  BENZYL sp3 Li-CH2-Ar        treated separately (not in v6.0 main fit)
  CARBENOID sp3 Li-CHX        ditto
  OTHER                       fallback (e.g., heteroaryl with N in ring)

Only n>=5 classes get full Bayesian fit (Phase 3). C4/C5 use conservative defaults.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem

BASE = Path(__file__).parent

# ============================================================
# Hammett σ table — covers all substituents in training set
# Source: Hansch, Leo & Taft 1991 (consolidated in UNIFIED_MODEL_ANALYSIS.md)
# ============================================================
SIGMA_TABLE = {
    # EWG (σ > 0, ordered descending)
    'C#N':            {'sigma_p': 0.66, 'sigma_m': 0.56, 'type': 'EWG_CN'},
    '[N+](=O)[O-]':   {'sigma_p': 0.78, 'sigma_m': 0.71, 'type': 'EWG_NO2'},
    'C(F)(F)F':       {'sigma_p': 0.54, 'sigma_m': 0.43, 'type': 'EWG_CF3'},
    'C(=O)OC':        {'sigma_p': 0.45, 'sigma_m': 0.37, 'type': 'EWG_ester'},
    'C(=O)OCC':       {'sigma_p': 0.45, 'sigma_m': 0.37, 'type': 'EWG_ester'},
    'C(=O)OC(C)C':    {'sigma_p': 0.45, 'sigma_m': 0.37, 'type': 'EWG_ester'},
    'C(=O)OC(C)(C)C': {'sigma_p': 0.45, 'sigma_m': 0.37, 'type': 'EWG_ester'},
    'C(=O)C':         {'sigma_p': 0.50, 'sigma_m': 0.38, 'type': 'EWG_ketone'},
    'C(=O)H':         {'sigma_p': 0.42, 'sigma_m': 0.35, 'type': 'EWG_CHO'},
    'C(=O)CC':        {'sigma_p': 0.50, 'sigma_m': 0.38, 'type': 'EWG_ketone'},
    'I':              {'sigma_p': 0.18, 'sigma_m': 0.35, 'type': 'halide_heavy'},
    'Br':             {'sigma_p': 0.23, 'sigma_m': 0.39, 'type': 'halide_heavy'},
    # Weak / inert
    'Cl':             {'sigma_p': 0.23, 'sigma_m': 0.37, 'type': 'halide_inert'},
    'F':              {'sigma_p': 0.06, 'sigma_m': 0.34, 'type': 'halide_inert'},
    'OC':             {'sigma_p': -0.27, 'sigma_m': 0.12, 'type': 'EDG_OMe'},
    'OCC':            {'sigma_p': -0.24, 'sigma_m': 0.10, 'type': 'EDG_OR'},
    'C':              {'sigma_p': -0.17, 'sigma_m': -0.07, 'type': 'EDG_alkyl'},
    'CC':             {'sigma_p': -0.15, 'sigma_m': -0.07, 'type': 'EDG_alkyl'},
    # phenyl/biphenyl (treat as inert)
    'c1ccccc1':       {'sigma_p': -0.01, 'sigma_m': 0.06, 'type': 'inert_aryl'},
}

EWG_THRESHOLD = 0.20   # |σ_p| > 0.20 → counts as remote-EWG for C2 classification


def get_substituent_smarts(mol, atom_idx, parent_idx):
    """Get the SMARTS / SMILES fragment of substituent rooted at atom_idx,
    where parent_idx is the aromatic ring atom it's attached to."""
    # Traverse from atom_idx outward, excluding parent_idx
    visited = {parent_idx}
    queue = [atom_idx]
    frag_atoms = []
    while queue:
        i = queue.pop(0)
        if i in visited: continue
        visited.add(i)
        frag_atoms.append(i)
        a = mol.GetAtomWithIdx(i)
        for n in a.GetNeighbors():
            if n.GetIdx() not in visited and n.GetIdx() != parent_idx:
                queue.append(n.GetIdx())
    # Build SMILES fragment via rdkit
    try:
        emol = Chem.RWMol(mol)
        # Get a SMILES of just these atoms
        return Chem.MolFragmentToSmiles(mol, frag_atoms, canonical=False, allHsExplicit=False)
    except Exception:
        return None


def match_substituent(frag_smiles):
    """Return σ table entry for substituent fragment, or None."""
    if frag_smiles is None:
        return None
    # Direct match
    if frag_smiles in SIGMA_TABLE:
        return SIGMA_TABLE[frag_smiles]
    # Normalize: try alternate forms
    # Single atom cases
    if frag_smiles in ('F', 'Cl', 'Br', 'I'):
        return SIGMA_TABLE.get(frag_smiles)
    # Try with canonical RDKit
    try:
        mol = Chem.MolFromSmiles(frag_smiles)
        if mol is None:
            return None
        canonical = Chem.MolToSmiles(mol, canonical=True)
        for key, val in SIGMA_TABLE.items():
            try:
                ref = Chem.MolToSmiles(Chem.MolFromSmiles(key), canonical=True)
                if ref == canonical:
                    return val
            except Exception:
                pass
    except Exception:
        pass
    return None


def find_li(mol):
    for a in mol.GetAtoms():
        if a.GetSymbol() == 'Li':
            return a
    return None


def find_li_c(mol):
    """Find the C atom directly bonded to Li."""
    li = find_li(mol)
    if li is None: return None, None
    for n in li.GetNeighbors():
        if n.GetSymbol() == 'C':
            return li, n
    return li, None


def classify(smi):
    """Return (class, info_dict).

    info_dict contains:
      - ortho_substs: list of {pos, frag, sigma_p, sigma_m, type}
      - remote_substs: list of same with pos in {meta, para}
      - sigma_p_sum, sigma_m_sum
      - is_aromatic_LiC: bool
      - has_3ring: bool
      - notes: list of free-form notes
    """
    info = {'ortho': [], 'remote': [], 'sigma_p_sum': 0.0, 'sigma_m_sum': 0.0,
            'notes': [], 'is_aromatic': False, 'has_3ring': False}

    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return 'INVALID', info

    li, c_ipso = find_li_c(mol)
    if li is None:
        return 'NO_LI', info
    if c_ipso is None:
        return 'NO_LIC', info

    info['is_aromatic'] = c_ipso.GetIsAromatic()

    # OXI: c_ipso in a 3-member ring
    ri = mol.GetRingInfo()
    for r in ri.AtomRings():
        if c_ipso.GetIdx() in r and len(r) == 3:
            info['has_3ring'] = True
            return 'OXI', info

    # Non-aromatic c_ipso: BENZYL or CARBENOID or other
    if not c_ipso.GetIsAromatic():
        halo_count = sum(1 for n in c_ipso.GetNeighbors()
                         if n.GetSymbol() in ('F', 'Cl', 'Br', 'I'))
        if halo_count > 0:
            return 'CARBENOID', info
        aryl_neighbor = any(n.GetIsAromatic() for n in c_ipso.GetNeighbors())
        if aryl_neighbor:
            return 'BENZYL', info
        return 'OTHER', info

    # Aromatic ArLi: find the ring containing c_ipso
    parent_ring = None
    for r in ri.AtomRings():
        if c_ipso.GetIdx() in r and len(r) == 6:
            parent_ring = list(r)
            break
    if parent_ring is None:
        # Non-6 ring (5-ring heteroaryl): treat as OTHER for now
        # Check if 5-ring with heteroatom (thiophene/furan/pyrrole) → OTHER class
        for r in ri.AtomRings():
            if c_ipso.GetIdx() in r and len(r) == 5:
                return 'OTHER', info
        return 'OTHER', info

    # Determine positions: ortho = adjacent in ring, meta = 2 away, para = 3 away (opposite)
    # Build ring graph
    ipso_idx = c_ipso.GetIdx()
    ring_set = set(parent_ring)

    # BFS distance from ipso
    dist = {ipso_idx: 0}
    queue = [ipso_idx]
    while queue:
        x = queue.pop(0)
        ax = mol.GetAtomWithIdx(x)
        for nb in ax.GetNeighbors():
            ni = nb.GetIdx()
            if ni in ring_set and ni not in dist:
                dist[ni] = dist[x] + 1
                queue.append(ni)

    # Heteroatoms in ring (other than C and the ipso C)?
    has_ring_heteroatom = any(
        mol.GetAtomWithIdx(i).GetSymbol() != 'C' for i in parent_ring
    )
    if has_ring_heteroatom:
        info['notes'].append('heteroaryl_ring')
        # Heteroaryl ArLi (e.g. pyridyl) — keep as OTHER unless we explicitly fit
        return 'OTHER', info

    # Look at each ring atom (except ipso) for substituents
    for ridx in parent_ring:
        if ridx == ipso_idx: continue
        d = dist[ridx]
        if d == 1: pos = 'ortho'
        elif d == 2: pos = 'meta'
        elif d == 3: pos = 'para'
        else: continue
        atom = mol.GetAtomWithIdx(ridx)
        for nb in atom.GetNeighbors():
            if nb.GetIdx() in ring_set: continue
            if nb.GetSymbol() == 'H': continue
            frag = get_substituent_smarts(mol, nb.GetIdx(), ridx)
            entry = match_substituent(frag) if frag else None
            sub_info = {
                'pos': pos, 'frag': frag,
                'sigma_p': entry['sigma_p'] if entry else None,
                'sigma_m': entry['sigma_m'] if entry else None,
                'type': entry['type'] if entry else 'unknown',
            }
            if pos == 'ortho':
                info['ortho'].append(sub_info)
            else:
                info['remote'].append(sub_info)
            if entry:
                # Add sigma to sum
                if pos == 'para':
                    info['sigma_p_sum'] += entry['sigma_p']
                elif pos == 'meta':
                    info['sigma_m_sum'] += entry['sigma_m']

    # ---- Apply class rules in priority order ----

    # C5: ortho benzyne (Br or I)
    for o in info['ortho']:
        if o['type'] == 'halide_heavy':
            return 'C5', info

    # C4: ortho 5-exo (CN, NO2, CHO, ketone)
    for o in info['ortho']:
        if o['type'] in ('EWG_CN', 'EWG_NO2', 'EWG_CHO', 'EWG_ketone'):
            return 'C4', info

    # C3: ortho chelation (ester, ether, amine)
    for o in info['ortho']:
        if o['type'] in ('EWG_ester', 'EDG_OMe', 'EDG_OR'):
            return 'C3', info

    # Check remote EWG for C2
    has_remote_EWG = any(
        r['sigma_p'] is not None and abs(r['sigma_p']) > EWG_THRESHOLD
        and 'EWG' in (r['type'] or '')
        for r in info['remote']
    )
    if has_remote_EWG:
        return 'C2', info

    return 'C1', info


def main():
    # ---- Load all substrates ----
    ga = pd.read_csv(BASE / 'global_arrhenius.csv')
    rows = []
    for _, r in ga.iterrows():
        smi = r['smi']
        cls, info = classify(smi)
        rows.append({
            'intermediate': r['intermediate'],
            'smi': smi,
            'class_v60': cls,
            'is_aromatic': info['is_aromatic'],
            'sigma_p_sum': round(info['sigma_p_sum'], 3),
            'sigma_m_sum': round(info['sigma_m_sum'], 3),
            'ortho_types': ';'.join(o['type'] for o in info['ortho']) if info['ortho'] else '',
            'remote_types': ';'.join(r['type'] for r in info['remote']) if info['remote'] else '',
            'notes': ';'.join(info['notes']),
            'Ea_f': r['Ea_f'], 'lnA_f': r['lnA_f'],
            'Ea_d': r['Ea_d'], 'lnA_d': r['lnA_d'], 'y_max': r['y_max'],
            'r2_global': r['r2_global'], 'n_temps': r['n_temps'],
        })

    df = pd.DataFrame(rows)

    # ---- Class distribution ----
    print("=" * 80)
    print("v6.0 Phase 1 — Mechanism classification of global_arrhenius substrates")
    print("=" * 80)
    print("\nClass distribution:")
    print(df['class_v60'].value_counts().to_string())

    print("\n--- Per class detail ---")
    for c in ['C1', 'C2', 'C3', 'C4', 'C5', 'OXI', 'BENZYL', 'CARBENOID', 'OTHER']:
        sub = df[df['class_v60'] == c]
        if len(sub) == 0: continue
        print(f"\n[{c}] n={len(sub)}:")
        for _, r in sub.iterrows():
            print(f"  {r['intermediate'][:55]:55} | σp={r['sigma_p_sum']:+.2f} σm={r['sigma_m_sum']:+.2f}"
                  f" | ortho={r['ortho_types'][:20]:20} | remote={r['remote_types'][:25]:25}")

    # Save
    out = BASE / 'v60_classified_substrates.csv'
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}")

    # ---- Classify validation substrates (4-Br-FC6H4, 5-Br-2-F-CN) ----
    print("\n" + "=" * 80)
    print("Validation substrates (blind test)")
    print("=" * 80)
    for name, smi in [
        ('4-Br-FC6H4 → p-F-PhLi (from 4-bromo-fluorobenzene)', '[Li]c1ccc(F)cc1'),
        ('5-Br-2-F-CN → m-CN-p-F-PhLi (from 5-bromo-2-fluorobenzonitrile)',
         '[Li]c1cc(C#N)c(F)cc1'),
        ('3-CN-PhLi (training set anchor)', '[Li]c1cccc(C#N)c1'),
        ('3,5-Br2-CN dual-EWG (out-of-domain test)', '[Li]c1cc(Br)cc(C#N)c1'),
    ]:
        cls, info = classify(smi)
        print(f"\n  {name}")
        print(f"    SMI: {smi}")
        print(f"    class={cls}  σp={info['sigma_p_sum']:+.2f}  σm={info['sigma_m_sum']:+.2f}")
        print(f"    ortho: {[o['type'] for o in info['ortho']]}")
        print(f"    remote: {[r['type'] for r in info['remote']]}")


if __name__ == '__main__':
    main()
