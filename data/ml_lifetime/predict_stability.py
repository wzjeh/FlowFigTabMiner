"""
Organolithium Stability Prediction Tool
========================================

Input:  SMILES + solvent + temperature
Output: predicted Ea, t½, reactor recommendation

Three-layer model:
  Layer 1: Structure → Ea_THF (3-param DFT: q_C_ipso + d_LiC + %Vbur)
  Layer 2: Solvent flag (THF-based validated; Et2O/DME as literature reference)
  Layer 3: Ea → lnA (per-class compensation) → t½

Usage:
  cd /Users/zhaowenyuan/Projects/FlowFigTabMiner
  source flowfigtabminer/bin/activate
  python data/ml_lifetime/predict_stability.py '[Li]c1ccc(F)cc1' THF -40
  python data/ml_lifetime/predict_stability.py --batch data/ml_lifetime/predict_input.csv
"""

import sys, os, json, warnings, numpy as np
import pandas as pd
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
from morfeus import Sterimol, BuriedVolume
from tblite.interface import Calculator

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).resolve().parent
R = 8.314e-3  # kJ/(mol·K)
BOHR = 1.8897259886

# ══════════════════════════════════════════════════
# Model coefficients (trained on 26 Tier 1 substrates)
# ══════════════════════════════════════════════════
EA_COEFS = {
    'dft_charge_C_ipso': 152.59,
    'dft_LiC_bond_A': -337.98,
    'buried_vol_Li': 73.08,
    'intercept': 700.58,
}

LNA_COMPENSATION = {
    'ArLi':        {'slope': 0.6799, 'intercept': -7.93},
    'oxiranylLi':  {'slope': 0.5483, 'intercept': -2.85},
    'benzylLi':    {'slope': 0.5689, 'intercept': -3.62},
    'default':     {'slope': 0.5862, 'intercept': -4.10},
}

SOLVENT_NOTES = {
    'THF': 'Validated. Model trained on THF-based data.',
    'Et2O': 'Literature suggests Et2O stabilizes n-BuLi (Ea -12 kJ/mol vs THF) but not t-BuLi. Use THF Ea as upper bound.',
    'DME': 'DME is aggressive: t-BuLi Ea drops ~30 kJ/mol vs THF. Expect much shorter t½.',
    'THF+TMEDA': 'TMEDA accelerates decomposition: n-BuLi Ea drops ~21 kJ/mol vs THF.',
    '2-MeTHF': 'Similar to THF. Slightly more stable (literature). Use THF Ea as approximation.',
    'hydrocarbon': 'Very stable in hydrocarbons. Ea >> THF value (n-BuLi: +43 kJ/mol).',
}


# ══════════════════════════════════════════════════
# Descriptor computation
# ══════════════════════════════════════════════════

class _SuppressStdout:
    def __enter__(self):
        self._fd = os.dup(1)
        self._devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(self._devnull, 1)
    def __exit__(self, *a):
        os.dup2(self._fd, 1)
        os.close(self._fd)
        os.close(self._devnull)


ELEM = {"H":1,"Li":3,"B":5,"C":6,"N":7,"O":8,"F":9,"Si":14,"P":15,"S":16,"Cl":17,"Br":35,"I":53}


def compute_descriptors(smiles):
    """Compute q(C_ipso), d(Li-C), %Vbur from SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, "Invalid SMILES"

    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    if AllChem.EmbedMolecule(mol, params) != 0:
        return None, "3D embedding failed"
    AllChem.UFFOptimizeMolecule(mol, maxIters=1000)

    # Find Li and C_ipso
    li_idx, c_idx = None, None
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == 'Li':
            li_idx = atom.GetIdx()
            for nbr in atom.GetNeighbors():
                if nbr.GetSymbol() == 'C':
                    c_idx = nbr.GetIdx()
                    break
            break

    if li_idx is None or c_idx is None:
        return None, "No Li-C bond found"

    n_atoms = mol.GetNumAtoms()
    symbols = [mol.GetAtomWithIdx(i).GetSymbol() for i in range(n_atoms)]
    coords = mol.GetConformer().GetPositions()

    # GFN2-xTB optimization + charges
    numbers = np.array([ELEM.get(s, 0) for s in symbols])
    if 0 in numbers:
        return None, f"Unsupported element in {smiles}"

    from scipy.optimize import minimize as sp_minimize
    pos_bohr = coords * BOHR

    def energy_grad(flat):
        p = flat.reshape(-1, 3)
        with _SuppressStdout():
            calc = Calculator("GFN2-xTB", numbers, p)
            calc.set("verbosity", 0)
            res = calc.singlepoint()
        return res.get("energy"), res.get("gradient").flatten()

    cache = [None, None, None]
    def en(x):
        if cache[0] is None or not np.array_equal(x, cache[0]):
            e, g = energy_grad(x)
            cache[0] = x.copy(); cache[1] = e; cache[2] = g
        return cache[1]
    def gr(x):
        if cache[0] is None or not np.array_equal(x, cache[0]):
            e, g = energy_grad(x)
            cache[0] = x.copy(); cache[1] = e; cache[2] = g
        return cache[2]

    result = sp_minimize(en, pos_bohr.flatten(), jac=gr, method='L-BFGS-B',
                        options={'maxiter': 200, 'gtol': 5e-4})
    opt_pos_bohr = result.x.reshape(-1, 3)

    # Final single-point for charges
    with _SuppressStdout():
        calc = Calculator("GFN2-xTB", numbers, opt_pos_bohr)
        calc.set("verbosity", 0)
        res = calc.singlepoint()

    charges = res.get("charges")
    if charges is None:
        charges = np.zeros(n_atoms)
    q_C_ipso = charges[c_idx]

    opt_pos_ang = opt_pos_bohr / BOHR
    d_LiC = np.linalg.norm(opt_pos_ang[li_idx] - opt_pos_ang[c_idx])

    # Buried volume (from xTB optimized geometry)
    try:
        bv = BuriedVolume(symbols, opt_pos_ang, li_idx + 1, radius=3.5)
        vbur = bv.fraction_buried_volume
    except:
        vbur = 0.25  # fallback

    # Classify intermediate
    li_on_aromatic = mol.GetAtomWithIdx(c_idx).GetIsAromatic() if c_idx is not None else False
    has_oxirane = any(
        len(ring) == 3 and 'O' in {mol.GetAtomWithIdx(i).GetSymbol() for i in ring}
        for ring in mol.GetRingInfo().AtomRings()
    )
    is_benzylic = not li_on_aromatic and any(
        nbr.GetIsAromatic() for nbr in mol.GetAtomWithIdx(c_idx).GetNeighbors() if nbr.GetIdx() != li_idx
    )

    if has_oxirane:
        cls = 'oxiranylLi'
    elif li_on_aromatic:
        cls = 'ArLi'
    elif is_benzylic:
        cls = 'benzylLi'
    else:
        cls = 'other'

    return {
        'smiles': smiles,
        'q_C_ipso': round(q_C_ipso, 4),
        'd_LiC': round(d_LiC, 4),
        'Vbur': round(vbur, 4),
        'class': cls,
    }, None


# ══════════════════════════════════════════════════
# Prediction
# ══════════════════════════════════════════════════

def predict(smiles, solvent='THF', T_C=-40):
    """Full prediction pipeline."""
    desc, err = compute_descriptors(smiles)
    if desc is None:
        return {'error': err}

    # Layer 1: Ea prediction
    Ea = (EA_COEFS['dft_charge_C_ipso'] * desc['q_C_ipso'] +
          EA_COEFS['dft_LiC_bond_A'] * desc['d_LiC'] +
          EA_COEFS['buried_vol_Li'] * desc['Vbur'] +
          EA_COEFS['intercept'])

    # Layer 2: Solvent flag
    solvent_note = SOLVENT_NOTES.get(solvent, f'Unknown solvent "{solvent}". Using THF model (may be inaccurate).')

    # Layer 3: per-class compensation → lnA → t½
    cls = desc['class']
    comp = LNA_COMPENSATION.get(cls, LNA_COMPENSATION['default'])
    lnA = comp['slope'] * Ea + comp['intercept']

    T_K = T_C + 273.15
    kd = np.exp(lnA - Ea / (R * T_K))
    t_half = np.log(2) / kd

    # Reactor recommendation
    if t_half < 1:
        reactor = 'FLASH chemistry (<1s)'
    elif t_half < 60:
        reactor = 'FLOW microreactor (1-60s)'
    elif t_half < 3600:
        reactor = 'FLOW or BATCH (1-60 min)'
    else:
        reactor = 'BATCH compatible (>1h)'

    return {
        'smiles': smiles,
        'intermediate_class': cls,
        'descriptors': desc,
        'Ea_THF_kJ_mol': round(Ea, 1),
        'lnA': round(lnA, 2),
        'solvent': solvent,
        'solvent_note': solvent_note,
        'T_C': T_C,
        't_half_s': round(t_half, 3),
        't_half_readable': format_time(t_half),
        'reactor_recommendation': reactor,
        'model_info': 'LOO-R²(Ea)=0.69, n=26, THF-based. t½ uncertainty ~10x.',
    }


def format_time(t):
    if t < 0.001: return f"{t*1e6:.0f} μs"
    if t < 1: return f"{t*1000:.0f} ms"
    if t < 60: return f"{t:.1f} s"
    if t < 3600: return f"{t/60:.1f} min"
    if t < 86400: return f"{t/3600:.1f} h"
    return f"{t/86400:.1f} d"


# ══════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nExamples:")
        print("  python predict_stability.py '[Li]c1ccc(F)cc1' THF -40")
        print("  python predict_stability.py '[Li]c1ccccc1I' THF -78")
        print("  python predict_stability.py '[Li]C1CO1' THF 0")
        return

    if sys.argv[1] == '--batch':
        # Batch mode
        df = pd.read_csv(sys.argv[2])
        results = []
        for _, row in df.iterrows():
            smi = row.get('smiles', row.get('SMILES'))
            solvent = row.get('solvent', 'THF')
            T = row.get('T_C', -40)
            r = predict(smi, solvent, T)
            results.append(r)
            status = f"Ea={r['Ea_THF_kJ_mol']} kJ/mol, t½={r['t_half_readable']}" if 'error' not in r else r['error']
            print(f"  {smi[:40]:40s} → {status}")
        return

    # Single prediction
    smiles = sys.argv[1]
    solvent = sys.argv[2] if len(sys.argv) > 2 else 'THF'
    T_C = float(sys.argv[3]) if len(sys.argv) > 3 else -40

    print(f"\n{'='*60}")
    print(f"  Organolithium Stability Prediction")
    print(f"{'='*60}")
    print(f"  Input:    {smiles}")
    print(f"  Solvent:  {solvent}")
    print(f"  Temp:     {T_C}°C")

    r = predict(smiles, solvent, T_C)

    if 'error' in r:
        print(f"\n  ERROR: {r['error']}")
        return

    print(f"\n  ── Descriptors ──")
    print(f"  Class:         {r['intermediate_class']}")
    print(f"  q(C_ipso):     {r['descriptors']['q_C_ipso']}")
    print(f"  d(Li-C):       {r['descriptors']['d_LiC']} Å")
    print(f"  %Vbur(Li):     {r['descriptors']['Vbur']}")

    print(f"\n  ── Prediction ──")
    print(f"  Ea (THF):      {r['Ea_THF_kJ_mol']} kJ/mol")
    print(f"  ln(A):         {r['lnA']}")
    print(f"  t½ @ {T_C}°C:   {r['t_half_readable']}")
    print(f"  Reactor:       {r['reactor_recommendation']}")

    print(f"\n  ── Solvent ──")
    print(f"  {r['solvent_note']}")

    print(f"\n  ── Caveats ──")
    print(f"  {r['model_info']}")
    print(f"{'='*60}")

    # Multi-temperature
    print(f"\n  t½ at other temperatures:")
    for T in [-78, -40, 0, 25]:
        T_K = T + 273.15
        kd = np.exp(r['lnA'] - r['Ea_THF_kJ_mol'] / (R * T_K))
        t = np.log(2) / kd
        marker = ' ←' if T == T_C else ''
        print(f"    {T:+4d}°C: {format_time(t):>10s}{marker}")


if __name__ == "__main__":
    main()
