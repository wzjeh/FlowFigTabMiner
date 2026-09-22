"""Is a structure reading a plausible organic molecule?

MolNexTR's misreads have a signature: an element no flow-chemistry paper draws
("NC" -> [Na] / [Ni], "S" -> [SH]([Mo])), or a real molecule dragging shards
("CBr.[I-].c1ccc(-c2ccccc2)cc1").  Both are decidable without the image and
mark the readings worth a second reader.
"""
from __future__ import annotations

from typing import Optional

ORGANIC_ELEMENTS = {"C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "B", "Si", "Sn", "Li", "Mg", "Zn",
                    "Cu", "Pd", "Al", "Ti", "Fe", "Se", "H", "*"}


def suspicious_smiles(smiles: Optional[str]) -> Optional[str]:
    """None for a plausible reading, else why it is suspect:
    ``"invalid"`` (RDKit rejects it), ``"odd:Mo,W"`` (elements outside
    ORGANIC_ELEMENTS), ``"shards"`` (one molecule of >= 6 heavy atoms plus
    fragments of <= 3).  A salt ([Na+].[Cl-]) is odd, not shards; a Markush
    core with ``*`` is plausible."""
    if not smiles or not str(smiles).strip():
        return "invalid"
    try:
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog("rdApp.*")
    except Exception:  # pragma: no cover
        return None
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return "invalid"
    odd = sorted({a.GetSymbol() for a in mol.GetAtoms()} - ORGANIC_ELEMENTS)
    if odd:
        return "odd:" + ",".join(odd)
    frags = str(smiles).split(".")
    if len(frags) > 1:
        heavy = sorted((Chem.MolFromSmiles(f).GetNumHeavyAtoms() if Chem.MolFromSmiles(f) else 0) for f in frags)
        if heavy[-2] <= 3 and heavy[-1] >= 6:
            return "shards"
    return None
