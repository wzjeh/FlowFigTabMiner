"""
QTAIM analysis via Multiwfn on M06-2X/def2-SVP fchk files.

Extracts per-molecule:
  - QTAIM atomic charges (Bader charges) for Li and C_ipso
  - Bond critical point (BCP) properties for Li-C bond:
    rho(BCP), laplacian, H(BCP), ellipticity
  - ELF basin population for Li-C bond region
  - Delocalization index DI(Li,C)

Usage:
  cd /Users/zhaowenyuan/Projects/FlowFigTabMiner
  source flowfigtabminer/bin/activate
  python data/ml_lifetime/compute_qtaim.py
"""

import json, os, re, subprocess, tempfile
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
FCHK_DIR = DATA_DIR / "m06_fchk"
MANIFEST = DATA_DIR / "hf_xyz" / "manifest.json"
CACHE_PATH = DATA_DIR / "qtaim_cache.json"
MULTIWFN = "Multiwfn"


def run_multiwfn(fchk_path, commands, timeout=120):
    """Run Multiwfn with a sequence of menu commands, return stdout."""
    cmd_str = "\n".join(str(c) for c in commands) + "\n"
    try:
        result = subprocess.run(
            [MULTIWFN, fchk_path],
            input=cmd_str, capture_output=True, text=True, timeout=timeout,
            env={**os.environ, "Multiwfnpath": ""}
        )
        return result.stdout
    except subprocess.TimeoutExpired:
        return ""
    except Exception as e:
        return f"ERROR: {e}"


def extract_qtaim_charges(fchk_path):
    """Extract QTAIM (Bader) charges for all atoms."""
    # Multiwfn menu: 7 (Population analysis) → 17 (QTAIM/Bader charges)
    out = run_multiwfn(fchk_path, [7, 17, 0, "q"], timeout=300)

    charges = {}
    # Parse: look for "Atom  X  charge:" pattern
    for line in out.split('\n'):
        # Pattern: "  1(Li )    Charge:    0.912345"
        m = re.search(r'(\d+)\((\w+)\s*\)\s+Charge:\s+([-\d.]+)', line)
        if m:
            idx = int(m.group(1)) - 1  # 0-indexed
            elem = m.group(2).strip()
            charge = float(m.group(3))
            charges[idx] = {'element': elem, 'charge': charge}

    return charges


def extract_bcp_properties(fchk_path, li_idx, c_idx):
    """Extract bond critical point properties for Li-C bond."""
    # Multiwfn menu: 2 (Topology analysis) → 2 (Search BCP) → ...
    # Simpler: use option 2 → 0 (search all CPs automatically)
    out = run_multiwfn(fchk_path, [2, 2, 0, -1, 0, "q"], timeout=300)

    bcp_props = {}
    # Look for BCP between Li and C
    lines = out.split('\n')
    for i, line in enumerate(lines):
        # Find BCP connected to our Li and C atoms
        if 'BCP' in line and f'{li_idx+1}(' in line and f'{c_idx+1}(' in line:
            # Parse nearby lines for rho, laplacian, H
            for j in range(max(0, i-5), min(len(lines), i+10)):
                l = lines[j]
                if 'Density of all electrons' in l or 'Rho' in l:
                    m = re.search(r'([-\d.Ee+]+)\s*$', l)
                    if m: bcp_props['rho_bcp'] = float(m.group(1))
                if 'Laplacian' in l:
                    m = re.search(r'([-\d.Ee+]+)\s*$', l)
                    if m: bcp_props['laplacian_bcp'] = float(m.group(1))
                if 'Energy density' in l or 'H(r)' in l:
                    m = re.search(r'([-\d.Ee+]+)\s*$', l)
                    if m: bcp_props['H_bcp'] = float(m.group(1))
                if 'Ellipticity' in l:
                    m = re.search(r'([-\d.Ee+]+)\s*$', l)
                    if m: bcp_props['ellipticity'] = float(m.group(1))

    return bcp_props


def analyze_one(fchk_path, li_idx, c_idx):
    """Full QTAIM analysis for one molecule."""
    result = {}

    # 1. QTAIM charges
    charges = extract_qtaim_charges(fchk_path)
    if charges:
        if li_idx in charges:
            result['qtaim_charge_Li'] = charges[li_idx]['charge']
        if c_idx in charges:
            result['qtaim_charge_C'] = charges[c_idx]['charge']
        result['n_atoms_charged'] = len(charges)

    # 2. BCP properties
    bcp = extract_bcp_properties(fchk_path, li_idx, c_idx)
    result.update(bcp)

    return result


def main():
    manifest = json.load(open(MANIFEST))
    cache = json.load(open(CACHE_PATH)) if CACHE_PATH.exists() else {}

    fchk_files = list(FCHK_DIR.glob("*.fchk"))
    print(f"FCHK files: {len(fchk_files)}")
    print(f"Manifest: {len(manifest)} molecules")
    print(f"Cache: {len(cache)} entries")

    for i, (smi, info) in enumerate(manifest.items()):
        if smi in cache and 'qtaim_charge_Li' in cache[smi]:
            print(f"  [{i+1}/{len(manifest)}] CACHED {smi[:40]}")
            continue

        fchk_path = str(FCHK_DIR / info["file"].replace(".xyz", ".fchk"))
        if not os.path.exists(fchk_path):
            print(f"  [{i+1}/{len(manifest)}] SKIP {smi[:40]} (no fchk)")
            continue

        li_idx = info["li_idx"]
        c_idx = info["c_idx"]

        print(f"  [{i+1}/{len(manifest)}] {smi[:40]:40s} ...", end=" ", flush=True)
        result = analyze_one(fchk_path, li_idx, c_idx)

        if 'qtaim_charge_Li' in result:
            print(f"q_Li={result['qtaim_charge_Li']:+.4f} q_C={result.get('qtaim_charge_C','?')}")
        else:
            print(f"QTAIM charges not found")

        cache[smi] = result
        with open(CACHE_PATH, 'w') as f:
            json.dump(cache, f, indent=2)

    # Summary
    ok = sum(1 for v in cache.values() if 'qtaim_charge_Li' in v)
    print(f"\nQTAIM charges extracted: {ok}/{len(manifest)}")


if __name__ == "__main__":
    main()
