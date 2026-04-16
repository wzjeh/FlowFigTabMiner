"""
Run HF/6-31+G* on pre-generated xyz files. No RDKit dependency.

  conda run -n base python data/ml_lifetime/run_psi4_hf.py
"""
import json, os, sys, time, glob
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
XYZ_DIR = DATA_DIR / "hf_xyz"
CACHE_PATH = DATA_DIR / "hf_charge_cache.json"

def run_one(xyz_path, li_idx, c_idx):
    import psi4
    psi4.core.set_output_file("/dev/null")
    psi4.set_memory("2 GB")
    psi4.set_num_threads(4)

    with open(xyz_path) as f:
        geom_str = f.read()

    try:
        mol = psi4.geometry(geom_str)
        e, wfn = psi4.energy("hf/6-31+g*", molecule=mol, return_wfn=True)

        # Mulliken
        psi4.oeprop(wfn, "MULLIKEN_CHARGES")
        mulliken = np.array([wfn.atomic_point_charges().np[i] for i in range(mol.natom())])

        # Lowdin
        psi4.oeprop(wfn, "LOWDIN_CHARGES")
        lowdin = np.array([wfn.atomic_point_charges().np[i] for i in range(mol.natom())])

        # Orbitals
        eps = wfn.epsilon_a().np
        occ = wfn.occupation_a().np
        homo_i = np.where(occ > 0.5)[0][-1]
        homo = float(eps[homo_i] * 27.211)
        lumo = float(eps[homo_i + 1] * 27.211)

        psi4.core.clean()

        return {
            "energy": float(e),
            "mulliken_Li": float(mulliken[li_idx]) if li_idx is not None else None,
            "mulliken_C": float(mulliken[c_idx]) if c_idx is not None else None,
            "lowdin_Li": float(lowdin[li_idx]) if li_idx is not None else None,
            "lowdin_C": float(lowdin[c_idx]) if c_idx is not None else None,
            "HOMO_eV": homo,
            "LUMO_eV": lumo,
            "gap_eV": lumo - homo,
            "mulliken_all": [float(x) for x in mulliken],
            "lowdin_all": [float(x) for x in lowdin],
        }
    except Exception as ex:
        psi4.core.clean()
        return {"error": str(ex)}


def main():
    manifest = json.load(open(XYZ_DIR / "manifest.json"))
    print(f"Molecules: {len(manifest)}")

    cache = {}
    if CACHE_PATH.exists():
        cache = json.load(open(CACHE_PATH))
        print(f"Cache: {len(cache)} entries")

    for i, (smi, info) in enumerate(manifest.items()):
        if smi in cache and "error" not in cache[smi]:
            print(f"  [{i+1}/{len(manifest)}] {smi[:45]:45s} CACHED")
            continue

        xyz_path = XYZ_DIR / info["file"]
        li_idx = info["li_idx"]
        c_idx = info["c_idx"]

        t0 = time.time()
        result = run_one(str(xyz_path), li_idx, c_idx)
        dt = time.time() - t0

        if "error" in result:
            print(f"  [{i+1}/{len(manifest)}] {smi[:45]:45s} ERROR: {result['error'][:60]} ({dt:.0f}s)")
        else:
            qc = result["mulliken_C"]
            ql = result["mulliken_Li"]
            homo = result["HOMO_eV"]
            print(f"  [{i+1}/{len(manifest)}] {smi[:45]:45s} q_C={qc:+.4f} q_Li={ql:+.4f} HOMO={homo:.2f}eV ({dt:.0f}s)")

        cache[smi] = result
        with open(CACHE_PATH, "w") as f:
            json.dump(cache, f, indent=2)

    # Summary
    ok = sum(1 for v in cache.values() if "error" not in v)
    print(f"\nDone: {ok}/{len(manifest)} successful")


if __name__ == "__main__":
    main()
