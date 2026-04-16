"""
M06-2X/def2-SVP single-point on xTB-optimized geometries.
Best DFT functional for organolithium (Ramachandran 2010).

  conda run -n base python data/ml_lifetime/run_m06_def2svp.py
"""
import json, os, sys, time
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
XYZ_DIR = DATA_DIR / "hf_xyz"  # reuse existing xyz files
CACHE_PATH = DATA_DIR / "m06_def2svp_cache.json"

def run_one(xyz_path, li_idx, c_idx):
    import psi4
    psi4.core.set_output_file("/dev/null")
    psi4.set_memory("2 GB")
    psi4.set_num_threads(4)

    with open(xyz_path) as f:
        geom_str = f.read()

    try:
        mol = psi4.geometry(geom_str)
        e, wfn = psi4.energy("m06-2x/def2-svp", molecule=mol, return_wfn=True)

        psi4.oeprop(wfn, "MULLIKEN_CHARGES")
        mulliken = np.array([wfn.atomic_point_charges().np[i] for i in range(mol.natom())])

        psi4.oeprop(wfn, "LOWDIN_CHARGES")
        lowdin = np.array([wfn.atomic_point_charges().np[i] for i in range(mol.natom())])

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
            "HOMO_eV": homo, "LUMO_eV": lumo, "gap_eV": lumo - homo,
        }
    except Exception as ex:
        psi4.core.clean()
        return {"error": str(ex)}

def main():
    manifest = json.load(open(XYZ_DIR / "manifest.json"))
    cache = json.load(open(CACHE_PATH)) if CACHE_PATH.exists() else {}
    print(f"Molecules: {len(manifest)}, Cached: {len(cache)}")

    for i, (smi, info) in enumerate(manifest.items()):
        if smi in cache and "error" not in cache[smi]:
            print(f"  [{i+1}/{len(manifest)}] CACHED {smi[:40]}")
            continue

        t0 = time.time()
        result = run_one(str(XYZ_DIR / info["file"]), info["li_idx"], info["c_idx"])
        dt = time.time() - t0

        if "error" in result:
            print(f"  [{i+1}/{len(manifest)}] ERROR {smi[:40]}: {result['error'][:50]} ({dt:.0f}s)")
        else:
            print(f"  [{i+1}/{len(manifest)}] {smi[:40]:40s} q_C={result['mulliken_C']:+.4f} q_Li={result['mulliken_Li']:+.4f} ({dt:.0f}s)")

        cache[smi] = result
        with open(CACHE_PATH, "w") as f:
            json.dump(cache, f, indent=2)

    ok = sum(1 for v in cache.values() if "error" not in v)
    print(f"\nDone: {ok}/{len(manifest)}")

if __name__ == "__main__":
    main()
