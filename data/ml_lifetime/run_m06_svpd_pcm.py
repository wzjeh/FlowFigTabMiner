"""
M06-2X/def2-SVPD + PCM(THF) on representative organolithiums.
Tests: gas phase vs solvated CDFT descriptors.

  conda run -n base python data/ml_lifetime/run_m06_svpd_pcm.py
"""
import psi4, json, numpy as np, time, os
from pathlib import Path

os.chdir("/Users/zhaowenyuan/Projects/FlowFigTabMiner")
DATA_DIR = Path("data/ml_lifetime")
XYZ_DIR = DATA_DIR / "hf_xyz"
manifest = json.load(open(XYZ_DIR / "manifest.json"))
CACHE = DATA_DIR / "m06_svpd_pcm_test.json"

psi4.set_memory("4 GB")
psi4.set_num_threads(4)

test = [
    ("[Li]Cc1ccc(C(=O)CC)cc1", 2.5),      # benzylLi (lowest Ea)
    ("[Li]c1ccc(C#N)cc1", 4.5),            # p-CN (problem case)
    ("[Li]c1ccc([N+](=O)[O-])cc1", 28.9),  # p-NO2
    ("[Li]c1ccccc1C(=O)OC", 39.6),         # o-CO2Me (middle)
    ("[Li]c1ccccc1I", 57.4),               # o-I (benzyne)
    ("[Li]C1(c2ccc(Cl)cc2)CO1", 64.9),     # oxiranylLi (highest)
]

PCM_STR = """
Units = Angstrom
Medium {
    SolverType = IEFPCM
    Solvent = THF
}
Cavity {
    Type = GePol
    Area = 0.3
}
"""

results = json.load(open(CACHE)) if CACHE.exists() else {}

for smi, ea in test:
    if smi in results:
        print(f"CACHED {smi[:40]}")
        continue
    if smi not in manifest:
        print(f"SKIP {smi[:40]}")
        continue

    info = manifest[smi]
    with open(str(XYZ_DIR / info["file"])) as f:
        geom = f.read()
    li_idx, c_idx = info["li_idx"], info["c_idx"]

    t0 = time.time()

    # Gas phase
    psi4.core.set_output_file("/dev/null")
    mol_g = psi4.geometry(geom)
    psi4.set_options({"basis": "def2-svpd", "pcm": False})
    e_g, wfn_g = psi4.energy("m06-2x", molecule=mol_g, return_wfn=True)
    psi4.oeprop(wfn_g, "MULLIKEN_CHARGES")
    q_g = np.array([wfn_g.atomic_point_charges().np[i] for i in range(mol_g.natom())])
    eps_g = wfn_g.epsilon_a().np
    occ_g = wfn_g.occupation_a().np
    hi = np.where(occ_g > 0.5)[0][-1]
    homo_g = float(eps_g[hi] * 27.211)
    lumo_g = float(eps_g[hi + 1] * 27.211)
    psi4.core.clean()

    # PCM(THF)
    mol_s = psi4.geometry(geom)
    psi4.set_options({"basis": "def2-svpd", "pcm": True, "pcm_scf_type": "total"})
    psi4.pcm_helper(PCM_STR)
    e_s, wfn_s = psi4.energy("m06-2x", molecule=mol_s, return_wfn=True)
    psi4.oeprop(wfn_s, "MULLIKEN_CHARGES")
    q_s = np.array([wfn_s.atomic_point_charges().np[i] for i in range(mol_s.natom())])
    eps_s = wfn_s.epsilon_a().np
    occ_s = wfn_s.occupation_a().np
    hi_s = np.where(occ_s > 0.5)[0][-1]
    homo_s = float(eps_s[hi_s] * 27.211)
    lumo_s = float(eps_s[hi_s + 1] * 27.211)
    psi4.core.clean()

    dt = time.time() - t0

    # CDFT
    def cdft(h, l):
        g = l - h
        mu = (h + l) / 2
        eta = g / 2
        return mu, eta, mu**2 / (2 * eta) if eta > 0 else 0

    mu_g, eta_g, omega_g = cdft(homo_g, lumo_g)
    mu_s, eta_s, omega_s = cdft(homo_s, lumo_s)

    results[smi] = {
        "ea": ea,
        "homo_g": homo_g, "lumo_g": lumo_g, "mu_g": mu_g, "eta_g": eta_g, "omega_g": omega_g,
        "q_C_g": float(q_g[c_idx]), "q_Li_g": float(q_g[li_idx]),
        "homo_s": homo_s, "lumo_s": lumo_s, "mu_s": mu_s, "eta_s": eta_s, "omega_s": omega_s,
        "q_C_s": float(q_s[c_idx]), "q_Li_s": float(q_s[li_idx]),
        "Gsolv_kJ": float((e_s - e_g) * 2625.5),
    }

    hw_g = "⚠️" if homo_g > 0 else "✓"
    hw_s = "⚠️" if homo_s > 0 else "✓"
    print(f"{smi[:40]:40s} Ea={ea:5.1f} ({dt:.0f}s)")
    print(f"  GAS:  HOMO={homo_g:+.3f}{hw_g} ω={omega_g:.3f} q_C={q_g[c_idx]:+.4f}")
    print(f"  THF:  HOMO={homo_s:+.3f}{hw_s} ω={omega_s:.3f} q_C={q_s[c_idx]:+.4f}")
    print()

    json.dump(results, open(CACHE, "w"), indent=2)

print(f"\nDone: {len(results)} molecules → {CACHE}")
