"""
Extended C2 (remote-EWG) virtual ArLi library for stronger external veto of the
HOMO+Gsolv model. Denser sampling of remote-EWG chemical space (more EWG types,
multi-substituted, wider σ/HOMO range, esp. high-σ end).

Usage:
  python compute_c2_extended.py --test 2
  python compute_c2_extended.py            -> virtual_c2_extended.csv
"""
import sys
from pathlib import Path
import pandas as pd
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
from compute_virtual_arli import compute_one
from build_v60_classifier import classify

BASE = Path(__file__).parent

# remote-EWG fragments + Hammett σ_p, σ_m
EWG = {
    "CN":   ("C#N",            0.66, 0.56),
    "NO2":  ("[N+](=O)[O-]",   0.78, 0.71),
    "CF3":  ("C(F)(F)F",       0.54, 0.43),
    "CO2Me":("C(=O)OC",        0.45, 0.37),
    "CO2Et":("C(=O)OCC",       0.45, 0.37),
    "CO2tBu":("C(=O)OC(C)(C)C",0.45, 0.37),
    "COMe": ("C(C)=O",         0.50, 0.38),
    "COPh": ("C(=O)c1ccccc1",  0.43, 0.34),
    "CONMe2":("C(=O)N(C)C",    0.36, 0.35),
    "SO2Me":("S(=O)(=O)C",     0.72, 0.60),
    "SO2CF3":("S(=O)(=O)C(F)(F)F",0.96,0.83),
    "Cl":   ("Cl",             0.23, 0.37),
    "F":    ("F",              0.06, 0.34),
    "Br":   ("Br",             0.23, 0.39),
}

def build():
    rows, seen = [], set()
    def add(name, smi, sp, sm):
        m = Chem.MolFromSmiles(smi)
        if m is None: return
        cs = Chem.MolToSmiles(m)
        if cs in seen: return
        seen.add(cs); rows.append(dict(name=name, smi=cs, sigma_p_sum=sp, sigma_m_sum=sm))
    # mono m / p
    for k,(fr,sp,sm) in EWG.items():
        add(f"m-{k}", f"[Li]c1cccc({fr})c1", 0.0, sm)
        add(f"p-{k}", f"[Li]c1ccc({fr})cc1", sp,  0.0)
    # 3,5-di (both meta)
    strong = ["CN","NO2","CF3","CO2Me","COMe","SO2Me","Cl","F"]
    for a in strong:
        fa,spa,sma = EWG[a]
        add(f"3,5-di{a}", f"[Li]c1cc({fa})cc({fa})c1", 0.0, 2*sma)
    # 3-X-5-Y mixed meta
    mix = [("CN","NO2"),("CN","CF3"),("CN","CO2Me"),("NO2","CF3"),("CN","Cl"),
           ("CN","F"),("CO2Me","CF3"),("NO2","CO2Me"),("CN","SO2Me"),("CF3","CO2Me")]
    for a,b in mix:
        fa,_,sma = EWG[a]; fb,_,smb = EWG[b]
        add(f"3-{a}-5-{b}", f"[Li]c1cc({fa})cc({fb})c1", 0.0, sma+smb)
    # 3,4-di and 2,4 (mixed positions, still remote-dominated)
    for a,b in [("CN","F"),("CN","Cl"),("CF3","F"),("CN","CF3"),("CO2Me","F")]:
        fa,spa,sma=EWG[a]; fb,spb,smb=EWG[b]
        add(f"3-{a}-4-{b}", f"[Li]c1ccc({fb})c({fa})c1", spb, sma)   # 4-b para, 3-a meta
    # 3,4,5-tri
    for a in ["F","CN","CF3"]:
        fa,sp,sm=EWG[a]
        add(f"3,4,5-tri{a}", f"[Li]c1cc({fa})c({fa})c({fa})c1", sp, 2*sm)
    return rows


def main():
    lib = build()
    n_test = None
    if "--test" in sys.argv:
        n_test = int(sys.argv[sys.argv.index("--test")+1]); lib = lib[:n_test]
    # keep only those classified C2
    kept = []
    for r in lib:
        try: c = classify(r["smi"])[0]
        except Exception: c = "ERR"
        r["cls"] = c
        if c == "C2" or n_test: kept.append(r)
    print(f"Generated {len(lib)}, classified-C2 kept {len(kept)}"
          + (f" (TEST {n_test})" if n_test else ""), flush=True)
    out = []
    for i, r in enumerate(kept):
        print(f"[{i+1}/{len(kept)}] {r['name']:14s} {r['smi']}", flush=True)
        d = compute_one(r["name"], r["smi"])
        if d is None: print("    FAILED"); continue
        out.append({**r, **d})
        print(f"    HOMO={d['HOMO_eV']:.3f} Gsolv={d['Gsolv_kJ']:.1f} σ={r['sigma_p_sum']+r['sigma_m_sum']:.2f}", flush=True)
    df = pd.DataFrame(out)
    suf = f"_test{n_test}" if n_test else ""
    p = BASE / f"virtual_c2_extended{suf}.csv"
    df.to_csv(p, index=False); print(f"\nSaved {len(df)} -> {p}")


if __name__ == "__main__":
    main()
