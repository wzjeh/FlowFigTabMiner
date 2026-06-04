"""
ArBr PRECURSOR descriptors for C2 formation (Br/Li exchange) kinetics.
Formation rate depends on Ar-Br + n-BuLi, NOT on Ar-Li. So compute descriptors of the
Ar-Br precursor (replace [Li] -> Br): C-Br bond length, C-Br homolytic BDE, ArBr LUMO/gap,
ipso-C and Br Mulliken charges, dipole, Gsolv. Then screen formation Ea_f / lnA_f.
"""
import subprocess, tempfile, itertools
from pathlib import Path
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
from rdkit.Chem import AllChem
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut
from scipy import stats
from compute_aggregation_descriptors import opt_tblite, write_std_xyz, XTB

BASE = Path(__file__).parent
HKJ = 2625.499

def to3d(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    m = Chem.AddHs(m); p = AllChem.ETKDGv3(); p.randomSeed = 42
    if AllChem.EmbedMolecule(m, p) != 0: return None
    AllChem.UFFOptimizeMolecule(m, maxIters=2000)
    return [a.GetSymbol() for a in m.GetAtoms()], np.array(m.GetConformer().GetPositions())

def xtb(elems, coords, alpb=None, uhf=0):
    with tempfile.TemporaryDirectory() as td:
        td = Path(td); write_std_xyz(td/"m.xyz", elems, coords)
        cmd = [XTB, "m.xyz", "--gfn", "2", "--sp"]
        if alpb: cmd += ["--alpb", alpb]
        if uhf: cmd += ["--uhf", str(uhf)]
        r = subprocess.run(cmd, cwd=td, capture_output=True, text=True, timeout=600)
        if r.returncode != 0: return None
        E = HOMO = LUMO = dip = None; L = r.stdout.splitlines()
        for i, ln in enumerate(L):
            if "TOTAL ENERGY" in ln:
                try: E = float(ln.split()[-3])
                except: pass
            if "(HOMO)" in ln:
                t = ln.split()
                try: HOMO = float(t[t.index("(HOMO)")-1])
                except: pass
            if "(LUMO)" in ln:
                t = ln.split()
                try: LUMO = float(t[t.index("(LUMO)")-1])
                except: pass
            if "molecular dipole" in ln:
                for j in range(i, min(i+4, len(L))):
                    if "full:" in L[j]:
                        try: dip = float(L[j].split()[-1])
                        except: pass
        q = None; cf = td/"charges"
        if cf.exists(): q = [float(x) for x in cf.read_text().split()]
        return dict(E=E, HOMO=HOMO, LUMO=LUMO, dip=dip, q=q)

tr = pd.read_csv("v60_training_set.csv"); tr = tr[(~tr.is_virtual) & (tr.class_v60 == "C2")]
rows = []
for _, r in tr.iterrows():
    arbr = r["smi"].replace("[Li]", "Br")
    g = to3d(arbr)
    if g is None: print("skip", arbr, flush=True); continue
    elems, c0 = g; opt, _, _ = opt_tblite(elems, c0)
    bi = next((i for i, e in enumerate(elems) if e == "Br"), None)
    ci = min((i for i, e in enumerate(elems) if e == "C"), key=lambda i: np.linalg.norm(opt[i]-opt[bi]))
    dCBr = float(np.linalg.norm(opt[bi]-opt[ci]))
    vac = xtb(elems, opt); thf = xtb(elems, opt, alpb="thf")
    # homolytic C-Br BDE: ArBr -> Ar. + Br.
    ar_e = [e for i, e in enumerate(elems) if i != bi]; ar_c = np.delete(opt, bi, 0)
    e_ar = xtb(ar_e, ar_c, uhf=1); e_br = xtb(["Br"], np.array([[0., 0., 0.]]), uhf=1)
    bde = (e_ar["E"] + e_br["E"] - vac["E"]) * HKJ if (e_ar and e_br) else np.nan
    rows.append(dict(intermediate=r["intermediate"], Ea_f=r["Ea_f"], lnA_f=r["lnA_f"],
        sig=r["sigma_p_sum"]+r["sigma_m_sum"],
        HOMO_br=vac["HOMO"], LUMO_br=vac["LUMO"], gap_br=vac["LUMO"]-vac["HOMO"], dip_br=vac["dip"],
        qCipso_br=vac["q"][ci] if vac["q"] else np.nan, qBr=vac["q"][bi] if vac["q"] else np.nan,
        dCBr=dCBr, BDE_CBr=bde, Gsolv_br=(thf["E"]-vac["E"])*HKJ))
    print(f"{r['intermediate'][:24]:24s} LUMO={vac['LUMO']:+.2f} qCipso={rows[-1]['qCipso_br']:+.3f} "
          f"dCBr={dCBr:.3f} BDE={bde:.0f}", flush=True)
df = pd.DataFrame(rows); df.to_csv(BASE/"arbr_c2_descriptors.csv", index=False)
print(f"\nSaved {len(df)} -> arbr_c2_descriptors.csv\n")

# ---- screen formation Ea_f / lnA_f against ArBr descriptors ----
POOL = ["LUMO_br", "gap_br", "qCipso_br", "qBr", "dCBr", "BDE_CBr", "dip_br", "HOMO_br", "sig"]
def loo(X, y):
    p = np.zeros(len(y))
    for a, b in LeaveOneOut().split(X): p[b] = LinearRegression().fit(X[a], y[a]).predict(X[b])
    return 1 - np.sum((y-p)**2)/np.sum((y-y.mean())**2)
for tgt in ["Ea_f", "lnA_f"]:
    print(f"=== formation {tgt} vs ArBr descriptors ===")
    print("  单变量 |r|:")
    for x in POOL:
        s = df.dropna(subset=[x, tgt])
        if len(s) >= 6 and s[x].std() > 0:
            rr, pp = stats.pearsonr(s[x], s[tgt]); print(f"    {x:9s} r={rr:+.2f} (p={pp:.3f})")
    res = []
    for k in (1, 2):
        for f in itertools.combinations(POOL, k):
            fl = list(f); s = df.dropna(subset=fl+[tgt])
            if len(s) < 9 or any(s[c].std() == 0 for c in fl): continue
            res.append((loo(s[fl].values, s[tgt].values), fl))
    res.sort(reverse=True)
    print("  穷举 top3 LOO:")
    for r2, fl in res[:3]: print(f"    LOO={r2:+.2f}  {'+'.join(fl)}")
    print()
