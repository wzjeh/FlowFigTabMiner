"""
GFN2-xTB aggregation descriptors for 3-Br-5-Li-C6H3-CN (m-ArLi from 3,5-dibromobenzonitrile).
SMILES: [Li]c1cc(Br)cc(C#N)c1

Outputs: ΔE_dim, ΔH_dim, ΔG_dim, ΔS_dim, Vbur_mono, Vbur_dim, ΔVbur_dim
Then plug into v4.6 m-ArLi formulas (Ea_f, Ea_d, lnA_f, lnA_d).
"""
import sys, time, json
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from compute_aggregation_descriptors import (
    smiles_to_3d, opt_tblite, xtb_hess_alpb, write_std_xyz,
    construct_dimer, find_li_indices, compute_vbur,
    HARTREE_TO_KJ, T_REF, GEOM_DIR
)

SMI = "[Li]c1cc(Br)cc(C#N)c1"   # 3-Br-5-Li-benzonitrile  (m-ArLi)
NAME = "3Br5LiCN"


def main():
    print(f"Target: {SMI}  (3-Br-5-Li-C6H3-CN, m-ArLi from 3,5-Br2-CN)")
    t0 = time.time()

    res = smiles_to_3d(SMI)
    if res is None:
        print("ERROR: smiles_to_3d failed"); return
    elems_m, coords_m, li_idx = res
    print(f"  monomer atoms: {len(elems_m)}  Li_idx={li_idx}")

    print("  [1/4] monomer opt (tblite L-BFGS-B)...")
    opt_m, _, _ = opt_tblite(elems_m, coords_m)
    write_std_xyz(GEOM_DIR / f"mono_{NAME}.xyz", elems_m, opt_m, comment=SMI)

    print("  [2/4] monomer xtb --hess --alpb thf...")
    mono = xtb_hess_alpb(elems_m, opt_m)
    if mono is None:
        print("ERROR: monomer hess failed"); return
    print(f"    E={mono['E']:.6f}  H={mono['H']:.6f}  G={mono['G']:.6f} Hartree")

    print("  [3/4] dimer opt (4-center)...")
    d_elems, d_init = construct_dimer(elems_m, opt_m, li_idx)
    opt_d, _, _ = opt_tblite(d_elems, d_init, maxiter=500)
    write_std_xyz(GEOM_DIR / f"dim_{NAME}.xyz", d_elems, opt_d, comment=f"dimer of {SMI}")

    print("  [4/4] dimer xtb --hess --alpb thf...")
    dim = xtb_hess_alpb(d_elems, opt_d)
    if dim is None:
        print("ERROR: dimer hess failed"); return
    print(f"    E={dim['E']:.6f}  H={dim['H']:.6f}  G={dim['G']:.6f} Hartree")

    dE = (dim['E'] - 2*mono['E']) * HARTREE_TO_KJ
    dH = (dim['H'] - 2*mono['H']) * HARTREE_TO_KJ
    dG = (dim['G'] - 2*mono['G']) * HARTREE_TO_KJ
    dS = (dH - dG) / T_REF * 1000

    vbur_m = compute_vbur(elems_m, opt_m, li_idx)
    li_d = find_li_indices(d_elems)
    vbur_d_vals = [v for v in (compute_vbur(d_elems, opt_d, i) for i in li_d) if v is not None]
    vbur_d = float(np.mean(vbur_d_vals)) if vbur_d_vals else None
    dVbur = (vbur_d - vbur_m) if (vbur_m is not None and vbur_d is not None) else None

    elapsed = time.time() - t0
    print()
    print(f"=== Aggregation descriptors for 3-Br-5-Li-C6H3-CN (took {elapsed:.0f}s) ===")
    print(f"  ΔE_dim   = {dE:+8.2f} kJ/mol")
    print(f"  ΔH_dim   = {dH:+8.2f} kJ/mol")
    print(f"  ΔG_dim   = {dG:+8.2f} kJ/mol")
    print(f"  ΔS_dim   = {dS:+8.2f} J/(mol·K)")
    print(f"  Vbur_mono= {vbur_m:.4f}")
    print(f"  Vbur_dim = {vbur_d:.4f}")
    print(f"  ΔVbur    = {dVbur:+.4f}")

    # ---- monomer descriptors from analog proxies in master CSV ----
    df_master = pd.read_csv(Path(__file__).parent / "intermediates_master.csv")
    # Analogs: m-Br-PhLi and m-CN-PhLi (closest m-substituted analogs)
    smi_brLi = "[Li]c1cccc(Br)c1"
    smi_cnLi = "[Li]c1cccc(C#N)c1"
    proxies = {}
    for s in (smi_brLi, smi_cnLi):
        rows = df_master[df_master['intermediate_smiles_canonical'] == s]
        if len(rows):
            proxies[s] = rows.iloc[0]

    def avg(col):
        vals = [r[col] for r in proxies.values() if pd.notna(r.get(col, None))]
        return float(np.mean(vals)) if vals else None

    print()
    print("=== Proxy monomer descriptors (avg of m-Br-PhLi + m-CN-PhLi from master.csv) ===")
    BDE   = avg("dft_LiC_BDE_kJ")
    vol   = avg("mol_volume")
    d_LiC = avg("dft_LiC_bond_A")
    LUMO  = avg("dft_LUMO_eV")
    Gsolv = avg("dft_Gsolv_kJ")
    print(f"  BDE   = {BDE}     kJ/mol")
    print(f"  vol   = {vol}      Å³")
    print(f"  d_LiC = {d_LiC}    Å")
    print(f"  LUMO  = {LUMO}     eV")
    print(f"  Gsolv = {Gsolv}    kJ/mol")

    # ---- v4.6 m-ArLi predictions ----
    print()
    print("=== v4.6 m-ArLi predictions ===")
    Ea_f  = 0.4821*BDE + 0.0853*vol + 0.1776*dH - 165.5877
    Ea_d  = 4.1346*Gsolv + 1.8525*vol - 429.595*dVbur + 204.2715
    lnA_f = -428.999*d_LiC + 0.0494*vol + 0.0601*dG + 837.3001
    lnA_d = -4.7318*LUMO - 1.0510*Gsolv + 8.9372*dVbur - 102.7021
    print(f"  Ea_f  = {Ea_f:.2f} kJ/mol")
    print(f"  Ea_d  = {Ea_d:.2f} kJ/mol")
    print(f"  lnA_f = {lnA_f:.2f}")
    print(f"  lnA_d = {lnA_d:.2f}")

    R = 8.314e-3
    for T_C in [-65, -50, -25, 0, 20]:
        Tk = T_C + 273.15
        kf = np.exp(lnA_f - Ea_f/(R*Tk))
        kd = np.exp(lnA_d - Ea_d/(R*Tk))
        print(f"  T={T_C:+4}°C  k_f={kf:.2e}  k_d={kd:.2e}  s⁻¹")

    out = pd.DataFrame([{
        'smi': SMI, 'name': NAME,
        'dE_dim_kJ': round(dE,2), 'dH_dim_kJ': round(dH,2),
        'dG_dim_kJ': round(dG,2), 'dS_dim_J_K': round(dS,2),
        'Vbur_mono': round(vbur_m,4), 'Vbur_dim': round(vbur_d,4),
        'dVbur': round(dVbur,4),
        'BDE_proxy': BDE, 'vol_proxy': vol, 'd_LiC_proxy': d_LiC,
        'LUMO_proxy': LUMO, 'Gsolv_proxy': Gsolv,
        'Ea_f_pred':  round(Ea_f, 2),
        'Ea_d_pred':  round(Ea_d, 2),
        'lnA_f_pred': round(lnA_f, 2),
        'lnA_d_pred': round(lnA_d, 2),
    }])
    out.to_csv(Path(__file__).parent / "35brcn_v46_prediction.csv", index=False)
    print(f"\nSaved: 35brcn_v46_prediction.csv")


if __name__ == "__main__":
    main()
