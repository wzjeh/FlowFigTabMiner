"""
Single-compound run: compute aggregation descriptors for p-FC6H4Li
([Li]c1ccc(F)cc1) so we can apply HYBRID v2 p-ArLi Ea_d formula:
  Ea_d = -1254·d_LiC + 29.07·B5 + 159.96·ΔVbur + 2308

Reuses helpers from compute_aggregation_descriptors.py.
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

SMI = "[Li]c1ccc(F)cc1"

def main():
    print(f"Target: {SMI}  (p-fluorophenyllithium)")
    t0 = time.time()

    # 1) initial 3D from SMILES
    res = smiles_to_3d(SMI)
    if res is None:
        print("ERROR: smiles_to_3d failed"); return
    elems_m, coords_m, li_idx = res
    print(f"  monomer atoms: {len(elems_m)}  Li_idx={li_idx}")

    # 2) monomer opt + hess
    print("  [1/4] monomer opt (tblite L-BFGS-B)...")
    opt_m, _, _ = opt_tblite(elems_m, coords_m)
    write_std_xyz(GEOM_DIR / "mono_pFArLi.xyz", elems_m, opt_m, comment=SMI)

    print("  [2/4] monomer xtb --hess --alpb thf...")
    mono = xtb_hess_alpb(elems_m, opt_m)
    if mono is None:
        print("ERROR: monomer hess failed"); return
    print(f"    E_mono={mono['E']:.6f}  H_mono={mono['H']:.6f}  G_mono={mono['G']:.6f} Hartree")

    # 3) dimer
    print("  [3/4] dimer opt (4-center)...")
    d_elems, d_init = construct_dimer(elems_m, opt_m, li_idx)
    opt_d, _, _ = opt_tblite(d_elems, d_init, maxiter=500)
    write_std_xyz(GEOM_DIR / "dim_pFArLi.xyz", d_elems, opt_d, comment=f"dimer of {SMI}")

    print("  [4/4] dimer xtb --hess --alpb thf...")
    dim = xtb_hess_alpb(d_elems, opt_d)
    if dim is None:
        print("ERROR: dimer hess failed"); return
    print(f"    E_dim ={dim['E']:.6f}  H_dim ={dim['H']:.6f}  G_dim ={dim['G']:.6f} Hartree")

    # 4) deltas
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
    print(f"=== Aggregation descriptors for p-FC6H4Li (took {elapsed:.0f}s) ===")
    print(f"  ΔE_dim   = {dE:+8.2f} kJ/mol")
    print(f"  ΔH_dim   = {dH:+8.2f} kJ/mol")
    print(f"  ΔG_dim   = {dG:+8.2f} kJ/mol")
    print(f"  ΔS_dim   = {dS:+8.2f} J/(mol·K)")
    print(f"  Vbur_mono= {vbur_m:.4f}")
    print(f"  Vbur_dim = {vbur_d:.4f}")
    print(f"  ΔVbur    = {dVbur:+.4f}")

    # ---- HYBRID v2 p-ArLi Ea_d / Ea_f / lnA_d / lnA_f predictions ----
    # Need d_LiC, B5 from descriptor table; pull from intermediates_master.csv
    df = pd.read_csv(Path(__file__).parent / "intermediates_master.csv")
    row = df[df['intermediate_smiles_canonical'] == SMI].iloc[0]
    d_LiC = row['dft_LiC_bond_A']
    B5    = row['sterimol_B5']
    print()
    print(f"=== Predicted with HYBRID v2 (now ΔVbur known) ===")
    print(f"  d_LiC = {d_LiC} Å,  B5 = {B5},  ΔVbur = {dVbur:+.4f}")
    Ea_d = -1254*d_LiC + 29.07*B5 + 159.96*dVbur + 2308
    print(f"  Ea_d (p-ArLi formula) = {Ea_d:.2f} kJ/mol")
    print()
    print(f"  Experiment (Scenario A, 17 pts): Ea_d = 24.7 kJ/mol")
    print(f"  Difference: {abs(Ea_d - 24.7):.1f} kJ/mol")

    # save
    out = pd.DataFrame([{
        'smi': SMI,
        'dE_dim_kJ': round(dE,2), 'dH_dim_kJ': round(dH,2),
        'dG_dim_kJ': round(dG,2), 'dS_dim_J_K': round(dS,2),
        'Vbur_mono': round(vbur_m,4), 'Vbur_dim': round(vbur_d,4),
        'dVbur': round(dVbur,4),
        'Ea_d_HYBRIDv2_pred': round(Ea_d,2),
        'Ea_d_experiment': 24.7,
    }])
    out.to_csv(Path(__file__).parent / "pFArLi_aggregation_result.csv", index=False)
    print(f"\nSaved: pFArLi_aggregation_result.csv")


if __name__ == "__main__":
    main()
