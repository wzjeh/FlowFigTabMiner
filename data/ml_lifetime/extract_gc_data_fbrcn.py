"""
Extract GC peak data for 5-bromo-2-fluorobenzonitrile + n-BuLi + MeOH quench experiments.

Mapping (per Zhao 2026-05-24):
  260522 (T=-25/0/+20°C):
    vial 1-5:    T=-25, L=3/10/25/50/100 cm
    vial 6-10:   T=  0, L=3/10/25/50/100 cm
    vial 11-15:  T=+20, L=3/10/25/50/100 cm
  260523 (T=-70/-50°C):
    vial 1-5:    T=-70, L=3/10/25/50/100 cm
    vial 6-10:   T=-50, L=3/10/25/50/100 cm

GC peaks of interest:
   ~9.14 min  → C12 (n-dodecane, internal standard)
  ~10.08 min  → 2-fluorobenzonitrile (PRODUCT after MeOH quench)
  ~11.87 min  → 5-bromo-2-fluorobenzonitrile (SUBSTRATE, unreacted)
   12.15 min  → product peak tailing at low conc (Zhao: add to product yield)

Calibration (Zhao 2026-05-24, y = Area_Ratio = Area_X/Area_C12, x = Mass_Ratio = m_X/m_C12):
  SM (5-Br-2-F-CN):  y = 0.8886 x − 0.1214  → m/m_C12 = (y + 0.1214) / 0.8886
  Product (2-F-CN):  y = 0.5070 x − 0.0701  → m/m_C12 = (y + 0.0701) / 0.5070

Molecular weights:
  5-Br-2-F-CN (SM):    C7H3BrFN  = 200.01 g/mol
  2-F-CN     (prod):   C7H4FN    = 121.11 g/mol
  n-C12      (IS):     C12H26    = 170.34 g/mol

Yield estimation:
  mol_X / mol_C12 (in GC vial) = (mass_X/mass_C12) × (MW_C12 / MW_X)

  ⚠ V_reaction correction (Zhao 2026-05-25):
  GC sample prep differs between 260522 and 260523:
    260522: 200 μL reaction + 50 μL C12 + 1250 μL EtOAc  → V_reaction/V_C12 = 4
    260523: 250 μL reaction + 50 μL C12 + 1200 μL EtOAc  → V_reaction/V_C12 = 5
  Same [X] in reaction gives 1.25× larger n_X/n_C12 in 260523's GC vial vs 260522's.
  → Must normalize per-dataset reference: n_SM0_260522 = n_SM0_260523 × (200/250)

  After correction:
    yield% (260522) = mol_prod_eff_GC / 7.048 × 100   (where 7.048 = 8.81 × 4/5)
    yield% (260523) = mol_prod_eff_GC / 8.81  × 100   (reference dataset, unchanged)
"""

# V_reaction per dataset (μL), V_C12 fixed at 50 μL
V_REACTION = {260522: 200, 260523: 250}
V_C12 = 50
import re
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path(__file__).parent
DATA_DIR = BASE / "experiment results" / "fbrcnbenzene"

# Vial → (T_C, L_cm) mapping
PLAN = {}
for i in range(15):
    if i < 5:    T = -25
    elif i < 10: T = 0
    else:        T = 20
    L = [3, 10, 25, 50, 100][i % 5]
    PLAN[f"WY-260522-{i+1}"] = (T, L)
for i in range(10):
    T = -70 if i < 5 else -50
    L = [3, 10, 25, 50, 100][i % 5]
    PLAN[f"WY-260523-{i+1}"] = (T, L)

# Zhao 2026-05-24: T=-25 vials 1 and 2 are mis-labeled — swap them
# (vial 1 actually L=10, vial 2 actually L=3)
PLAN["WY-260522-1"] = (-25, 10)
PLAN["WY-260522-2"] = (-25, 3)


def t_tag(T_C):
    """File tag for temperature: -25 → 'T-25', 20 → 'T20'."""
    return f"T{T_C}"


def resolve_pdf(sample, T_C, L_cm):
    """Look for renamed file first, then legacy {sample}.pdf."""
    renamed = DATA_DIR / f"{sample}_{t_tag(T_C)}_L{L_cm}.pdf"
    if renamed.exists(): return renamed
    legacy = DATA_DIR / f"{sample}.pdf"
    return legacy if legacy.exists() else None


def rename_pdfs():
    """Rename WY-260522-1.pdf → WY-260522-1_T-25_L10.pdf etc. Idempotent."""
    import shutil
    for sample, (T_C, L_cm) in PLAN.items():
        legacy = DATA_DIR / f"{sample}.pdf"
        new = DATA_DIR / f"{sample}_{t_tag(T_C)}_L{L_cm}.pdf"
        if new.exists() and not legacy.exists():
            continue
        if legacy.exists():
            shutil.move(str(legacy), str(new))
            print(f"  {legacy.name} → {new.name}")
        else:
            print(f"  SKIP: {sample} (no source file)")

# tR(s) per reactor length — same as 4-Br-F-C6H4 experiments
L_TO_TR = {3: 0.0471, 10: 0.1571, 25: 0.3927, 50: 0.7854, 100: 1.5708}

# Peak retention windows (min)
PEAK_C12   = (8.90,  9.40)    # C12 IS
PEAK_PROD  = (9.85, 10.35)    # 2-F-benzonitrile (product only)
# 2026-05-25 (Zhao decision): use PEAK_SUB = (11.70, 12.45) — absorbs both 12.05 and 12.15
# clusters as substrate-related. Lowest MAE (7.64 pp) configuration. GC-MS to be done to
# identify these neighboring peaks definitively.
PEAK_SUB   = (11.70, 12.45)   # SM main 11.87 + 12.05 cluster + 12.15 cluster
PEAK_SIDE  = (9.99,  9.99)    # disabled (kept for code compat)


def parse_peaks(pdf_path):
    """Run pdftotext -layout and parse peak table."""
    txt = subprocess.run(
        ["pdftotext", "-layout", str(pdf_path), "-"],
        capture_output=True, text=True
    ).stdout
    # Peak lines look like: "  1   9.142   218645   125256   0.000   S"
    pat = re.compile(r"^\s*\d+\s+(\d+\.\d+)\s+(\d+)\s+(\d+)\b")
    peaks = []
    for line in txt.splitlines():
        m = pat.match(line)
        if m:
            rt, area, h = float(m.group(1)), int(m.group(2)), int(m.group(3))
            peaks.append((rt, area, h))
    return peaks


def assign_area(peaks, window):
    """Return summed area of peaks within retention window."""
    return sum(a for (rt, a, h) in peaks if window[0] <= rt <= window[1])


# Calibration curves (Zhao 2026-05-24): y = Area_Ratio, x = Mass_Ratio = m_X/m_C12
CAL_SM   = dict(slope=0.8886, intercept=-0.1214)   # 5-Br-2-F-CN
CAL_PROD = dict(slope=0.5070, intercept=-0.0701)   # 2-F-CN

MW_SM   = 200.01   # 5-Br-2-F-benzonitrile
MW_PROD = 121.11   # 2-F-benzonitrile
MW_C12  = 170.34   # n-dodecane


def area_ratio_to_mass_ratio(area_ratio, cal):
    """Invert y = m·x + b → x = (y − b) / m. Clip non-negative."""
    x = (area_ratio - cal["intercept"]) / cal["slope"]
    return max(x, 0.0)


def main():
    import sys
    if "--rename" in sys.argv:
        print("Renaming PDF files to include T and L tags...")
        rename_pdfs()
        print("Done renaming.\n")

    rows = []
    for sample, (T_C, L_cm) in PLAN.items():
        pdf = resolve_pdf(sample, T_C, L_cm)
        if pdf is None:
            print(f"MISSING: {sample}"); continue
        peaks = parse_peaks(pdf)
        c12 = assign_area(peaks, PEAK_C12)
        prod = assign_area(peaks, PEAK_PROD)
        sub  = assign_area(peaks, PEAK_SUB)
        side = assign_area(peaks, PEAK_SIDE)   # disabled (always 0)
        prod_eff = prod   # 2026-05-25: 12.05/12.15 now in PEAK_SUB, not added here
        ar_sub  = sub / c12 if c12 else np.nan
        ar_prod = prod_eff / c12 if c12 else np.nan
        # Mass ratios (mass_X / mass_C12) via calibration inversion
        mr_sub  = area_ratio_to_mass_ratio(ar_sub,  CAL_SM)
        mr_prod = area_ratio_to_mass_ratio(ar_prod, CAL_PROD)
        # Mole ratios (n_X / n_C12) = (m_X / m_C12) × (MW_C12 / MW_X)
        nr_sub  = mr_sub  * MW_C12 / MW_SM
        nr_prod = mr_prod * MW_C12 / MW_PROD
        rows.append({
            "sample": sample,
            "date": int(sample.split('-')[1]),
            "vial": int(sample.split('-')[2]),
            "T_C": T_C, "L_cm": L_cm, "tR_s": L_TO_TR[L_cm],
            "c12_area": c12, "prod_area": prod, "sub_area": sub, "side_area": side,
            "prod_eff_area": prod_eff,
            "ar_sub": round(ar_sub, 4), "ar_prod": round(ar_prod, 4),
            "mr_sub": round(mr_sub, 4), "mr_prod": round(mr_prod, 4),
            "nr_sub": round(nr_sub, 4), "nr_prod": round(nr_prod, 4),
            "nr_total": round(nr_sub + nr_prod, 4),
        })
    df = pd.DataFrame(rows).sort_values(["T_C", "L_cm"]).reset_index(drop=True)

    # ---- V_reaction correction ----
    # Determine reference n_SM0 from the dataset with largest V_reaction
    # (260523's max nr_total gives best estimate of true c0/c_C12 ratio)
    ref_date = max(V_REACTION, key=V_REACTION.get)   # 260523
    ref_V = V_REACTION[ref_date]
    ref_subset = df[df['date'] == ref_date]
    n_SM0_ref = ref_subset['nr_total'].max()         # 8.81 from 260523 -70°C L=25
    best_row = df.loc[df["nr_total"].idxmax()]

    # Per-dataset n_SM0 scaled by V_reaction
    df['V_reaction_uL'] = df['date'].map(V_REACTION)
    df['n_SM0_dataset'] = n_SM0_ref * df['V_reaction_uL'] / ref_V

    df["yield_pct"]        = (df["nr_prod"] / df['n_SM0_dataset'] * 100).round(2)
    df["conversion_pct"]   = ((df['n_SM0_dataset'] - df["nr_sub"]) / df['n_SM0_dataset'] * 100).clip(0, 100).round(2)
    df["residual_pct"]     = (df["nr_sub"]  / df['n_SM0_dataset'] * 100).round(2)
    df["mass_balance_pct"] = (df["nr_total"] / df['n_SM0_dataset'] * 100).round(2)
    n_SM0 = n_SM0_ref   # for compat with print below

    print("=" * 110)
    print("5-bromo-2-fluorobenzonitrile + n-BuLi + MeOH quench")
    print(f"Calibration: SM y={CAL_SM['slope']}x{CAL_SM['intercept']:+.4f},  "
          f"Prod y={CAL_PROD['slope']}x{CAL_PROD['intercept']:+.4f}  (y=Area_X/Area_C12, x=m_X/m_C12)")
    print(f"Side peak (12.15) added to product (tailing).")
    print(f"V_reaction correction (Zhao 2026-05-25):")
    print(f"  Reference: {ref_date} (V_reaction={ref_V}μL), n_SM0_ref={n_SM0_ref:.4f}")
    print(f"  Per-dataset n_SM0 = n_SM0_ref × (V_reaction / {ref_V}):")
    for d, v in V_REACTION.items():
        print(f"    {d}: V_reaction={v}μL, n_SM0={n_SM0_ref * v / ref_V:.4f}")
    print(f"  Reference row: {best_row['sample']}, T={best_row['T_C']:+}°C, L={best_row['L_cm']}cm")
    print("=" * 110)

    cols_show = ["sample", "T_C", "L_cm", "tR_s",
                 "ar_sub", "ar_prod", "mr_sub", "mr_prod",
                 "nr_sub", "nr_prod", "nr_total",
                 "yield_pct", "conversion_pct", "mass_balance_pct"]
    print(df[cols_show].to_string(index=False))

    out_csv = BASE / "experiment_fbrcn_summary.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    # Yield matrix
    print("\n=== Yield (%) matrix [rows=T, cols=L_cm] ===")
    pivot = df.pivot(index="T_C", columns="L_cm", values="yield_pct")
    print(pivot.to_string())
    print("\n=== Conversion (%) matrix ===")
    print(df.pivot(index="T_C", columns="L_cm", values="conversion_pct").to_string())
    print("\n=== Mass balance (n_total / n_SM0 × 100) ===")
    print(df.pivot(index="T_C", columns="L_cm", values="mass_balance_pct").to_string())


if __name__ == "__main__":
    main()
