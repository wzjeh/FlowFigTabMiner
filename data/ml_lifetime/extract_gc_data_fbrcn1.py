"""
Extract GC peak data for 260602 batch — 5-bromo-2-fluorobenzonitrile + n-BuLi + MeOH quench
(new run; reuses the 260522/260523 calibration and yield-correction conventions).

Vial → (T_C, L_cm) mapping (Zhao 2026-06-04):
  vial 1-5:    T = +20°C,  L = 3 / 10 / 25 / 50 / 100 cm
  vial 6-10:   T =   0°C,  L = 3 / 10 / 25 / 50 / 100 cm
  vial 11-15:  T = −25°C,  L = 3 / 10 / 25 / 50 / 100 cm
  vial 16-20:  T = −50°C,  L = 3 / 10 / 25 / 50 / 100 cm
  vial 21-25:  T = −78°C,  L = 3 / 10 / 25 / 50 / 100 cm

V_reaction = 250 μL (same as 260523).  V_C12 = 50 μL.
Calibration / MW / peak windows are identical to the 260522/260523 protocol
(see extract_gc_data_fbrcn.py for derivation).
"""
import re, subprocess
from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path(__file__).parent
DATA_DIR = BASE / "experiment results" / "fbrcnbenzene1"

# (T, L) grid: 5 T (warm → cold) × 5 L
PLAN = {}
T_LIST = [20, 0, -25, -50, -78]
L_LIST = [3, 10, 25, 50, 100]
for ti, T in enumerate(T_LIST):
    for li, L in enumerate(L_LIST):
        vial = ti * 5 + li + 1
        PLAN[f"WY-260602-{vial}"] = (T, L)

# Reactor length → residence time (s), same flow conditions as substrate 1
L_TO_TR = {3: 0.0471, 10: 0.1571, 25: 0.3927, 50: 0.7854, 100: 1.5708}

# Peak retention windows (min) — identical to 260522/260523
PEAK_C12  = (8.90,  9.40)   # C12 internal standard (~9.14 min)
PEAK_PROD = (9.85, 10.35)   # 2-fluorobenzonitrile, the MeOH-trapping product
PEAK_SUB  = (11.70, 12.45)  # 5-bromo-2-fluorobenzonitrile + 12.05/12.15 tailing cluster

# Calibration curves (Zhao 2026-05-24)
CAL_SM   = dict(slope=0.8886, intercept=-0.1214)   # SM:   y = 0.8886 x − 0.1214
CAL_PROD = dict(slope=0.5070, intercept=-0.0701)   # Prod: y = 0.5070 x − 0.0701

MW_SM, MW_PROD, MW_C12 = 200.01, 121.11, 170.34   # g/mol

V_REACTION_UL = 250   # μL — Zhao 2026-06-04: 260602 follows 260523 protocol


def parse_peaks(pdf):
    """pdftotext → list of (rt_min, area, height)."""
    txt = subprocess.run(["pdftotext", "-layout", str(pdf), "-"],
                         capture_output=True, text=True).stdout
    pat = re.compile(r"^\s*\d+\s+(\d+\.\d+)\s+(\d+)\s+(\d+)\b")
    return [(float(m.group(1)), int(m.group(2)), int(m.group(3)))
            for line in txt.splitlines()
            if (m := pat.match(line))]


def assign_area(peaks, window):
    return sum(a for (rt, a, _) in peaks if window[0] <= rt <= window[1])


def mass_ratio(area_ratio, cal):
    """Invert y = m·x + b → x = (y − b) / m, clipped non-negative."""
    return max((area_ratio - cal["intercept"]) / cal["slope"], 0.0)


def main():
    rows = []
    for sample, (T_C, L_cm) in PLAN.items():
        pdf = DATA_DIR / f"{sample}.pdf"
        if not pdf.exists():
            print(f"MISSING: {pdf.name}")
            continue
        peaks = parse_peaks(pdf)
        c12  = assign_area(peaks, PEAK_C12)
        prod = assign_area(peaks, PEAK_PROD)
        sub  = assign_area(peaks, PEAK_SUB)
        ar_sub  = sub  / c12 if c12 else np.nan
        ar_prod = prod / c12 if c12 else np.nan
        mr_sub  = mass_ratio(ar_sub,  CAL_SM)
        mr_prod = mass_ratio(ar_prod, CAL_PROD)
        nr_sub  = mr_sub  * MW_C12 / MW_SM
        nr_prod = mr_prod * MW_C12 / MW_PROD
        rows.append({
            "sample": sample, "T_C": T_C, "L_cm": L_cm, "tR_s": L_TO_TR[L_cm],
            "c12_area": c12, "prod_area": prod, "sub_area": sub,
            "ar_sub": round(ar_sub, 4), "ar_prod": round(ar_prod, 4),
            "mr_sub": round(mr_sub, 4), "mr_prod": round(mr_prod, 4),
            "nr_sub": round(nr_sub, 4), "nr_prod": round(nr_prod, 4),
            "nr_total": round(nr_sub + nr_prod, 4),
        })
    df = pd.DataFrame(rows).sort_values(["T_C", "L_cm"]).reset_index(drop=True)

    # ---- yield correction (reuse 260523 reference) ----
    # Reference n_SM0 from the old 260523 dataset (V=250 μL) → 8.81
    # 260602 also uses V=250 μL, so no V scaling needed: n_SM0 = 8.81 directly.
    N_SM0_REF_260523 = 8.81
    df["n_SM0"] = N_SM0_REF_260523 * V_REACTION_UL / 250
    df["yield_pct"]        = (df["nr_prod"] / df["n_SM0"] * 100).round(2)
    df["conversion_pct"]   = ((df["n_SM0"] - df["nr_sub"]) / df["n_SM0"] * 100).clip(0, 100).round(2)
    df["residual_pct"]     = (df["nr_sub"] / df["n_SM0"] * 100).round(2)
    df["mass_balance_pct"] = (df["nr_total"] / df["n_SM0"] * 100).round(2)

    print("=" * 90)
    print("5-bromo-2-fluorobenzonitrile + n-BuLi + MeOH quench  (260602 batch)")
    print(f"Calibration: SM y={CAL_SM['slope']}x{CAL_SM['intercept']:+.4f},  "
          f"Prod y={CAL_PROD['slope']}x{CAL_PROD['intercept']:+.4f}")
    print(f"V_reaction = {V_REACTION_UL} μL,  n_SM0_ref = {N_SM0_REF_260523} (from 260523, V=250)")
    print("=" * 90)
    print(df[["sample","T_C","L_cm","tR_s","ar_sub","ar_prod",
              "yield_pct","conversion_pct","mass_balance_pct"]].to_string(index=False))

    out_csv = BASE / "experiment_fbrcn1_summary.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    print("\n=== Yield (%) matrix [rows=T, cols=L_cm] ===")
    print(df.pivot(index="T_C", columns="L_cm", values="yield_pct").to_string())
    print("\n=== Conversion (%) matrix ===")
    print(df.pivot(index="T_C", columns="L_cm", values="conversion_pct").to_string())
    print("\n=== Mass balance (%) ===")
    print(df.pivot(index="T_C", columns="L_cm", values="mass_balance_pct").to_string())


if __name__ == "__main__":
    main()
