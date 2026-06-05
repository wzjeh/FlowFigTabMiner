"""
Re-extract GC peak data for the OLD 260522 + 260523 MeOD-quench experiment of
5-bromo-2-fluorobenzonitrile + n-BuLi, but with the **deuterated-product**
calibration curve obtained from a purified fluorobenzonitrile-5-d standard
(Zhao 2026-06-04):

    y = 0.9092 x − 0.1201        (y = Area_X/Area_C12, x = mass_X/mass_C12)

vs. the original miscalibrated curve (purified non-deuterated 2-F-C6H4-CN):

    y = 0.5070 x − 0.0701        (used in extract_gc_data_fbrcn.py)

Everything else is reused verbatim from extract_gc_data_fbrcn.py — vial→(T,L)
mapping (incl. the manual 260522-1/2 swap), V_reaction per dataset (200/250 µL),
the n_SM0 reference logic, peak windows, and SM (5-Br-2-F-CN) calibration.
The SM is still non-deuterated so its calibration is unchanged; only the
product (2-F-CN-d) calibration and its MW are updated.

The previous summary file is intentionally **not** overwritten — the new run
goes to experiment_fbrcn_d_calibrated_summary.csv and tags every row with the
previous yield_pct so Zhao can quickly read the systematic shift.
"""
import re, subprocess
from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path(__file__).parent
DATA_DIR = BASE / "experiment results" / "fbrcnbenzene"

# ---- vial → (T_C, L_cm) mapping (identical to extract_gc_data_fbrcn.py) ----
PLAN = {}
for i in range(15):
    T = -25 if i < 5 else (0 if i < 10 else 20)
    L = [3, 10, 25, 50, 100][i % 5]
    PLAN[f"WY-260522-{i+1}"] = (T, L)
for i in range(10):
    T = -70 if i < 5 else -50
    L = [3, 10, 25, 50, 100][i % 5]
    PLAN[f"WY-260523-{i+1}"] = (T, L)
# Zhao 2026-05-24: T=-25 vials 1/2 were mis-labelled — swap (same as original)
PLAN["WY-260522-1"] = (-25, 10)
PLAN["WY-260522-2"] = (-25, 3)

# ---- residence-time table & peak windows (identical to original) ----
L_TO_TR = {3: 0.0471, 10: 0.1571, 25: 0.3927, 50: 0.7854, 100: 1.5708}
PEAK_C12  = (8.90,  9.40)
PEAK_PROD = (9.85, 10.35)
PEAK_SUB  = (11.70, 12.45)

# ---- calibrations: SM unchanged (non-D); PROD updated to D standard ----
CAL_SM   = dict(slope=0.8886, intercept=-0.1214)       # 5-Br-2-F-CN (SM)
CAL_PROD = dict(slope=0.9092, intercept=-0.1201)       # 2-F-CN-5-d (NEW, Zhao 2026-06-04)

MW_SM   = 200.01           # 5-Br-2-F-CN, g/mol
MW_PROD = 122.12           # 2-F-CN-5-d, g/mol (121.11 + 1.006 for D ↔ H)
MW_C12  = 170.34           # n-C12, g/mol

V_REACTION = {260522: 200, 260523: 250}   # GC prep volumes (µL)


def parse_peaks(pdf):
    txt = subprocess.run(["pdftotext", "-layout", str(pdf), "-"],
                         capture_output=True, text=True).stdout
    pat = re.compile(r"^\s*\d+\s+(\d+\.\d+)\s+(\d+)\s+(\d+)\b")
    return [(float(m.group(1)), int(m.group(2)), int(m.group(3)))
            for line in txt.splitlines() if (m := pat.match(line))]


def assign_area(peaks, window):
    return sum(a for (rt, a, _) in peaks if window[0] <= rt <= window[1])


def mass_ratio(area_ratio, cal):
    return max((area_ratio - cal["intercept"]) / cal["slope"], 0.0)


def resolve_pdf(sample, T_C, L_cm):
    """Reuse the same fallback as the original: renamed file first, legacy second."""
    tag = f"T{T_C}"
    renamed = DATA_DIR / f"{sample}_{tag}_L{L_cm}.pdf"
    if renamed.exists():
        return renamed
    legacy = DATA_DIR / f"{sample}.pdf"
    return legacy if legacy.exists() else None


def main():
    rows = []
    for sample, (T_C, L_cm) in PLAN.items():
        pdf = resolve_pdf(sample, T_C, L_cm)
        if pdf is None:
            print(f"MISSING: {sample}"); continue
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
            "sample": sample,
            "date": int(sample.split('-')[1]),
            "vial": int(sample.split('-')[2]),
            "T_C": T_C, "L_cm": L_cm, "tR_s": L_TO_TR[L_cm],
            "c12_area": c12, "prod_area": prod, "sub_area": sub,
            "ar_sub": round(ar_sub, 4), "ar_prod": round(ar_prod, 4),
            "mr_sub": round(mr_sub, 4), "mr_prod_D": round(mr_prod, 4),
            "nr_sub": round(nr_sub, 4), "nr_prod_D": round(nr_prod, 4),
            "nr_total": round(nr_sub + nr_prod, 4),
        })
    df = pd.DataFrame(rows).sort_values(["T_C", "L_cm"]).reset_index(drop=True)

    # ---- V_reaction correction (same protocol as original) ----
    ref_date = max(V_REACTION, key=V_REACTION.get)   # 260523, V=250 µL
    ref_V    = V_REACTION[ref_date]
    n_SM0_ref = df[df["date"] == ref_date]["nr_total"].max()
    df["V_reaction_uL"]  = df["date"].map(V_REACTION)
    df["n_SM0_dataset"]  = n_SM0_ref * df["V_reaction_uL"] / ref_V

    df["yield_pct"]        = (df["nr_prod_D"] / df["n_SM0_dataset"] * 100).round(2)
    df["conversion_pct"]   = ((df["n_SM0_dataset"] - df["nr_sub"]) / df["n_SM0_dataset"] * 100).clip(0, 100).round(2)
    df["mass_balance_pct"] = (df["nr_total"]  / df["n_SM0_dataset"] * 100).round(2)

    # ---- attach previous (non-D-calibrated) yield for direct comparison ----
    prev = pd.read_csv(BASE / "experiment_fbrcn_summary.csv")[["sample", "yield_pct"]]
    prev = prev.rename(columns={"yield_pct": "prev_yield_pct"})
    df = df.merge(prev, on="sample", how="left")
    df["delta_pp"] = (df["yield_pct"] - df["prev_yield_pct"]).round(2)

    print("=" * 100)
    print("5-bromo-2-fluorobenzonitrile + n-BuLi + MeOD quench, RE-EXTRACTED with D-product calibration")
    print(f"  SM   cal (unchanged): y = {CAL_SM['slope']}x{CAL_SM['intercept']:+.4f}")
    print(f"  Prod cal (NEW, D):    y = {CAL_PROD['slope']}x{CAL_PROD['intercept']:+.4f}")
    print(f"  MW_PROD: 121.11 (H) → {MW_PROD} (D), +{MW_PROD-121.11:.3f} g/mol")
    print(f"  n_SM0_ref = {n_SM0_ref:.4f}  (from {ref_date}, V_reaction={ref_V} µL)")
    print("=" * 100)
    cols = ["sample", "T_C", "L_cm", "tR_s", "ar_sub", "ar_prod",
            "yield_pct", "prev_yield_pct", "delta_pp",
            "conversion_pct", "mass_balance_pct"]
    print(df[cols].to_string(index=False))

    out = BASE / "experiment_fbrcn_d_calibrated_summary.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}")

    print("\n=== D-calibrated yield (%) [rows=T, cols=L_cm] ===")
    print(df.pivot(index="T_C", columns="L_cm", values="yield_pct").to_string())
    print("\n=== Δ vs old non-D calibration (pp) ===")
    print(df.pivot(index="T_C", columns="L_cm", values="delta_pp").to_string())
    print("\n=== ratio new/old (mean across all cells) ===")
    r = (df["yield_pct"] / df["prev_yield_pct"]).dropna()
    print(f"  mean  = {r.mean():.3f}")
    print(f"  std   = {r.std():.3f}")
    print(f"  range = [{r.min():.3f}, {r.max():.3f}]")


if __name__ == "__main__":
    main()
