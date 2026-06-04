"""
Extract GC data for 3,5-Dibromobenzonitrile · n-BuLi · TMSCl-trap experiment
(WY-260515 batch, 25 vials, T×L = 5×5 grid).

Peak assignments (Zhao 2026-05-18):
  5.27 min   : TMSCl-related impurity (vanishes at T ≥ 0°C)
  9.36 min   : C12 internal standard
  11.92 min  : m-bromobenzonitrile (3-Br-C6H4-CN, proto-de-Li by-product)
  12.79 min  : SUBSTRATE 3,5-dibromobenzonitrile
  13.27 min  : Misc side product (irregular)
  14.23 min  : MAIN PRODUCT 3-Br-5-TMS-C6H3-CN
  14.44 min  : ortho-CN-deprotonation / TMS-trap by-product
               i.e. 3,5-Dibromo-2-(trimethylsilyl)benzonitrile
  15.4-15.7  : high-boiling side products (benzyne pathway at +20°C)

Concentration (2026-05-18): substrate 0.10 M in THF, flow 6 mL/min
                            n-BuLi 0.43 M in hexane, flow 1.5 mL/min
                            TMSCl quench (3 mL/min)
Vial 5 (-65°C, L=100) EXCLUDED — anomalous GC trace (multiple extra peaks).

Calibration (ECN-based RRF):
  mol_subst (30 s)  = 0.10 M × 6 mL/min × 0.5 min = 0.30 mmol
  mol_C12  (1 mL)   = 0.0523 mmol
  mol_ratio @ 0% conv = 5.74
  ECN_subst (3,5-Br2-CN) ≈ 5.0   → RRF = 5.0/12 = 0.417
  ECN_prod  (3-Br-5-TMS-CN) ≈ 7.2 → RRF = 0.60 (used for product yield)
  area_ratio @ 0% conv  = 5.74 × 0.417 = 2.39
  area_ratio @ 100% yld = 5.74 × 0.60  = 3.44
"""

import re
from pathlib import Path

import fitz
import numpy as np
import pandas as pd

PDF_DIR = Path(__file__).parent / "experiment results" / "35brcnbenzene"
OUT_CSV = Path(__file__).parent / "experiment_35brcn_summary.csv"

# Peak windows (min)
PK_IMPURITY = (5.10, 5.40)
PK_C12      = (9.20, 9.50)
PK_ARH      = (11.70, 12.10)   # 11.92 — m-bromobenzonitrile
PK_SUBST    = (12.65, 12.95)   # 12.79 — 3,5-dibromobenzonitrile (substrate)
PK_MISC     = (13.15, 13.45)   # 13.27 — irregular
PK_PRODUCT  = (14.10, 14.35)   # 14.23 — main 3-Br-5-TMS-CN
PK_ORTHO    = (14.36, 14.55)   # 14.44 — ortho-deprot/TMS

# Filename pattern with Chinese fullwidth parens
PAT = re.compile(r"WY-(\d{6})-(\d+)（(-?\d+\.?\d*)-(\d+\.?\d*)）\.pdf")

# Tube geometry
TUBE_AREA_CM2 = np.pi * 0.025 ** 2
TOTAL_FLOW_ML_S = 7.5 / 60      # substrate + n-BuLi only (reactor section)

# Calibration constants
SUBST_M = 0.10
SUBST_ML_MIN = 6.0
SAMPLE_MIN = 0.5
C12_STOCK_PCT = 0.89         # 0.89 g per 100 mL
C12_SPIKE_ML = 1.0
MW_C12 = 170.34

mol_subst_30s = SUBST_M * SUBST_ML_MIN * SAMPLE_MIN              # 0.30 mmol
mol_C12_1mL = (C12_STOCK_PCT / 100.0) / MW_C12 * C12_SPIKE_ML * 1000.0  # 0.0523 mmol
MOL_RATIO_AT_0_CONV = mol_subst_30s / mol_C12_1mL                # 5.74

# ECN-based RRF
ECN_C12 = 12.0
ECN_SUBST = 5.0     # 3,5-Br2-C6H3-CN (rough estimate)
ECN_PROD = 7.2      # 3-Br-5-TMS-C6H3-CN (rough estimate)
RRF_SUBST = ECN_SUBST / ECN_C12
RRF_PROD = ECN_PROD / ECN_C12

AREA_RATIO_AT_0_CONV  = MOL_RATIO_AT_0_CONV * RRF_SUBST   # ≈ 2.39
AREA_RATIO_AT_100_YLD = MOL_RATIO_AT_0_CONV * RRF_PROD    # ≈ 3.44

# Exclude vial 5
EXCLUDE_VIALS = {5}


def extract_section(text, start, end):
    if start not in text: return []
    after = text.split(start, 1)[1]
    if end in after: after = after.split(end, 1)[0]
    return after.split()


def extract_peaks(pdf_path):
    doc = fitz.open(pdf_path)
    text = "\n".join(p.get_text() for p in doc)
    doc.close()
    if "ピークレポート" not in text:
        return []
    text = text.split("ピークレポート", 1)[1]
    rt_toks = extract_section(text, "保持時間", "面積")
    area_toks = extract_section(text, "面積", "高さ")
    rts = [float(t) for t in rt_toks if re.fullmatch(r"\d+\.?\d*", t)]
    areas = [int(t) for t in area_toks if re.fullmatch(r"\d+", t)]
    if len(areas) > len(rts):
        areas = areas[:len(rts)]
    return [{"rt": rt, "area": areas[i] if i < len(areas) else None}
            for i, rt in enumerate(rts)]


def find_peak(peaks, lo, hi):
    matches = [p for p in peaks if lo <= p["rt"] <= hi]
    return max(matches, key=lambda p: p["area"] or 0) if matches else None


def main():
    rows = []
    for pdf in sorted(PDF_DIR.glob("WY-*.pdf")):
        m = PAT.match(pdf.name)
        if not m: continue
        date, vial, T_s, L_s = m.groups()
        vial = int(vial)
        if vial in EXCLUDE_VIALS:
            continue
        T_C = float(T_s); L_cm = float(L_s)
        tR_s = TUBE_AREA_CM2 * L_cm / TOTAL_FLOW_ML_S

        peaks = extract_peaks(pdf)
        p_imp  = find_peak(peaks, *PK_IMPURITY)
        p_c12  = find_peak(peaks, *PK_C12)
        p_arh  = find_peak(peaks, *PK_ARH)
        p_sub  = find_peak(peaks, *PK_SUBST)
        p_mis  = find_peak(peaks, *PK_MISC)
        p_prd  = find_peak(peaks, *PK_PRODUCT)
        p_ort  = find_peak(peaks, *PK_ORTHO)

        c12_a = p_c12["area"] if p_c12 else None
        if p_sub and c12_a:
            r_sub = p_sub["area"] / c12_a
            residual_pct = 100.0 * r_sub / AREA_RATIO_AT_0_CONV
            conv_pct = max(0.0, 100.0 - residual_pct)
        else:
            r_sub = residual_pct = conv_pct = None

        if p_prd and c12_a:
            r_prd = p_prd["area"] / c12_a
            yield_pct = 100.0 * r_prd / AREA_RATIO_AT_100_YLD
        else:
            r_prd = yield_pct = None

        if p_arh and c12_a:
            r_arh = p_arh["area"] / c12_a
            arh_pct = 100.0 * r_arh / AREA_RATIO_AT_0_CONV   # use subst RRF as proxy
        else:
            r_arh = arh_pct = None

        if p_ort and c12_a:
            r_ort = p_ort["area"] / c12_a
            ortho_pct = 100.0 * r_ort / AREA_RATIO_AT_100_YLD
        else:
            r_ort = ortho_pct = None

        rows.append({
            "date": int(date), "vial": vial,
            "T_C": T_C, "L_cm": L_cm, "tR_s": round(tR_s, 4),
            "imp_area":   p_imp["area"] if p_imp else None,
            "c12_area":   c12_a,
            "arh_area":   p_arh["area"] if p_arh else None,
            "subst_area": p_sub["area"] if p_sub else None,
            "misc_area":  p_mis["area"] if p_mis else None,
            "prod_area":  p_prd["area"] if p_prd else None,
            "ortho_area": p_ort["area"] if p_ort else None,
            "r_sub": round(r_sub, 4) if r_sub is not None else None,
            "r_prd": round(r_prd, 4) if r_prd is not None else None,
            "r_arh": round(r_arh, 4) if r_arh is not None else None,
            "r_ort": round(r_ort, 4) if r_ort is not None else None,
            "conversion_pct":   round(conv_pct, 2) if conv_pct is not None else None,
            "trap_yield_pct":   round(yield_pct, 2) if yield_pct is not None else None,
            "proto_deLi_pct":   round(arh_pct, 2) if arh_pct is not None else None,
            "ortho_deprot_pct": round(ortho_pct, 2) if ortho_pct is not None else None,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values(["T_C", "L_cm"], ascending=[False, True]).reset_index(drop=True)
    df.to_csv(OUT_CSV, index=False)

    cols = ["vial","T_C","L_cm","tR_s","subst_area","c12_area","prod_area",
            "conversion_pct","trap_yield_pct","proto_deLi_pct","ortho_deprot_pct"]
    print(f"\nCalibration:")
    print(f"  mol_ratio @ 0% conv      = {MOL_RATIO_AT_0_CONV:.3f}")
    print(f"  RRF_subst (ECN={ECN_SUBST}) = {RRF_SUBST:.3f}")
    print(f"  RRF_prod  (ECN={ECN_PROD}) = {RRF_PROD:.3f}")
    print(f"  area_ratio @ 0% conv     = {AREA_RATIO_AT_0_CONV:.3f}")
    print(f"  area_ratio @ 100% yield  = {AREA_RATIO_AT_100_YLD:.3f}")
    print(f"\nSaved: {OUT_CSV}   ({len(df)} vials, vial 5 excluded)")
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
