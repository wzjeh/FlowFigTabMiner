"""
Extract GC data for new PhCHO-trap experiment (2026-05-12 batch).

Peak identification:
  ~6.75 min  →  4-Br-FC6H4 (substrate residual)
  ~8.85 min  →  PhCHO (苯甲醛, BP 178°C, trap reagent residue)
  ~9.16 min  →  C12 (n-dodecane, internal standard)
  ~14.0 min  →  (4-F-C6H4)(Ph)CHOH  trap product ⭐
  ~15.4 min  →  side product (minor)

Output: extracts substrate area, PhCHO area, C12 area, trap-product area,
side-product area for each PDF.  Maps vial → (T, L) using a default
hypothesis (5 L × 4 T + 1 -65°C); user can override.
"""

import re
from pathlib import Path

import fitz
import numpy as np
import pandas as pd

PDF_DIR = Path(__file__).parent / "experiment results"
OUT_CSV = Path(__file__).parent / "experiment_phcho_summary.csv"

# Peak windows (min)
PK_SUBST   = (6.6, 6.95)
PK_PHCHO   = (8.5, 9.05)
PK_C12     = (9.05, 9.35)
PK_PRODUCT = (13.5, 14.5)
PK_SIDE    = (15.0, 15.8)

# Default hypothesis: WY-260512-{i} mapping (please verify with Zhao)
# 1..5  : T=+20°C, L=3,10,25,50,100
# 6..10 : T=  0°C, L=3,10,25,50,100
# 11..15: T=-25°C, L=3,10,25,50,100
# 16..20: T=-50°C, L=3,10,25,50,100
# 21    : T=-65°C, L=3
DEFAULT_MAP = {}
T_ORDER = [20, 0, -25, -50]
L_ORDER = [3, 10, 25, 50, 100]
i = 1
for T in T_ORDER:
    for L in L_ORDER:
        DEFAULT_MAP[i] = (T, L)
        i += 1
DEFAULT_MAP[21] = (-65, 3)
# Vials 22-25 appeared 2026-05-13; tentative mapping: -65°C × L=10,25,50,100
DEFAULT_MAP[22] = (-65, 10)
DEFAULT_MAP[23] = (-65, 25)
DEFAULT_MAP[24] = (-65, 50)
DEFAULT_MAP[25] = (-65, 100)


# Reagent / dilution constants (same as before)
MW_SUBSTRATE = 175.0
MW_C12 = 170.34            # n-dodecane
SOLVENT_VOL_ML = 100.0     # new batch: 1.75 g / 100 mL = 0.10 M
SUBST_FLOW = 6.0
NBU_FLOW   = 1.5
PHCHO_FLOW = 3.0
TOTAL_FLOW = SUBST_FLOW + NBU_FLOW + PHCHO_FLOW   # 10.5 mL/min

# NEW protocol (260512): 30s sample (whole quench output) + 1 mL of 0.89 g C12 / 100 mL EtOAc
SAMPLE_TIME_S = 30.0
C12_STOCK_G_PER_100ML = 0.89
C12_SPIKE_ML = 1.0
# mol_subst in 30 s of substrate flow (0.10 M × 6 mL/min × 30 s = 0.30 mmol):
mol_subst_per_sample_mmol = 0.10 * SUBST_FLOW * (SAMPLE_TIME_S / 60.0)
# mol C12 added: 0.89 g / 100 mL × 1 mL spike = 8.9 mg / 170.34 = 0.0523 mmol:
mol_C12_per_vial_mmol = (C12_STOCK_G_PER_100ML * (C12_SPIKE_ML / 100.0)) / MW_C12 * 1000.0
MOL_RATIO_AT_0_CONV = mol_subst_per_sample_mmol / mol_C12_per_vial_mmol  # ≈ 5.74

# ECN-based RRF
ECN_C12 = 12.0
ECN_SUB = 5.85       # 4-Br-FC6H4 ≈ ECN 5.85 (Br ≈ 0, F ≈ 0, 6 aromatic C)
ECN_PRD = 12.0       # (4-F-C6H4)(Ph)CHOH ≈ 13 aromatic C - 1 oxidation correction ≈ 12
RRF_SUB = ECN_SUB / ECN_C12     # 0.488
RRF_PRD = ECN_PRD / ECN_C12     # 1.0


def extract_section(text, start, end):
    if start not in text: return []
    after = text.split(start, 1)[1]
    if end in after: after = after.split(end, 1)[0]
    return after.split()


def extract_peaks(pdf_path):
    doc = fitz.open(pdf_path)
    text = "\n".join(p.get_text() for p in doc)
    doc.close()
    if "ピークレポート" not in text: return []
    text = text.split("ピークレポート", 1)[1]
    rt_toks   = extract_section(text, "保持時間", "面積")
    area_toks = extract_section(text, "面積", "高さ")
    rts   = [float(t) for t in rt_toks   if re.fullmatch(r"\d+\.?\d*", t)]
    areas = [int(t)   for t in area_toks if re.fullmatch(r"\d+", t)]
    if len(areas) > len(rts): areas = areas[:len(rts)]
    return [{"rt": rt, "area": areas[i] if i < len(areas) else None}
            for i, rt in enumerate(rts)]


def find_peak(peaks, lo, hi):
    matches = [p for p in peaks if lo <= p["rt"] <= hi]
    return max(matches, key=lambda p: p["area"] or 0) if matches else None


def main():
    rows = []
    for pdf in sorted(PDF_DIR.glob("WY-260512-*.pdf")):
        m = re.search(r"WY-260512-(\d+)\.pdf", pdf.name)
        if not m: continue
        vial = int(m.group(1))

        peaks = extract_peaks(pdf)
        ps = find_peak(peaks, *PK_SUBST)
        pp = find_peak(peaks, *PK_PHCHO)
        pc = find_peak(peaks, *PK_C12)
        pr = find_peak(peaks, *PK_PRODUCT)
        psi = find_peak(peaks, *PK_SIDE)

        T_C, L_cm = DEFAULT_MAP.get(vial, (None, None))
        if L_cm is not None:
            tube_vol_per_cm = np.pi * (0.025 ** 2)
            tR_s = tube_vol_per_cm * L_cm / (7.5 / 60)
        else:
            tR_s = None

        # ---- NEW protocol calibration (2026-05-13 fix) ----
        # Zhao adds 1 mL of 0.89 g/100 mL C12 spike DIRECTLY into the entire 30s sample.
        # mol_subst (in 30s sample at 0% conv) = 0.30 mmol
        # mol_C12 spiked = 0.0523 mmol
        # → mol_ratio @ 0% conv = 5.74, area_ratio = 5.74 × RRF_SUB ≈ 2.80
        area_ratio_0 = MOL_RATIO_AT_0_CONV * RRF_SUB

        if ps and pc and pc["area"]:
            r_sub_obs = ps["area"] / pc["area"]
            residual_pct = 100 * r_sub_obs / area_ratio_0
            conversion_pct = 100 - residual_pct
        elif pc and pc["area"] and ps is None:
            # No substrate peak found → fully consumed → conversion ≈ 100%
            r_sub_obs = 0.0
            residual_pct = 0.0
            conversion_pct = 100.0
        else:
            r_sub_obs = residual_pct = conversion_pct = None

        # trap product yield with new protocol
        # mol_product (100% yield) = 0.30 mmol, same C12 spike
        area_ratio_prd_100 = MOL_RATIO_AT_0_CONV * RRF_PRD

        if pr and pc and pc["area"]:
            r_prd_obs = pr["area"] / pc["area"]
            yield_pct = 100 * r_prd_obs / area_ratio_prd_100
        else:
            r_prd_obs = yield_pct = None

        rows.append({
            "vial": vial,
            "T_C_guess": T_C, "L_cm_guess": L_cm, "tR_s_guess": tR_s,
            "subst_rt":  ps["rt"]   if ps else None,
            "subst_area":ps["area"] if ps else None,
            "phcho_rt":  pp["rt"]   if pp else None,
            "phcho_area":pp["area"] if pp else None,
            "c12_rt":    pc["rt"]   if pc else None,
            "c12_area":  pc["area"] if pc else None,
            "prod_rt":   pr["rt"]   if pr else None,
            "prod_area": pr["area"] if pr else None,
            "side_rt":   psi["rt"]  if psi else None,
            "side_area": psi["area"]if psi else None,
            "r_sub":     round(r_sub_obs, 5) if r_sub_obs else None,
            "r_prd":     round(r_prd_obs, 5) if r_prd_obs else None,
            "residual_pct":   round(residual_pct, 2) if residual_pct else None,
            "conversion_pct": round(conversion_pct, 2) if conversion_pct is not None else None,
            "trap_yield_pct": round(yield_pct, 2) if yield_pct else None,
        })

    df = pd.DataFrame(rows).sort_values("vial")
    df.to_csv(OUT_CSV, index=False)

    print(f"area_ratio_at_0%_conv (substrate)  = {area_ratio_0:.4f}")
    print(f"area_ratio_at_100%_yield (product) = {area_ratio_prd_100:.4f}")
    print()
    print(f"Extracted {len(df)} rows. Saved: {OUT_CSV}\n")
    cols = ["vial","T_C_guess","L_cm_guess","tR_s_guess",
            "subst_area","phcho_area","c12_area","prod_area",
            "conversion_pct","trap_yield_pct"]
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
