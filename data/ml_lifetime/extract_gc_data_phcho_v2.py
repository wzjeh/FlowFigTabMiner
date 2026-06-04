"""
Extract GC data from PhCHO-trap experiment, parsing the new filename
convention WY-{date}-{N}（T-L）.pdf where (T-L) encodes °C and cm.

L = 2.5 cm  → half-bore tube (Zhao's L=10 label, kept as separate tR point)
L = 3, 10, 25, 50, 100 cm → standard 500 μm tube

Latest-wins policy: WY-260514-* re-runs override original vials at the
same (T, L).

Peak windows:
  6.6-6.95 min  →  substrate 4-Br-FC6H4
  8.5-9.05 min  →  PhCHO trap reagent residual
  9.05-9.35 min →  C12 internal standard
  11.4-11.9 min →  Bu-CH(OH)-Ph (n-BuLi residue × PhCHO)
  13.5-14.5 min →  (4-F-C6H4)(Ph)CHOH  main trap product
  15.0-15.8 min →  minor side product

Calibration (2026-05-13 fix):
  mol_substrate (30 s)  = 0.10 M × 6 mL/min × 0.5 min = 0.30 mmol
  mol_C12 (1 mL spike)  = 0.89/170.34 × 0.01 × 1000   = 0.0523 mmol
  → mol_ratio @ 0% conv = 5.74
"""

import re
from pathlib import Path

import fitz
import numpy as np
import pandas as pd

PDF_DIR = Path(__file__).parent / "experiment results"
OUT_CSV = Path(__file__).parent / "experiment_phcho_summary_v2.csv"

PK_SUBST   = (6.6, 6.95)
PK_PHCHO   = (8.5, 9.05)
PK_C12     = (9.05, 9.35)
PK_BU_PRD  = (11.4, 11.9)
PK_PRODUCT = (13.5, 14.5)
PK_SIDE    = (15.0, 15.8)

# Filename pattern: (T-L)  with Chinese parens
PAT = re.compile(r"WY-(\d{6})-(\d+)（(-?\d+\.?\d*)-(\d+\.?\d*)）\.pdf")

TUBE_AREA_CM2 = np.pi * 0.025 ** 2
TOTAL_FLOW_ML_S = 7.5 / 60      # subst + n-BuLi only (reaction reactor)
# Calibration constants
MW_C12 = 170.34
mol_subst_30s = 0.10 * 6.0 * 0.5             # 0.30 mmol
mol_C12_1mL   = (0.89/100.0) / MW_C12 * 1.0 * 1000   # 0.0523 mmol
MOL_RATIO_AT_0_CONV = mol_subst_30s / mol_C12_1mL    # 5.74
RRF_SUB = 5.85 / 12.0
AREA_RATIO_AT_0_CONV = MOL_RATIO_AT_0_CONV * RRF_SUB     # 2.80
# Yield: empirical anchor (mean r_prd at conv ≥ 95% = 6.19)
AREA_RATIO_AT_100_YIELD = 6.19


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
    rt_toks   = extract_section(text, "保持時間", "面積")
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
        if not m:
            continue
        date, vial, T_s, L_s = m.groups()
        T_C = float(T_s); L_cm = float(L_s)

        # Effective tR: L=2.5 is half-bore-tube error (already in cm-equiv)
        tR_s = TUBE_AREA_CM2 * L_cm / TOTAL_FLOW_ML_S
        # L=2.5 cm came from half-bore tube at L=10cm (volume × 1/4 → equivalent 2.5 cm)
        # tR formula already correct because L_cm = 2.5

        peaks = extract_peaks(pdf)
        ps = find_peak(peaks, *PK_SUBST)
        pp = find_peak(peaks, *PK_PHCHO)
        pc = find_peak(peaks, *PK_C12)
        pb = find_peak(peaks, *PK_BU_PRD)
        pr = find_peak(peaks, *PK_PRODUCT)
        psi = find_peak(peaks, *PK_SIDE)

        if ps and pc and pc["area"]:
            r_sub_obs = ps["area"] / pc["area"]
            residual_pct = 100 * r_sub_obs / AREA_RATIO_AT_0_CONV
            conv_pct = 100 - residual_pct
        elif pc and pc["area"] and ps is None:
            r_sub_obs = 0.0; residual_pct = 0.0; conv_pct = 100.0
        else:
            r_sub_obs = residual_pct = conv_pct = None

        if pr and pc and pc["area"]:
            r_prd_obs = pr["area"] / pc["area"]
            yield_pct = 100 * r_prd_obs / AREA_RATIO_AT_100_YIELD
        else:
            r_prd_obs = yield_pct = None

        rows.append({
            "date": date, "vial": int(vial),
            "T_C": T_C, "L_cm": L_cm, "tR_s": round(tR_s, 4),
            "subst_area": ps["area"] if ps else None,
            "phcho_area": pp["area"] if pp else None,
            "c12_area":   pc["area"] if pc else None,
            "bu_prd_area": pb["area"] if pb else None,
            "prod_area":  pr["area"] if pr else None,
            "side_area":  psi["area"] if psi else None,
            "r_sub": round(r_sub_obs, 5) if r_sub_obs is not None else None,
            "r_prd": round(r_prd_obs, 5) if r_prd_obs is not None else None,
            "residual_pct":   round(residual_pct, 2) if residual_pct is not None else None,
            "conversion_pct": round(conv_pct, 2) if conv_pct is not None else None,
            "trap_yield_pct": round(yield_pct, 2) if yield_pct is not None else None,
        })

    df = pd.DataFrame(rows)

    # Latest-wins by default; but Zhao prefers ORIGINAL data for these (T, L)
    # because the new run came in lower (suspected n-BuLi stock issue 05-14):
    PREFER_OLD = {(0.0, 3.0)}                # use 05-12 vial #6 = 85%, not 05-14 = 79%
    df["date"] = df["date"].astype(int)
    df_old = df[df["date"] == 260512].copy()
    df_new = df[df["date"] != 260512].copy()
    # Drop new rows that should defer to old at same (T, L)
    df_new = df_new[~df_new.apply(
        lambda r: (r["T_C"], r["L_cm"]) in PREFER_OLD, axis=1)]
    df = pd.concat([df_old, df_new], ignore_index=True)
    df = (df.sort_values(["T_C","L_cm","date"])
            .drop_duplicates(["T_C","L_cm"], keep="last")
            .sort_values(["T_C","L_cm"], ascending=[False, True]))
    df.to_csv(OUT_CSV, index=False)
    cols = ["date","vial","T_C","L_cm","tR_s","subst_area","c12_area",
            "prod_area","conversion_pct","trap_yield_pct"]
    print(f"Saved: {OUT_CSV}   ({len(df)} unique (T,L) points)")
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
