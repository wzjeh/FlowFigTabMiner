"""
Extract peak data from Shimadzu GC PDFs and compute SUBSTRATE CONVERSION.

NOTE (2026-05-12): the 6.7 min peak is the SUBSTRATE (4-bromofluorobenzene),
NOT the product fluorobenzene. So area_ratio measures residual substrate.
Therefore:
    residual_subst%  = (area_ratio_obs / area_ratio_initial) × 100
    conversion%      = 100 - residual_subst%

Since 1 ArBr → 1 ArLi → 1 ArH (perfect mass balance) and ECN(4-BrFC6H4) ≈
ECN(FC6H5) ≈ 5.85, the same denominator (mol_ratio × RRF ≈ 0.488)
approximates both "100% yield" and "0% conversion". Hence
conversion% ≈ 100 - old_yield_pct_ECN. The systematic RRF error
(Br vs F atom) is < 6%.

Reaction setup (φ500μm, T-junction):
  substrate (4-bromofluorobenzene, MW=175.0, in 50mL THF): 6 mL/min
  n-BuLi (0.42 M in hexane):                              1.5 mL/min
  MeOH (0.6 M in THF, quench):                            3 mL/min
  Total quenched flow: 10.5 mL/min

Internal-standard workup:
  300 µL 0.05 M C12 (n-dodecane) + 300 µL reaction-organic-phase → GC vial
  → C12 in vial = 0.025 M (constant)
  → substrate in vial @ 0% conversion = [substrate]₀ × (6/10.5) × 0.5

Substrate weights per T group (Zhao's notebook).

GC FID response factor (ECN method):
  ECN(C12)=12.0,  ECN(4-BrFC6H4)≈5.85  →  RRF ≈ 0.488
  ⚠ ECN-estimated; absolute conversion% has ±6% systematic error.
"""

import re
from pathlib import Path

import fitz
import numpy as np
import pandas as pd

PDF_DIR = Path(__file__).parent / "experiment results"
OUT_CSV = Path(__file__).parent / "experiment_yield_summary.csv"

SUBSTRATE_RT = (6.6, 6.95)    # 4-bromofluorobenzene (was mis-labeled "product")
C12_RT       = (9.0, 9.35)

MW_SUBSTRATE = 175.0
SOLVENT_VOL_ML = 50.0
SUBST_FLOW = 6.0
NBU_FLOW   = 1.5
MEOH_FLOW  = 3.0
TOTAL_FLOW = SUBST_FLOW + NBU_FLOW + MEOH_FLOW   # 10.5 mL/min

C12_CONC_VIAL = 0.025          # M (300µL 0.05M / 600µL)
RRF_PhF_C12   = 5.85 / 12.0    # ECN-based, 0.4875

# --- substrate weights (mg) by (T_C, L_cm) ---
WEIGHT_MAP: dict[tuple[int,int], float] = {}
for L in (3, 10, 25, 50):
    WEIGHT_MAP[(0, L)] = 834
WEIGHT_MAP[(0, 100)] = 864
for L in (3, 10, 25, 50, 100):
    WEIGHT_MAP[(-25, L)] = 864
for L in (3, 10, 25, 50):
    WEIGHT_MAP[(-50, L)] = 887
WEIGHT_MAP[(-50, 100)] = 870
WEIGHT_MAP[(-78, 3)] = 834
for L in (10, 25, 50, 100):
    WEIGHT_MAP[(-78, L)] = 870
for L in (3, 10, 25, 50, 100):
    WEIGHT_MAP[(20, L)] = 834


def parse_filename(name: str):
    if name.startswith("Lithium"):
        m = re.search(r"T(-?\d+)-(\d+)cm", name)
        return (int(m.group(1)), int(m.group(2))) if m else None
    if name.startswith("WY-260509-"):
        m = re.search(r"WY-260509-(\d+)-(\d+)", name)
        if not m:
            return None
        t_raw, l = int(m.group(1)), int(m.group(2))
        return (t_raw if t_raw == 20 else -t_raw, l)
    return None


def extract_section(text: str, start: str, end: str):
    if start not in text:
        return []
    after = text.split(start, 1)[1]
    if end in after:
        after = after.split(end, 1)[0]
    return after.split()


def extract_peaks(pdf_path: Path):
    doc = fitz.open(pdf_path)
    text = "\n".join(p.get_text() for p in doc)
    doc.close()
    if "ピークレポート" not in text:
        return []
    text = text.split("ピークレポート", 1)[1]

    rt_toks   = extract_section(text, "保持時間", "面積")
    area_toks = extract_section(text, "面積", "高さ")
    rts   = [float(t) for t in rt_toks   if re.fullmatch(r"\d+\.?\d*", t)]
    areas = [int(t)   for t in area_toks if re.fullmatch(r"\d+", t)]
    if len(areas) > len(rts):
        areas = areas[:len(rts)]
    return [{"rt": rt, "area": areas[i] if i < len(areas) else None}
            for i, rt in enumerate(rts)]


def find_peak_in_window(peaks, lo, hi):
    matches = [p for p in peaks if lo <= p["rt"] <= hi]
    return max(matches, key=lambda p: p["area"] or 0) if matches else None


def main():
    rows = []

    for pdf in sorted(PDF_DIR.glob("*.pdf")):
        name = pdf.name
        peaks = extract_peaks(pdf)

        if "F-benzene" in name:
            print(f"[REF standard] {name}: {[(p['rt'], p['area']) for p in peaks]}")
            continue

        parsed = parse_filename(name)
        if parsed is None:
            print(f"[SKIP] {name}")
            continue
        T_C, L_cm = parsed
        weight_mg = WEIGHT_MAP.get((T_C, L_cm))
        if weight_mg is None:
            print(f"[WARN] no weight for ({T_C}, {L_cm})")

        subst = find_peak_in_window(peaks, *SUBSTRATE_RT)
        c12   = find_peak_in_window(peaks, *C12_RT)

        # tR for φ500µm, total flow 7.5 mL/min in reaction segment
        tube_vol_per_cm = np.pi * (0.025 ** 2)        # mL/cm  (r=250µm)
        tR_s = tube_vol_per_cm * L_cm / (7.5 / 60)

        # substrate concentration in storage solution (50 mL)
        subst_conc = (weight_mg / MW_SUBSTRATE) / SOLVENT_VOL_ML * 1000 / 1000  # M
        # in vial @ 0% conversion: all initial substrate intact
        subst_conc_vial_0pct = subst_conc * (SUBST_FLOW / TOTAL_FLOW)
        mol_ratio_0pct = subst_conc_vial_0pct / 0.05
        # expected area ratio if no reaction occurred (RRF for 4-BrFC6H4)
        area_ratio_0pct = mol_ratio_0pct * RRF_PhF_C12

        if subst and c12 and c12["area"]:
            area_ratio_obs = subst["area"] / c12["area"]
            residual_pct   = 100.0 * area_ratio_obs / area_ratio_0pct
            conversion_pct = 100.0 - residual_pct
        else:
            area_ratio_obs = None
            residual_pct   = None
            conversion_pct = None

        rows.append({
            "file": name,
            "T_C": T_C,
            "L_cm": L_cm,
            "tR_s": round(tR_s, 4),
            "weight_mg": weight_mg,
            "subst_M": round(subst_conc, 4),
            "subst_rt": subst["rt"] if subst else None,
            "subst_area": subst["area"] if subst else None,
            "c12_rt": c12["rt"] if c12 else None,
            "c12_area": c12["area"] if c12 else None,
            "area_ratio_obs": round(area_ratio_obs, 5) if area_ratio_obs else None,
            "area_ratio_0pct": round(area_ratio_0pct, 5),
            "residual_subst_pct_ECN":  round(residual_pct, 2) if residual_pct else None,
            "conversion_pct_ECN":      round(conversion_pct, 2) if conversion_pct is not None else None,
            # keep legacy column name for backward compat with downstream scripts
            "yield_pct_ECN":           round(residual_pct, 2) if residual_pct else None,
        })

    df = pd.DataFrame(rows).sort_values(["T_C", "L_cm"], ascending=[False, True])
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved: {OUT_CSV}  ({len(df)} rows)")
    cols = ["T_C", "L_cm", "tR_s", "weight_mg", "subst_M",
            "area_ratio_obs", "area_ratio_0pct",
            "residual_subst_pct_ECN", "conversion_pct_ECN"]
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
