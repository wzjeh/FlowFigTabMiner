"""Backfill ``layout.json`` + ``context/`` for intermediates produced BEFORE
TF-ID started persisting crop geometry.

Each existing crop PNG is an exact sub-image of the page rendered by pdfium at
scale 4 (see ``ActiveAreaDetector.save_crops``), so its position is recovered
deterministically with normalised cross-correlation template matching — no
re-run of Florence-2, and every existing evidence file stays untouched.

Usage (inside the venv, from the project root):
  flowfigtabminer/bin/python scripts/backfill_layout.py "data/input/verify_2/*.pdf"
  flowfigtabminer/bin/python scripts/backfill_layout.py --all      # every intermediate with a locatable PDF
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import pypdfium2 as pdfium

from src.parsing.caption_locator import write_contexts

INTERMEDIATE = "data/intermediate"
COARSE = 4          # coarse search at 1/4 of the 4x render (= 1 px/pt)
MIN_SCORE = 0.85


def _pdf_index() -> dict:
    return {os.path.splitext(os.path.basename(p))[0]: p for p in glob.glob("data/input/**/*.pdf", recursive=True)}


def _crops(idir: str):
    for kind in ("figure", "table"):
        for p in sorted(glob.glob(os.path.join(idir, f"{kind}s", f"page_*_{kind}_*.png"))):
            m = re.match(r"page_(\d+)_(figure|table)_(\d+)\.png$", os.path.basename(p))
            if m:
                yield int(m.group(1)), kind, os.path.splitext(os.path.basename(p))[0], p


def _match(page_img: np.ndarray, crop: np.ndarray):
    """Return (x, y, score) of ``crop`` inside ``page_img`` (both 4x renders, gray)."""
    ph, pw = page_img.shape[:2]
    ch, cw = crop.shape[:2]
    if ch > ph or cw > pw:
        return None
    small_page = cv2.resize(page_img, (pw // COARSE, ph // COARSE), interpolation=cv2.INTER_AREA)
    small_crop = cv2.resize(crop, (max(1, cw // COARSE), max(1, ch // COARSE)), interpolation=cv2.INTER_AREA)
    res = cv2.matchTemplate(small_page, small_crop, cv2.TM_CCOEFF_NORMED)
    _, score, _, loc = cv2.minMaxLoc(res)
    x, y = loc[0] * COARSE, loc[1] * COARSE
    # refine at full resolution in a small window around the coarse hit
    pad = COARSE * 3
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(pw, x + cw + pad), min(ph, y + ch + pad)
    win = page_img[y0:y1, x0:x1]
    if win.shape[0] >= ch and win.shape[1] >= cw:
        res2 = cv2.matchTemplate(win, crop, cv2.TM_CCOEFF_NORMED)
        _, s2, _, l2 = cv2.minMaxLoc(res2)
        if s2 >= score - 0.05:
            x, y, score = x0 + l2[0], y0 + l2[1], max(score, s2)
    return x, y, float(score)


def backfill(pdf_path: str, idir: str) -> dict:
    layout = {"pdf": os.path.abspath(pdf_path), "detect_scale": 2, "crop_scale": 4, "sources": []}
    pdf = pdfium.PdfDocument(pdf_path)
    renders: dict = {}
    n_ok = n_bad = 0
    try:
        for page_no, kind, sid, crop_path in _crops(idir):
            if page_no - 1 >= len(pdf):
                continue
            if page_no not in renders:
                page = pdf[page_no - 1]
                pil = page.render(scale=4).to_pil().convert("L")
                renders[page_no] = (np.array(pil), page.get_size())
            page_img, (w_pt, h_pt) = renders[page_no]
            crop = cv2.imread(crop_path, cv2.IMREAD_GRAYSCALE)
            if crop is None:
                continue
            hit = _match(page_img, crop)
            if not hit or hit[2] < MIN_SCORE:
                n_bad += 1
                print(f"   ! {sid}: no confident match (score={hit[2] if hit else None})")
                continue
            x, y, score = hit
            ch, cw = crop.shape[:2]
            bbox_px = [x, y, x + cw, y + ch]
            layout["sources"].append({
                "source_id": sid, "kind": kind, "page": page_no,
                "page_size_pt": [round(w_pt, 2), round(h_pt, 2)],
                "bbox_px": [float(v) for v in bbox_px],
                "bbox_pt": [round(v / 4.0, 2) for v in bbox_px],
                "crop_path": crop_path,
                "geometry_source": "backfill_template_match",
                "match_score": round(score, 3),
            })
            n_ok += 1
    finally:
        pdf.close()
    with open(os.path.join(idir, "layout.json"), "w") as f:
        json.dump(layout, f, indent=2)
    print(f"[backfill] {os.path.basename(idir)}: {n_ok} located, {n_bad} unmatched")
    return layout


def main(argv):
    index = _pdf_index()
    if argv and argv[0] == "--all":
        targets = [(index[b], os.path.join(INTERMEDIATE, b)) for b in sorted(os.listdir(INTERMEDIATE))
                   if b in index and os.path.isdir(os.path.join(INTERMEDIATE, b))]
    else:
        pdfs = [p for a in argv for p in (glob.glob(a) or [a])]
        targets = [(p, os.path.join(INTERMEDIATE, os.path.splitext(os.path.basename(p))[0])) for p in pdfs]
    total_res = total_src = 0
    for pdf_path, idir in targets:
        if not os.path.isdir(idir) or not os.path.exists(pdf_path):
            print(f"[backfill] skip {idir} (missing dir or pdf)")
            continue
        layout = backfill(pdf_path, idir)
        total_src += len(layout["sources"])
        total_res += write_contexts(pdf_path, idir)
    print(f"[backfill] DONE: {total_res}/{total_src} sources resolved to PDF-text captions")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    main(sys.argv[1:])
