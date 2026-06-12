"""Issue #14 Phase-2 — MolNexTR concurrency / micro-batch feasibility experiment.

READ-ONLY w.r.t. the pipeline: this script does NOT touch the production code
path.  It loads the SAME singleton MolNexTR model and runs a fixed set of real
molecule crops (cut from a structure-dense table with the SAME crop geometry the
pipeline uses) through three strategies, then reports SMILES consistency and
wall-clock speedup:

  1. baseline  — per-box  predict_images([crop])     (current pipeline behaviour)
  2. microbatch — one call predict_images(all, bs=N)  (uses native batch path)
  3. concurrent-K — ThreadPoolExecutor(K) each calling predict_images([crop])

Consistency = RDKit-canonical SMILES per box vs the baseline (ground truth).
A strategy is only viable if it is byte-for-byte consistent AND faster.

Usage:
  flowfigtabminer/bin/python scripts/exp_molnextr_concurrency.py \
      --table "data/intermediate/.../page_6_table_0/page_6_table_0_body_main.png" \
      --limit 27 --workers 2 4 --batch 8 16
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.extraction.common.molnextr.molnextr import MolNexTRSingleton  # noqa: E402
from src.adjudication.entity_pool import canonical_smiles  # noqa: E402

YOLO_MODEL = "models/yolo11s-tab-molecule-0207/runs/detect/train/weights/best.pt"


def cut_crops(table_path: str, conf=0.25):
    """Replicate MoleculeProcessor's YOLO detect + crop geometry exactly
    (margin=10, white pad=30, upscale if short side < 192px, max 3x)."""
    from ultralytics import YOLO
    import torch

    if torch.get_num_threads() > 1:
        torch.set_num_threads(1)
    model = YOLO(YOLO_MODEL)
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(dev)

    img = cv2.imread(table_path)
    h, w = img.shape[:2]
    res = model(img, conf=conf, imgsz=1024, rect=False, verbose=False)[0]
    crops = []
    for box in res.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        m = 10
        tc = img[max(0, y1 - m):min(h, y2 + m), max(0, x1 - m):min(w, x2 + m)].copy()
        pad = 30
        crop = cv2.copyMakeBorder(tc, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        hc, wc = crop.shape[:2]
        if hc < 192:
            sf = min(300 / hc, 3.0)
            if sf > 1.0:
                crop = cv2.resize(crop, (int(wc * sf), int(hc * sf)), interpolation=cv2.INTER_CUBIC)
        # MolNexTR expects RGB (content_recognizer does BGR->RGB before predict)
        crops.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    return crops


def run_baseline(model, crops):
    out = []
    t0 = time.perf_counter()
    for c in crops:
        out.append(model.predict_images([c])[0].get("predicted_smiles", ""))
    return out, time.perf_counter() - t0


def run_microbatch(model, crops, bs):
    t0 = time.perf_counter()
    res = model.predict_images(crops, batch_size=bs)
    out = [r.get("predicted_smiles", "") for r in res]
    return out, time.perf_counter() - t0


def run_concurrent(model, crops, k):
    def one(c):
        return model.predict_images([c])[0].get("predicted_smiles", "")
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=k) as ex:
        out = list(ex.map(one, crops))
    return out, time.perf_counter() - t0


def consistency(base, other):
    """#boxes whose canonical SMILES differs from baseline."""
    mism = 0
    for b, o in zip(base, other):
        if canonical_smiles(b) != canonical_smiles(o):
            mism += 1
    return mism


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--limit", type=int, default=0, help="cap #crops (0=all)")
    ap.add_argument("--workers", type=int, nargs="*", default=[2, 4])
    ap.add_argument("--batch", type=int, nargs="*", default=[8, 16])
    args = ap.parse_args()

    dev, dev_name = MolNexTRSingleton.get_device()
    print(f"[device] MolNexTR runs on: {dev_name} ({dev})", flush=True)

    print(f"[crops] cutting molecule boxes from {os.path.basename(args.table)} ...", flush=True)
    crops = cut_crops(args.table)
    if args.limit:
        crops = crops[:args.limit]
    n = len(crops)
    print(f"[crops] {n} molecule boxes ready", flush=True)
    if n == 0:
        print("no molecule boxes detected — pick a structure-dense table")
        return

    model = MolNexTRSingleton.get_instance()

    # Warmup (exclude lazy graph build / first-call cost from timings).
    print("[warmup] one inference to exclude cold-start ...", flush=True)
    model.predict_images([crops[0]])

    print(f"\n=== baseline (per-box, current pipeline) — {n} boxes ===", flush=True)
    base, t_base = run_baseline(model, crops)
    print(f"  time={t_base:.1f}s  ({t_base/n:.2f}s/box)", flush=True)

    rows = [("baseline", t_base, t_base / n, 1.0, 0)]

    for bs in args.batch:
        print(f"\n=== microbatch bs={bs} ===", flush=True)
        out, t = run_microbatch(model, crops, bs)
        mism = consistency(base, out)
        print(f"  time={t:.1f}s  speedup={t_base/t:.2f}x  mismatch={mism}/{n}", flush=True)
        rows.append((f"microbatch-{bs}", t, t / n, t_base / t, mism))

    for k in args.workers:
        print(f"\n=== concurrent workers={k} ===", flush=True)
        out, t = run_concurrent(model, crops, k)
        mism = consistency(base, out)
        print(f"  time={t:.1f}s  speedup={t_base/t:.2f}x  mismatch={mism}/{n}", flush=True)
        rows.append((f"concurrent-{k}", t, t / n, t_base / t, mism))

    print(f"\n================ SUMMARY ({n} boxes, device={dev_name}) ================")
    print(f"{'strategy':16} {'time':>8} {'s/box':>7} {'speedup':>8} {'mismatch':>9}")
    for name, t, per, sp, mism in rows:
        flag = "" if mism == 0 else "  <-- INCONSISTENT"
        print(f"{name:16} {t:>7.1f}s {per:>6.2f}s {sp:>7.2f}x {mism:>4}/{n}{flag}")


if __name__ == "__main__":
    main()
