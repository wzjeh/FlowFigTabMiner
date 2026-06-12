"""Issue #17 Phase 1.5 — split caption/note OCR time into det vs rec + box count.

READ-ONLY probe (like ``exp_molnextr_concurrency.py``): does NOT touch the
production code path.  It loads the SAME PaddleOCR (``get_ocr_instance``) the
table pipeline uses, applies the SAME 3x upscale + 50px white pad the pipeline
applies to caption/note crops (``pipeline.py`` step 7), then for each crop
reports, using only stable PaddleOCR/PaddleX APIs:

  total = wall time of the full ``.ocr()`` (det+rec) — exactly what the
          pipeline pays for one caption/note crop
  det   = wall time of the detection sub-model alone (``text_det_model``)
  boxes = number of detected text-line polygons (``dt_polys``)
  rec   ≈ total - det   (recognition + cropping + textline-orientation)

Goal: decide whether the per-note minutes live in DETECTION, in RECOGNITION
(a tall multi-line note → many text-line boxes → rec runs per line), or
elsewhere.  This script DRAWS NO CONCLUSION; it just produces the three numbers
issue #17 needs before any fix is chosen.

Process ONE crop at a time (no full pipeline) so there is no OOM risk — pass
only crops known to complete; do NOT feed the giant page_2 note that OOM-killed
the full run.

Usage:
  flowfigtabminer/bin/python scripts/exp_ocr_detrec_split.py <crop.png> [<crop2.png> ...]
"""
import os
import sys
import time

os.environ.setdefault("DISABLE_MODEL_SOURCE_CHECK", "True")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2  # noqa: E402
from src.extraction.common.ocr_backend import get_ocr_instance  # noqa: E402


def upscale_pad(img):
    """Replicate the pipeline's caption/note preprocessing (pipeline.py step 7)."""
    img = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    pad = 50
    img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    return img


def count_boxes(det_results):
    n = 0
    for item in det_results:
        try:
            n += len(item["dt_polys"])
        except Exception:
            pass
    return n


def main():
    crops = sys.argv[1:]
    if not crops:
        print("usage: exp_ocr_detrec_split.py <crop.png> [...]")
        return

    ocr = get_ocr_instance(lang="en", enable_mkldnn=False)
    det_model = ocr.paddlex_pipeline.text_det_model
    print(f"[omp] OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}", flush=True)

    # Warmup so lazy init / first-call cost is excluded from the timings.
    w = cv2.imread(crops[0])
    if w is not None:
        _ = list(det_model(cv2.cvtColor(upscale_pad(w), cv2.COLOR_BGR2RGB)))

    for path in crops:
        img0 = cv2.imread(path)
        if img0 is None:
            print(f"\n{os.path.basename(path)}: <unreadable>")
            continue
        h0, w0 = img0.shape[:2]
        rgb = cv2.cvtColor(upscale_pad(img0), cv2.COLOR_BGR2RGB)
        h1, w1 = rgb.shape[:2]

        # det alone (boxes + det time)
        t = time.perf_counter()
        det_res = list(det_model(rgb))
        det_t = time.perf_counter() - t
        boxes = count_boxes(det_res)

        # total: the exact full det+rec call the pipeline makes
        t = time.perf_counter()
        _ = ocr.ocr(rgb)
        total_t = time.perf_counter() - t

        rec_t = max(0.0, total_t - det_t)
        print(
            f"\n{os.path.basename(path)}\n"
            f"  input={w0}x{h0}  resized(3x+pad)={w1}x{h1}\n"
            f"  boxes={boxes}\n"
            f"  total={total_t:.1f}s  det={det_t:.1f}s  rec≈{rec_t:.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
