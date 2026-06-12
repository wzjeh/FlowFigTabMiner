"""Opt-in OCR timing instrumentation — issue #17 Phase 1 (profiling only).

Enabled ONLY when ``OCR_PROFILE=1``.  When disabled, every hook is a cheap
no-op, so pipeline behaviour, OCR inputs/outputs, and control flow are
byte-identical.  This module *measures*; it never changes what the pipeline
does — it must not be used to justify an optimization on its own (see the
issue #17 "measure first" discipline).

What it records, per table:
  - per OCR loop (caption / note): #crops, input size + post-resize size
    (P50 / P95 / max by area), OCR inference time, OCR postprocess time
  - named buckets for non-loop sections: ``serialize``, ``vlm``, ...

Usage (host code):
    from src.extraction.common.ocr_profile import profiler
    profiler.start_loop("caption")
    for crop in crops:
        profiler.record_crop("caption", (w0, h0), (w1, h1))
        text = recognizer._recognize_text(crop)   # times itself internally
    profiler.end_loop()
    with profiler.bucket("serialize"):
        ...
    profiler.report(table="page_3_table_0")
"""
import os
import time
from contextlib import contextmanager


class _OCRProfiler:
    def __init__(self):
        # Captured once at construction; the env is set by the caller before the
        # subprocess starts, so this is stable for the life of a run.
        self.enabled = os.environ.get("OCR_PROFILE", "0") == "1"
        self.reset()

    def reset(self):
        self._loops = {}     # name -> {crops, input[], resized[], ocr_inference, ocr_postprocess}
        self._buckets = {}   # name -> seconds
        self._current = None

    def _loop(self, name):
        return self._loops.setdefault(
            name, {"crops": 0, "input": [], "resized": [],
                   "ocr_inference": 0.0, "ocr_postprocess": 0.0})

    # ── loop scoping (host sets which OCR loop is active) ──
    def start_loop(self, name):
        if not self.enabled:
            return
        self._loop(name)
        self._current = name

    def end_loop(self):
        if not self.enabled:
            return
        self._current = None

    def record_crop(self, name, input_wh, resized_wh):
        if not self.enabled:
            return
        d = self._loop(name)
        d["crops"] += 1
        d["input"].append((int(input_wh[0]), int(input_wh[1])))
        d["resized"].append((int(resized_wh[0]), int(resized_wh[1])))

    # ── timing context managers ──
    @contextmanager
    def ocr_phase(self, key):
        """Attribute an OCR sub-timing to the active loop.

        ``key`` is ``"ocr_inference"`` or ``"ocr_postprocess"``.  Calls made
        outside any loop (e.g. the molecule OCR fallback) land in ``_other``.
        """
        if not self.enabled:
            yield
            return
        t = time.perf_counter()
        try:
            yield
        finally:
            d = self._loop(self._current or "_other")
            d[key] = d.get(key, 0.0) + (time.perf_counter() - t)

    @contextmanager
    def bucket(self, name):
        """Time a non-loop section (e.g. ``serialize``, ``vlm``)."""
        if not self.enabled:
            yield
            return
        t = time.perf_counter()
        try:
            yield
        finally:
            self._buckets[name] = self._buckets.get(name, 0.0) + (time.perf_counter() - t)

    # ── reporting ──
    @staticmethod
    def _rep_size(sizes, p):
        """Representative WxH at the area percentile p (nearest-rank)."""
        if not sizes:
            return "-"
        order = sorted(range(len(sizes)), key=lambda i: sizes[i][0] * sizes[i][1])
        k = max(0, min(len(order) - 1, int(round((p / 100.0) * (len(order) - 1)))))
        w, h = sizes[order[k]]
        return f"{w}x{h}"

    def report(self, table=""):
        if not self.enabled:
            return
        lines = [f"[OCR_PROFILE] table={table}"]
        for name, d in self._loops.items():
            if d["crops"] == 0 and d["ocr_inference"] == 0.0 and d["ocr_postprocess"] == 0.0:
                continue
            inp, rsz = d["input"], d["resized"]
            lines.append(
                f"  {name}: crops={d['crops']}  "
                f"ocr_inf={d['ocr_inference']:.1f}s  ocr_post={d['ocr_postprocess']:.1f}s  "
                f"input p50={self._rep_size(inp, 50)} p95={self._rep_size(inp, 95)} max={self._rep_size(inp, 100)}  "
                f"resized p50={self._rep_size(rsz, 50)} p95={self._rep_size(rsz, 95)} max={self._rep_size(rsz, 100)}"
            )
        for name, secs in self._buckets.items():
            lines.append(f"  {name}: {secs:.2f}s")
        print("\n".join(lines), flush=True)
        self.reset()


# Module-level singleton.
profiler = _OCRProfiler()
