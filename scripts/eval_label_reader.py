"""Evaluate the VLM label reader (design: VLM reads symbols, geometry stays with YOLO).

Re-runs ONLY the figure stage on existing crops for the given papers with the
reader set by ``--reader off|gemini|claude`` (via FFTM_LABEL_READER), then
Steps 5-6 (reassemble, vars reused), and writes a metrics report:
axis-fit rejections, fit inlier ratio, OCR×VLM agreement / conflict counts,
synthesized-record fidelity, has_outcome share, per-figure cost.

Usage (inside the venv, from the project root, .env loaded):
  python scripts/eval_label_reader.py --reader gemini --papers data/input/verify_2/*.pdf
  python scripts/eval_label_reader.py --reader gemini --papers-from eval/audit/audit_error_chain_2026-09-17_60.xlsx
  python scripts/eval_label_reader.py --report-only --reader gemini --papers ...   # metrics from existing outputs
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PRICE_PER_M = {"gemini-2.5-flash": (0.30, 2.50), "claude-sonnet-5": (2.00, 10.00), "claude-sonnet-4-6": (3.00, 15.00)}


def _papers_from_audit(xlsx: str) -> list[str]:
    import pandas as pd
    df = pd.read_excel(xlsx)
    return sorted({p for p in df["pdf"].dropna().astype(str) if os.path.exists(p)})


def run_stage(paths: list[str], reader: str) -> None:
    os.environ["FFTM_LABEL_READER"] = reader
    import src.pipeline._threadcaps  # noqa: F401
    from src.llm.providers.gemini import GeminiProvider
    from src.llm.config import load_vlm_config
    from src.extraction.figure.metadata_vlm import FigureMetadataExtractor
    from src.pipeline.figure_pipeline import FigurePipeline
    from src.pipeline.main import _build_label_reader
    from scripts.reassemble import main as reassemble_main
    provider = GeminiProvider()
    fp = FigurePipeline(metadata_extractor=FigureMetadataExtractor(vlm=provider, cfg=load_vlm_config("config.yaml")),
                        post_extract_hooks=[], label_reader=_build_label_reader(provider))
    for p in paths:
        base = os.path.splitext(os.path.basename(p))[0]
        idir = os.path.join("data/intermediate", base)
        figs = sorted(glob.glob(os.path.join(idir, "figures", "*.png")))
        print(f"[eval] === {base} ({len(figs)} crops, reader={reader})", flush=True)
        if figs:
            try:
                fp.process_images(figs, idir)
            except Exception as exc:
                print(f"[eval] figure stage error {base}: {exc}", flush=True)
    reassemble_main(paths, rebuild_vars=False)


def _num(v):
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else v


def collect(paths: list[str]) -> dict:
    m = collections.Counter()
    fit_inl = fit_tot = 0
    cost_in = cost_out = 0; calls = 0; models = collections.Counter()
    fidelity_ok = fidelity_tot = 0
    for p in paths:
        base = os.path.splitext(os.path.basename(p))[0]
        idir = os.path.join("data/intermediate", base)
        for lg in glob.glob(os.path.join(idir, "macro_cleaned", "*_coordmap_log.txt")):
            txt = open(lg).read()
            m["figures"] += 1
            m["fit_rejected"] += txt.count("Fit REJECTED (x)") + txt.count("Fit REJECTED (y_left)")
            m["dual_axis_rejected"] += txt.count("Dual axis REJECTED")
        for evp in glob.glob(os.path.join(idir, "macro_cleaned", "*_evidence.json")):
            ev = json.load(open(evp)); f = (ev.get("meta") or {}).get("facts") or {}
            for q in (f.get("fit_quality") or {}).values():
                fit_inl += q.get("inliers", 0); fit_tot += q.get("total", 0)
            lr = f.get("label_reader") or {}
            for k in ("tick_boxes", "tick_agree", "tick_conflict", "tick_vlm_only", "tick_ocr_only",
                      "value_boxes", "value_agree", "value_conflict", "value_from_vlm", "value_from_ocr"):
                m[k] += int(lr.get(k) or 0)
            for callmeta in (lr, lr.get("value_call") or {}):
                if callmeta.get("model") and not callmeta.get("cache_hit"):
                    calls += 1; models[callmeta["model"]] += 1
                    cost_in += callmeta.get("tokens_in") or 0; cost_out += callmeta.get("tokens_out") or 0
            m["points"] += len(ev.get("raw_data") or [])
            m["points_with_label"] += sum(1 for r in ev.get("raw_data") or [] if _num(r.get("Y_Right/Data_Value")) is not None)
            # Series recovery (xy plots only): how many points still carry no series.
            if f.get("chart_type") == "xy":
                raw = ev.get("raw_data") or []
                m["xy_points"] += len(raw)
                m["xy_points_default"] += sum(1 for r in raw if (r.get("Series") or "Default") == "Default")
                n_names = len((ev.get("text_evidence") or {}).get("legend_text") or [])
                m["xy_figs_multi_series"] += int(n_names >= 2)
                m[f"series_source_{f.get('series_source') or 'na'}"] += 1
        fin = f"data/final_output/{base}_normalized.json"
        if os.path.exists(fin):
            recs = json.load(open(fin))
            figs = [r for r in recs if r.get("__synthesized")]
            m["records"] += len(recs); m["figure_records"] += len(figs)
            m["has_outcome"] += sum(1 for r in recs if r.get("has_outcome"))
            by = collections.defaultdict(list)
            for r in figs: by[r["__source_id"]].append(r)
            for sid, g in by.items():
                evp = os.path.join(idir, "macro_cleaned", f"{sid}_evidence.json")
                if not os.path.exists(evp): continue
                raw = json.load(open(evp))["raw_data"]
                for r, pt in zip(g, raw):
                    fidelity_tot += 1; c = r["conditions"]
                    x, yl, dv = _num(pt.get("X")), _num(pt.get("Y_Left")), _num(pt.get("Y_Right/Data_Value"))
                    vals = [c.get("temperature_C"), c.get("residence_time_s"), r.get("yield_pct")] + list((r.get("other_metrics") or {}).values())
                    has = lambda v, tol: v is None or any(isinstance(w, (int, float)) and abs(w - v) <= tol for w in vals)
                    # X of heatmap columns is clustered in log space (snap_levels_log) → log tolerance.
                    has_x = lambda v: v is None or has(v, 1e-6) or (v > 0 and any(
                        isinstance(w, (int, float)) and w > 0 and abs(math.log10(w) - math.log10(v)) <= 0.15 for w in vals))
                    y_ok = dv is None or dv > 100 or dv < 0 or any(isinstance(w, (int, float)) and abs(w - dv) < 1e-6 for w in vals)
                    fidelity_ok += bool(has_x(x) and has(yl, 2.5) and y_ok)
    model = models.most_common(1)[0][0] if models else None
    pin, pout = PRICE_PER_M.get(model or "", (0, 0))
    return {**m, "fit_inlier_ratio": round(fit_inl / fit_tot, 3) if fit_tot else None,
            "fidelity": f"{fidelity_ok}/{fidelity_tot}", "fidelity_pct": round(100 * fidelity_ok / fidelity_tot, 1) if fidelity_tot else None,
            "has_outcome_pct": round(100 * m["has_outcome"] / m["records"], 1) if m["records"] else None,
            "vlm_calls": calls, "vlm_model": model, "tokens_in": cost_in, "tokens_out": cost_out,
            "est_cost_usd": round(cost_in / 1e6 * pin + cost_out / 1e6 * pout, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reader", choices=["off", "gemini", "claude"], required=True)
    ap.add_argument("--papers", nargs="*", default=[])
    ap.add_argument("--papers-from", help="audit xlsx (uses its 'pdf' column)")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    paths = [p for pat in a.papers for p in (glob.glob(pat) or [pat])]
    if a.papers_from:
        paths += _papers_from_audit(a.papers_from)
    paths = sorted(set(paths))
    if not paths:
        ap.error("no papers")
    t0 = time.time()
    if not a.report_only:
        run_stage(paths, a.reader)
    metrics = collect(paths)
    metrics["papers"] = len(paths); metrics["wall_s"] = round(time.time() - t0, 1); metrics["reader"] = a.reader
    out = a.out or f"eval/label_reader/{a.reader}_{time.strftime('%Y-%m-%d')}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(metrics, open(out, "w"), indent=2)
    print(json.dumps(metrics, indent=2)); print("->", out)


if __name__ == "__main__":
    main()
