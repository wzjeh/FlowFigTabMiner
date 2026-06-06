"""Aggregate a batch stress-test run into one report.

Reads the batch checkpoint (data/batch_progress.json) and each paper's
normalized output (data/final_output/{basename}_normalized.json), and
summarises:
- success / failure counts and the failed list
- per-paper record counts
- field fill-rates (SMILES / yield / selectivity / conditions)
- per-paper timing (parsed from data/intermediate/{basename}/pipeline.log)

Usage (from project root):
    flowfigtabminer/bin/python scripts/aggregate_stress_report.py
    flowfigtabminer/bin/python scripts/aggregate_stress_report.py --pdf-dir data/input/test_10
"""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime

CHECKPOINT = "data/batch_progress.json"
FINAL_DIR = "data/final_output"
INTER_DIR = "data/intermediate"


def _fill_rate(records: list, field: str, nested: str | None = None) -> int:
    n = 0
    for r in records:
        v = r.get(nested, {}).get(field) if nested else r.get(field)
        if v not in (None, "", [], {}):
            n += 1
    return n


def _parse_runtime(basename: str) -> float | None:
    """Best-effort wall-clock from the per-PDF pipeline.log mtime vs ctime."""
    log = os.path.join(INTER_DIR, basename, "pipeline.log")
    if not os.path.exists(log):
        return None
    try:
        st = os.stat(log)
        return round(st.st_mtime - st.st_ctime, 1)
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf-dir", default=None,
                    help="If set, aggregate exactly the PDFs in this dir (else use checkpoint)")
    ap.add_argument("--out", default="data/final_output/stress_report.json")
    args = ap.parse_args()

    # Resolve the paper list.
    if args.pdf_dir and os.path.isdir(args.pdf_dir):
        basenames = sorted(
            os.path.splitext(f)[0]
            for f in os.listdir(args.pdf_dir)
            if f.lower().endswith(".pdf")
        )
        done, failed = [], []
    else:
        cp = json.load(open(CHECKPOINT)) if os.path.exists(CHECKPOINT) else {}
        done = [os.path.splitext(b)[0] for b in cp.get("done", [])]
        failed = [os.path.splitext(b)[0] for b in cp.get("failed", [])]
        basenames = done + failed

    rows = []
    tot_records = 0
    agg = {k: [0, 0] for k in
           ("product_smiles", "reactant1_smiles", "yield_pct",
            "selectivity_pct", "conversion_pct")}
    cond_agg = {k: [0, 0] for k in ("temperature_C", "solvent", "catalyst", "pressure_bar")}

    for b in basenames:
        norm = os.path.join(FINAL_DIR, f"{b}_normalized.json")
        if not os.path.exists(norm):
            rows.append({"paper": b[:60], "status": "no_output", "records": 0})
            continue
        try:
            recs = json.load(open(norm))
        except Exception:
            rows.append({"paper": b[:60], "status": "bad_json", "records": 0})
            continue
        n = len(recs)
        tot_records += n
        row = {"paper": b[:60], "status": "ok", "records": n,
               "runtime_s": _parse_runtime(b)}
        for f in agg:
            c = _fill_rate(recs, f)
            agg[f][0] += c
            agg[f][1] += n
            row[f] = f"{c}/{n}"
        for f in cond_agg:
            c = _fill_rate(recs, f, nested="conditions")
            cond_agg[f][0] += c
            cond_agg[f][1] += n
        rows.append(row)

    n_papers = len(basenames)
    n_ok = sum(1 for r in rows if r["status"] == "ok")
    report = {
        "generated_at": datetime.now().isoformat(),
        "papers_total": n_papers,
        "papers_with_output": n_ok,
        "papers_failed": failed,
        "total_records": tot_records,
        "avg_records_per_paper": round(tot_records / max(1, n_ok), 1),
        "fill_rates": {
            f: (f"{c}/{t} = {100*c//max(1,t)}%") for f, (c, t) in agg.items()
        },
        "condition_fill_rates": {
            f: (f"{c}/{t} = {100*c//max(1,t)}%") for f, (c, t) in cond_agg.items()
        },
        "per_paper": rows,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Console summary
    print(f"\n{'='*60}")
    print(f"STRESS REPORT  ({n_papers} papers, {n_ok} produced output)")
    print(f"{'='*60}")
    print(f"total records: {tot_records}   avg/paper: {report['avg_records_per_paper']}")
    if failed:
        print(f"FAILED: {failed}")
    print("\nfill rates:")
    for f, v in report["fill_rates"].items():
        print(f"  {f:18} {v}")
    print("conditions:")
    for f, v in report["condition_fill_rates"].items():
        print(f"  {f:18} {v}")
    print("\nper-paper:")
    for r in rows:
        rt = f"{r.get('runtime_s')}s" if r.get("runtime_s") else "-"
        print(f"  [{r['status']:9}] {r['records']:3} rec  {rt:>8}  {r['paper']}")
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
