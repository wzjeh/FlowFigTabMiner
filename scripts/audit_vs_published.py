"""Audit pipeline output against a published, human-curated dataset.

The published rows are the answer key.  For every published row (paper, yield,
T, tR1/tR2, product SMILES) the script looks for a pipeline record of the same
paper whose yield / temperature / residence time agree, then checks whether
that record carries the same product structure.

Usage:
  flowfigtabminer/bin/python scripts/audit_vs_published.py \
      --truth eval/robustness/dataset_numbers_export.csv \
      --papers eval/robustness/pub25_pdf_chosen.json \
      [--final-dir data/final_output] [--out eval/robustness]

--truth   CSV with columns paper_id, yield_pct, T1_C, tR1_s, tR2_s,
          product_smiles_canonical (an export of dataset.numbers; never edit
          the .numbers file itself).
--papers  JSON {paper_id: path/to/paper.pdf}; the pipeline output is
          data/final_output/{pdf basename}_normalized.json.

Match rules (per paper):
  yield |d| <= 2 pct; temperature |d| <= 3 C; the pipeline's residence_time_s
  or residence_time_2_s within 25 % (or 0.01 s) of the published tR1 or tR2,
  whichever step the paper varied; several candidates -> nearest yield.

Writes audit_summary_vs_published.csv (per paper) and
audit_pairs_vs_published.csv (per matched row) into --out and prints the four
headline percentages.  Exit code 0 always; this is a measurement, not a gate.
"""
import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.adjudication.entity_pool import canonical_smiles  # noqa: E402


def _close(a, b, rel=0.25, abs_=0.01):
    if a is None or b is None or pd.isna(a) or pd.isna(b):
        return False
    return abs(a - b) <= abs_ or abs(a - b) / max(abs(b), 1e-9) <= rel


def load_records(final_dir, basename):
    path = os.path.join(final_dir, f"{basename}_normalized.json")
    if not os.path.exists(path):
        return []
    data = json.load(open(path))
    recs = data if isinstance(data, list) else data.get("records", [])
    out = []
    for x in recs:
        if x.get("yield_pct") is None:
            continue
        c = x.get("conditions") or {}
        out.append({
            "y": float(x["yield_pct"]),
            "T": c.get("temperature_C"),
            "tR": c.get("residence_time_s"),
            "tR2": c.get("residence_time_2_s"),
            "psm": canonical_smiles(x.get("product_smiles")) if x.get("product_smiles") else None,
            "src": x.get("__source_id"),
            "pname": x.get("product_name"),
            "plabel": x.get("product_label"),
        })
    return out


def match_row(row, records):
    """Best pipeline record for one published row, or None."""
    y, T, t1, t2 = row.yield_pct, row.T1_C, row.tR1_s, row.tR2_s
    steps = [v for v in (t1, t2) if pd.notna(v) and v > 0]
    best = None
    for p in records:
        if abs(p["y"] - y) > 2:
            continue
        if pd.notna(T) and (p["T"] is None or abs(p["T"] - T) > 3):
            continue
        if steps and not any(_close(p[k], v) for k in ("tR", "tR2") for v in steps):
            continue
        d = abs(p["y"] - y)
        if best is None or d < best[0]:
            best = (d, p)
    return best[1] if best else None


def audit(truth, papers, final_dir):
    summary, pairs = [], []
    for pid, g in truth.groupby("paper_id"):
        pdf = papers.get(pid)
        basename = os.path.splitext(os.path.basename(pdf))[0] if pdf else None
        records = load_records(final_dir, basename) if basename else []
        row_stats = dict(paper_id=pid, pub_rows=len(g), pipe_outcome=len(records),
                         yield_match=0, cond_match=0, smiles_present=0, smiles_agree=0)
        for _, row in g.iterrows():
            if pd.isna(row.yield_pct):
                continue
            if any(abs(p["y"] - row.yield_pct) <= 2 for p in records):
                row_stats["yield_match"] += 1
            p = match_row(row, records)
            if p is None:
                continue
            row_stats["cond_match"] += 1
            pub_smi = canonical_smiles(row.product_smiles_canonical) if pd.notna(row.product_smiles_canonical) else None
            agree = p["psm"] is not None and pub_smi is not None and p["psm"] == pub_smi
            row_stats["smiles_present"] += p["psm"] is not None
            row_stats["smiles_agree"] += agree
            pairs.append(dict(paper_id=pid, pub_y=row.yield_pct, pub_T=row.T1_C, pub_tR1=row.tR1_s, pub_tR2=row.tR2_s,
                              pub_prod=row.get("product"), pub_smiles=pub_smi, pipe_src=p["src"], pipe_T=p["T"],
                              pipe_tR=p["tR"], pipe_tR2=p["tR2"], pipe_smiles=p["psm"], pipe_pname=p["pname"],
                              pipe_plabel=p["plabel"], smiles_agree=agree))
        summary.append(row_stats)
    return pd.DataFrame(summary), pd.DataFrame(pairs)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--papers", required=True)
    ap.add_argument("--final-dir", default="data/final_output")
    ap.add_argument("--out", default="eval/robustness")
    args = ap.parse_args(argv)

    truth = pd.read_csv(args.truth)
    papers = json.load(open(args.papers))
    summary, pairs = audit(truth, papers, args.final_dir)

    os.makedirs(args.out, exist_ok=True)
    summary.to_csv(os.path.join(args.out, "audit_summary_vs_published.csv"), index=False)
    pairs.to_csv(os.path.join(args.out, "audit_pairs_vs_published.csv"), index=False)

    pd.set_option("display.width", 200)
    print(summary.to_string(index=False))
    n = summary.pub_rows.sum()
    cm = summary.cond_match.sum()
    print(f"\npublished rows {n} | yield found {summary.yield_match.sum()} ({summary.yield_match.sum() / n:.0%})"
          f" | yield+T+tR match {cm} ({cm / n:.0%})"
          f" | of matched: SMILES present {summary.smiles_present.sum()} ({summary.smiles_present.sum() / max(cm, 1):.0%}),"
          f" SMILES agree {summary.smiles_agree.sum()} ({summary.smiles_agree.sum() / max(cm, 1):.0%})")


if __name__ == "__main__":
    main()
