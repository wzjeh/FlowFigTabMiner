"""Regression gate: snapshot the final outputs of a paper set, compare two
snapshots source by source, and summarise the health of a batch run.

Why: a one-line prompt change re-rolls the assembly of unrelated sources
(2026-09-21: a reworded candidate block cost 31 reactant SMILES).  Overall
percentages hide that; a per-source diff against the previous snapshot shows it.

Usage (project root, inside the venv):
    python scripts/regression_gate.py snapshot NAME --papers-from eval/paper_sets/eval30.txt
    python scripts/regression_gate.py compare BASE [NEW]        # NEW omitted = live data/final_output
    python scripts/regression_gate.py health --pdf-dir data/input/robust_b1 [--out report.json]

Snapshots live in eval/snapshots/NAME/ (not tracked by git).
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import shutil
import subprocess
import time
from typing import Dict, Iterable, List, Tuple

FINAL_DIR = "data/final_output"
INTER_DIR = "data/intermediate"
SNAP_DIR = "eval/snapshots"
FIELDS = ("n", "T", "tR", "tR2", "t_batch", "solvent", "yield", "outcome", "psm", "rsm", "pname")


# ── pure functions ───────────────────────────────────────────────────────────
def source_kind(rec: dict) -> str:
    sid = str(rec.get("__source_id") or "")
    if "figure" in sid or (not sid and "Figure" in str(rec.get("source_table_or_figure") or "")):
        return "figure"
    return "table"


def field_counts(records: Iterable[dict]) -> collections.Counter:
    c: collections.Counter = collections.Counter()
    for r in records:
        cd = r.get("conditions") or {}
        c["n"] += 1
        c["T"] += cd.get("temperature_C") is not None
        c["tR"] += cd.get("residence_time_s") is not None
        c["tR2"] += cd.get("residence_time_2_s") is not None
        c["t_batch"] += cd.get("reaction_time_s") is not None
        c["solvent"] += bool(cd.get("solvent"))
        c["yield"] += r.get("yield_pct") is not None
        c["outcome"] += bool(r.get("has_outcome"))
        c["psm"] += bool(r.get("product_smiles"))
        c["rsm"] += bool(r.get("reactant1_smiles"))
        c["pname"] += bool(r.get("product_name"))
    return c


def summarise(by_paper: Dict[str, List[dict]]) -> Tuple[dict, dict]:
    """→ ({kind: counts}, {(paper, source label): counts})."""
    kinds = {"table": collections.Counter(), "figure": collections.Counter()}
    sources: Dict[Tuple[str, str], collections.Counter] = {}
    for paper, recs in by_paper.items():
        groups: Dict[Tuple[str, str], List[dict]] = collections.defaultdict(list)
        for r in recs:
            groups[(paper, str(r.get("source_table_or_figure"))[:40])].append(r)
            kinds[source_kind(r)] += field_counts([r])
        for key, rs in groups.items():
            sources[key] = field_counts(rs)
    return kinds, sources


def diff_sources(old: dict, new: dict, min_abs: int = 3, min_frac: float = 0.2) -> List[dict]:
    """Per-source field changes worth a look: |Δ| ≥ min_abs and ≥ min_frac of the
    source's record count; sources that appeared or vanished are always listed."""
    rows = []
    for key in sorted(set(old) | set(new)):
        a, b = old.get(key), new.get(key)
        if a is None or b is None:
            rows.append({"paper": key[0], "source": key[1], "field": "n",
                         "old": (a or {}).get("n", 0), "new": (b or {}).get("n", 0),
                         "note": "source appeared" if a is None else "source vanished"})
            continue
        n = max(a["n"], b["n"], 1)
        for f in FIELDS:
            d = b[f] - a[f]
            if abs(d) >= min_abs and abs(d) >= min_frac * n:
                rows.append({"paper": key[0], "source": key[1], "field": f, "old": a[f], "new": b[f], "note": ""})
    return rows


def health(papers: List[str], inter_dir: str = INTER_DIR, final_dir: str = FINAL_DIR) -> dict:
    """Did every paper finish, where did sources die, how long did it take."""
    out = {"papers": len(papers), "no_intermediate": [], "not_ok": [], "no_output": [], "zero_records": [],
           "stage_outcomes": collections.Counter(), "failures": [], "timing_s": {}, "records": 0}
    for b in papers:
        idir = os.path.join(inter_dir, b)
        if not os.path.isdir(idir):
            out["no_intermediate"].append(b)
            continue
        timing = {}
        tp = os.path.join(idir, "timing.json")
        if os.path.exists(tp):
            try:
                timing = json.load(open(tp))
            except Exception:
                timing = {}
        if timing.get("status") != "ok":
            out["not_ok"].append({"paper": b, "status": timing.get("status", "no timing.json (crash or still running)")})
        if timing.get("total") is not None:
            out["timing_s"][b] = timing
        for sp in sorted(glob.glob(os.path.join(idir, "status", "*.json"))):
            try:
                s = json.load(open(sp))
            except Exception:
                continue
            out["stage_outcomes"][f"{s.get('stage')}:{s.get('outcome')}"] += 1
            if s.get("outcome") == "failed":
                out["failures"].append({"paper": b, "source": s.get("source_id"), "stage": s.get("stage"),
                                        "reason": str(s.get("reason"))[:200]})
        fp = os.path.join(final_dir, f"{b}_normalized.json")
        if not os.path.exists(fp):
            if timing.get("status") == "ok":
                out["no_output"].append(b)
            continue
        try:
            n = len(json.load(open(fp)))
        except Exception:
            out["failures"].append({"paper": b, "source": "-", "stage": "final_output", "reason": "unreadable JSON"})
            continue
        out["records"] += n
        if n == 0:
            out["zero_records"].append(b)
    out["stage_outcomes"] = dict(out["stage_outcomes"])
    return out


# ── I/O ──────────────────────────────────────────────────────────────────────
def _basenames(papers_from: str | None, pdf_dir: str | None) -> List[str]:
    if pdf_dir:
        return sorted(os.path.splitext(f)[0] for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf"))
    names = [l.strip() for l in open(papers_from) if l.strip()]
    return [os.path.splitext(os.path.basename(n))[0] if n.lower().endswith(".pdf") else os.path.basename(n) for n in names]


def _load(records_dir: str, papers: List[str] | None = None) -> Dict[str, List[dict]]:
    by_paper = {}
    for fp in sorted(glob.glob(os.path.join(records_dir, "*_normalized.json"))):
        b = os.path.basename(fp)[: -len("_normalized.json")]
        if papers is not None and b not in papers:
            continue
        try:
            by_paper[b] = json.load(open(fp))
        except Exception:
            by_paper[b] = []
    return by_paper


def _print_kinds(title: str, old: dict, new: dict) -> None:
    print(f"\n{title}")
    print(f"{'field':10s} " + " ".join(f"{k + ' old':>11s} {k + ' new':>11s}" for k in ("table", "figure")))
    for f in FIELDS:
        cells = []
        for k in ("table", "figure"):
            a, b = old[k][f], new[k][f]
            mark = "" if a == b else (" ▲" if b > a else " ▼")
            cells.append(f"{a:11d} {str(b) + mark:>11s}")
        print(f"{f:10s} " + " ".join(cells))


def cmd_snapshot(a) -> None:
    papers = _basenames(a.papers_from, a.pdf_dir)
    dest = os.path.join(SNAP_DIR, a.name, "records")
    os.makedirs(dest, exist_ok=True)
    copied = 0
    for b in papers:
        fp = os.path.join(FINAL_DIR, f"{b}_normalized.json")
        if os.path.exists(fp):
            shutil.copy2(fp, dest)
            copied += 1
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.strip())
    meta = {"name": a.name, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "commit": commit, "dirty_tree": dirty,
            "papers": papers, "copied": copied, "health": health(papers)}
    json.dump(meta, open(os.path.join(SNAP_DIR, a.name, "meta.json"), "w"), indent=2, ensure_ascii=False)
    print(f"snapshot {a.name}: {copied}/{len(papers)} papers, commit {commit}{' (dirty tree)' if dirty else ''} -> {os.path.join(SNAP_DIR, a.name)}")


def cmd_compare(a) -> None:
    base_dir = os.path.join(SNAP_DIR, a.base, "records")
    papers = json.load(open(os.path.join(SNAP_DIR, a.base, "meta.json")))["papers"]
    old = _load(base_dir, papers)
    new = _load(os.path.join(SNAP_DIR, a.new, "records") if a.new else FINAL_DIR, papers)
    ko, so = summarise(old)
    kn, sn = summarise(new)
    _print_kinds(f"{a.base}  ->  {a.new or 'live ' + FINAL_DIR}   ({len(old)} / {len(new)} papers)", ko, kn)
    rows = diff_sources(so, sn, a.min_abs, a.min_frac)
    print(f"\nper-source changes (|Δ| ≥ {a.min_abs} and ≥ {a.min_frac:.0%} of the source): {len(rows)}")
    for r in rows:
        print(f"  {r['paper'][:28]:28s} {r['source'][:24]:24s} {r['field']:8s} {r['old']:4d} -> {r['new']:4d}  {r['note']}")
    net = collections.Counter()
    for key in set(so) | set(sn):
        for f in FIELDS:
            net[f] += (sn.get(key) or {}).get(f, 0) - (so.get(key) or {}).get(f, 0)
    print("\nnet change, all sources:", {f: net[f] for f in FIELDS if net[f]})
    if a.out:
        json.dump({"base": a.base, "new": a.new or "live", "kinds_old": ko, "kinds_new": kn, "rows": rows}, open(a.out, "w"),
                  indent=2, ensure_ascii=False)


def cmd_health(a) -> None:
    papers = _basenames(a.papers_from, a.pdf_dir)
    h = health(papers)
    done = [b for b in papers if b in h["timing_s"]]
    print(f"papers {h['papers']}  finished {len(done)}  records {h['records']}")
    for k in ("no_intermediate", "not_ok", "no_output", "zero_records"):
        if h[k]:
            print(f"  {k}: {h[k]}")
    print("  stage outcomes:", h["stage_outcomes"])
    if h["failures"]:
        print(f"  failed sources: {len(h['failures'])}")
        for f in h["failures"]:
            print(f"    {f['paper'][:24]:24s} {str(f['source'])[:22]:22s} {f['stage']:12s} {f['reason'][:110]}")
    if done:
        tot = sorted(h["timing_s"][b].get("total", 0) for b in done)
        stage_sum = collections.Counter()
        for b in done:
            for k, v in h["timing_s"][b].items():
                if isinstance(v, (int, float)) and k != "total":
                    stage_sum[k] += v
        print(f"  time per paper: median {tot[len(tot) // 2]:.0f} s, max {tot[-1]:.0f} s; by stage (s):",
              {k: round(v) for k, v in stage_sum.most_common()})
    if a.out:
        json.dump(h, open(a.out, "w"), indent=2, ensure_ascii=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("snapshot"); s.add_argument("name"); s.add_argument("--papers-from"); s.add_argument("--pdf-dir")
    s.set_defaults(fn=cmd_snapshot)
    c = sub.add_parser("compare"); c.add_argument("base"); c.add_argument("new", nargs="?")
    c.add_argument("--min-abs", type=int, default=3); c.add_argument("--min-frac", type=float, default=0.2); c.add_argument("--out")
    c.set_defaults(fn=cmd_compare)
    h = sub.add_parser("health"); h.add_argument("--papers-from"); h.add_argument("--pdf-dir"); h.add_argument("--out")
    h.set_defaults(fn=cmd_health)
    a = ap.parse_args()
    if a.cmd in ("snapshot", "health") and not (a.papers_from or a.pdf_dir):
        ap.error("give --papers-from or --pdf-dir")
    a.fn(a)


if __name__ == "__main__":
    main()
