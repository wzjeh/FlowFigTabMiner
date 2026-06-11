"""Unified dataset metrics panel — compare any FlowFigTabMiner output on one
fixed, layered yardstick so versions never get compared apples-to-oranges again.

Motivation: the "old pipeline had more info" impression came from comparing a
hand-cleaned / VLM-enriched release (clean_organolithium_unified.csv) against a
near-raw pipeline output. This script forces every dataset to declare its
PROCESSING LAYER and reports the same metrics on all of them.

Layers (declare with --layer):
  raw_pipeline      LLM direct output, pre PostProcessor
  +post             through PostProcessor (PubChem, entity_pool, is_hollow, normalisation)
  +manual_or_vlm    hand-cleaned or VLM-enriched (e.g. the old unified CSV, 25 manual fixes)

Inputs (auto-detected):
  - a directory of *_normalized.json   (optionally filtered by --basenames-from)
  - a single .json file
  - a .csv / .xlsx file (already-flat dataset)

Read-only. Outputs <out>.json + <out>.md.

Usage:
  flowfigtabminer/bin/python -m eval.dataset_metrics_panel \
      --input data/final_output --basenames-from "data/input/test_10" \
      --layer +post --label test_10 \
      --intermediate-dir data/intermediate --out eval/metrics_panel_test_10
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter
from typing import Any, Dict, List, Optional


# ── field aliases: canonical metric name → candidate column/key names ──────────
# Handles the new normalized.json schema AND the old unified-CSV schema so both
# land on the same metric rows.
FIELD_ALIASES: Dict[str, List[str]] = {
    "product_smiles":   ["product_smiles", "product_smiles_canonical"],
    "reactant_smiles":  ["reactant1_smiles", "substrate1_smiles", "substrate1_smiles_canonical"],
    "product_name":     ["product_name", "product"],
    "yield":            ["yield_pct", "yield"],
    "conversion":       ["conversion_pct", "conversion"],
    "selectivity":      ["selectivity_pct", "selectivity"],
    "ee":               ["ee_pct", "ee"],
    "temperature":      ["temperature_C", "T1_C", "temperature"],
    "residence_time":   ["residence_time_s", "tR1_s", "tR_s"],
    "solvent":          ["solvent"],
    "reactor_type":     ["reactor_type"],
    "flow_rate":        ["flow_rate_mL_min", "flow_rate_stream1_mL_min"],
    "reaction_class":   ["reaction_class", "reaction_class_paper_level"],
}

# Metric display order, grouped.
GROUPS = [
    ("Identity", ["product_smiles", "reactant_smiles", "product_name"]),
    ("Outcome",  ["yield", "conversion", "selectivity", "ee"]),
    ("Conditions", ["temperature", "residence_time", "solvent", "reactor_type", "flow_rate"]),
    ("Class",    ["reaction_class"]),
]

TIMING_STAGES = ["filter", "tfid", "figure", "table", "scheme", "local_vars", "assembly", "post", "total"]


def _is_empty(v: Any) -> bool:
    """A cell counts as missing if null / empty / the literal string 'null'/'nan'."""
    if v is None:
        return True
    if isinstance(v, float):
        # NaN
        return v != v
    s = str(v).strip().lower()
    return s in ("", "null", "nan", "none", "na")


def _flatten(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Lift nested conditions/other_metrics into the top level (normalized.json)."""
    out = dict(rec)
    for nested in ("conditions", "other_metrics"):
        block = out.pop(nested, None)
        if isinstance(block, dict):
            for k, v in block.items():
                out.setdefault(k, v)
    return out


def _get(rec: Dict[str, Any], metric: str) -> Any:
    for col in FIELD_ALIASES[metric]:
        if col in rec and not _is_empty(rec[col]):
            return rec[col]
    return None


# ── input adapters ─────────────────────────────────────────────────────────────
def _load_records(input_path: str, basenames: Optional[set]) -> tuple[List[Dict[str, Any]], int]:
    """Return (flat records, n_papers). n_papers is best-effort."""
    if os.path.isdir(input_path):
        return _load_json_dir(input_path, basenames)
    ext = os.path.splitext(input_path)[1].lower()
    if ext == ".json":
        recs = _read_json_records(input_path)
        return [_flatten(r) for r in recs], 1
    if ext in (".csv", ".xlsx", ".xls"):
        return _load_tabular(input_path)
    raise SystemExit(f"Unsupported input: {input_path}")


def _read_json_records(path: str) -> List[Dict[str, Any]]:
    data = json.load(open(path))
    if isinstance(data, list):
        return [r for r in data if isinstance(r, dict)]
    if isinstance(data, dict):
        return [data]
    return []


def _load_json_dir(d: str, basenames: Optional[set]) -> tuple[List[Dict[str, Any]], int]:
    files = sorted(glob.glob(os.path.join(d, "*_normalized.json")))
    recs: List[Dict[str, Any]] = []
    n_papers = 0
    for f in files:
        base = os.path.basename(f)[: -len("_normalized.json")]
        if basenames is not None and base not in basenames:
            continue
        rows = _read_json_records(f)
        if not rows:
            continue
        n_papers += 1
        recs.extend(_flatten(r) for r in rows)
    return recs, n_papers


def _load_tabular(path: str) -> tuple[List[Dict[str, Any]], int]:
    import pandas as pd
    df = pd.read_csv(path) if path.lower().endswith(".csv") else pd.read_excel(path)
    recs = df.to_dict(orient="records")
    # n_papers: prefer an explicit paper id column
    n_papers = 0
    for col in ("paper_id", "paper_doi", "paper_basename", "source_file"):
        if col in df.columns:
            n_papers = int(df[col].nunique())
            break
    return recs, n_papers


def _resolve_basenames(spec: Optional[str]) -> Optional[set]:
    """--basenames-from may be a directory of PDFs or a text file of names."""
    if not spec:
        return None
    names = set()
    if os.path.isdir(spec):
        for p in glob.glob(os.path.join(spec, "*.pdf")):
            names.add(os.path.splitext(os.path.basename(p))[0])
    elif os.path.isfile(spec):
        for line in open(spec):
            line = line.strip()
            if line:
                names.add(os.path.splitext(line)[0])
    return names or None


# ── metrics ─────────────────────────────────────────────────────────────────────
def _coverage(records: List[Dict[str, Any]], metrics: List[str]) -> Dict[str, Dict[str, Any]]:
    n = len(records)
    out = {}
    for m in metrics:
        filled = sum(1 for r in records if _get(r, m) is not None)
        out[m] = {"filled": filled, "pct": round(100.0 * filled / n, 1) if n else 0.0}
    return out


def _smiles_sources(records: List[Dict[str, Any]]) -> Dict[str, int]:
    """Distribution of __smiles_source for records that HAVE a product_smiles."""
    c = Counter()
    for r in records:
        if _is_empty(r.get("product_smiles")):
            continue
        src = r.get("__smiles_source")
        c[str(src) if not _is_empty(src) else "inline_or_unknown"] += 1
    return dict(c)


def _timing(basenames: Optional[set], intermediate_dir: str, attributable: bool) -> Dict[str, Any]:
    # Timing is only meaningful when we can map records to specific intermediate
    # dirs. A tabular dataset with no --basenames-from cannot be attributed to
    # any pipeline run (e.g. the hand-cleaned unified CSV) → report N/A rather
    # than silently summing unrelated timing.json files.
    if not attributable:
        return {}
    if not intermediate_dir or not os.path.isdir(intermediate_dir):
        return {}
    files = glob.glob(os.path.join(intermediate_dir, "*", "timing.json"))
    agg = {s: 0.0 for s in TIMING_STAGES}
    n = 0
    for f in files:
        base = os.path.basename(os.path.dirname(f))
        if basenames is not None and base not in basenames:
            continue
        try:
            t = json.load(open(f))
        except Exception:
            continue
        n += 1
        for s in TIMING_STAGES:
            if isinstance(t.get(s), (int, float)):
                agg[s] += float(t[s])
    if not n:
        return {}
    return {
        "papers_with_timing": n,
        "total_wall_s": round(agg["total"], 1),
        "mean_wall_s_per_paper": round(agg["total"] / n, 1),
        "stage_seconds_total": {s: round(agg[s], 1) for s in TIMING_STAGES if s != "total"},
    }


def build_panel(args) -> Dict[str, Any]:
    basenames = _resolve_basenames(args.basenames_from)
    records, n_papers = _load_records(args.input, basenames)
    n = len(records)
    # Timing is attributable when input is the json-dir (pipeline-native) or
    # when an explicit basename filter ties records to intermediate dirs.
    timing_attributable = os.path.isdir(args.input) or (basenames is not None)
    all_metrics = [m for _, ms in GROUPS for m in ms]

    n_hollow = sum(1 for r in records if r.get("is_hollow") is True)
    non_hollow = [r for r in records if r.get("is_hollow") is not True]

    panel = {
        "label": args.label,
        "layer": args.layer,
        "input": args.input,
        "basenames_filter": (sorted(basenames) if basenames else None),
        "scale": {
            "records": n,
            "papers": n_papers,
            "is_hollow": n_hollow,
            "is_hollow_pct": round(100.0 * n_hollow / n, 1) if n else 0.0,
            "non_hollow_records": len(non_hollow),
            "has_is_hollow_field": any("is_hollow" in r for r in records),
        },
        "coverage_all": _coverage(records, all_metrics),
        "coverage_non_hollow": _coverage(non_hollow, all_metrics) if non_hollow else {},
        "smiles_sources": _smiles_sources(records),
        "timing": _timing(basenames, args.intermediate_dir, timing_attributable),
        "token_cost": "N/A (not persisted historically; capture on re-run from per_source logs)",
    }
    return panel


# ── rendering ─────────────────────────────────────────────────────────────────
def _md_table(panel: Dict[str, Any]) -> str:
    s = panel["scale"]
    lines = [
        f"# Metrics panel — {panel['label']}",
        "",
        f"- **Layer**: `{panel['layer']}`  _(raw_pipeline / +post / +manual_or_vlm — do not compare across layers naively)_",
        f"- **Input**: `{panel['input']}`",
        f"- **Records**: {s['records']}  |  **Papers**: {s['papers']}",
        f"- **is_hollow**: {s['is_hollow']} ({s['is_hollow_pct']}%)  |  **non-hollow**: {s['non_hollow_records']}"
        + ("" if s["has_is_hollow_field"] else "  ⚠️ no is_hollow field in this dataset"),
        "",
        "## Field coverage (% non-null)",
        "",
        "| Group | Field | All records | Non-hollow |",
        "|---|---|---|---|",
    ]
    cov_all = panel["coverage_all"]
    cov_nh = panel["coverage_non_hollow"]
    for group, ms in GROUPS:
        for m in ms:
            a = cov_all.get(m, {})
            nh = cov_nh.get(m, {})
            nh_cell = f"{nh.get('pct', '—')}%" if nh else "—"
            lines.append(f"| {group} | {m} | {a.get('pct', 0.0)}% ({a.get('filled', 0)}) | {nh_cell} |")

    lines += ["", "## SMILES sources (records with product_smiles)", ""]
    src = panel["smiles_sources"]
    if src:
        lines += ["| Source | Count |", "|---|---|"]
        for k, v in sorted(src.items(), key=lambda kv: -kv[1]):
            lines.append(f"| {k} | {v} |")
    else:
        lines.append("_(no product_smiles in dataset)_")

    t = panel["timing"]
    lines += ["", "## Cost / latency", ""]
    if t:
        lines.append(f"- Papers with timing: {t['papers_with_timing']}")
        lines.append(f"- Total wall time: {t['total_wall_s']} s  |  mean/paper: {t['mean_wall_s_per_paper']} s")
        stages = ", ".join(f"{k}={v}s" for k, v in t["stage_seconds_total"].items())
        lines.append(f"- Stage totals: {stages}")
    else:
        lines.append("_(no timing.json found for these basenames)_")
    lines.append(f"- Token cost: {panel['token_cost']}")
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Unified, layered dataset metrics panel.")
    ap.add_argument("--input", required=True, help="dir of *_normalized.json, or a .json/.csv/.xlsx file")
    ap.add_argument("--layer", required=True, choices=["raw_pipeline", "+post", "+manual_or_vlm"])
    ap.add_argument("--label", required=True, help="short dataset name for the report")
    ap.add_argument("--basenames-from", default=None,
                    help="dir of PDFs or text file of names; filters json-dir input + timing")
    ap.add_argument("--intermediate-dir", default="data/intermediate")
    ap.add_argument("--out", default=None, help="output path stem (writes .json + .md)")
    args = ap.parse_args()

    panel = build_panel(args)
    md = _md_table(panel)
    print(md)

    if args.out:
        with open(args.out + ".json", "w") as f:
            json.dump(panel, f, indent=2, ensure_ascii=False)
        with open(args.out + ".md", "w") as f:
            f.write(md)
        print(f"\n[panel] wrote {args.out}.json and {args.out}.md")


if __name__ == "__main__":
    main()
