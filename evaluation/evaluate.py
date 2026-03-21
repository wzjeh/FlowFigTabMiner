"""
FigTabMiner Evaluation Framework

Compares a _normalized.json prediction against a manually annotated ground
truth file, and reports per-field Precision / Recall / F1.

Usage:
    flowfigtabminer/bin/python -m evaluation.evaluate \\
        --gt  evaluation/ground_truth/example.json \\
        --pred data/final_output/example_normalized.json

    # Save results to JSON:
    flowfigtabminer/bin/python -m evaluation.evaluate \\
        --gt  evaluation/ground_truth/example.json \\
        --pred data/final_output/example_normalized.json \\
        --out  evaluation/results/example_eval.json

Ground Truth Format:
    See evaluation/ground_truth/TEMPLATE.json for a filled example.

    Key rules:
    - Each record must have "source_table_or_figure" (matching key).
    - "entry_number" is optional but strongly recommended for accurate matching.
    - Only include fields you want to evaluate — absent/omitted fields are
      simply ignored and do NOT count as wrong.
    - Condition fields (temperature_C, solvent, ...) go inside "conditions": {}
      exactly as in _normalized.json.
    - Use "_gt_notes" for your own annotation notes (ignored by evaluator).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any


# ---------------------------------------------------------------------------
# Fields that live inside the "conditions" sub-dict
# ---------------------------------------------------------------------------
_CONDITION_FIELDS = {
    "temperature_C", "residence_time_s", "flow_rate_mL_min",
    "flow_rate_stream1_mL_min", "flow_rate_stream2_mL_min",
    "solvent", "solvent_list", "catalyst", "catalyst_metal",
    "catalyst_loading_pct", "ligand", "ligand_loading_pct",
    "additive", "pressure_bar", "reactor_type",
}

# Fields where numeric tolerance comparison applies
_NUMERIC_FIELDS = {
    "yield_pct", "conversion_pct", "selectivity_pct", "ee_pct",
    "batch_yield_pct", "temperature_C", "residence_time_s",
    "flow_rate_mL_min", "flow_rate_stream1_mL_min", "flow_rate_stream2_mL_min",
    "pressure_bar", "catalyst_loading_pct", "ligand_loading_pct",
}

# Fields to exclude from evaluation (used only for matching)
_MATCH_KEY_FIELDS = {"source_table_or_figure", "entry_number"}

# Internal annotation fields (ignored)
_INTERNAL_FIELDS = {"_gt_notes"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flatten(record: dict) -> dict:
    """Hoist 'conditions' and 'other_metrics' sub-dicts to top level."""
    flat = {k: v for k, v in record.items()
            if k not in ("conditions", "other_metrics") and k not in _INTERNAL_FIELDS}
    flat.update(record.get("conditions") or {})
    flat.update(record.get("other_metrics") or {})
    return flat


def _norm_str(s: Any) -> str:
    if s is None:
        return ""
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def _is_match(gt_val: Any, pred_val: Any, field: str) -> bool:
    """True if predicted value matches ground truth for this field."""
    if gt_val is None:
        return pred_val is None
    if pred_val is None:
        return False

    # Numeric fields: within 1% relative or 0.01 absolute tolerance
    if field in _NUMERIC_FIELDS:
        try:
            gv, pv = float(gt_val), float(pred_val)
            return abs(gv - pv) <= max(0.01, abs(gv) * 0.01)
        except (TypeError, ValueError):
            pass

    # String fields: case-insensitive + whitespace-normalized
    gs, ps = _norm_str(gt_val), _norm_str(pred_val)
    if gs == ps:
        return True
    # Abbreviation variant: "tetrahydrofuran (thf)" matches "thf"
    if f"({ps})" in gs or f"({gs})" in ps:
        return True
    return False


def _match_key(flat: dict) -> str:
    """Build a (source, entry_number) lookup key."""
    src = _norm_str(flat.get("source_table_or_figure", ""))
    entry = _norm_str(flat.get("entry_number", ""))
    return f"{src}|{entry}"


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate(gt_path: str, pred_path: str) -> dict:
    with open(gt_path, encoding="utf-8") as f:
        gt_records = json.load(f)
    with open(pred_path, encoding="utf-8") as f:
        pred_records = json.load(f)

    gt_flats = [_flatten(r) for r in gt_records]
    pred_flats = [_flatten(r) for r in pred_records]

    # -- Match GT records to predicted records (greedy, one-to-one) ----------
    used_pred = set()
    matched_pairs: list[tuple[dict, dict]] = []
    unmatched_gt: list[dict] = []

    for g in gt_flats:
        gk = _match_key(g)
        gt_src = _norm_str(g.get("source_table_or_figure", ""))

        best_idx = None
        # Pass 1: exact (source + entry_number) match
        for j, p in enumerate(pred_flats):
            if j in used_pred:
                continue
            if _match_key(p) == gk:
                best_idx = j
                break

        # Pass 2: source-only match (first unused)
        if best_idx is None and gt_src:
            for j, p in enumerate(pred_flats):
                if j in used_pred:
                    continue
                if _norm_str(p.get("source_table_or_figure", "")) == gt_src:
                    best_idx = j
                    break

        if best_idx is not None:
            used_pred.add(best_idx)
            matched_pairs.append((g, pred_flats[best_idx]))
        else:
            unmatched_gt.append(g)

    # -- Determine which fields to evaluate ----------------------------------
    eval_fields: set[str] = set()
    for g in gt_flats:
        for k, v in g.items():
            if v is not None and k not in _MATCH_KEY_FIELDS:
                eval_fields.add(k)

    # -- Per-field metrics ---------------------------------------------------
    field_stats: dict[str, dict] = {}
    for field in sorted(eval_fields):
        tp = fp = fn = 0
        for g, p in matched_pairs:
            if field not in g or g[field] is None:
                continue  # GT doesn't specify this field for this record
            if _is_match(g[field], p.get(field), field):
                tp += 1
            else:
                fn += 1
                if p.get(field) is not None:
                    fp += 1

        # Unmatched GT records count as FN
        for g in unmatched_gt:
            if field in g and g[field] is not None:
                fn += 1

        defined = (
            sum(1 for g, _ in matched_pairs if field in g and g[field] is not None)
            + sum(1 for g in unmatched_gt if field in g and g[field] is not None)
        )
        if defined == 0:
            continue

        precision = tp / (tp + fp) if (tp + fp) > 0 else None
        recall    = tp / (tp + fn) if (tp + fn) > 0 else None
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and (precision + recall) > 0
            else None
        )
        field_stats[field] = {
            "defined": defined, "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1,
        }

    f1_values = [s["f1"] for s in field_stats.values() if s["f1"] is not None]
    macro_f1 = sum(f1_values) / len(f1_values) if f1_values else None

    return {
        "gt_count": len(gt_records),
        "pred_count": len(pred_records),
        "matched": len(matched_pairs),
        "unmatched_gt": len(unmatched_gt),
        "field_stats": field_stats,
        "macro_f1": macro_f1,
    }


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------

def print_report(results: dict, gt_path: str, pred_path: str) -> None:
    W = 75
    print(f"\n{'=' * W}")
    print("FigTabMiner Evaluation Report")
    print(f"  GT:   {gt_path}")
    print(f"  Pred: {pred_path}")
    print(f"{'=' * W}")
    print(
        f"GT records : {results['gt_count']:>4}  |  "
        f"Predicted : {results['pred_count']:>4}  |  "
        f"Matched : {results['matched']:>4}  |  "
        f"Unmatched GT : {results['unmatched_gt']:>4}"
    )
    print()

    if not results["field_stats"]:
        print("No fields to evaluate (ground truth is empty or has no non-null fields).")
        return

    hdr = f"{'Field':<30} {'Defined':>7} {'TP':>4} {'FP':>4} {'FN':>4} {'Precision':>10} {'Recall':>8} {'F1':>8}"
    print(hdr)
    print("-" * W)

    for field, s in results["field_stats"].items():
        p_s  = f"{s['precision']:.3f}" if s["precision"] is not None else "  n/a"
        r_s  = f"{s['recall']:.3f}"    if s["recall"]    is not None else "  n/a"
        f1_s = f"{s['f1']:.3f}"        if s["f1"]        is not None else "  n/a"
        print(
            f"{field:<30} {s['defined']:>7} {s['tp']:>4} {s['fp']:>4} {s['fn']:>4}"
            f" {p_s:>10} {r_s:>8} {f1_s:>8}"
        )

    print("-" * W)
    if results["macro_f1"] is not None:
        print(f"{'Macro F1 (mean over all fields):':<47} {results['macro_f1']:.3f}")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate FigTabMiner _normalized.json against ground truth."
    )
    parser.add_argument("--gt",   required=True, help="Ground truth JSON path")
    parser.add_argument("--pred", required=True, help="_normalized.json path")
    parser.add_argument("--out",  default=None,  help="Save results to this JSON path")
    args = parser.parse_args()

    for p, label in [(args.gt, "GT"), (args.pred, "Pred")]:
        if not os.path.exists(p):
            print(f"ERROR: {label} file not found: {p}")
            sys.exit(1)

    results = evaluate(args.gt, args.pred)
    print_report(results, args.gt, args.pred)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved → {args.out}")
