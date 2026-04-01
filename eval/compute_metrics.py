"""
eval/compute_metrics.py
========================
从填写完毕的三份标注 CSV 计算 Layer1 和 Layer2 指标。

使用方式（从项目根目录运行）：
    source flowfigtabminer/bin/activate
    python eval/compute_metrics.py

输入：
    eval/layer1_figure_annotation.csv   （填好 gt_X/gt_Y 或 is_match_within_tol）
    eval/layer1_table_annotation.csv    （填好 is_correct）
    eval/layer2_final_annotation.csv    （填好 is_correct）

输出：
    eval/eval_report.json    — 完整指标（机器可读）
    eval/eval_summary.csv    — 汇总表（Excel 可读）
    终端打印摘要
"""

import os
import sys
import csv
import json
import re
from collections import defaultdict

EVAL_DIR = os.path.dirname(__file__)
FIG_CSV  = os.path.join(EVAL_DIR, "layer1_figure_annotation.csv")
TAB_CSV  = os.path.join(EVAL_DIR, "layer1_table_annotation.csv")
FIN_CSV  = os.path.join(EVAL_DIR, "layer2_final_annotation.csv")
REPORT_JSON = os.path.join(EVAL_DIR, "eval_report.json")
SUMMARY_CSV = os.path.join(EVAL_DIR, "eval_summary.csv")

try:
    from rdkit import Chem
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    print("[提示] RDKit 未安装，SMILES canonical match 将跳过。")

NUMERIC_TOL = 0.05   # 数值字段允许 ±5% tolerance

# ─────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────

def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _canon(smiles: str):
    if not HAS_RDKIT or not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles.strip())
    return Chem.MolToSmiles(mol) if mol else None


def _prf(matched, n_pred, n_gt):
    p = matched / n_pred if n_pred > 0 else 0.0
    r = matched / n_gt   if n_gt   > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return round(p, 4), round(r, 4), round(f, 4)


def _load(path):
    if not os.path.exists(path):
        print(f"[错误] 找不到: {path}")
        sys.exit(1)
    with open(path, newline="", encoding="utf-8-sig", errors="replace") as f:
        return list(csv.DictReader(f))


def _is_annotated(row, key="is_correct"):
    return row.get(key, "").strip() in ("0", "1")


def _is_match_flag(row):
    return row.get("is_match_within_tol", "").strip()


# ─────────────────────────────────────────────────────────────
# Layer 1 — Figure
# ─────────────────────────────────────────────────────────────

def compute_figure_metrics(rows):
    by_label = defaultdict(list)
    for r in rows:
        by_label[r["label"]].append(r)

    per_sample, agg = {}, {"matched": 0, "n_pred": 0, "n_gt": 0}

    for label, pts in sorted(by_label.items()):
        has_coords = any(_f(p.get("gt_X")) is not None for p in pts)
        has_flag   = any(_is_match_flag(p) in ("0", "1") for p in pts)

        if not has_coords and not has_flag:
            per_sample[label] = {"status": "未标注", "n_predicted": len(pts)}
            continue

        if has_coords:
            # 模式 A：坐标 + 容差匹配
            pred_xs = [_f(p["pred_X"])    for p in pts]
            pred_ys = [_f(p["pred_Y_Left"]) for p in pts]
            gt_xs   = [_f(p.get("gt_X"))  for p in pts]
            gt_ys   = [_f(p.get("gt_Y_Left")) for p in pts]

            all_x = [v for v in pred_xs + gt_xs if v is not None]
            all_y = [v for v in pred_ys + gt_ys if v is not None]
            tol_x = 0.05 * (max(all_x) - min(all_x)) if len(all_x) >= 2 else 1e-6
            tol_y = 0.05 * (max(all_y) - min(all_y)) if len(all_y) >= 2 else 1e-6

            matched = sum(
                1 for px, py, gx, gy in zip(pred_xs, pred_ys, gt_xs, gt_ys)
                if None not in (px, py, gx, gy)
                and abs(px - gx) <= tol_x
                and abs(py - gy) <= tol_y
            )
            n_pred = len(pts)
            n_gt   = sum(1 for gx in gt_xs if gx is not None)
            mode   = "坐标容差"
        else:
            # 模式 B：人工判断
            annotated = [p for p in pts if _is_match_flag(p) in ("0", "1")]
            matched   = sum(1 for p in annotated if _is_match_flag(p) == "1")
            n_pred    = len(pts)
            n_gt      = n_pred
            mode      = "人工判断"

        p, r, f = _prf(matched, n_pred, n_gt)
        per_sample[label] = {
            "mode": mode, "n_predicted": n_pred, "n_gt": n_gt, "matched": matched,
            "precision": p, "recall": r, "f1": f,
        }
        agg["matched"] += matched
        agg["n_pred"]  += n_pred
        agg["n_gt"]    += n_gt

    op, or_, of = _prf(agg["matched"], agg["n_pred"], agg["n_gt"])
    return {
        "per_sample": per_sample,
        "overall": {
            "precision": op, "recall": or_, "f1": of,
            "total_predicted": agg["n_pred"], "total_gt": agg["n_gt"],
            "total_matched": agg["matched"],
        },
    }


# ─────────────────────────────────────────────────────────────
# Layer 1 — Table
# ─────────────────────────────────────────────────────────────

def compute_table_metrics(rows):
    by_label = defaultdict(list)
    for r in rows:
        by_label[r["label"]].append(r)

    per_sample = {}
    agg = defaultdict(int)

    for label, cells in sorted(by_label.items()):
        ann = [c for c in cells if _is_annotated(c)]
        if not ann:
            per_sample[label] = {"status": "未标注", "n_cells": len(cells)}
            continue

        text_c   = [c for c in ann if c["cell_type"] == "text"]
        smiles_c = [c for c in ann if c["cell_type"] == "smiles"]

        # Text accuracy
        tc_correct = sum(1 for c in text_c if c["is_correct"].strip() == "1")
        tc_acc = tc_correct / len(text_c) if text_c else None

        # SMILES valid rate
        if smiles_c and HAS_RDKIT:
            sv = sum(1 for c in smiles_c if _canon(c["predicted_value"]) is not None)
            sv_rate = sv / len(smiles_c)
        else:
            sv, sv_rate = None, None

        # SMILES canonical match (需要 gt_value)
        sm_with_gt = [c for c in smiles_c if c.get("gt_value", "").strip()]
        if sm_with_gt and HAS_RDKIT:
            sm = sum(
                1 for c in sm_with_gt
                if _canon(c["predicted_value"]) is not None
                and _canon(c["predicted_value"]) == _canon(c["gt_value"])
            )
            sm_rate = sm / len(sm_with_gt)
        else:
            sm, sm_rate = None, None

        overall_correct = sum(1 for c in ann if c["is_correct"].strip() == "1")
        overall_acc = overall_correct / len(ann)

        per_sample[label] = {
            "n_cells": len(ann),
            "text_accuracy":        round(tc_acc,  4) if tc_acc  is not None else "N/A",
            "smiles_valid_rate":    round(sv_rate,  4) if sv_rate is not None else "N/A",
            "smiles_canonical_match": round(sm_rate, 4) if sm_rate is not None else "N/A",
            "overall_cell_accuracy": round(overall_acc, 4),
        }

        if tc_acc  is not None: agg["tc_correct"] += tc_correct;  agg["tc_total"] += len(text_c)
        if sv_rate is not None: agg["sv_valid"] += sv;             agg["sv_total"] += len(smiles_c)
        if sm_rate is not None: agg["sm_match"] += sm;             agg["sm_total"] += len(sm_with_gt)
        agg["ov_correct"] += overall_correct; agg["ov_total"] += len(ann)

    def _safe(num, den):
        return round(num / den, 4) if den > 0 else "N/A"

    overall = {
        "text_accuracy":          _safe(agg["tc_correct"], agg["tc_total"]),
        "smiles_valid_rate":      _safe(agg["sv_valid"],   agg["sv_total"]),
        "smiles_canonical_match": _safe(agg["sm_match"],   agg["sm_total"]),
        "overall_cell_accuracy":  _safe(agg["ov_correct"], agg["ov_total"]),
        "total_cells":   agg["ov_total"],
        "text_cells":    agg["tc_total"],
        "smiles_cells":  agg["sv_total"],
    }
    return {"per_sample": per_sample, "overall": overall}


# ─────────────────────────────────────────────────────────────
# Layer 2 — Final output
# ─────────────────────────────────────────────────────────────

def compute_final_metrics(rows):
    """
    按字段分组：对每个 field（yield_pct / temperature_C / …）计算准确率。
    支持两种标注模式：
      - 只填 is_correct (1/0) → 直接统计
      - 填了 gt_value + is_correct → 数值字段自动用 ±5% tolerance 重新判断
    """
    by_field   = defaultdict(list)
    by_paper   = defaultdict(list)
    for r in rows:
        if _is_annotated(r):
            by_field[r["field"]].append(r)
            by_paper[r["paper_id"]].append(r)

    per_field, per_paper = {}, {}

    for field, cells in sorted(by_field.items()):
        # 尝试数值容差重判（如果 gt_value 有值）
        correct = 0
        for c in cells:
            pred_v = c["predicted_value"].strip()
            gt_v   = c.get("gt_value", "").strip()
            raw_flag = c["is_correct"].strip()

            if gt_v:
                # 数值字段
                pred_f, gt_f = _f(pred_v), _f(gt_v)
                if pred_f is not None and gt_f is not None:
                    if gt_f == 0:
                        match = abs(pred_f) < 1e-6
                    else:
                        match = abs(pred_f - gt_f) / abs(gt_f) <= NUMERIC_TOL
                    correct += int(match)
                    continue
                # SMILES 字段
                if HAS_RDKIT and _canon(pred_v) and _canon(gt_v):
                    correct += int(_canon(pred_v) == _canon(gt_v))
                    continue
            # 回退到人工标注
            correct += int(raw_flag == "1")

        acc = round(correct / len(cells), 4) if cells else 0.0
        per_field[field] = {"n": len(cells), "correct": correct, "accuracy": acc}

    for paper, cells in sorted(by_paper.items()):
        correct = sum(
            1 for c in cells
            if c["is_correct"].strip() == "1"
        )
        acc = round(correct / len(cells), 4) if cells else 0.0
        per_paper[paper] = {"n": len(cells), "correct": correct, "accuracy": acc}

    total_correct = sum(v["correct"] for v in per_field.values())
    total_n       = sum(v["n"]       for v in per_field.values())
    overall_acc   = round(total_correct / total_n, 4) if total_n > 0 else 0.0

    return {
        "per_field": per_field,
        "per_paper": per_paper,
        "overall": {
            "accuracy": overall_acc,
            "total_fields_evaluated": total_n,
            "total_correct": total_correct,
        },
    }


# ─────────────────────────────────────────────────────────────
# Summary CSV
# ─────────────────────────────────────────────────────────────

def write_summary_csv(fig_m, tab_m, fin_m, path):
    rows = []

    # Layer1 Figure per-sample
    for label, m in sorted(fig_m["per_sample"].items()):
        if m.get("status") == "未标注":
            rows.append({"layer": "L1-figure", "category": label, "metric": "status", "value": "未标注"})
        else:
            for k in ("precision", "recall", "f1"):
                rows.append({"layer": "L1-figure", "category": label, "metric": k, "value": m[k]})
    ov = fig_m["overall"]
    for k in ("precision", "recall", "f1"):
        rows.append({"layer": "L1-figure", "category": "OVERALL", "metric": k, "value": ov[k]})

    # Layer1 Table per-sample
    for label, m in sorted(tab_m["per_sample"].items()):
        if m.get("status") == "未标注":
            rows.append({"layer": "L1-table", "category": label, "metric": "status", "value": "未标注"})
        else:
            for k in ("text_accuracy", "smiles_valid_rate", "smiles_canonical_match", "overall_cell_accuracy"):
                rows.append({"layer": "L1-table", "category": label, "metric": k, "value": m[k]})
    ov = tab_m["overall"]
    for k in ("text_accuracy", "smiles_valid_rate", "smiles_canonical_match", "overall_cell_accuracy"):
        rows.append({"layer": "L1-table", "category": "OVERALL", "metric": k, "value": ov[k]})

    # Layer2 Final per-field
    for field, m in sorted(fin_m["per_field"].items()):
        rows.append({"layer": "L2-final", "category": field, "metric": "accuracy", "value": m["accuracy"]})
        rows.append({"layer": "L2-final", "category": field, "metric": "n",        "value": m["n"]})
    ov = fin_m["overall"]
    rows.append({"layer": "L2-final", "category": "OVERALL", "metric": "accuracy", "value": ov["accuracy"]})

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["layer", "category", "metric", "value"])
        writer.writeheader()
        writer.writerows(rows)


# ─────────────────────────────────────────────────────────────
# Print helper
# ─────────────────────────────────────────────────────────────

def _bar(v, width=30):
    if not isinstance(v, float):
        return ""
    filled = int(v * width)
    return "█" * filled + "░" * (width - filled) + f"  {v:.1%}"


def print_report(fig_m, tab_m, fin_m):
    sep = "─" * 56

    print(f"\n{'═'*56}")
    print("  Layer 1 — Figure 坐标提取")
    print(f"{'═'*56}")
    ov = fig_m["overall"]
    print(f"  Overall   Precision={ov['precision']}  Recall={ov['recall']}  F1={ov['f1']}")
    print(f"            matched={ov['total_matched']} / pred={ov['total_predicted']} / gt={ov['total_gt']}")
    print(sep)
    for label, m in sorted(fig_m["per_sample"].items()):
        if m.get("status") == "未标注":
            print(f"  {label}: 未标注")
        else:
            print(f"  {label} [{m.get('mode','')}]  P={m['precision']}  R={m['recall']}  "
                  f"F1={m['f1']}  ({m['matched']}/{m['n_predicted']})")

    print(f"\n{'═'*56}")
    print("  Layer 1 — Table Cell 提取")
    print(f"{'═'*56}")
    ov = tab_m["overall"]
    print(f"  Overall  text_acc={ov['text_accuracy']}  "
          f"smiles_valid={ov['smiles_valid_rate']}  "
          f"smiles_canonical={ov['smiles_canonical_match']}")
    print(f"           cell_acc={ov['overall_cell_accuracy']}  "
          f"(cells={ov['total_cells']}, text={ov['text_cells']}, smiles={ov['smiles_cells']})")
    print(sep)
    for label, m in sorted(tab_m["per_sample"].items()):
        if m.get("status") == "未标注":
            print(f"  {label}: 未标注")
        else:
            print(f"  {label}  text={m['text_accuracy']}  "
                  f"smiles_valid={m['smiles_valid_rate']}  "
                  f"canonical={m['smiles_canonical_match']}  "
                  f"overall={m['overall_cell_accuracy']}")

    print(f"\n{'═'*56}")
    print("  Layer 2 — 最终 JSON 字段准确率")
    print(f"{'═'*56}")
    ov = fin_m["overall"]
    print(f"  Overall  accuracy={ov['accuracy']}  "
          f"({ov['total_correct']}/{ov['total_fields_evaluated']} field-values)")
    print(sep)
    for field, m in sorted(fin_m["per_field"].items()):
        bar = _bar(m["accuracy"])
        print(f"  {field:<22} {bar}   n={m['n']}")
    print()


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("读取标注文件...")
    fig_rows = _load(FIG_CSV)
    tab_rows = _load(TAB_CSV)
    fin_rows = _load(FIN_CSV)

    fig_m = compute_figure_metrics(fig_rows)
    tab_m = compute_table_metrics(tab_rows)
    fin_m = compute_final_metrics(fin_rows)

    # 保存
    report = {"layer1_figure": fig_m, "layer1_table": tab_m, "layer2_final": fin_m}
    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"完整报告: {REPORT_JSON}")

    write_summary_csv(fig_m, tab_m, fin_m, SUMMARY_CSV)
    print(f"汇总 CSV:  {SUMMARY_CSV}")

    print_report(fig_m, tab_m, fin_m)
