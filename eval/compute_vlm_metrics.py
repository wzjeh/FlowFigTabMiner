"""
eval/compute_vlm_metrics.py
============================
解析 VLM (Gemini / Claude) 的原始 JSON 输出，与 Zhao 的人工标注 GT 对齐，
计算 Figure / Table 的 P/R/F1 和准确率，并与 FlowFigTabMiner 的结果并列对比。

使用方式（从项目根目录运行）：
    source flowfigtabminer/bin/activate
    python eval/compute_vlm_metrics.py

输入：
    eval/vlm_raw/gemini/F1.json ... T10.json
    eval/vlm_raw/claude/F1.json ... T10.json
    eval/layer1_figure_annotation.csv   （已有 GT）
    eval/layer1_table_annotation.csv    （已有 GT）
    eval/samples.json                   （样本配置）

输出：
    eval/vlm_comparison_report.json
    终端打印对比表
"""

import os
import sys
import csv
import json
import re
from collections import defaultdict

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR  = os.path.join(EVAL_DIR, "vlm_raw")
FIG_CSV  = os.path.join(EVAL_DIR, "layer1_figure_annotation.csv")
TAB_CSV  = os.path.join(EVAL_DIR, "layer1_table_annotation.csv")
SAMPLES  = os.path.join(EVAL_DIR, "samples.json")
REPORT   = os.path.join(EVAL_DIR, "vlm_comparison_report.json")

try:
    from rdkit import Chem
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False


# ─────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────

def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _prf(matched, n_pred, n_gt):
    p = matched / n_pred if n_pred > 0 else 0.0
    r = matched / n_gt   if n_gt   > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return round(p, 4), round(r, 4), round(f, 4)


def _load_csv(path):
    with open(path, newline="", encoding="utf-8-sig", errors="replace") as f:
        return list(csv.DictReader(f))


def _load_json_robust(path):
    """读取 VLM 输出 JSON，容忍 markdown 代码块包裹和尾部多余文字。"""
    if not os.path.exists(path):
        return None
    text = open(path, encoding="utf-8", errors="replace").read().strip()

    # 去掉 markdown ```json ... ```
    m = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()

    # 尝试直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 尝试找第一个 { ... } 或 [ ... ]
    for start_char, end_char in [('{', '}'), ('[', ']')]:
        idx = text.find(start_char)
        if idx < 0:
            continue
        depth = 0
        for i in range(idx, len(text)):
            if text[i] == start_char:
                depth += 1
            elif text[i] == end_char:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[idx:i+1])
                    except json.JSONDecodeError:
                        break
    print(f"  [警告] 无法解析 JSON: {path}")
    return None


# ─────────────────────────────────────────────────────────────
# Figure: 从 VLM JSON 提取点列表
# ─────────────────────────────────────────────────────────────

def extract_vlm_figure_points(data):
    """
    从 VLM 的 figure JSON 输出中提取 (series, X, Y_Left, Y_Right) 列表。
    兼容多种输出格式。
    """
    points = []
    if data is None:
        return points

    series_list = data.get("series", [])
    if not series_list and isinstance(data, dict):
        # 可能直接是 list 或其他格式
        for key in ("data", "datasets", "plots"):
            if key in data:
                series_list = data[key]
                break

    for s in series_list:
        name = s.get("name", s.get("label", s.get("series", "Unknown")))
        pts  = s.get("points", s.get("data", []))
        for pt in pts:
            x = _f(pt.get("X", pt.get("x", pt.get("X_value"))))
            yl = _f(pt.get("Y_Left", pt.get("y", pt.get("Y", pt.get("y_left")))))
            yr = _f(pt.get("Y_Right", pt.get("y_right")))
            if x is not None and yl is not None:
                points.append({"series": str(name), "X": x, "Y_Left": yl, "Y_Right": yr})

    return points


# ─────────────────────────────────────────────────────────────
# Table: 从 VLM JSON 提取单元格列表
# ─────────────────────────────────────────────────────────────

def extract_vlm_table_cells(data):
    """
    从 VLM 的 table JSON 输出中提取 (row, col, value) 列表。
    """
    cells = []
    if data is None:
        return cells

    columns = data.get("columns", [])
    rows = data.get("rows", data.get("data", []))

    for row_idx, row in enumerate(rows):
        if isinstance(row, dict):
            # {col_name: value} 格式
            for col_idx, col_name in enumerate(columns):
                val = str(row.get(col_name, "")).strip()
                if val:
                    cells.append({"row": row_idx + 1, "col": col_idx, "col_name": col_name, "value": val})
        elif isinstance(row, list):
            for col_idx, val in enumerate(row):
                val_str = str(val).strip() if val is not None else ""
                if val_str:
                    col_name = columns[col_idx] if col_idx < len(columns) else f"col_{col_idx}"
                    cells.append({"row": row_idx + 1, "col": col_idx, "col_name": col_name, "value": val_str})

    return cells


# ─────────────────────────────────────────────────────────────
# Figure 对比: VLM points vs GT
# ─────────────────────────────────────────────────────────────

def compute_figure_vlm_metrics(vlm_points, gt_rows):
    """
    用 GT 标注中的 is_match_within_tol 或 gt_X/gt_Y_Left 来评估 VLM 点。

    策略：用 GT 点坐标做最近邻匹配。
    GT 来源：layer1_figure_annotation.csv 中已人工标注过的行。
    """
    # 收集 GT 点（从人工标注中提取真实坐标）
    gt_points = []
    for r in gt_rows:
        flag = r.get("is_match_within_tol", "").strip()
        gt_x = _f(r.get("gt_X"))
        gt_y = _f(r.get("gt_Y_Left"))

        # 如果 is_match == "1"，说明 pipeline 的 pred 值就是正确的，可以当 GT
        if flag == "1":
            px = _f(r.get("pred_X"))
            py = _f(r.get("pred_Y_Left"))
            if px is not None and py is not None:
                gt_points.append({"X": px, "Y_Left": py})
        elif gt_x is not None and gt_y is not None:
            gt_points.append({"X": gt_x, "Y_Left": gt_y})
        # 对于 is_match == "0" 但没有 gt_X/gt_Y：这个点是 pipeline 的误判，
        # 我们无法知道 GT 坐标，所以跳过

    if not gt_points or not vlm_points:
        return {"status": "无法计算", "n_vlm": len(vlm_points), "n_gt": len(gt_points)}

    # 计算容差
    all_x = [p["X"] for p in gt_points + vlm_points]
    all_y = [p["Y_Left"] for p in gt_points + vlm_points]
    tol_x = 0.05 * (max(all_x) - min(all_x)) if len(all_x) >= 2 else 1e-6
    tol_y = 0.05 * (max(all_y) - min(all_y)) if len(all_y) >= 2 else 1e-6

    # 贪心最近邻匹配
    gt_used = [False] * len(gt_points)
    matched = 0
    for vp in vlm_points:
        best_dist, best_idx = float("inf"), -1
        for gi, gp in enumerate(gt_points):
            if gt_used[gi]:
                continue
            dx = abs(vp["X"] - gp["X"])
            dy = abs(vp["Y_Left"] - gp["Y_Left"])
            if dx <= tol_x and dy <= tol_y:
                dist = (dx / max(tol_x, 1e-9))**2 + (dy / max(tol_y, 1e-9))**2
                if dist < best_dist:
                    best_dist, best_idx = dist, gi
        if best_idx >= 0:
            gt_used[best_idx] = True
            matched += 1

    # 还需要加上 pipeline 发现但 VLM 也应该发现的"缺N个"的 GT 点
    # 从 description 字段中解析 "缺N个"
    extra_gt = 0
    seen_descriptions = set()
    for r in gt_rows:
        desc = r.get("description", "").strip()
        if desc and desc not in seen_descriptions:
            seen_descriptions.add(desc)
            m = re.search(r"缺(\d+)个", desc)
            if m:
                extra_gt += int(m.group(1))

    n_gt_total = len(gt_points) + extra_gt
    p, r_val, f = _prf(matched, len(vlm_points), n_gt_total)
    return {
        "n_vlm": len(vlm_points), "n_gt": n_gt_total,
        "matched": matched,
        "precision": p, "recall": r_val, "f1": f,
    }


# ─────────────────────────────────────────────────────────────
# Table 对比: VLM cells vs GT
# ─────────────────────────────────────────────────────────────

def compute_table_vlm_metrics(vlm_cells, gt_rows):
    """
    VLM 输出的 cell 列表 vs GT 标注行。
    按 (row, col) 位置匹配，比较值。
    """
    if not gt_rows:
        return {"status": "无GT"}

    # 构建 GT dict: (data_row, col_idx) -> {gt_value, predicted_value, is_correct}
    gt_dict = {}
    for r in gt_rows:
        key = (int(r["data_row"]), int(r["col_idx"]))
        gt_val = r.get("gt_value", "").strip()
        pred_val = r.get("predicted_value", "").strip()
        is_correct = r.get("is_correct", "").strip()

        # GT 值优先用 gt_value，没有就用 is_correct=1 时的 predicted_value
        if gt_val:
            gt_dict[key] = gt_val
        elif is_correct == "1" and pred_val:
            gt_dict[key] = pred_val

    if not gt_dict:
        return {"status": "GT 未标注"}

    # 对齐 VLM cells
    # VLM 的 row index 从 1 开始（data row），col 从 0 开始
    correct = 0
    total_compared = 0

    for vc in vlm_cells:
        key = (vc["row"], vc["col"])
        if key not in gt_dict:
            continue  # GT 里没有这个位置
        total_compared += 1

        vlm_val = vc["value"].strip()
        gt_val  = gt_dict[key]

        # 精确匹配
        if vlm_val == gt_val:
            correct += 1
            continue

        # 数值容差匹配 (±5%)
        vf, gf = _f(vlm_val), _f(gt_val)
        if vf is not None and gf is not None:
            if gf == 0:
                if abs(vf) < 1e-6:
                    correct += 1
            elif abs(vf - gf) / abs(gf) <= 0.05:
                correct += 1
            continue

        # 归一化后匹配（去空格、大小写）
        if vlm_val.lower().replace(" ", "") == gt_val.lower().replace(" ", ""):
            correct += 1
            continue

    acc = round(correct / total_compared, 4) if total_compared > 0 else 0.0
    return {
        "n_vlm_cells": len(vlm_cells),
        "n_gt_cells": len(gt_dict),
        "n_compared": total_compared,
        "correct": correct,
        "accuracy": acc,
    }


# ─────────────────────────────────────────────────────────────
# FlowFigTabMiner 已有结果
# ─────────────────────────────────────────────────────────────

def load_pipeline_figure_metrics(fig_csv_rows):
    """从 Zhao 的标注 CSV 中计算 FlowFigTabMiner 的 per-figure 指标。"""
    from compute_metrics import compute_figure_metrics
    return compute_figure_metrics(fig_csv_rows)


def _compute_pipeline_figure_simple(fig_rows):
    """简化版：直接从标注 CSV 算 pipeline 的 figure 指标。"""
    by_label = defaultdict(list)
    for r in fig_rows:
        by_label[r["label"]].append(r)

    results = {}
    for label, pts in sorted(by_label.items()):
        has_flag = any(r.get("is_match_within_tol", "").strip() in ("0", "1") for r in pts)
        if not has_flag:
            results[label] = {"status": "未标注"}
            continue

        annotated = [p for p in pts if p.get("is_match_within_tol", "").strip() in ("0", "1")]
        matched = sum(1 for p in annotated if p.get("is_match_within_tol", "").strip() == "1")
        n_pred = len(pts)

        # 计算 n_gt (包含"缺N个")
        extra_gt = 0
        seen_desc = set()
        for r in pts:
            desc = r.get("description", "").strip()
            if desc and desc not in seen_desc:
                seen_desc.add(desc)
                m = re.search(r"缺(\d+)个", desc)
                if m:
                    extra_gt += int(m.group(1))

        n_gt = n_pred + extra_gt
        p, r_val, f = _prf(matched, n_pred, n_gt)
        results[label] = {"precision": p, "recall": r_val, "f1": f,
                          "matched": matched, "n_pred": n_pred, "n_gt": n_gt}
    return results


def _compute_pipeline_table_simple(tab_rows):
    """简化版：直接从标注 CSV 算 pipeline 的 table 指标。"""
    by_label = defaultdict(list)
    for r in tab_rows:
        by_label[r["label"]].append(r)

    results = {}
    for label, cells in sorted(by_label.items()):
        ann = [c for c in cells if c.get("is_correct", "").strip() in ("0", "1")]
        if not ann:
            results[label] = {"status": "未标注"}
            continue
        correct = sum(1 for c in ann if c.get("is_correct", "").strip() == "1")
        acc = round(correct / len(ann), 4) if ann else 0.0
        results[label] = {"accuracy": acc, "correct": correct, "n_cells": len(ann)}
    return results


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    # 加载配置和 GT
    with open(SAMPLES, encoding="utf-8") as f:
        samples = json.load(f)
    fig_rows = _load_csv(FIG_CSV)
    tab_rows = _load_csv(TAB_CSV)

    # Pipeline 已有指标
    pipeline_fig = _compute_pipeline_figure_simple(fig_rows)
    pipeline_tab = _compute_pipeline_table_simple(tab_rows)

    # 按 label 分组 GT
    fig_gt_by_label = defaultdict(list)
    for r in fig_rows:
        fig_gt_by_label[r["label"]].append(r)
    tab_gt_by_label = defaultdict(list)
    for r in tab_rows:
        tab_gt_by_label[r["label"]].append(r)

    # 检测哪些模型有数据
    models = []
    for d in sorted(os.listdir(RAW_DIR)):
        dp = os.path.join(RAW_DIR, d)
        if os.path.isdir(dp) and any(f.endswith(".json") for f in os.listdir(dp)):
            models.append(d)

    if not models:
        print("[错误] eval/vlm_raw/ 下没有找到任何模型输出目录。")
        print("       请先将 VLM 输出保存到 eval/vlm_raw/gemini/ 或 eval/vlm_raw/claude/")
        sys.exit(1)

    print(f"检测到模型: {models}")

    # ── Figure 对比 ──
    figure_comparison = {}
    figure_labels = [s["label"] for s in samples["figures"]]

    for label in figure_labels:
        gt = fig_gt_by_label.get(label, [])
        entry = {"pipeline": pipeline_fig.get(label, {"status": "N/A"})}

        for model in models:
            path = os.path.join(RAW_DIR, model, f"{label}.json")
            data = _load_json_robust(path)
            if data is None:
                entry[model] = {"status": "未提交"}
                continue
            vlm_pts = extract_vlm_figure_points(data)
            entry[model] = compute_figure_vlm_metrics(vlm_pts, gt)

        figure_comparison[label] = entry

    # ── Table 对比 ──
    table_comparison = {}
    table_labels = [s["label"] for s in samples["tables"]]

    for label in table_labels:
        gt = tab_gt_by_label.get(label, [])
        entry = {"pipeline": pipeline_tab.get(label, {"status": "N/A"})}

        for model in models:
            path = os.path.join(RAW_DIR, model, f"{label}.json")
            data = _load_json_robust(path)
            if data is None:
                entry[model] = {"status": "未提交"}
                continue
            vlm_cells = extract_vlm_table_cells(data)
            entry[model] = compute_table_vlm_metrics(vlm_cells, gt)

        table_comparison[label] = entry

    # ── 汇总 ──
    summary = {"figure_avg_f1": {}, "table_avg_accuracy": {}}

    # Pipeline figure avg F1
    fig_f1s = [v.get("f1", 0) for v in pipeline_fig.values() if "f1" in v]
    summary["figure_avg_f1"]["pipeline"] = round(sum(fig_f1s) / len(fig_f1s), 4) if fig_f1s else "N/A"

    # Pipeline table avg accuracy
    tab_accs = [v.get("accuracy", 0) for v in pipeline_tab.values() if "accuracy" in v]
    summary["table_avg_accuracy"]["pipeline"] = round(sum(tab_accs) / len(tab_accs), 4) if tab_accs else "N/A"

    for model in models:
        # Figure
        f1s = []
        for label in figure_labels:
            m = figure_comparison[label].get(model, {})
            if "f1" in m:
                f1s.append(m["f1"])
        summary["figure_avg_f1"][model] = round(sum(f1s) / len(f1s), 4) if f1s else "N/A"

        # Table
        accs = []
        for label in table_labels:
            m = table_comparison[label].get(model, {})
            if "accuracy" in m:
                accs.append(m["accuracy"])
        summary["table_avg_accuracy"][model] = round(sum(accs) / len(accs), 4) if accs else "N/A"

    report = {
        "figure_comparison": figure_comparison,
        "table_comparison": table_comparison,
        "summary": summary,
    }

    with open(REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n完整对比报告: {REPORT}")

    # ── 打印表格 ──
    print(f"\n{'═'*70}")
    print("  Figure 数据点提取 — 对比")
    print(f"{'═'*70}")
    header = f"  {'Label':<6} {'Pipeline':>28}"
    for m in models:
        header += f" {'│ ' + m:>28}"
    print(header)
    print(f"  {'─'*6} {'─'*28}" + "".join(f" {'─'*28}" for _ in models))

    for label in figure_labels:
        row = f"  {label:<6}"
        for source in ["pipeline"] + models:
            d = figure_comparison[label].get(source, {})
            if "f1" in d:
                row += f"  P={d['precision']:.2f} R={d.get('recall', 0):.2f} F1={d['f1']:.2f}"
            elif "status" in d:
                row += f"  {d['status']:>26}"
            else:
                row += f"  {'—':>26}"
        print(row)

    # 汇总行
    row = f"  {'AVG':<6}"
    for source in ["pipeline"] + models:
        val = summary["figure_avg_f1"].get(source, "N/A")
        if isinstance(val, float):
            row += f"  {'F1=' + f'{val:.2f}':>26}"
        else:
            row += f"  {str(val):>26}"
    print(f"  {'─'*6} {'─'*28}" + "".join(f" {'─'*28}" for _ in models))
    print(row)

    print(f"\n{'═'*70}")
    print("  Table 单元格提取 — 对比")
    print(f"{'═'*70}")
    header = f"  {'Label':<6} {'Pipeline':>20}"
    for m in models:
        header += f" {'│ ' + m:>20}"
    print(header)
    print(f"  {'─'*6} {'─'*20}" + "".join(f" {'─'*20}" for _ in models))

    for label in table_labels:
        row = f"  {label:<6}"
        for source in ["pipeline"] + models:
            d = table_comparison[label].get(source, {})
            if "accuracy" in d:
                row += f"  acc={d['accuracy']:.2f} ({d.get('correct', '?')}/{d.get('n_compared', d.get('n_cells', '?'))})"
            elif "status" in d:
                row += f"  {d['status']:>18}"
            else:
                row += f"  {'—':>18}"
        print(row)

    row = f"  {'AVG':<6}"
    for source in ["pipeline"] + models:
        val = summary["table_avg_accuracy"].get(source, "N/A")
        if isinstance(val, float):
            row += f"  {'acc=' + f'{val:.2f}':>18}"
        else:
            row += f"  {str(val):>18}"
    print(f"  {'─'*6} {'─'*20}" + "".join(f" {'─'*20}" for _ in models))
    print(row)

    print()


if __name__ == "__main__":
    main()
