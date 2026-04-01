"""
eval/generate_vlm_annotation.py
================================
解析 VLM (Gemini / Claude) 的 JSON 输出，生成与 pipeline 标注格式一致的 CSV，
供 Zhao 人工标注 VLM 提取结果是否正确。

使用方式（从项目根目录运行）：
    source flowfigtabminer/bin/activate
    python eval/generate_vlm_annotation.py

输入：
    eval/images/figures/F{n}.json          — Gemini figure
    eval/images/figures/F{n}_claude.json   — Claude figure
    eval/images/tables/T{n}.json           — Gemini table
    eval/images/tables/T{n}_claude.json    — Claude table

输出：
    eval/vlm_figure_annotation_gemini.csv
    eval/vlm_figure_annotation_claude.csv
    eval/vlm_table_annotation_gemini.csv
    eval/vlm_table_annotation_claude.csv
"""

import os
import json
import csv
import re

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(EVAL_DIR, "images")
FIG_DIR = os.path.join(IMAGES_DIR, "figures")
TAB_DIR = os.path.join(IMAGES_DIR, "tables")

# 当前可用的标签
FIGURE_LABELS = [f"F{i}" for i in range(1, 11)]
TABLE_LABELS  = [f"T{i}" for i in range(1, 11)]


# ─────────────────────────────────────────────────────────────
# JSON 加载（容忍 markdown 包裹）
# ─────────────────────────────────────────────────────────────

def load_json(path):
    if not os.path.exists(path):
        return None
    text = open(path, encoding="utf-8", errors="replace").read().strip()
    # 去掉 ```json ... ```
    m = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 尝试提取第一个 { ... }
    idx = text.find("{")
    if idx >= 0:
        depth = 0
        for i in range(idx, len(text)):
            if text[i] == "{": depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[idx:i+1])
                    except json.JSONDecodeError:
                        break
    print(f"  [警告] 无法解析: {path}")
    return None


# ─────────────────────────────────────────────────────────────
# Figure: 提取点列表
# ─────────────────────────────────────────────────────────────

def _float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def extract_figure_points(data):
    """
    从 VLM figure JSON 提取统一格式的点列表。
    返回: (x_axis_title, y_axis_title, [(series_name, X, Y_Left, Y_Right), ...])
    """
    if data is None:
        return "", "", []

    points = []

    # ── Claude 格式: {figures: [{series/points, ...}]} ──
    if "figures" in data:
        x_title = ""
        y_title = ""
        for fig in data["figures"]:
            fx = fig.get("x_axis_title", "")
            fy = fig.get("y_axis_title", "")
            if fx: x_title = fx
            if fy: y_title = fy
            fig_label = fig.get("label", fig.get("title", ""))

            # 普通 series 格式
            if "series" in fig:
                for s in fig["series"]:
                    name = s.get("name", s.get("label", "Unknown"))
                    for pt in s.get("points", []):
                        x  = _float(pt.get("X", pt.get("x")))
                        yl = _float(pt.get("Y_Left", pt.get("y", pt.get("Y"))))
                        yr = _float(pt.get("Y_Right", pt.get("y_right")))
                        if x is not None and yl is not None:
                            series_name = f"({fig_label}) {name}" if fig_label else name
                            points.append((series_name, x, yl, yr))

            # Heatmap 格式 (Claude): points 直接在 figure 下
            elif "points" in fig:
                name = fig_label or "heatmap"
                for pt in fig["points"]:
                    # 语义字段: t_R1, T1, yield
                    x  = _float(pt.get("t_R1", pt.get("X", pt.get("x"))))
                    yl = _float(pt.get("T1", pt.get("Y_Left", pt.get("y"))))
                    yr = _float(pt.get("yield", pt.get("Y_Right", pt.get("Z_Value"))))
                    if x is not None and yl is not None:
                        points.append((name, x, yl, yr))

        return x_title, y_title, points

    # ── Gemini 格式: {series: [{name, points}]} ──
    x_title = data.get("x_axis_title", "")
    y_title = data.get("y_axis_title", "")

    for s in data.get("series", []):
        name = s.get("name", s.get("label", "Unknown"))
        for pt in s.get("points", []):
            x  = _float(pt.get("X", pt.get("x")))
            yl = _float(pt.get("Y_Left", pt.get("y", pt.get("Y"))))
            yr = _float(pt.get("Y_Right", pt.get("y_right", pt.get("Z_Value"))))
            if x is not None and yl is not None:
                points.append((name, x, yl, yr))

    return x_title, y_title, points


# ─────────────────────────────────────────────────────────────
# Table: 提取单元格列表
# ─────────────────────────────────────────────────────────────

def _looks_like_smiles(s):
    """粗略判断字符串是否像 SMILES。"""
    if not s or len(s) < 4:
        return False
    smiles_chars = set("CNOSFPIBrcnos=#()[]@+\\/-12345678.%")
    ratio = sum(1 for c in s if c in smiles_chars) / len(s)
    return ratio > 0.7 and any(c in s for c in "=#()[]")


def extract_table_cells(data):
    """
    从 VLM table JSON 提取单元格列表。
    返回: (columns, [(row_idx, col_idx, col_name, value, cell_type), ...])
    """
    if data is None:
        return [], []

    columns = data.get("columns", [])
    rows = data.get("rows", data.get("data", []))

    cells = []
    for row_idx, row in enumerate(rows):
        if isinstance(row, list):
            for col_idx, val in enumerate(row):
                val_str = str(val).strip() if val is not None else ""
                if not val_str:
                    continue
                col_name = columns[col_idx] if col_idx < len(columns) else f"col_{col_idx}"
                cell_type = "smiles" if _looks_like_smiles(val_str) else "text"
                cells.append((row_idx + 1, col_idx, col_name, val_str, cell_type))
        elif isinstance(row, dict):
            for col_idx, col_name in enumerate(columns):
                val_str = str(row.get(col_name, "")).strip()
                if not val_str:
                    continue
                cell_type = "smiles" if _looks_like_smiles(val_str) else "text"
                cells.append((row_idx + 1, col_idx, col_name, val_str, cell_type))

    return columns, cells


# ─────────────────────────────────────────────────────────────
# 生成 Figure 标注 CSV
# ─────────────────────────────────────────────────────────────

def generate_figure_csv(model_name, suffix, out_path):
    """
    model_name: 'gemini' 或 'claude'
    suffix: '' (gemini) 或 '_claude' (claude)
    """
    fieldnames = [
        "label", "model", "figure_id", "x_axis_title", "y_axis_title",
        "series", "pred_X", "pred_Y_Left", "pred_Y_Right",
        "point_index", "gt_X", "gt_Y_Left", "is_match_within_tol",
        "ref_image", "description",
    ]

    all_rows = []
    for label in FIGURE_LABELS:
        json_path = os.path.join(FIG_DIR, f"{label}{suffix}.json")
        data = load_json(json_path)
        if data is None:
            continue

        x_title, y_title, points = extract_figure_points(data)
        ref_image = f"eval/images/figures/{label}.png"

        for idx, (series, x, yl, yr) in enumerate(points):
            all_rows.append({
                "label":       label,
                "model":       model_name,
                "figure_id":   label,
                "x_axis_title": x_title,
                "y_axis_title": y_title,
                "series":      series,
                "pred_X":      round(x, 4) if x is not None else "",
                "pred_Y_Left": round(yl, 4) if yl is not None else "",
                "pred_Y_Right": round(yr, 4) if yr is not None else "",
                "point_index": idx,
                "gt_X":        "",
                "gt_Y_Left":   "",
                "is_match_within_tol": "",
                "ref_image":   ref_image,
                "description": "",
            })

    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"  → {out_path}  ({len(all_rows)} 行)")
    return len(all_rows)


# ─────────────────────────────────────────────────────────────
# 生成 Table 标注 CSV
# ─────────────────────────────────────────────────────────────

def generate_table_csv(model_name, suffix, out_path):
    fieldnames = [
        "label", "model", "table_id", "data_row", "col_idx", "col_name",
        "cell_type", "predicted_value", "gt_value", "is_correct",
        "ref_image", "description",
    ]

    all_rows = []
    for label in TABLE_LABELS:
        json_path = os.path.join(TAB_DIR, f"{label}{suffix}.json")
        data = load_json(json_path)
        if data is None:
            continue

        columns, cells = extract_table_cells(data)
        ref_image = f"eval/images/tables/{label}.png"

        for row_idx, col_idx, col_name, val, cell_type in cells:
            all_rows.append({
                "label":           label,
                "model":           model_name,
                "table_id":        label,
                "data_row":        row_idx,
                "col_idx":         col_idx,
                "col_name":        col_name,
                "cell_type":       cell_type,
                "predicted_value": val,
                "gt_value":        "",
                "is_correct":      "",
                "ref_image":       ref_image,
                "description":     "",
            })

    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"  → {out_path}  ({len(all_rows)} 行)")
    return len(all_rows)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("生成 VLM Figure 标注模板...")
    n1 = generate_figure_csv("gemini", "",        os.path.join(EVAL_DIR, "vlm_figure_annotation_gemini.csv"))
    n2 = generate_figure_csv("claude", "_claude",  os.path.join(EVAL_DIR, "vlm_figure_annotation_claude.csv"))

    print("\n生成 VLM Table 标注模板...")
    n3 = generate_table_csv("gemini", "",          os.path.join(EVAL_DIR, "vlm_table_annotation_gemini.csv"))
    n4 = generate_table_csv("claude", "_claude",    os.path.join(EVAL_DIR, "vlm_table_annotation_claude.csv"))

    print(f"\n完成！共生成 {n1 + n2 + n3 + n4} 行标注数据。")
    print("\n标注方式：")
    print("  Figure: 对照 ref_image 中的原图，在 is_match_within_tol 填 1(正确) 或 0(不正确)")
    print("  Table:  对照 ref_image 中的原图，在 is_correct 填 1(正确) 或 0(不正确)")
