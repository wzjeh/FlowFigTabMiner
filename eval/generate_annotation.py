"""
eval/generate_annotation.py
============================
生成三份人工标注模板 CSV。完全独立，不依赖 src/ 任何代码。

使用方式（从项目根目录运行）：
    source flowfigtabminer/bin/activate
    python eval/generate_annotation.py

输出（都在 eval/ 目录下）：
    eval/layer1_figure_annotation.csv   — Layer1: 图坐标提取（中间体）
    eval/layer1_table_annotation.csv    — Layer1: 表格 cell 提取（中间体）
    eval/layer2_final_annotation.csv    — Layer2: 最终 JSON 组装字段

标注说明见 eval/README.md
"""

import os
import sys
import json
import csv
import re

# 脚本运行时需确保工作目录是项目根目录
SAMPLES_FILE = os.path.join(os.path.dirname(__file__), "samples.json")
OUT_DIR = os.path.dirname(__file__)

# Layer1 table: 每张表最多取前 N 个数据行（跳过 header row）
MAX_DATA_ROWS_PER_TABLE = 10


def load_samples():
    with open(SAMPLES_FILE, encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────

def _looks_like_smiles(text: str) -> bool:
    if not text or len(text) < 4:
        return False
    return bool(re.search(r"[=#@\[\(]|[A-Z][a-z]?\(|c1|C1|>>", text))


def _str(v):
    """将 None/float/str 统一转为字符串。"""
    if v is None:
        return ""
    return str(v)


# ─────────────────────────────────────────────────────────────
# Layer 1 — Figure
# ─────────────────────────────────────────────────────────────

def generate_layer1_figure(samples, out_path):
    rows = []
    for s in samples["figures"]:
        path = s["evidence_json"]
        if not os.path.exists(path):
            print(f"  [跳过] 文件不存在: {path}")
            continue

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        text_ev = data.get("text_evidence", {})
        x_title = (text_ev.get("x_axis_title") or [{}])[0].get("text", "")
        y_title = (text_ev.get("y_axis_title") or [{}])[0].get("text", "")
        raw_data = data.get("raw_data", [])

        for i, pt in enumerate(raw_data):
            rows.append({
                "label":            s["label"],
                "paper_id":         s["paper_id"],
                "figure_id":        data.get("meta", {}).get("figure_id", ""),
                "x_axis_title":     x_title,
                "y_axis_title":     y_title,
                "series":           pt.get("Series", ""),
                "pred_X":           _str(pt.get("X")),
                "pred_Y_Left":      _str(pt.get("Y_Left")),
                "pred_Y_Right":     _str(pt.get("Y_Right/Data_Value")),
                "point_index":      i,
                # ── Zhao 填写 ──────────────────────────────────
                # 方式 A（精确）：填入从原图读出的真实 X / Y
                "gt_X":             "",
                "gt_Y_Left":        "",
                # 方式 B（快速）：直接判断这个点是否正确
                # 1 = 提取正确（X/Y 在 ±5% 容差内）; 0 = 错误
                "is_match_within_tol": "",
                # ────────────────────────────────────────────────
                "ref_image":        s.get("crop_image", ""),
                "description":      s["description"],
            })

    fieldnames = [
        "label", "paper_id", "figure_id", "x_axis_title", "y_axis_title",
        "series", "pred_X", "pred_Y_Left", "pred_Y_Right", "point_index",
        "gt_X", "gt_Y_Left", "is_match_within_tol",
        "ref_image", "description",
    ]
    _write_csv(out_path, fieldnames, rows)
    return len(rows)


# ─────────────────────────────────────────────────────────────
# Layer 1 — Table
# ─────────────────────────────────────────────────────────────

def generate_layer1_table(samples, out_path):
    rows = []
    for s in samples["tables"]:
        path = s["extracted_csv"]
        if not os.path.exists(path):
            print(f"  [跳过] 文件不存在: {path}")
            continue

        with open(path, newline="", encoding="utf-8", errors="replace") as f:
            all_rows = list(csv.reader(f))

        if not all_rows:
            continue

        header = all_rows[0]
        # 只取数据行（跳过 row 0 header），最多 MAX_DATA_ROWS_PER_TABLE 行
        data_rows = all_rows[1: 1 + MAX_DATA_ROWS_PER_TABLE]

        for row_idx, row in enumerate(data_rows, start=1):
            for col_idx, cell in enumerate(row):
                cell = cell.strip()
                if not cell:       # 跳过空 cell
                    continue
                col_name = header[col_idx].strip() if col_idx < len(header) else f"col_{col_idx}"
                cell_type = "smiles" if _looks_like_smiles(cell) else "text"

                rows.append({
                    "label":           s["label"],
                    "paper_id":        s["paper_id"],
                    "table_id":        os.path.basename(path).replace("_extracted.csv", ""),
                    "data_row":        row_idx,
                    "col_idx":         col_idx,
                    "col_name":        col_name,
                    "cell_type":       cell_type,
                    "predicted_value": cell,
                    # ── Zhao 填写 ────────────────────────────────
                    # gt_value：从 PDF 原表读出的正确内容（可选，留空则只看 is_correct）
                    "gt_value":        "",
                    # is_correct: 1 = 提取正确; 0 = 错误/缺失
                    "is_correct":      "",
                    # ────────────────────────────────────────────
                    "ref_image":       s.get("table_image", ""),
                    "description":     s["description"],
                })

    fieldnames = [
        "label", "paper_id", "table_id", "data_row", "col_idx", "col_name",
        "cell_type", "predicted_value",
        "gt_value", "is_correct",
        "ref_image", "description",
    ]
    _write_csv(out_path, fieldnames, rows)
    return len(rows)


# ─────────────────────────────────────────────────────────────
# Layer 2 — Final assembled output
# ─────────────────────────────────────────────────────────────

# 只评测这些字段（排除 LLM 推断字段 reaction_class / reaction_smiles）
# 每项: (normalized 字段名, final.json 兼容字段名, 位置)
EVAL_FIELDS = [
    ("yield_pct",          "yield_pct",        "top"),
    ("conversion_pct",     "conversion_pct",   "top"),
    ("reactant1_smiles",   "reactant_smiles",  "top"),   # final.json 用 reactant_smiles
    ("product_smiles",     "product_smiles",   "top"),
    ("temperature_C",      "temperature_C",    "conditions"),
    ("residence_time_s",   "residence_time_s", "conditions"),
    ("solvent",            "solvent",          "conditions"),
    ("catalyst",           "catalyst",         "conditions"),
]


def _get_field(rec, field_normalized, field_final, location):
    """从 record 取字段值，自动兼容 normalized 和 final.json 两种 schema。"""
    cond = rec.get("conditions") or {}
    if location == "top":
        v = rec.get(field_normalized)
        if v is None:
            v = rec.get(field_final)   # fallback for _final.json schema
    else:
        v = cond.get(field_normalized)
        if v is None:
            v = cond.get(field_final)
    return _str(v)


def _find_intermediate(paper_id, source):
    """尽力找到 source 对应的 intermediate 文件路径。

    LLM 使用 1-indexed PDF 页码（p.4 = PDF 第4页），
    而中间文件使用 0-indexed 页码（page_3 = 第4页）。
    因此搜索时同时尝试 page_{n} 和 page_{n-1}。
    """
    m = re.search(r'p\.(\d+)', source)
    if not m:
        return ""
    pdf_page = int(m.group(1))          # 1-indexed PDF 页码
    candidates = [pdf_page - 1, pdf_page]  # 0-indexed 优先，也试 1-indexed
    base = f"data/intermediate/{paper_id}"
    if "Table" in source or "table" in source:
        d = os.path.join(base, "tables")
        if os.path.exists(d):
            for pg in candidates:
                files = sorted(f for f in os.listdir(d)
                               if f.startswith(f"page_{pg}_table") and f.endswith("_extracted.csv"))
                if files:
                    return os.path.join(d, files[0])
    elif "Figure" in source or "figure" in source:
        d = os.path.join(base, "macro_cleaned")
        if os.path.exists(d):
            for pg in candidates:
                files = sorted(f for f in os.listdir(d)
                               if f.startswith(f"page_{pg}_figure") and f.endswith("_evidence.json"))
                if files:
                    return os.path.join(d, files[0])
    # source 本身就是文件名的情况
    direct = os.path.join(base, "tables", source)
    if os.path.exists(direct):
        return direct
    return ""


def generate_layer2_final(samples, out_path):
    rows = []
    for s in samples["final_output"]:
        path = s["normalized_json"]
        if not os.path.exists(path):
            path = path.replace("_normalized.json", "_final.json")
        if not os.path.exists(path):
            print(f"  [跳过] 文件不存在: {path}")
            continue

        with open(path, encoding="utf-8") as f:
            records = json.load(f)

        if not isinstance(records, list):
            print(f"  [跳过] 非列表格式: {path}")
            continue

        max_n = s.get("max_records", 8)
        for rec_idx, rec in enumerate(records[:max_n]):
            source = rec.get("source_table_or_figure", "")
            inter  = _find_intermediate(s["paper_id"], source)

            for field_norm, field_fin, location in EVAL_FIELDS:
                pred_val = _get_field(rec, field_norm, field_fin, location)

                # 跳过预测值为空的字段（LLM 没有提取到，无法评测）
                if not pred_val:
                    continue

                rows.append({
                    "paper_id":         s["paper_id"],
                    "record_idx":       rec_idx,
                    "source":           source,
                    "intermediate_file": inter,
                    "field":            field_norm,
                    "predicted_value":  pred_val,
                    # ── Zhao 填写 ────────────────────────────────
                    # gt_value：从 PDF 原文读出的真实值（可选）
                    "gt_value":         "",
                    # is_correct: 1 = 正确; 0 = 错误或幻觉
                    # 数值字段（yield/temp/rt）允许 ±5% tolerance
                    # SMILES 字段按 canonical SMILES 比对
                    "is_correct":     "",
                    # ────────────────────────────────────────────
                })

    fieldnames = [
        "paper_id", "record_idx", "source", "intermediate_file",
        "field", "predicted_value",
        "gt_value", "is_correct",
    ]
    _write_csv(out_path, fieldnames, rows)
    return len(rows)


# ─────────────────────────────────────────────────────────────
# CSV 写入工具
# ─────────────────────────────────────────────────────────────

def _write_csv(path, fieldnames, rows):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    samples = load_samples()

    fig_out = os.path.join(OUT_DIR, "layer1_figure_annotation.csv")
    tab_out = os.path.join(OUT_DIR, "layer1_table_annotation.csv")
    fin_out = os.path.join(OUT_DIR, "layer2_final_annotation.csv")

    print("生成 Layer1 Figure 模板...")
    n = generate_layer1_figure(samples, fig_out)
    print(f"  → {fig_out}  ({n} 行 predicted points)")

    print("生成 Layer1 Table 模板...")
    n = generate_layer1_table(samples, tab_out)
    print(f"  → {tab_out}  ({n} 行 non-empty cells)")

    print("生成 Layer2 Final 模板...")
    n = generate_layer2_final(samples, fin_out)
    print(f"  → {fin_out}  ({n} 行 field values)")

    print("\n下一步：阅读 eval/README.md 了解如何填写标注。")
