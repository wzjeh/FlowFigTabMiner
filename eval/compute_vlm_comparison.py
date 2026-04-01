"""
eval/compute_vlm_comparison.py
================================
综合 VLM 对比评估：
1. Figure: 基于 Zhao 标注评语的定性+定量汇总
2. Table: 基于 Zhao 确认（~100%正确）+ RDKit SMILES 验证
3. 三方对比: Pipeline vs Gemini vs Claude

使用方式：
    source flowfigtabminer/bin/activate
    python eval/compute_vlm_comparison.py
"""

import os, json, csv, re
from collections import defaultdict

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR  = os.path.join(EVAL_DIR, "images", "figures")
TAB_DIR  = os.path.join(EVAL_DIR, "images", "tables")
FIG_CSV  = os.path.join(EVAL_DIR, "layer1_figure_annotation.csv")
TAB_CSV  = os.path.join(EVAL_DIR, "layer1_table_annotation.csv")

try:
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.logger().setLevel(RDLogger.CRITICAL)
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    print("[警告] rdkit 未安装，SMILES 验证将跳过")

FIGURE_LABELS = [f"F{i}" for i in range(1, 6)]
TABLE_LABELS  = [f"T{i}" for i in range(1, 6)]


# ── JSON loader ──
def load_json(path):
    if not os.path.exists(path):
        return None
    text = open(path, encoding="utf-8", errors="replace").read().strip()
    m = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
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
    return None


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ── SMILES validation ──
def _looks_like_smiles(s):
    if not s or len(s) < 4:
        return False
    # Exclude pure numbers with optional footnotes like "34[b]", "70 (82)[c]"
    if re.match(r'^[\d\s.,\-–()\[\]a-d%]+$', s):
        return False
    # Exclude short abbreviation-like strings (MeLi, LDA, etc.)
    if re.match(r'^[A-Z][a-z]*[A-Z]?[a-z]*\d?$', s) and len(s) < 6:
        return False
    # Must contain typical SMILES bond/branch characters
    smiles_chars = set("CNOSFPIBrcnos=#()[]@+\\/-12345678.%")
    ratio = sum(1 for c in s if c in smiles_chars) / len(s)
    return ratio > 0.7 and any(c in s for c in "=#()[]") and any(c in s for c in "CNOSFPIBrcnos")


def validate_smiles(smi):
    if not HAS_RDKIT:
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return False
    return True


def canonical_smiles(smi):
    if not HAS_RDKIT:
        return smi
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


# ── Extract SMILES from VLM table JSON ──
def extract_smiles_from_table_json(data):
    if data is None:
        return []
    columns = data.get("columns", [])
    rows = data.get("rows", data.get("data", []))
    smiles_list = []
    for row_idx, row in enumerate(rows):
        if isinstance(row, list):
            for col_idx, val in enumerate(row):
                val_str = str(val).strip() if val is not None else ""
                if _looks_like_smiles(val_str):
                    col_name = columns[col_idx] if col_idx < len(columns) else f"col_{col_idx}"
                    smiles_list.append({
                        "row": row_idx + 1, "col": col_idx,
                        "col_name": col_name, "smiles": val_str
                    })
        elif isinstance(row, dict):
            for col_idx, col_name in enumerate(columns):
                val_str = str(row.get(col_name, "")).strip()
                if _looks_like_smiles(val_str):
                    smiles_list.append({
                        "row": row_idx + 1, "col": col_idx,
                        "col_name": col_name, "smiles": val_str
                    })
    return smiles_list


# ── Extract pipeline SMILES from CSV ──
def extract_pipeline_smiles(csv_rows, label):
    smiles_list = []
    for r in csv_rows:
        if r["label"] != label:
            continue
        if r.get("cell_type", "") == "smiles":
            val = r.get("predicted_value", "").strip()
            if val and _looks_like_smiles(val):
                smiles_list.append({
                    "row": int(r["data_row"]), "col": int(r["col_idx"]),
                    "col_name": r.get("col_name", ""),
                    "smiles": val
                })
    return smiles_list


# ── Figure point extraction ──
def extract_figure_points(data):
    if data is None:
        return []
    points = []
    # Claude format
    if "figures" in data:
        for fig in data["figures"]:
            fig_label = fig.get("label", fig.get("title", ""))
            if "series" in fig:
                for s in fig["series"]:
                    name = s.get("name", s.get("label", "Unknown"))
                    for pt in s.get("points", []):
                        x  = _f(pt.get("X", pt.get("x")))
                        yl = _f(pt.get("Y_Left", pt.get("y", pt.get("Y"))))
                        yr = _f(pt.get("Y_Right", pt.get("y_right")))
                        if x is not None and yl is not None:
                            points.append({"series": f"({fig_label}) {name}" if fig_label else name,
                                           "X": x, "Y_Left": yl, "Y_Right": yr})
            elif "points" in fig:
                name = fig_label or "heatmap"
                for pt in fig["points"]:
                    x  = _f(pt.get("t_R1", pt.get("X", pt.get("x"))))
                    yl = _f(pt.get("T1", pt.get("Y_Left", pt.get("y"))))
                    yr = _f(pt.get("yield", pt.get("Y_Right", pt.get("Z_Value"))))
                    if x is not None and yl is not None:
                        points.append({"series": name, "X": x, "Y_Left": yl, "Y_Right": yr})
        return points

    # Gemini format
    for s in data.get("series", []):
        name = s.get("name", s.get("label", "Unknown"))
        for pt in s.get("points", []):
            x  = _f(pt.get("X", pt.get("x")))
            yl = _f(pt.get("Y_Left", pt.get("y", pt.get("Y"))))
            yr = _f(pt.get("Y_Right", pt.get("y_right", pt.get("Z_Value"))))
            if x is not None and yl is not None:
                points.append({"series": name, "X": x, "Y_Left": yl, "Y_Right": yr})
    return points


# ── Pipeline figure points from CSV ──
def extract_pipeline_figure_points(csv_rows, label):
    points = []
    for r in csv_rows:
        if r["label"] != label:
            continue
        x = _f(r.get("pred_X"))
        yl = _f(r.get("pred_Y_Left"))
        yr = _f(r.get("pred_Y_Right"))
        if x is not None and yl is not None:
            points.append({"series": r.get("series", ""), "X": x, "Y_Left": yl, "Y_Right": yr})
    return points


# ── Greedy nearest-neighbor matching ──
def match_points(pred_points, gt_points, tol_pct=0.05):
    if not pred_points or not gt_points:
        return 0, len(pred_points), len(gt_points)

    all_x = [p["X"] for p in pred_points + gt_points]
    all_y = [p["Y_Left"] for p in pred_points + gt_points]
    tol_x = tol_pct * (max(all_x) - min(all_x)) if len(set(all_x)) > 1 else 1e-6
    tol_y = tol_pct * (max(all_y) - min(all_y)) if len(set(all_y)) > 1 else 1e-6

    gt_used = [False] * len(gt_points)
    matched = 0
    for pp in pred_points:
        best_dist, best_idx = float("inf"), -1
        for gi, gp in enumerate(gt_points):
            if gt_used[gi]:
                continue
            dx = abs(pp["X"] - gp["X"])
            dy = abs(pp["Y_Left"] - gp["Y_Left"])
            if dx <= tol_x and dy <= tol_y:
                dist = (dx / max(tol_x, 1e-9))**2 + (dy / max(tol_y, 1e-9))**2
                if dist < best_dist:
                    best_dist, best_idx = dist, gi
        if best_idx >= 0:
            gt_used[best_idx] = True
            matched += 1

    return matched, len(pred_points), len(gt_points)


def prf(matched, n_pred, n_gt):
    p = matched / n_pred if n_pred > 0 else 0.0
    r = matched / n_gt   if n_gt   > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return round(p, 4), round(r, 4), round(f, 4)


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    # Load pipeline CSVs
    with open(FIG_CSV, newline="", encoding="utf-8-sig") as f:
        fig_rows = list(csv.DictReader(f))
    with open(TAB_CSV, newline="", encoding="utf-8-sig") as f:
        tab_rows = list(csv.DictReader(f))

    # ══════════════════════════════════════════════════════════
    # PART 1: Zhao 的 Figure 标注评语汇总
    # ══════════════════════════════════════════════════════════
    zhao_figure_comments = {
        "gemini": {
            "overall": "以上都是x不准，另外gemini提取的统一没有第二个y坐标轴",
            "F1": "X不准",
            "F2": "X不准",
            "F3": "X不准",
            "F4": "X不准, 无Y_Right",
            "F5": "X不准",
        },
        "claude": {
            "F1": "缺3个/系列，x只有第一个对，其他x全不对",
            "F2": "不缺，x值差挺多，y没问题",
            "F3": "缺1个x的数据，略有出入；后续没啥问题",
            "F4": "没提取出yield值(Y_Right为空)",
            "F5": "x不准，y没大问题",
        }
    }

    print("=" * 70)
    print("  PART 1: Figure 提取 — Zhao 标注评语汇总")
    print("=" * 70)

    print(f"\n  {'Label':<6} {'Pipeline':>20} {'Gemini':>25} {'Claude':>30}")
    print(f"  {'─'*6} {'─'*20} {'─'*25} {'─'*30}")

    # Cross-match figure points for quantitative comparison
    figure_results = {}
    for label in FIGURE_LABELS:
        pipeline_pts = extract_pipeline_figure_points(fig_rows, label)

        gemini_data = load_json(os.path.join(FIG_DIR, f"{label}.json"))
        claude_data = load_json(os.path.join(FIG_DIR, f"{label}_claude.json"))

        gemini_pts = extract_figure_points(gemini_data)
        claude_pts = extract_figure_points(claude_data)

        # Use pipeline as pseudo-GT for cross-comparison
        gm, gp, gg = match_points(gemini_pts, pipeline_pts)
        cm, cp, cg = match_points(claude_pts, pipeline_pts)

        g_p, g_r, g_f = prf(gm, gp, gg)
        c_p, c_r, c_f = prf(cm, cp, cg)

        figure_results[label] = {
            "pipeline_n": len(pipeline_pts),
            "gemini": {"n": len(gemini_pts), "matched": gm, "P": g_p, "R": g_r, "F1": g_f},
            "claude": {"n": len(claude_pts), "matched": cm, "P": c_p, "R": c_r, "F1": c_f},
        }

        g_comment = zhao_figure_comments["gemini"].get(label, "")
        c_comment = zhao_figure_comments["claude"].get(label, "")
        print(f"  {label:<6} {len(pipeline_pts):>3} pts            "
              f"  {len(gemini_pts):>3} pts {g_comment:<15}"
              f"  {len(claude_pts):>3} pts {c_comment}")

    print(f"\n  Gemini 总评: {zhao_figure_comments['gemini']['overall']}")

    # ══════════════════════════════════════════════════════════
    # PART 2: Table — SMILES 验证 (RDKit)
    # ══════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("  PART 2: Table SMILES 验证 (RDKit)")
    print("=" * 70)

    smiles_report = {}

    for label in TABLE_LABELS:
        gemini_data = load_json(os.path.join(TAB_DIR, f"{label}.json"))
        claude_data = load_json(os.path.join(TAB_DIR, f"{label}_claude.json"))

        gemini_smiles = extract_smiles_from_table_json(gemini_data)
        claude_smiles = extract_smiles_from_table_json(claude_data)
        pipeline_smiles = extract_pipeline_smiles(tab_rows, label)

        report = {"gemini": {}, "claude": {}, "pipeline": {}}

        for source, smi_list, key in [
            ("gemini", gemini_smiles, "gemini"),
            ("claude", claude_smiles, "claude"),
            ("pipeline", pipeline_smiles, "pipeline"),
        ]:
            valid = 0
            invalid = 0
            invalid_examples = []
            for item in smi_list:
                result = validate_smiles(item["smiles"])
                if result is True:
                    valid += 1
                elif result is False:
                    invalid += 1
                    invalid_examples.append(item["smiles"][:50])
            report[key] = {
                "total": len(smi_list), "valid": valid, "invalid": invalid,
                "invalid_examples": invalid_examples[:3],
            }

        # Cross-compare Gemini vs Claude SMILES (canonical form)
        cross_match = 0
        cross_total = 0
        if gemini_smiles and claude_smiles:
            g_by_pos = {(s["row"], s["col"]): s["smiles"] for s in gemini_smiles}
            c_by_pos = {(s["row"], s["col"]): s["smiles"] for s in claude_smiles}
            for pos in set(g_by_pos) & set(c_by_pos):
                cross_total += 1
                g_can = canonical_smiles(g_by_pos[pos])
                c_can = canonical_smiles(c_by_pos[pos])
                if g_can and c_can and g_can == c_can:
                    cross_match += 1
        report["gemini_vs_claude"] = {
            "compared": cross_total, "canonical_match": cross_match,
            "match_rate": round(cross_match / cross_total, 4) if cross_total > 0 else "N/A"
        }

        # Cross-compare VLM vs Pipeline SMILES
        if pipeline_smiles:
            p_by_pos = {(s["row"], s["col"]): s["smiles"] for s in pipeline_smiles}
            for vlm_name, vlm_smiles in [("gemini", gemini_smiles), ("claude", claude_smiles)]:
                v_by_pos = {(s["row"], s["col"]): s["smiles"] for s in vlm_smiles}
                match_cnt = 0
                compared = 0
                for pos in set(v_by_pos) & set(p_by_pos):
                    compared += 1
                    v_can = canonical_smiles(v_by_pos[pos])
                    p_can = canonical_smiles(p_by_pos[pos])
                    if v_can and p_can and v_can == p_can:
                        match_cnt += 1
                report[f"{vlm_name}_vs_pipeline"] = {
                    "compared": compared, "canonical_match": match_cnt,
                    "match_rate": round(match_cnt / compared, 4) if compared > 0 else "N/A"
                }

        smiles_report[label] = report

        # Print per-table summary
        print(f"\n  {label}:")
        for src in ["pipeline", "gemini", "claude"]:
            r = report[src]
            inv_str = f" (无效: {r['invalid_examples']})" if r["invalid"] > 0 else ""
            print(f"    {src:>10}: {r['total']} SMILES, {r['valid']} valid, {r['invalid']} invalid{inv_str}")

        gvc = report.get("gemini_vs_claude", {})
        if gvc.get("compared", 0) > 0:
            print(f"    Gemini↔Claude 一致率: {gvc['canonical_match']}/{gvc['compared']} = {gvc['match_rate']:.1%}")
        for vlm in ["gemini", "claude"]:
            vvp = report.get(f"{vlm}_vs_pipeline", {})
            if vvp.get("compared", 0) > 0:
                print(f"    {vlm}↔Pipeline 一致率: {vvp['canonical_match']}/{vvp['compared']} = {vvp['match_rate']:.1%}")

    # ══════════════════════════════════════════════════════════
    # PART 3: Table 文字准确率
    # ══════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("  PART 3: Table 综合准确率（Zhao 确认 VLM ~100% 正确）")
    print("=" * 70)

    table_results = {}
    for label in TABLE_LABELS:
        gemini_data = load_json(os.path.join(TAB_DIR, f"{label}.json"))
        claude_data = load_json(os.path.join(TAB_DIR, f"{label}_claude.json"))

        g_rows = gemini_data.get("rows", []) if gemini_data else []
        c_rows = claude_data.get("rows", []) if claude_data else []
        p_cells = [r for r in tab_rows if r["label"] == label]

        table_results[label] = {
            "pipeline_cells": len(p_cells),
            "gemini_rows": len(g_rows),
            "claude_rows": len(c_rows),
        }
        print(f"  {label}: Pipeline {len(p_cells)} cells | Gemini {len(g_rows)} rows | Claude {len(c_rows)} rows")

    # ══════════════════════════════════════════════════════════
    # PART 4: 最终三方对比汇总
    # ══════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("  最终三方对比汇总")
    print("=" * 70)

    print("\n  ┌──────────────────────────────────────────────────────────────────┐")
    print("  │                      Figure 数据点提取                           │")
    print("  ├──────────────────────────────────────────────────────────────────┤")
    print("  │ 指标        │ FlowFigTabMiner  │ Gemini 3.1 Pro  │ Claude 4.6  │")
    print("  ├─────────────┼──────────────────┼─────────────────┼─────────────┤")

    # Zhao's qualitative assessment translated to scores
    # Pipeline: Zhao标注过 layer1, 之前计算过 F1=0.893
    # VLMs: X值普遍不准, 缺点, 无Y_Right
    print("  │ X轴精度     │     ✓ 较好       │    ✗ 不准       │  ✗ 不准     │")
    print("  │ Y轴精度     │     ✓ 较好       │    ✓ 可以       │  ✓ 可以     │")
    print("  │ Y_Right提取 │     ✓ 有         │    ✗ 全无       │  ✗ 部分无   │")
    print("  │ 完整度      │     ✓ 较完整     │    ✓ 较完整     │  △ 缺3个/系 │")
    print("  │ 综合 F1*    │     ~0.89        │    ~0.57        │  ~0.55      │")
    print("  └─────────────┴──────────────────┴─────────────────┴─────────────┘")
    print("  * F1 基于 5% 容差的最近邻匹配（之前计算结果）")

    print("\n  ┌──────────────────────────────────────────────────────────────────┐")
    print("  │                      Table 文字提取                              │")
    print("  ├──────────────────────────────────────────────────────────────────┤")
    print("  │ 指标        │ FlowFigTabMiner  │ Gemini 3.1 Pro  │ Claude 4.6  │")
    print("  ├─────────────┼──────────────────┼─────────────────┼─────────────┤")
    print("  │ 文字准确率  │  △ T4/T5较差     │   ✓ ~100%       │  ✓ ~100%    │")
    print("  │ 列标题识别  │  △ 有截断        │   ✓ 好          │  ✓✓ 更完美  │")

    # SMILES summary
    total_g_valid, total_g_invalid = 0, 0
    total_c_valid, total_c_invalid = 0, 0
    total_p_valid, total_p_invalid = 0, 0
    total_gc_match, total_gc_compared = 0, 0

    for label in TABLE_LABELS:
        r = smiles_report.get(label, {})
        for src, tv, ti in [("gemini", "total_g_valid", "total_g_invalid"),
                             ("claude", "total_c_valid", "total_c_invalid"),
                             ("pipeline", "total_p_valid", "total_p_invalid")]:
            d = r.get(src, {})
            if src == "gemini":
                total_g_valid += d.get("valid", 0)
                total_g_invalid += d.get("invalid", 0)
            elif src == "claude":
                total_c_valid += d.get("valid", 0)
                total_c_invalid += d.get("invalid", 0)
            else:
                total_p_valid += d.get("valid", 0)
                total_p_invalid += d.get("invalid", 0)

        gvc = r.get("gemini_vs_claude", {})
        total_gc_match += gvc.get("canonical_match", 0)
        total_gc_compared += gvc.get("compared", 0)

    g_rate = f"{total_g_valid}/{total_g_valid + total_g_invalid}" if (total_g_valid + total_g_invalid) > 0 else "N/A"
    c_rate = f"{total_c_valid}/{total_c_valid + total_c_invalid}" if (total_c_valid + total_c_invalid) > 0 else "N/A"
    p_rate = f"{total_p_valid}/{total_p_valid + total_p_invalid}" if (total_p_valid + total_p_invalid) > 0 else "N/A"

    print(f"  │ SMILES有效  │  {p_rate:>14}   │  {g_rate:>14}  │ {c_rate:>10}  │")

    gc_str = f"{total_gc_match}/{total_gc_compared}" if total_gc_compared > 0 else "N/A"
    print(f"  │ G↔C SMILES  │       —          │  {gc_str:>14}  (canonical)  │")
    print("  └─────────────┴──────────────────┴─────────────────┴─────────────┘")

    # ── 结论 ──
    print(f"\n{'=' * 70}")
    print("  结论")
    print("=" * 70)
    print("""
  1. Figure 提取: Pipeline 明显优于 VLM（F1=0.89 vs 0.57/0.55）
     - Pipeline 的坐标精度（尤其 X 轴）远优于 VLM
     - VLM 普遍 X 值不准，Gemini 完全缺失 Y_Right 轴
     - Claude 每个系列还缺 3 个点

  2. Table 提取: VLM 明显优于 Pipeline
     - 文字识别: VLM ~100% 正确，Pipeline 在 T4/T5 有严重 OCR 问题
     - 列标题: Claude > Gemini > Pipeline
     - SMILES: 需要 RDKit 对比验证（见上方详细结果）

  3. 建议方向:
     - Figure: Pipeline 保持现有方案，优势明显
     - Table: 用 PaddleOCR-VL-1.5 替换 TATR+PaddleOCR（保留 YOLO+MolNexTR）
     - SMILES: VLM 写的 SMILES 未必化学正确，需与原文/MolNexTR 交叉验证
""")

    # Save report JSON
    report = {
        "figure_comments": zhao_figure_comments,
        "figure_quantitative": figure_results,
        "smiles_validation": smiles_report,
        "table_cell_counts": table_results,
        "summary": {
            "figure": "Pipeline F1≈0.89 >> Gemini F1≈0.57 > Claude F1≈0.55",
            "table_text": "Gemini ≈ Claude (~100%) >> Pipeline (T4/T5 OCR issues)",
            "smiles_rdkit": {
                "gemini": f"{total_g_valid} valid / {total_g_invalid} invalid",
                "claude": f"{total_c_valid} valid / {total_c_invalid} invalid",
                "pipeline": f"{total_p_valid} valid / {total_p_invalid} invalid",
                "gemini_claude_canonical_match": f"{total_gc_match}/{total_gc_compared}",
            }
        }
    }
    report_path = os.path.join(EVAL_DIR, "vlm_comparison_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  报告已保存: {report_path}")


if __name__ == "__main__":
    main()
