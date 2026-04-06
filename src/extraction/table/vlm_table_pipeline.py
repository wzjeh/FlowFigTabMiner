"""
VLM-based Table Pipeline — Phase 1 (Semi-automated)

Workflow:
  1. prepare_vlm_batch()  → 为每张表生成 YOLO 分子检测 + mol_meta.json + 待识别图片
  2. (手动)                → Zhao 将图片送入 VLM (Gemini/Claude) 客户端，保存 JSON 输出
  3. run_vlm_merge()      → 解析 VLM JSON + 回填 MolNexTR SMILES → CSV + evidence JSON
"""

import os
import sys
import json
import glob
import shutil
import logging
import cv2
import numpy as np
import pandas as pd
from rdkit import Chem

sys.path.insert(0, os.getcwd())

from src.utils.config import load_config
from src.parsing.table_filter import TableFilter
from src.extraction.common.molecule_processor import MoleculeProcessor
from src.extraction.common.content_recognizer import ContentRecognizer

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# 1. prepare_vlm_batch
# ──────────────────────────────────────────────────────────────

def prepare_vlm_batch(intermediate_dir, prompt_path="eval/vlm_prompts/table_prompt.txt"):
    """
    对 intermediate_dir/tables/ 下的表格图片运行 YOLO 分子检测 + MolNexTR，
    输出 vlm_input/ 目录供手动送 VLM。

    Args:
        intermediate_dir: e.g. "data/intermediate/example1"
        prompt_path: VLM prompt 文件路径
    """
    tables_dir = os.path.join(intermediate_dir, "tables")
    if not os.path.exists(tables_dir):
        print(f"[VLM-Table] 未找到 tables 目录: {tables_dir}")
        return

    vlm_input_dir = os.path.join(intermediate_dir, "vlm_input")
    os.makedirs(vlm_input_dir, exist_ok=True)

    # 复制 prompt
    if os.path.exists(prompt_path):
        shutil.copy2(prompt_path, os.path.join(vlm_input_dir, "prompt.txt"))

    # 找所有表格图片 (排除 debug crops)
    table_imgs = glob.glob(os.path.join(tables_dir, "*.png"))
    table_imgs = [f for f in table_imgs if "_body" not in os.path.basename(f)
                  and "_crop" not in os.path.basename(f)
                  and "_debug" not in os.path.basename(f)
                  and "_masked" not in os.path.basename(f)]

    if not table_imgs:
        print(f"[VLM-Table] 未找到表格图片: {tables_dir}")
        return

    # 加载模型
    cfg = load_config()
    tables_cfg = cfg.get("tables", {})

    print("[VLM-Table] 加载 ContentRecognizer (PaddleOCR + MolNexTR)...")
    content_rec = ContentRecognizer()

    mol_det_cfg = tables_cfg.get("molecule_detection", {})
    mol_model_path = mol_det_cfg.get("model_path")
    mol_conf = mol_det_cfg.get("confidence_threshold", 0.25)
    print(f"[VLM-Table] 加载 MoleculeProcessor: {mol_model_path}")
    mol_processor = MoleculeProcessor(model_path=mol_model_path, conf_threshold=mol_conf)

    for img_path in sorted(table_imgs):
        table_name = os.path.splitext(os.path.basename(img_path))[0]
        print(f"\n[VLM-Table] 处理: {table_name}")

        # 复制原始表格图到 vlm_input (这就是要送 VLM 的图)
        dst_img = os.path.join(vlm_input_dir, f"{table_name}.png")
        shutil.copy2(img_path, dst_img)

        # YOLO 分子检测 + MolNexTR
        _, mol_meta = mol_processor.process_image(
            img_path, content_rec, mask_only=True
        )

        # 保存 mol_meta
        mol_meta_serializable = []
        for m in mol_meta:
            mol_meta_serializable.append({
                "box": m["box"],        # [x1, y1, x2, y2] 像素坐标
                "smiles": m.get("smiles", ""),
                "conf": m.get("conf", 0.0),
            })

        meta_path = os.path.join(vlm_input_dir, f"{table_name}_mol_meta.json")
        with open(meta_path, "w") as f:
            json.dump(mol_meta_serializable, f, indent=2)
        print(f"   → 检测到 {len(mol_meta_serializable)} 个分子，保存到 {meta_path}")

    print(f"\n[VLM-Table] ✓ 准备完成。输出目录: {vlm_input_dir}")
    print(f"   → 将 vlm_input/*.png 逐张送入 VLM 客户端")
    print(f"   → VLM 输出 JSON 保存到 vlm_output/ 目录（文件名与图片一致）")


# ──────────────────────────────────────────────────────────────
# 2. parse_vlm_output
# ──────────────────────────────────────────────────────────────

def parse_vlm_output(vlm_json_str):
    """
    解析 VLM 返回的 JSON 字符串 → pandas DataFrame。

    Expected format:
    {
      "columns": ["col1", "col2", ...],
      "rows": [["cell1", "cell2", ...], ...]
    }
    """
    data = json.loads(vlm_json_str)
    columns = data.get("columns", [])
    rows = data.get("rows", [])

    if not rows:
        return pd.DataFrame()

    # 确保所有行长度一致
    n_cols = len(columns) if columns else max(len(r) for r in rows)
    normalized_rows = []
    for row in rows:
        if len(row) < n_cols:
            row = row + [""] * (n_cols - len(row))
        elif len(row) > n_cols:
            row = row[:n_cols]
        normalized_rows.append(row)

    if columns and len(columns) == n_cols:
        df = pd.DataFrame(normalized_rows, columns=columns)
    else:
        df = pd.DataFrame(normalized_rows)

    return df


# ──────────────────────────────────────────────────────────────
# 3. merge_smiles — 核心回填逻辑
# ──────────────────────────────────────────────────────────────

def _is_structure_cell(val):
    """判断 cell 值是否为分子结构占位符或 VLM 尝试写的 SMILES。"""
    s = str(val).strip()
    if not s:
        return False
    # 明确的占位符
    if s.lower() in ("[structure]", "[mol]", "[化学结构]", "[分子结构]"):
        return True
    # VLM 尝试写的 SMILES: 含有典型 SMILES 字符且 >10 字符
    if len(s) > 10:
        smiles_chars = set("CNOSPFClBr=#()[]/@+-.0123456789cnos")
        ratio = sum(1 for c in s if c in smiles_chars) / len(s)
        if ratio > 0.7 and any(c in s for c in "=#()[]"):
            return True
    return False


def _cluster_by_y(mol_meta, n_rows, img_height):
    """
    按 Y 坐标间距聚类将分子分组（同行分子 Y 坐标非常接近）。
    然后将分组按顺序映射到 VLM DataFrame 中含分子的行。

    Returns:
        list of lists: row_groups[i] = 第 i 行的分子列表（按 X 排序）
    """
    if not mol_meta:
        return [[] for _ in range(n_rows)]

    # 按 Y 中心排序
    sorted_mols = sorted(mol_meta, key=lambda m: (m["box"][1] + m["box"][3]) / 2)

    # 间距聚类：同一行内分子 Y 差 < gap_threshold
    gap_threshold = 30  # 像素
    groups = []
    current_group = [sorted_mols[0]]

    for i in range(1, len(sorted_mols)):
        cy_prev = (current_group[-1]["box"][1] + current_group[-1]["box"][3]) / 2
        cy_curr = (sorted_mols[i]["box"][1] + sorted_mols[i]["box"][3]) / 2
        if cy_curr - cy_prev > gap_threshold:
            groups.append(current_group)
            current_group = [sorted_mols[i]]
        else:
            current_group.append(sorted_mols[i])
    groups.append(current_group)

    # 每组内按 X 排序
    for group in groups:
        group.sort(key=lambda m: (m["box"][0] + m["box"][2]) / 2)

    logger.info(f"[_cluster_by_y] {len(mol_meta)} mols → {len(groups)} spatial groups (table has {n_rows} rows)")

    # 映射到 DataFrame 行：groups 按 Y 顺序排列
    # 填入长度为 n_rows 的列表，多余 group 丢弃
    row_groups = [[] for _ in range(n_rows)]
    # groups 直接按索引对应 — 将在 merge_smiles 中与 structure_rows 对齐
    for i, group in enumerate(groups):
        if i < n_rows:
            row_groups[i] = group

    return row_groups


def merge_smiles(df, mol_meta, img_height):
    """
    将 MolNexTR 的 SMILES 回填到 VLM 解析的 DataFrame 中。

    Args:
        df: VLM 解析出的 DataFrame
        mol_meta: [{box:[x1,y1,x2,y2], smiles:"...", conf:float}, ...]
        img_height: 表格图片高度（像素）
    Returns:
        df: 回填后的 DataFrame
        stats: 回填统计 dict
    """
    if df.empty or not mol_meta:
        return df, {"replaced": 0, "kept_vlm": 0, "no_match": 0}

    n_rows = len(df)

    # Step 1: 识别分子列 + 含结构 cell 的行
    mol_col_indices = set()
    # structure_rows: [(row_idx, [col_indices])] — 仅含结构 cell 的行
    structure_rows = []
    for row_idx in range(n_rows):
        row_struct_cols = []
        for col_idx in range(len(df.columns)):
            if _is_structure_cell(df.iloc[row_idx, col_idx]):
                mol_col_indices.add(col_idx)
                row_struct_cols.append(col_idx)
        if row_struct_cols:
            structure_rows.append((row_idx, row_struct_cols))

    mol_col_indices = sorted(mol_col_indices)

    if not structure_rows:
        logger.info("[merge_smiles] 未检测到分子列，跳过回填")
        return df, {"replaced": 0, "kept_vlm": 0, "no_match": 0}

    logger.info(f"[merge_smiles] 分子列索引: {mol_col_indices}, "
                f"含结构行: {len(structure_rows)}/{n_rows}")

    # Step 2: 按 Y 坐标聚类 — 组数对齐 structure_rows（而非全部行）
    row_groups = _cluster_by_y(mol_meta, len(structure_rows), img_height)

    # Step 3: 逐行回填 — spatial_groups[i] 对应 structure_rows[i]
    # 策略: VLM valid SMILES 优先保留，MolNexTR 仅在 VLM invalid 时 fallback
    replaced = 0
    kept_vlm = 0
    no_match = 0

    def _rdkit_valid(smi):
        """检查 SMILES 是否可被 RDKit 解析为有效分子。"""
        if not smi or not isinstance(smi, str):
            return False
        try:
            return Chem.MolFromSmiles(smi) is not None
        except Exception:
            return False

    for group_idx, (row_idx, col_indices) in enumerate(structure_rows):
        # 该行检测到的分子（已按 X 排序）
        row_mols = row_groups[group_idx] if group_idx < len(row_groups) else []

        # 一一对应回填
        for i, col_idx in enumerate(col_indices):
            vlm_val = str(df.iloc[row_idx, col_idx]).strip()
            vlm_valid = _rdkit_valid(vlm_val)

            if i < len(row_mols):
                mol_smi = row_mols[i].get("smiles", "")
                mol_valid = _rdkit_valid(mol_smi)

                if vlm_valid:
                    # VLM 写了有效 SMILES → 保留 VLM
                    kept_vlm += 1
                elif mol_valid:
                    # VLM invalid, MolNexTR valid → 用 MolNexTR
                    df.iloc[row_idx, col_idx] = mol_smi
                    replaced += 1
                else:
                    # 两者都 invalid → 保留 VLM 原文
                    kept_vlm += 1
            else:
                # 无对应分子（YOLO 漏检），保留 VLM 原文
                no_match += 1

    stats = {"replaced": replaced, "kept_vlm": kept_vlm, "no_match": no_match}
    logger.info(f"[merge_smiles] 回填统计: {stats}")
    return df, stats


# ──────────────────────────────────────────────────────────────
# 4. run_vlm_merge — 整合流程
# ──────────────────────────────────────────────────────────────

def run_vlm_merge(intermediate_dir, vlm_output_dir=None):
    """
    读取 VLM 输出 JSON + mol_meta → 回填 SMILES → 生成 CSV + evidence JSON。

    Args:
        intermediate_dir: e.g. "data/intermediate/example1"
        vlm_output_dir: VLM JSON 输出目录。默认 intermediate_dir/vlm_output/
    """
    vlm_input_dir = os.path.join(intermediate_dir, "vlm_input")
    if vlm_output_dir is None:
        vlm_output_dir = os.path.join(intermediate_dir, "vlm_output")

    if not os.path.exists(vlm_output_dir):
        print(f"[VLM-Table] 未找到 VLM 输出目录: {vlm_output_dir}")
        print(f"   → 请先将 vlm_input/*.png 送入 VLM，保存 JSON 到 {vlm_output_dir}/")
        return

    tables_dir = os.path.join(intermediate_dir, "tables")

    # 找所有 VLM 输出 JSON
    vlm_jsons = glob.glob(os.path.join(vlm_output_dir, "*.json"))
    if not vlm_jsons:
        print(f"[VLM-Table] vlm_output 目录为空: {vlm_output_dir}")
        return

    # ContentRecognizer 懒加载（仅在需要 caption/note OCR 时加载）
    content_rec = None

    results = []

    for vlm_json_path in sorted(vlm_jsons):
        table_name = os.path.splitext(os.path.basename(vlm_json_path))[0]
        print(f"\n[VLM-Table] 合并: {table_name}")

        # 1. 解析 VLM JSON
        with open(vlm_json_path, "r") as f:
            vlm_json_str = f.read()

        try:
            df = parse_vlm_output(vlm_json_str)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"   ✗ JSON 解析失败: {e}")
            continue

        if df.empty:
            print(f"   ✗ VLM 输出为空")
            continue

        print(f"   → VLM 表格: {len(df)} 行 × {len(df.columns)} 列")

        # 2. 读取 mol_meta
        mol_meta_path = os.path.join(vlm_input_dir, f"{table_name}_mol_meta.json")
        mol_meta = []
        if os.path.exists(mol_meta_path):
            with open(mol_meta_path, "r") as f:
                mol_meta = json.load(f)
            print(f"   → mol_meta: {len(mol_meta)} 个分子")

        # 3. 获取图片高度（用于 Y 坐标映射）
        img_path = os.path.join(vlm_input_dir, f"{table_name}.png")
        if not os.path.exists(img_path):
            img_path = os.path.join(tables_dir, f"{table_name}.png")
        img = cv2.imread(img_path)
        img_height = img.shape[0] if img is not None else 1000

        # 4. SMILES 回填
        df, stats = merge_smiles(df, mol_meta, img_height)
        print(f"   → 回填: {stats['replaced']} 替换, {stats['kept_vlm']} 保留VLM, {stats['no_match']} 无匹配")

        # 5. 输出 CSV
        table_output_dir = os.path.join(tables_dir, table_name)
        os.makedirs(table_output_dir, exist_ok=True)

        csv_path = os.path.join(table_output_dir, f"{table_name}_vlm_extracted.csv")
        df.to_csv(csv_path, index=False)
        print(f"   → CSV: {csv_path}")

        # 6. Caption/Note OCR (复用已有逻辑, 懒加载 ContentRecognizer)
        context_data = {"caption": [], "table_note": []}
        cap_pattern = os.path.join(table_output_dir, f"{table_name}_table_caption_*.png")
        note_pattern = os.path.join(table_output_dir, f"{table_name}_table_note_*.png")
        cap_files = sorted(glob.glob(cap_pattern))
        note_files = sorted(glob.glob(note_pattern))

        if cap_files or note_files:
            if content_rec is None:
                print("[VLM-Table] 加载 ContentRecognizer (用于 caption/note OCR)...")
                content_rec = ContentRecognizer()

            for c_path in cap_files:
                c_img = cv2.imread(c_path)
                if c_img is not None:
                    c_img = cv2.resize(c_img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
                    pad = 50
                    c_img = cv2.copyMakeBorder(c_img, pad, pad, pad, pad,
                                               cv2.BORDER_CONSTANT, value=(255, 255, 255))
                    c_rgb = cv2.cvtColor(c_img, cv2.COLOR_BGR2RGB)
                    txt = content_rec._recognize_text(c_rgb)
                    if txt.strip():
                        context_data["caption"].append(txt)

            for n_path in note_files:
                n_img = cv2.imread(n_path)
                if n_img is not None:
                    n_img = cv2.resize(n_img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
                    pad = 50
                    n_img = cv2.copyMakeBorder(n_img, pad, pad, pad, pad,
                                               cv2.BORDER_CONSTANT, value=(255, 255, 255))
                    n_rgb = cv2.cvtColor(n_img, cv2.COLOR_BGR2RGB)
                    txt = content_rec._recognize_text(n_rgb)
                    if txt.strip():
                        context_data["table_note"].append(txt)

        # 7. Evidence JSON
        evidence = {
            "csv_path": csv_path,
            "num_extracted": int(df.size),
            "caption_text": " ".join(context_data["caption"]),
            "table_note_text": " ".join(context_data["table_note"]),
            "is_relevant": True,
            "vlm_merge_stats": stats,
        }
        json_path = os.path.join(table_output_dir, f"{table_name}_vlm_evidence.json")
        with open(json_path, "w") as f:
            json.dump(evidence, f, indent=2)
        print(f"   → Evidence: {json_path}")

        results.append({
            "table_name": table_name,
            "csv_path": csv_path,
            "shape": list(df.shape),
            "merge_stats": stats,
        })

    # 汇总
    print(f"\n[VLM-Table] ✓ 完成。共处理 {len(results)} 张表格")
    summary_path = os.path.join(vlm_output_dir, "_merge_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"   → 汇总: {summary_path}")
    return results


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VLM Table Pipeline (Phase 1)")
    sub = parser.add_subparsers(dest="command")

    # prepare
    p_prep = sub.add_parser("prepare", help="准备 VLM 输入（YOLO 分子检测）")
    p_prep.add_argument("intermediate_dir", help="e.g. data/intermediate/example1")
    p_prep.add_argument("--prompt", default="eval/vlm_prompts/table_prompt.txt")

    # merge
    p_merge = sub.add_parser("merge", help="合并 VLM 输出 + MolNexTR SMILES")
    p_merge.add_argument("intermediate_dir", help="e.g. data/intermediate/example1")
    p_merge.add_argument("--vlm-output", default=None, help="VLM JSON 目录（默认 intermediate_dir/vlm_output/）")

    args = parser.parse_args()

    if args.command == "prepare":
        prepare_vlm_batch(args.intermediate_dir, prompt_path=args.prompt)
    elif args.command == "merge":
        run_vlm_merge(args.intermediate_dir, vlm_output_dir=args.vlm_output)
    else:
        parser.print_help()
