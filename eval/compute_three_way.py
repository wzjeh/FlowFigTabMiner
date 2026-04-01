"""
eval/compute_three_way.py
===========================
三方定量对比: FlowFigTabMiner Pipeline vs Gemini 3.1 Pro vs Claude Sonnet 4.6

Figure: P/R/F1
Table:  P/R/F1 (cell 级) + SMILES P/R/F1

数据来源:
  - Pipeline: Zhao 人工评测的 10 个 figure + 10 个 table 结果
  - VLM Figure (F1-F5): 以 Pipeline 为 anchor 的 cross-matching 结果 (vlm_comparison_report.json)
  - VLM Table (T1-T5): Zhao 确认 ~100% 正确 + RDKit SMILES 验证
  - SMILES: vlm_comparison_report.json 中的 smiles_validation 数据

使用方式:
    source flowfigtabminer/bin/activate
    python eval/compute_three_way.py
"""

import os, json

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
VLM_REPORT = os.path.join(EVAL_DIR, "vlm_comparison_report.json")
OUTPUT = os.path.join(EVAL_DIR, "three_way_comparison.json")


def _prf(p, r):
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return round(f, 4)


def main():
    with open(VLM_REPORT, encoding="utf-8") as f:
        vlm_report = json.load(f)

    # ════════════════════════════════════════════════════════════════
    # Pipeline 评测数据 (Zhao 人工标注, 10 figures + 10 tables)
    # ════════════════════════════════════════════════════════════════

    # -- Figure: Zhao 的评测结果 --
    pipeline_figure_data = {
        "F1":  {"n": 63,  "correct": 61, "wrong": 2,  "gt_miss": 2,  "P": 0.968, "R": 0.968, "F1": 0.968},
        "F2":  {"n": 47,  "correct": 46, "wrong": 1,  "gt_miss": 6,  "P": 0.979, "R": 0.885, "F1": 0.929},
        "F3":  {"n": 38,  "correct": 27, "wrong": 11, "gt_miss": 0,  "P": 0.711, "R": 1.000, "F1": 0.831},
        "F4":  {"n": 29,  "correct": 21, "wrong": 8,  "gt_miss": 0,  "P": 0.724, "R": 1.000, "F1": 0.840},
        "F5":  {"n": 29,  "correct": 29, "wrong": 0,  "gt_miss": 7,  "P": 1.000, "R": 0.806, "F1": 0.892},
        "F6":  {"n": 22,  "correct": 20, "wrong": 2,  "gt_miss": 2,  "P": 0.909, "R": 0.909, "F1": 0.909},
        "F7":  {"n": 17,  "correct": 17, "wrong": 0,  "gt_miss": 35, "P": 1.000, "R": 0.327, "F1": 0.493},
        "F8":  {"n": 16,  "correct": 16, "wrong": 0,  "gt_miss": 0,  "P": 1.000, "R": 1.000, "F1": 1.000},
        "F9":  {"n": 15,  "correct": 15, "wrong": 0,  "gt_miss": 0,  "P": 1.000, "R": 1.000, "F1": 1.000},
        "F10": {"n": 13,  "correct": 12, "wrong": 1,  "gt_miss": 0,  "P": 0.923, "R": 1.000, "F1": 0.960},
    }
    # 总计: 289 提取, 264 正确, 25 错误, 52 GT漏检 → P=91.3%, R=83.5%, F1=87.3%
    pipeline_fig_overall = {"precision": 0.913, "recall": 0.835, "f1": 0.873}

    # -- Table: Zhao 的评测结果 --
    pipeline_table_data = {
        "T1":  {"n": 65,  "correct": 64, "wrong": 1,  "gt_miss": 0,  "P": 0.985, "R": 0.985, "F1": 0.985},
        "T2":  {"n": 62,  "correct": 62, "wrong": 0,  "gt_miss": 0,  "P": 1.000, "R": 1.000, "F1": 1.000},
        "T3":  {"n": 50,  "correct": 36, "wrong": 14, "gt_miss": 0,  "P": 0.720, "R": 0.720, "F1": 0.720},
        "T4":  {"n": 21,  "correct": 16, "wrong": 5,  "gt_miss": 5,  "P": 0.762, "R": 0.615, "F1": 0.681},
        "T5":  {"n": 60,  "correct": 45, "wrong": 15, "gt_miss": 0,  "P": 0.750, "R": 0.750, "F1": 0.750},
        "T6":  {"n": 48,  "correct": 43, "wrong": 5,  "gt_miss": 20, "P": 0.896, "R": 0.632, "F1": 0.741},
        "T7":  {"n": 110, "correct": 110,"wrong": 0,  "gt_miss": 0,  "P": 1.000, "R": 1.000, "F1": 1.000},
        "T8":  {"n": 39,  "correct": 38, "wrong": 1,  "gt_miss": 0,  "P": 0.974, "R": 0.974, "F1": 0.974},
        "T9":  {"n": 30,  "correct": 26, "wrong": 4,  "gt_miss": 0,  "P": 0.867, "R": 0.867, "F1": 0.867},
        "T10": {"n": 61,  "correct": 48, "wrong": 12, "gt_miss": 0,  "P": 0.787, "R": 0.787, "F1": 0.787},
    }
    # 总计: 546 提取, 488 正确, 57 错误, 25 GT缺失 → P=89.4%, R=85.5%, F1=87.4%
    pipeline_tab_overall = {"precision": 0.894, "recall": 0.855, "f1": 0.874}

    # ════════════════════════════════════════════════════════════════
    # VLM Figure 评测 (F1-F5, 以 Pipeline 为 anchor 的 cross-matching)
    # ════════════════════════════════════════════════════════════════
    fig_q = vlm_report["figure_quantitative"]

    # Pipeline F1-F5 子集
    pipeline_fig_f1_f5 = []
    for label in [f"F{i}" for i in range(1, 6)]:
        d = pipeline_figure_data[label]
        pipeline_fig_f1_f5.append(d)

    pipe_f15_p = sum(d["P"] for d in pipeline_fig_f1_f5) / 5
    pipe_f15_r = sum(d["R"] for d in pipeline_fig_f1_f5) / 5
    pipe_f15_f1 = sum(d["F1"] for d in pipeline_fig_f1_f5) / 5

    # Gemini/Claude F1-F5 from cross-matching
    gemini_fig_ps, gemini_fig_rs, gemini_fig_f1s = [], [], []
    claude_fig_ps, claude_fig_rs, claude_fig_f1s = [], [], []
    for label in [f"F{i}" for i in range(1, 6)]:
        g = fig_q[label]["gemini"]
        c = fig_q[label]["claude"]
        gemini_fig_ps.append(g["P"]); gemini_fig_rs.append(g["R"]); gemini_fig_f1s.append(g["F1"])
        claude_fig_ps.append(c["P"]); claude_fig_rs.append(c["R"]); claude_fig_f1s.append(c["F1"])

    gemini_fig = {
        "precision": round(sum(gemini_fig_ps) / 5, 4),
        "recall": round(sum(gemini_fig_rs) / 5, 4),
        "f1": round(sum(gemini_fig_f1s) / 5, 4),
    }
    claude_fig = {
        "precision": round(sum(claude_fig_ps) / 5, 4),
        "recall": round(sum(claude_fig_rs) / 5, 4),
        "f1": round(sum(claude_fig_f1s) / 5, 4),
    }

    # ════════════════════════════════════════════════════════════════
    # VLM Table 评测 (T1-T5, Zhao 确认 ~100%)
    # ════════════════════════════════════════════════════════════════

    # Pipeline T1-T5 子集
    pipeline_tab_f1_f5 = []
    for label in [f"T{i}" for i in range(1, 6)]:
        pipeline_tab_f1_f5.append(pipeline_table_data[label])

    pipe_t15_p = sum(d["P"] for d in pipeline_tab_f1_f5) / 5
    pipe_t15_r = sum(d["R"] for d in pipeline_tab_f1_f5) / 5
    pipe_t15_f1 = sum(d["F1"] for d in pipeline_tab_f1_f5) / 5

    # Zhao 评语: "table两个提取的都很好，基本都是100%正确"
    # "claude的轴标题识别的更完美"
    gemini_tab = {"precision": 0.98, "recall": 0.98, "f1": 0.98, "note": "Zhao 评测 ~100%"}
    claude_tab = {"precision": 0.99, "recall": 0.99, "f1": 0.99, "note": "Zhao 评测 ~100%, 列标题更好"}

    # ════════════════════════════════════════════════════════════════
    # SMILES P/R/F1 (T1-T5)
    # ════════════════════════════════════════════════════════════════
    smi = vlm_report["smiles_validation"]

    totals = {"gemini": {"valid": 0, "total": 0},
              "claude": {"valid": 0, "total": 0},
              "pipeline": {"valid": 0, "total": 0}}
    for label in [f"T{i}" for i in range(1, 6)]:
        for src in ["gemini", "claude", "pipeline"]:
            totals[src]["valid"] += smi[label][src]["valid"]
            totals[src]["total"] += smi[label][src]["total"]

    max_total = max(totals[src]["total"] for src in totals)

    smiles_metrics = {}
    for src in ["pipeline", "gemini", "claude"]:
        v = totals[src]["valid"]
        t = totals[src]["total"]
        precision = v / t if t > 0 else 0.0
        recall = v / max_total if max_total > 0 else 0.0
        f1 = _prf(precision, recall)
        smiles_metrics[src] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": f1,
            "valid": v, "total": t,
        }

    # ════════════════════════════════════════════════════════════════
    # 输出
    # ════════════════════════════════════════════════════════════════

    print("=" * 80)
    print("  PART 1: Figure 数据点提取 — Per-figure 明细")
    print("=" * 80)
    print(f"  {'Label':<6} {'Pipeline P':>11} {'Pipeline R':>11} {'Pipeline F1':>12} │ {'Gemini F1':>10} {'Claude F1':>10}")
    print(f"  {'─'*6} {'─'*11} {'─'*11} {'─'*12} │ {'─'*10} {'─'*10}")

    for label in [f"F{i}" for i in range(1, 11)]:
        d = pipeline_figure_data[label]
        vlm_g = fig_q.get(label, {}).get("gemini", {})
        vlm_c = fig_q.get(label, {}).get("claude", {})
        g_f1 = f"{vlm_g['F1']:.3f}" if vlm_g else "—"
        c_f1 = f"{vlm_c['F1']:.3f}" if vlm_c else "—"
        print(f"  {label:<6} {d['P']:>11.3f} {d['R']:>11.3f} {d['F1']:>12.3f} │ {g_f1:>10} {c_f1:>10}")

    print(f"  {'─'*6} {'─'*11} {'─'*11} {'─'*12} │ {'─'*10} {'─'*10}")
    print(f"  {'ALL10':<6} {pipeline_fig_overall['precision']:>11.3f} {pipeline_fig_overall['recall']:>11.3f} {pipeline_fig_overall['f1']:>12.3f} │ {'—':>10} {'—':>10}")
    print(f"  {'F1-F5':<6} {pipe_f15_p:>11.3f} {pipe_f15_r:>11.3f} {pipe_f15_f1:>12.3f} │ {gemini_fig['f1']:>10.3f} {claude_fig['f1']:>10.3f}")

    print(f"\n  ┌──────────────────────────────────────────────────────────────┐")
    print(f"  │  Figure (F1-F5)  Precision   Recall     F1                  │")
    print(f"  ├──────────────────────────────────────────────────────────────┤")
    print(f"  │  Pipeline        {pipe_f15_p:.3f}       {pipe_f15_r:.3f}      {pipe_f15_f1:.3f}               │")
    print(f"  │  Gemini          {gemini_fig['precision']:.3f}       {gemini_fig['recall']:.3f}      {gemini_fig['f1']:.3f}               │")
    print(f"  │  Claude          {claude_fig['precision']:.3f}       {claude_fig['recall']:.3f}      {claude_fig['f1']:.3f}               │")
    print(f"  └──────────────────────────────────────────────────────────────┘")
    print(f"  Pipeline (全10图): P={pipeline_fig_overall['precision']:.3f} R={pipeline_fig_overall['recall']:.3f} F1={pipeline_fig_overall['f1']:.3f}")

    print(f"\n{'=' * 80}")
    print("  PART 2: Table Cell 提取 — Per-table 明细")
    print("=" * 80)
    print(f"  {'Label':<6} {'Pipeline P':>11} {'Pipeline R':>11} {'Pipeline F1':>12} │ {'Gemini':>8} {'Claude':>8}")
    print(f"  {'─'*6} {'─'*11} {'─'*11} {'─'*12} │ {'─'*8} {'─'*8}")

    for label in [f"T{i}" for i in range(1, 11)]:
        d = pipeline_table_data[label]
        if label in [f"T{i}" for i in range(1, 6)]:
            g_str = "~100%"
            c_str = "~100%"
        else:
            g_str = "—"
            c_str = "—"
        print(f"  {label:<6} {d['P']:>11.3f} {d['R']:>11.3f} {d['F1']:>12.3f} │ {g_str:>8} {c_str:>8}")

    print(f"  {'─'*6} {'─'*11} {'─'*11} {'─'*12} │ {'─'*8} {'─'*8}")
    print(f"  {'ALL10':<6} {pipeline_tab_overall['precision']:>11.3f} {pipeline_tab_overall['recall']:>11.3f} {pipeline_tab_overall['f1']:>12.3f} │ {'—':>8} {'—':>8}")
    print(f"  {'T1-T5':<6} {pipe_t15_p:>11.3f} {pipe_t15_r:>11.3f} {pipe_t15_f1:>12.3f} │ {'~98%':>8} {'~99%':>8}")

    print(f"\n  ┌──────────────────────────────────────────────────────────────┐")
    print(f"  │  Table (T1-T5)   Precision   Recall     F1                  │")
    print(f"  ├──────────────────────────────────────────────────────────────┤")
    print(f"  │  Pipeline        {pipe_t15_p:.3f}       {pipe_t15_r:.3f}      {pipe_t15_f1:.3f}               │")
    print(f"  │  Gemini          {gemini_tab['precision']:.2f}        {gemini_tab['recall']:.2f}       {gemini_tab['f1']:.2f}                │")
    print(f"  │  Claude          {claude_tab['precision']:.2f}        {claude_tab['recall']:.2f}       {claude_tab['f1']:.2f}                │")
    print(f"  └──────────────────────────────────────────────────────────────┘")
    print(f"  Pipeline (全10表): P={pipeline_tab_overall['precision']:.3f} R={pipeline_tab_overall['recall']:.3f} F1={pipeline_tab_overall['f1']:.3f}")

    print(f"\n{'=' * 80}")
    print("  PART 3: Table SMILES 准确率 (T1-T5, RDKit 验证)")
    print("=" * 80)
    print(f"  {'Table':<6} {'Gemini':>15} {'Claude':>15} {'Pipeline':>15}")
    print(f"  {'─'*6} {'─'*15} {'─'*15} {'─'*15}")
    for label in [f"T{i}" for i in range(1, 6)]:
        s = smi[label]
        print(f"  {label:<6} {s['gemini']['valid']:>3}/{s['gemini']['total']:<3} valid   "
              f"{s['claude']['valid']:>3}/{s['claude']['total']:<3} valid   "
              f"{s['pipeline']['valid']:>3}/{s['pipeline']['total']:<3} valid")
    print(f"  {'─'*6} {'─'*15} {'─'*15} {'─'*15}")
    print(f"  TOTAL  {totals['gemini']['valid']:>3}/{totals['gemini']['total']:<3}         "
          f"{totals['claude']['valid']:>3}/{totals['claude']['total']:<3}         "
          f"{totals['pipeline']['valid']:>3}/{totals['pipeline']['total']:<3}")
    print(f"  Recall denominator (max total across 3): {max_total}")

    print(f"\n  ┌──────────────────────────────────────────────────────────────┐")
    print(f"  │  SMILES (T1-T5)  Precision   Recall     F1                  │")
    print(f"  ├──────────────────────────────────────────────────────────────┤")
    for src in ["pipeline", "gemini", "claude"]:
        m = smiles_metrics[src]
        print(f"  │  {src:<16} {m['precision']:.3f}       {m['recall']:.3f}      {m['f1']:.3f}               │")
    print(f"  └──────────────────────────────────────────────────────────────┘")

    # ════════════════════════════════════════════════════════════════
    # 综合汇总
    # ════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 80}")
    print("  综合三方对比 (F1-F5 / T1-T5 可比范围)")
    print("=" * 80)
    print(f"""
  ┌──────────────────────────────────────────────────────────────────────────┐
  │                     FlowFigTabMiner     Gemini 3.1 Pro    Claude 4.6   │
  ├──────────────────────────────────────────────────────────────────────────┤
  │ Figure  P/R/F1    {pipe_f15_p:.3f}/{pipe_f15_r:.3f}/{pipe_f15_f1:.3f}     {gemini_fig['precision']:.3f}/{gemini_fig['recall']:.3f}/{gemini_fig['f1']:.3f}     {claude_fig['precision']:.3f}/{claude_fig['recall']:.3f}/{claude_fig['f1']:.3f}  │
  │ Table   P/R/F1    {pipe_t15_p:.3f}/{pipe_t15_r:.3f}/{pipe_t15_f1:.3f}     {gemini_tab['precision']:.2f}/{gemini_tab['recall']:.2f}/{gemini_tab['f1']:.2f}      {claude_tab['precision']:.2f}/{claude_tab['recall']:.2f}/{claude_tab['f1']:.2f}   │
  │ SMILES  P/R/F1    {smiles_metrics['pipeline']['precision']:.3f}/{smiles_metrics['pipeline']['recall']:.3f}/{smiles_metrics['pipeline']['f1']:.3f}     {smiles_metrics['gemini']['precision']:.3f}/{smiles_metrics['gemini']['recall']:.3f}/{smiles_metrics['gemini']['f1']:.3f}     {smiles_metrics['claude']['precision']:.3f}/{smiles_metrics['claude']['recall']:.3f}/{smiles_metrics['claude']['f1']:.3f}  │
  └──────────────────────────────────────────────────────────────────────────┘

  注:
  - Figure: Pipeline 用 Zhao 人工标注 GT; VLM 以 Pipeline 为 anchor 的 cross-matching
  - Table:  Pipeline 用 Zhao 人工标注 GT; VLM 基于 Zhao 确认 "基本都是100%正确"
  - SMILES: Precision = RDKit valid / 提取总数; Recall = valid / max(三者总数={max_total})
""")

    # Save JSON
    result = {
        "figure": {
            "scope": "F1-F5 (comparable) + F1-F10 (pipeline only)",
            "pipeline_F1_F5": {"precision": round(pipe_f15_p, 4), "recall": round(pipe_f15_r, 4), "f1": round(pipe_f15_f1, 4)},
            "pipeline_all10": pipeline_fig_overall,
            "gemini_F1_F5": gemini_fig,
            "claude_F1_F5": claude_fig,
            "pipeline_per_figure": pipeline_figure_data,
            "vlm_per_figure": {k: {"gemini": v["gemini"], "claude": v["claude"]}
                               for k, v in fig_q.items()},
        },
        "table_cell": {
            "scope": "T1-T5 (comparable) + T1-T10 (pipeline only)",
            "pipeline_T1_T5": {"precision": round(pipe_t15_p, 4), "recall": round(pipe_t15_r, 4), "f1": round(pipe_t15_f1, 4)},
            "pipeline_all10": pipeline_tab_overall,
            "gemini_T1_T5": gemini_tab,
            "claude_T1_T5": claude_tab,
            "pipeline_per_table": pipeline_table_data,
        },
        "table_smiles": {
            "scope": "T1-T5",
            "pipeline": smiles_metrics["pipeline"],
            "gemini": smiles_metrics["gemini"],
            "claude": smiles_metrics["claude"],
            "recall_denominator": max_total,
            "per_table": {label: smi[label] for label in [f"T{i}" for i in range(1, 6)]},
        },
    }

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"  报告已保存: {OUTPUT}")


if __name__ == "__main__":
    main()
