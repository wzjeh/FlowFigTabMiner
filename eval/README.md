# FlowFigTabMiner 评测框架

本目录完全独立于 `src/` 和 `scripts/`，可随时整体删除（`rm -rf eval/`）。

---

## 目录结构

```
eval/
├── README.md                     ← 本文件（操作说明）
├── samples.json                  ← 评测样本配置（修改此文件来换样本）
├── generate_annotation.py        ← Step 1：生成标注模板
├── compute_metrics.py            ← Step 3：计算指标
│
├── (运行后生成)
├── layer1_figure_annotation.csv  ← Layer1 图坐标标注模板
├── layer1_table_annotation.csv   ← Layer1 表格 cell 标注模板
├── layer2_final_annotation.csv   ← Layer2 最终 JSON 字段标注模板
├── eval_report.json              ← 完整指标（JSON）
└── eval_summary.csv              ← 汇总表（Excel 查看）
```

---

## 评测架构说明

```
Layer 1 — 中间体提取质量
  ├── Figure：evidence.json → 预测坐标 vs 原图真实坐标
  └── Table：extracted.csv → 预测 cell 内容 vs 原 PDF 表格

Layer 2 — 最终组装质量（跳过 LLM 推断字段）
  └── normalized.json → 8 个关键字段 vs 原 PDF
      yield_pct / conversion_pct / reactant1_smiles / product_smiles
      temperature_C / residence_time_s / solvent / catalyst
```

---

## 操作步骤

### Step 0：进入 venv，切换到项目根目录

```bash
cd /path/to/FlowFigTabMiner
source flowfigtabminer/bin/activate
```

---

### Step 1：生成标注模板

```bash
python eval/generate_annotation.py
```

生成三个 CSV 文件（见上方目录结构）。

---

### Step 2：人工标注（用 Excel / Numbers 打开 CSV）

#### 2a — Layer1 Figure（`layer1_figure_annotation.csv`）

**每行 = pipeline 提取的一个数据点。**

参考文件：
- `ref_image` 列：打开这张图片（原始 PDF 裁剪图），对比预测值
- 同时对照原始 PDF，找到该图所在页面

填写方式（二选一）：

| 填法 | 操作 | 适用场景 |
|------|------|---------|
| **快速法（推荐）** | 在 `is_match_within_tol` 列填 `1`（正确）或 `0`（错误）| 只想快速判断对不对 |
| **精确法** | 在 `gt_X` 和 `gt_Y_Left` 列填从图上读出的真实值 | 想得到精确误差分析 |

> **评判标准**：X 值和 Y 值都在 ±5% 轴范围内 → 正确（`1`）
> 例：Y 轴 0-100%，则 ±5 以内算正确；Y 轴 0-200°C，则 ±10°C 以内算正确

**注意事项**：
- `series` 列是 pipeline 提取的图例标签，可顺带检查是否正确
- F7（example/page_6_figure_0）有双 Y 轴，`pred_Y_Right` 列是右轴值

---

#### 2b — Layer1 Table（`layer1_table_annotation.csv`）

**每行 = 表格中的一个非空 cell（最多取前 10 个数据行）。**

参考文件：
- `ref_image` 列：打开这张表格图片
- `cell_type` 列：`text`（文字/数字）或 `smiles`（分子结构）
- 对照原 PDF 找到对应表格

填写方式：

1. 在 `is_correct` 列填 `1`（正确）或 `0`（错误）
2. 可选：在 `gt_value` 列填真实值（用于 SMILES 精确比对）

> **评判标准**：
> - **text/数字 cell**：提取内容与原表一致（允许细微格式差异，如 `-78` vs `−78`）
> - **smiles cell**：与原结构代表同一个分子（不要求字符完全一致，canonical SMILES 相同即可）

**注意事项**：
- T3（107/page_8_table_0）Product 列出现两次，是 TATR 结构识别误判，可标 `0`
- T4（d3cc00756a）OCR 噪声重，`?à` 这类明显错误标 `0`

---

#### 2c — Layer2 Final（`layer2_final_annotation.csv`）

**每行 = 最终 JSON 中某条反应记录的某个字段值。**

参考：
- `paper_id` + `source` 列：定位是哪篇文章的哪张表/图
- `field` 列：字段名称
- `predicted_value` 列：pipeline 提取的值

填写方式：

1. 在 `is_correct` 列填 `1` 或 `0`
2. 可选：在 `gt_value` 列填 PDF 原文的真实值

> **评判标准（按字段）**：
> | 字段 | 标准 |
> |------|------|
> | `yield_pct` / `conversion_pct` | ±5% 相对误差以内 |
> | `temperature_C` / `residence_time_s` | ±5% 相对误差以内 |
> | `reactant1_smiles` / `product_smiles` | RDKit canonical SMILES 相同 |
> | `solvent` / `catalyst` | 主要成分名称正确（允许缩写差异）|

---

### Step 3：计算指标

```bash
python eval/compute_metrics.py
```

终端会打印：

```
Layer 1 — Figure 坐标提取
  Overall   Precision=0.xx  Recall=0.xx  F1=0.xx
  F1 [人工判断]  P=0.xx  R=0.xx  F1=0.xx  (matched/pred)
  ...

Layer 1 — Table Cell 提取
  Overall  text_acc=0.xx  smiles_valid=0.xx  smiles_canonical=0.xx
  ...

Layer 2 — 最终 JSON 字段准确率
  Overall  accuracy=0.xx
  yield_pct              ████████████░░░░░░░  63%   n=80
  temperature_C          ████████████████░░░  82%   n=75
  ...
```

同时生成 `eval/eval_report.json` 和 `eval/eval_summary.csv`。

---

## 指标含义

### Layer 1 Figure

| 指标 | 含义 |
|------|------|
| **Precision** | 提取的点中，有多少比例是正确的（避免幻觉点）|
| **Recall** | 图中真实点中，有多少比例被正确提取（避免漏掉）|
| **F1** | P 和 R 的调和平均，综合指标 |

> 容差：X 和 Y 各自 ±5% 轴范围

### Layer 1 Table

| 指标 | 含义 |
|------|------|
| **text_accuracy** | 文字/数字 cell 提取正确率 |
| **smiles_valid_rate** | 提取的 SMILES 可被 RDKit 解析的比例（合法性）|
| **smiles_canonical_match** | 与 GT canonical SMILES 完全匹配的比例（正确性，需填 gt_value）|
| **overall_cell_accuracy** | 所有 cell 的综合正确率 |

### Layer 2 Final

| 指标 | 含义 |
|------|------|
| **per-field accuracy** | 每个字段（yield/temp/smiles 等）的提取准确率 |
| **overall accuracy** | 全部评测字段的综合准确率 |

> 不包含 `reaction_class` / `reaction_smiles` 等 LLM 推断字段

---

## 期望结果参考

基于已有数据的初步观察：

- **T3**（107/page_8_table_0，最干净）→ smiles_valid_rate 应接近 **100%**
- **T1**（example1，MolNexTR）→ smiles_valid_rate 约 **80-90%**，OCR 产率有误
- **F1**（102，4系列）→ 如 series 正确、坐标精度高，F1 可达 **70-85%**
- **Layer2 yield_pct** → 预计 **60-80%**（LLM 可能调整数值）
- **Layer2 temperature_C / residence_time_s** → 预计 **75-90%**（直接来自表格提取）

---

## 修改评测范围

编辑 `eval/samples.json`：
- 换图/表样本：修改 `figures` / `tables` 数组
- 换 final output 论文：修改 `final_output` 数组
- 调整每篇 final 的取样数量：修改 `max_records`

修改后重新运行 `python eval/generate_annotation.py` 即可。
