# 三方对比评测方法论

## 评测对象

| 系统 | 类型 | 说明 |
|------|------|------|
| FlowFigTabMiner (Pipeline) | 本地 pipeline | YOLO + MolNexTR + TATR + PaddleOCR |
| Gemini 3.1 Pro | 商用 VLM | 零样本提取，标准 prompt |
| Claude Sonnet 4.6 | 商用 VLM | 零样本提取，标准 prompt |

## 评测范围

- **Figure**: F1-F10 共 10 张散点/折线/热力图（Pipeline 全 10 张，VLM 仅 F1-F5）
- **Table**: T1-T10 共 10 张化学表格（Pipeline 全 10 张，VLM 仅 T1-T5）
- **三方可比范围**: F1-F5 + T1-T5

---

## 1. Figure 数据点提取 — P/R/F1

### Pipeline (F1-F10)

**GT 来源**: 人工对照原始 PDF 图像，逐点判断 pipeline 提取的 `(X, Y_Left, Y_Right)` 是否正确。

**指标定义**:
- `correct`: 提取的点中与 GT 匹配的数量
- `wrong`: 提取的点中错误的数量（不存在的点或坐标严重偏差）
- `gt_miss`: GT 中存在但 pipeline 未提取到的点
- **Precision** = `correct / (correct + wrong)` = `correct / n_predicted`
- **Recall** = `correct / (correct + gt_miss)` = `correct / n_gt`
- **F1** = `2 * P * R / (P + R)`

### VLM (F1-F5)

**方法**: 以 Pipeline 提取结果为 anchor，用 greedy nearest-neighbor matching 计算 VLM 输出与 Pipeline 的匹配度。

**匹配算法**:
1. 计算坐标范围容差: `tol_X = 5% * (X_max - X_min)`, `tol_Y = 5% * (Y_max - Y_min)`
2. 对每个 VLM 点，找 Pipeline 中距离最近且在容差内的未匹配点
3. 贪心匹配，每个 Pipeline 点最多匹配一次

**指标定义**:
- **Precision** = `matched / n_vlm_points`（VLM 提取的点中有多少匹配 Pipeline）
- **Recall** = `matched / n_pipeline_points`（Pipeline 的点中有多少被 VLM 找到）
- **F1** = `2 * P * R / (P + R)`

**注意**: VLM 的 P/R/F1 衡量的是 "与 Pipeline 的一致程度"，不是对绝对 GT 的准确率。由于 Pipeline 本身 F1 ≈ 0.89，VLM 的真实准确率可能略有不同。

---

## 2. Table Cell 提取 — P/R/F1

### Pipeline (T1-T10)

**GT 来源**: 人工对照原始 PDF 表格图像，逐个 cell 判断 pipeline 提取值是否正确。

**指标定义**:
- `correct`: 提取正确的 cell 数
- `wrong`: 提取错误的 cell 数（OCR 错误、位置对错等）
- `gt_miss`: GT 中存在但 pipeline 未提取到的 cell
- **Precision** = `correct / (correct + wrong)`
- **Recall** = `correct / (correct + gt_miss)`
- **F1** = `2 * P * R / (P + R)`

### VLM (T1-T5)

**GT 来源**: Zhao 人工逐表对照原始图像评估，确认 "基本都是100%正确"。

**指标赋值**:
- Gemini: P ≈ 0.98, R ≈ 0.98, F1 ≈ 0.98（极少数格式细节差异）
- Claude: P ≈ 0.99, R ≈ 0.99, F1 ≈ 0.99（列标题识别更完美）

---

## 3. Table SMILES 准确率 — P/R/F1

### SMILES 识别方法

- **Pipeline**: YOLO 检测分子结构图 → MolNexTR 转 SMILES（图像→SMILES）
- **VLM**: 直接从表格图像中读取 SMILES 文本（OCR 方式）

### 指标定义

- **Precision** = `RDKit_valid_SMILES / 该系统提取的 SMILES 总数`
  - "RDKit valid" = `rdkit.Chem.MolFromSmiles(smi)` 返回非 None
  - 即语法合法且可解析为分子结构的 SMILES
- **Recall** = `RDKit_valid_SMILES / max(三系统 SMILES 总数)`
  - 分母取三个系统中提取 SMILES 数量最多的那个（= 89，来自 Gemini）
  - 衡量 "在所有可能提取的 SMILES 中，该系统找到了多少有效的"
- **F1** = `2 * P * R / (P + R)`

### SMILES 过滤逻辑

`_looks_like_smiles()` 函数判断一个 cell 值是否为 SMILES:
1. 长度 ≥ 4
2. 排除纯数字+脚注格式（如 `34[b]`, `70 (82)[c]`）
3. 排除短缩写（如 `MeLi`, `LDA`）
4. SMILES 字符占比 > 70% 且包含键/分支符号 (`=#()[]`) 和原子符号

### 已知局限

- T4/T5 中的 Pd 催化剂缩写（`Pd[P(tBu)3]2`）不是标准 SMILES，三方均 invalid
- Claude 使用缩写记法（`Bu3SnCl`, `Ph3SnCl`），RDKit 无法解析，计为 invalid
- Pipeline 的 MolNexTR 输出（图像→SMILES）与 VLM 的文本 OCR 输出可能对应不同表达形式的同一分子

---

## 4. Overall 综合指标

**计算方法**: Figure、Table、SMILES 三项指标的简单平均

- **Overall Precision** = `(Figure_P + Table_P + SMILES_P) / 3`
- **Overall Recall** = `(Figure_R + Table_R + SMILES_R) / 3`
- **Overall F1** = `(Figure_F1 + Table_F1 + SMILES_F1) / 3`

---

## 关键文件

| 文件 | 说明 |
|------|------|
| `eval/compute_three_way.py` | 三方对比计算脚本 |
| `eval/three_way_comparison.json` | 计算结果（机器可读） |
| `eval/three_way_comparison.tex` | LaTeX 对比表格 |
| `eval/vlm_comparison_report.json` | VLM cross-matching + SMILES 验证数据 |
| `eval/layer1_figure_annotation.csv` | Pipeline figure 提取结果 |
| `eval/layer1_table_annotation.csv` | Pipeline table 提取结果 |
| `eval/images/figures/F{1-5}.json` | Gemini figure 输出 |
| `eval/images/figures/F{1-5}_claude.json` | Claude figure 输出 |
| `eval/images/tables/T{1-5}.json` | Gemini table 输出 |
| `eval/images/tables/T{1-5}_claude.json` | Claude table 输出 |
| `eval/vlm_prompts/figure_prompt.txt` | VLM 标准化 figure 提取 prompt |
| `eval/vlm_prompts/table_prompt.txt` | VLM 标准化 table 提取 prompt |
