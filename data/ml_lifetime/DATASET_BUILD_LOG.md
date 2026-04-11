# clean_organolithium_unified.csv 数据集构建日志

记录数据集从原始 PDF 到最终 CSV 的全过程，包括数据来源、处理步骤、修正历史。

---

## 1. 数据来源

### 1.1 论文集（26 篇）

所有论文 PDF 存放在 `data/ml_lifetime/papers/`（未入 git，160MB）。

来源两个方向：
- **现有 tR 数据集**：`data/final_output/organolithium_tr_subdataset_vlm_enriched.csv`（1,470 行，20 篇论文）— 由 FlowFigTabMiner 的 figure pipeline + VLM 提取
- **Clean organolithium 补充**：20 篇 PDF，从 `data/input/Clean organolithium/` 复制，去重后约 13 篇新增

### 1.2 提取管线

每篇论文经过以下管线：

1. **FlowFigTabMiner pipeline**（`src/pipeline/main.py`）
   - TF-ID (Florence-2) 检测 PDF 中的 figure/table
   - YOLO macro/micro 检测 → 坐标映射 → 热图数据提取
   - SchemeSegParser → `compound_pool.json`（化合物标签→SMILES 映射）
   - 输出：`data/intermediate/{paper_name}/`

2. **Gemini VLM 表格提取**（`scripts/ml_lifetime/extract_tables_gemini.py`）
   - 对 SI PDF 的每页表格调用 Gemini 2.5 Flash
   - 输出：`data/ml_lifetime/gemini_table_extractions/{paper_key}/_all_tables.json`
   - 每个表格含 `columns`, `rows`, `reaction_context`, `structures`

3. **GlobalAssembly LLM 组装**（`src/adjudication/global_assembly.py`）
   - 将 PDF 全文 + figure evidence + table data + compound_pool 送给 Qwen-Plus
   - 输出：`data/final_output/{paper_name}_final.json`
   - 注：2026-04 因 API 端点迁移（DashScope 新加坡）重新执行

4. **Gemini 化合物解析**（手动 + API）
   - 对无法从 pipeline 获得 SMILES 的化合物标签（如 "3a", "c-9/t-9"），
     将论文全文发送给 Gemini 解析化合物指代
   - 结果存储：`data/ml_lifetime/gemini_compound_resolution.json`

### 1.3 SI 表格下载

`scripts/ml_lifetime/download_si.py` — 从 Wiley/ACS 下载 Supporting Information PDF
`scripts/ml_lifetime/extract_si_tables.py` — 调用 Gemini 提取 SI 表格

---

## 2. 构建脚本

**主脚本**: `scripts/ml_lifetime/build_clean_dataset.py`

### 2.1 数据流

```
gemini_table_extractions/    ──┐
                               ├─→ build_clean_dataset.py ──→ clean_organolithium_unified.csv
organolithium_tr_subdataset  ──┘
     _vlm_enriched.csv              (2609 行 × 43 列)
```

### 2.2 处理步骤

1. 遍历 `gemini_table_extractions/` 中 26 篇论文的 `_all_tables.json`
2. 按 paper_config 映射：paper_id, DOI, intermediate, reaction_class 等元数据
3. 每行从 Gemini 表格中提取：tR1, T1, yield, substrate, electrophile, product
4. 从 `structures` 数组和 `reaction_context` 中查找 SMILES
5. 应用 25 个修正（Fix 1–25），见下文
6. RDKit 规范化所有 SMILES 列
7. 输出排序后的 CSV

### 2.3 Schema（43 列）

| 类别 | 列 |
|------|-----|
| 标识 | paper_id, paper_doi, quality_flag, data_source_type |
| 中间体 | intermediate, intermediate_smiles, intermediate_smiles_canonical |
| 底物 | substrate1, substrate1_smiles, substrate1_smiles_canonical |
| 试剂 | organolithium_reagent |
| 亲电试剂 | electrophile, electrophile_smiles, electrophile_smiles_canonical, electrophile_type |
| 产物 | product, product_smiles, product_smiles_canonical |
| 反应 | reaction_class |
| 条件 | tR_step, tR1_s, T1_C, tR2_s, T2_C, solvent, conc_substrate_M |
| 分类 | analysis_subset, intermediate_class |
| 描述符 | sigma_hammett, Es_taft, delta_ortho, delta_benzyne |
| DFT | dft_charge_Li, dft_charge_C_ipso, dft_HOMO_eV, dft_LUMO_eV, dft_LiC_bond_A, dft_method, dft_LiC_BDE_kJ, dft_wiberg_LiC, dft_dipole_D, dft_Gsolv_kJ |
| 结果 | yield_pct |

---

## 3. 修正历史 (Fix 1–25)

### Fix 1–11: 基础清洗
- Fix 1–4: 列重命名、空值处理、electrophile_type 分类
- Fix 5–8: paper_id 标准化（中间体+年份+作者格式）
- Fix 9–11: analysis_subset 分配（kd_clean/kd_valid/scope/k2_trapping）

### Fix 12: SMILES 修正映射
- 修复 Gemini 提取中的无效 SMILES（ring closure 冲突、"Ph" 缩写等）
- **2026-04-11 修正**: OxiranylLi product "3" 的异构体错误
  - 原: `CC1(O1)c2ccccc2` → `CC1OC1c1ccccc1`（2-methyl-**3**-phenyloxirane，错误）
  - 改: `CC1(O1)c2ccccc2` → `CC1(c2ccccc2)CO1`（2-methyl-**2**-phenyloxirane，正确）
  - 依据: α-phenyloxiranyllithium 去质子化在 α 碳（与 Ph 同碳），MeI 加成同一碳 → gem-disubstituted

### Fix 13: 特定论文底物/产物修正
- 13a: PyridylKetone 底物 SMILES
- 13b: PyridylKetone 底物名称
- 13c: ProtGroupFree Table S1 compound "1" SMILES
- 13d: ProtGroupFree 试剂 = MesLi

### Fix 14–15: 描述符填充
- Hammett σ, Taft Es, δ_ortho, δ_benzyne 映射

### Fix 16: 文本列中的 SMILES 检测
- 将 product/electrophile 文本列中实际为 SMILES 的值迁移到对应 _smiles 列

### Fix 17–23: 中间体 SMILES 派生
- Fix 17: tBuEsters 中间体 SMILES
- Fix 18: 通用 halogen→[Li] 中间体派生（82 行）
- Fix 19–22: OxiranylLi / SilyloxiranylLi / ProtGroupFree 特定中间体
- Fix 23: PyridineLi 中间体 SMILES（单锂化 + 双锂化）

### Fix 24: CHLiIF yield 列修正
- Table 1 的 yield 应取 "flow yield" 列而非 "batch yield" 列

### Fix 25: product_smiles 全面解析（520 行）

这是最大的一次修正，将 product_smiles 覆盖率从 80.1% 提升至 **100%**。

| 论文 | 行数 | 方法 |
|------|------|------|
| OxiranylLi | 339 | 论文全文 IUPAC 命名 → SMILES：compound 3 (2-methyl-2-phenyloxirane), c-9/t-9, c-11/t-11, c-16/t-16 + SI 表格底物推导（c-8→c-9, biphenylyl→methylated, 6→7）|
| AlkoxycarbonylFlow | 124 | Gemini structures (3a=tBu benzoate, 3b=iPr benzoate, 3c=Et benzoate, 3d=Me benzoate) + 动力学 quench 行→3b |
| CHLiIF | 28 | Gemini 化合物解析：CHLiIF + Weinreb amide → fluoromethyl ketone |
| BiaryLi | 16 | 中间体质子化：2-bromo-2'-lithiobiphenyl + MeOH → 2-bromobiphenyl |
| SilyloxiranylLi | 12 | Gemini 解析：product 2 = TMS-silyloxirane |
| FuncAlkylLi | 1 | PhCOCCCBr + MeOTf → valerophenone |

---

## 4. 化学审计（2026-04-11）

对全部 2609 行 product_smiles 执行以下检查：

| 检查项 | 结果 |
|--------|------|
| RDKit 有效性 | 2609/2609 ✓ |
| 无 [Li] 残留 | 0 行含 [Li] ✓ |
| 同产物→同 SMILES | 0 不一致 ✓ |
| product ≠ substrate | 0 行相同 ✓ |
| MW 范围合理 | 最大 609 (Bu₃Sn-C₆F₁₃) ✓ |
| quench 产物卤素数 | 6 行异常（PyridineLi/ThreeComp scope 误分类为 quench，pre-existing）|
| Fix 25 全部 16 化合物 | 氧丙烷环/酯基/卤素/F/Si 原子数均正确 ✓ |
| OxiranylLi gem 模式一致性 | 修正后 105 行统一为 `CC1(c2ccccc2)CO1` ✓ |

---

## 5. 最终数据集统计

```
文件:   data/ml_lifetime/clean_organolithium_unified.csv
行数:   2609
列数:   43
论文:   26 篇

SMILES 覆盖率:
  intermediate_smiles: 94.9%
  substrate1_smiles:   96.2%
  electrophile_smiles: 90.6%
  product_smiles:      100.0%

analysis_subset 分布:
  kd_clean:        1893 (72.5%)
  kd_valid:         350 (13.4%)
  scope:            219  (8.4%)
  kd_and_trapping:  122  (4.7%)
  k2_trapping:       25  (1.0%)
```

---

## 6. 已知限制

1. **electrophile_type 误分类**: PyridineLi/ThreeComp 的 6 行 scope 产物被标为 quench_probe
2. **CHLiIF 产物命名缺失**: 28 行 Bu₃SnCHFI 产物有 SMILES 但无 product name
3. **papers/ 未入 git**: 32 篇 PDF (160MB) 不在版本控制中
4. **DFT 描述符列**: 目前全为空，计划后续通过 Gaussian/ORCA 计算填充
