# 有机锂中间体稳定性预测：从分子机理到反应器推荐

> PPT 叙事线：6 个发现，从科学到工程的完整闭环。每页底部附参考文献。

---

## Page 1: 背景与目标

### 问题
有机锂中间体在流动化学中广泛使用，但热不稳定 [1]。选错反应器 → 中间体分解 → 产率崩塌。

### 目标
**输入 SMILES + 温度 → 推荐 flash / flow / batch 反应器**

### 数据集
- 75 个有机锂中间体，12 篇文献 [1–6]，2357 个 yield-vs-tR 数据点
- 40 个 Arrhenius 拟合（26 个 tier-1，≥3 温度），42 个化合物有 t_max 数据
- THF / THF-hexane 4:1 溶剂

### 图表
→ 分子结构总览图 (`ppt_fig_molecules_classified.png`)

> **Refs**: [1] De Gennaro 2014, Lithium Compounds in Organic Synthesis, Ch.18; [2–6] Nagaki et al. 2007–2019

---

## Page 2: Finding 1 — 机制异质性（跨类别预测失败）

### 核心结论
**不同结构类别的有机锂走完全不同的分解机制，不能用一个模型通吃。**

### 证据
全局 Ea 模型（q_C + d_LiC + %Vbur, n=24）LOO-R² = 0.694，看似不错。
但 LOCO 验证（训练 2 类 → 预测第 3 类）：

| 排除类别 | R² |
|---|---|
| m+p ArLi | −0.21 |
| o-ArLi | −0.04 |
| oxiranylLi | **−12.9** |

R² < 0 = 比直接猜均值还差。

### 图表
→ `finding1_LOCO.png`

### 物理解释
- m+p ArLi: THF 溶剂裂解（缔合型 TS）
- o-ArLi: 苯炔消除（解离型碎裂）
- oxiranylLi: 环氧开环

> **Refs**: [14] Collum 2007, Angew. Chem. — "substrate-dependent mechanisms may be the rule, not the exception"

---

## Page 3: Finding 2 — Ea-lnA 补偿（为什么 t½ 难预测）

### 核心结论
**Ea 可以从基态描述符预测，但 lnA（∝ ΔS‡）不可以。两者高度相关（r=0.94），相互抵消导致 t½ 预测失败。**

### 证据
- Ea 模型: LOO-R² = 0.694（全局），0.91–0.95（类别内）
- lnA: 无描述符组合能有效预测
- Ea 和 lnA 的补偿: 高 Ea（分解慢）的化合物有高 lnA（大前指因子），t½ 被"压平"

### 图表
→ `finding2_EalnA_compensation.png`

### 物理解释
lnA ∝ ΔS‡，ΔS‡ 受多重因素叠加 [14]：
- 解聚（+）：聚集体释放片段
- 溶剂重组（−）：THF 分子被组织到 TS 中
- TS 构象受限（−）

这些因素取决于溶液相动态行为，基态描述符无法捕捉。

> **Refs**: [14] Collum 2007; [7] Ramachandran 2010, J. Phys. Chem. A — 聚集态 +24 kJ/mol, 溶剂化 −31 kJ/mol

---

## Page 4: Finding 3 — 描述符层级悖论（xTB 优于 DFT）

### 核心结论
**更昂贵的计算方法（HF → M06-2X → QTAIM）并不带来更好的预测。最廉价的 xTB（秒级）反而最好。**

### 证据：28 个描述符 × 6 个计算层级穷举筛选

| 层级 | 代表描述符 | 成本 | 全局 Ea R² |
|---|---|---|---|
| **xTB (Level 1)** | q_C, d_LiC, %Vbur | **秒** | **0.694** |
| HF (Level 2) | q(C)_HF | 分钟 | 0.55 |
| M06-2X (Level 3) | q(C)_M06 | 10 分钟 | 0.41 |
| ADCH (Level 3) | q(C)_ADCH | 10 分钟 | 0.40 |
| QTAIM (Level 3) | ρ_BCP | 20 分钟 | ~0 |

### 图表
→ `ppt_fig1_full_correlation_matrix.png`（28 描述符全相关矩阵）

### 物理解释
预测瓶颈不在"基态电子结构精度"，而在"溶液相动力学复杂性"——聚集态、溶剂配位、动态熵效应 [14]。更精确的气相电子结构反而引入了与溶液行为不相关的噪声。

选中的 3 个描述符（q_C, d_LiC, L）互相关性低（r < 0.3），分别编码**电子效应、键强度、空间保护**——物理意义正交。

> **Refs**: [8] Bannwarth 2019 (GFN2-xTB); [11] Lu 2012 (Multiwfn); [16] Verkhov 2025 (QTAIM/ELF)

---

## Page 5: Finding 4 — 气相 TS 计算不可行

### 核心结论
**从第一性原理精确计算 Ea 在当前不可行。气相 TS 高估 3.8×，加隐式溶剂仅修正 23%。**

### 证据：3 个代表体系的 TS 计算 (M06-2X/def2-SVP, ORCA)

| 体系 | 机制 | 气相 ΔH‡ | +CPCM | 实验 Ea |
|---|---|---|---|---|
| oxiranylLi | 环开裂 | 134.7 | 111.6 | 35.6 kJ/mol |
| PhLi+THF | α-H 转移 | ~96 | — | 36.5 kJ/mol |
| o-BrPhLi | 苯炔消除 | 无势垒 | — | 72.1 kJ/mol |

CPCM 对 ΔS‡ 完全无效：气相 +1.0 → CPCM +1.1 → 实验 −121 J/(mol·K)

### 图表
→ `finding4_5_TS_entropy.png` panel (a)

### 为什么不行？
*Implicit solvation partially corrects the enthalpy barrier but fails to capture entropy contributions, suggesting the importance of explicit solvent interactions and aggregation dynamics.* [14]

要做对需要：显式溶剂 + 聚集态采样 + AIMD → 每个化合物天-周级计算 → 不可扩展

> **Refs**: [7] Ramachandran 2010 (M06-2X benchmark); [12] Neese 2022 (ORCA); [14] Collum 2007 — "只考虑 TS 端溶剂效应 is complete nonsense"

---

## Page 6: Finding 5 — ΔS‡ 揭示机制多样性

### 核心结论
**实验 ΔS‡ 的符号直接区分缔合型 vs 解离型机制，与 LOCO 验证的跨类失败一致。**

### 证据

| 体系 | ΔS‡ [J/(mol·K)] | 机制类型 |
|---|---|---|
| oxiranylLi | −121 | 缔合型 TS（溶剂参与决速步） |
| PhLi | −125 | 缔合型 TS（THF 失去平动自由度） |
| o-BrPhLi | **+77** | 解离型 TS（碎裂产生平动熵） |

### 图表
→ `finding4_5_TS_entropy.png` panel (b)

### 物理解释
ΔS‡ 是多个效应的叠加 [14]：解聚（+）+ 溶剂重组（−）+ TS 构象受限（−）。
不同类别的净值符号不同 → 机制本质不同 → 统计验证（LOCO R²<0）与物理验证（ΔS‡ 符号）一致。

> **Refs**: [14] Collum 2007; [15] Seebach 1988 (聚集态结构); [17] Lucht & Collum 1999 (溶液结构)

---

## Page 7: Finding 6 — 反应器推荐工具（最终成果）

### 核心结论
**绕过 Ea-lnA 补偿，直接预测 t_max → flash/flow/batch 分类，89.3% 准确率。**

### 模型

```
log₁₀(t_max/s) = 10.86·q(C) − 13.09·d(Li-C) + 0.48·L + 336/T + 21.93

t_max < 0.1 s  → flash (微混合器)
0.1–60 s       → flow (管式反应器)  
> 60 s         → batch (常规操作)
```

LOO-CV 分类准确率 = **89.3%**（140 数据点，42 化合物，−78 ~ +25°C）

### 图表
→ `ppt_fig2_model_performance.png`（log-log parity + 特征重要性）

### 特征物理意义

| 描述符 | 重要性 | 系数 | 物理意义 |
|---|---|---|---|
| q(C_ipso) | 0.413 | + | C 上正电荷越多 → C-Li 键越稳定 → t_max 越长 |
| d(Li-C) | 0.298 | − | 键越长 → 越弱 → 分解越快 → t_max 越短 |
| Sterimol L | 0.174 | + | 取代基越长 → 空间保护 → t_max 越长 |
| 1/T | 0.115 | + | 温度越低 → Arrhenius 减速 → t_max 越长 |

3 个分子描述符从 xTB 计算（秒级）+ 1 个温度参数 → 涵盖 −78 ~ +25°C 全温度范围。

> **Refs**: [8] Bannwarth 2019 (GFN2-xTB); [9] Falivene 2016 (Sterimol/Vbur)

---

## Page 8: 总结 — 从分子到工程的完整闭环

```
科学层                    解释层                   工程层
                                              
Finding 1: 机制异质性      Finding 4: TS 不可行      Finding 6: t_max 工具
Finding 2: Ea-lnA 补偿     Finding 5: ΔS‡ 区分机制   89.3% 准确率
Finding 3: xTB > DFT                              flash/flow/batch
                                              
     为什么？                  为什么不能算？            怎么用？
```

### 一句话论文结论

*Activation parameters (Ea, lnA) provide mechanistic insight into the decomposition process, whereas t_max directly determines optimal operating conditions in flow systems. The combination of mechanistic analysis and data-driven prediction forms a complete framework from molecular understanding to process design.*

### 当前局限
- 数据量（42 化合物）是主要瓶颈，更多实验数据 >> 更高级计算方法
- "batch" 类化合物在训练集中缺失（都来自流动化学文献）
- 聚集态信息未显式纳入

> **Refs**: [1] De Gennaro 2014; [14] Collum 2007; [7] Ramachandran 2010

---

## 参考文献

[1] De Gennaro, L.; Fanelli, F.; Luisi, R. *Lithium Compounds in Organic Synthesis*, Wiley, 2014, Ch.18. — 有机锂分解动力学综述，LFER 基础。

[2] Nagaki, A. et al. *J. Am. Chem. Soc.* **2014**, 136, 12245. — o-BrPhLi, o-IPhLi 热稳定性。

[3] Nagaki, A. et al. *Angew. Chem. Int. Ed.* **2019**, 58, 4027. — oxiranylLi 分解动力学。

[4] Nagaki, A. et al. *J. Flow Chem.* **2008**. — o-CO₂R-ArLi Arrhenius 参数。

[5] Musci, P. et al. *React. Chem. Eng.* **2020**, 5, 935. — 卡宾类 LiCHXY 动力学。

[6] Stanetty, P.; Mihovilovic, M. *J. Org. Chem.* **1997**, 62, 1514. — 有机锂半衰期数据。

[7] Ramachandran, B. et al. *J. Phys. Chem. A* **2010**, 114, 8423. — M06-2X 最佳泛函；聚集态 +24 kJ/mol, 溶剂化 −31 kJ/mol。

[8] Bannwarth, C. et al. *J. Chem. Theory Comput.* **2019**, 15, 1652. — GFN2-xTB 方法。

[9] Falivene, L. et al. *Organometallics* **2016**, 35, 2286. — %Vbur 和 Sterimol 参数。

[10] Smith, D. G. A. et al. *J. Chem. Phys.* **2020**, 152, 184108. — Psi4 软件。

[11] Lu, T.; Chen, F. *J. Comput. Chem.* **2012**, 33, 580. — Multiwfn 波函数分析。

[12] Neese, F. *WIREs Comput. Mol. Sci.* **2022**, 12, e1606. — ORCA 软件。

[13] Hansch, C. et al. *Chem. Rev.* **1991**, 91, 165. — Hammett σ 取代基常数。

[14] Collum, D. B. et al. *Angew. Chem. Int. Ed.* **2007**, 46, 3002. — 有机锂溶液动力学经典综述：聚集态 TS、底物依赖机制、溶剂影响两端。

[15] Seebach, D. *Angew. Chem. Int. Ed. Engl.* **1988**, 27, 1624. — 有机锂聚集态晶体学。

[16] Verkhov, V. A. et al. *J. Chem. Phys.* **2025**, 162, 044114. — QTAIM/ELF 描述符。

[17] Lucht, B. L.; Collum, D. B. *Acc. Chem. Res.* **1999**, 32, 1035. — 有机锂溶液结构综述。

[18] Reich, H. J. *Chem. Rev.* **2013**, 113, 7130. — 聚集态与反应性综述。

[19] Harrison-Marchand, A.; Mongin, F. *Chem. Rev.* **2013**, 113, 7470. — 混合聚集体综述。
