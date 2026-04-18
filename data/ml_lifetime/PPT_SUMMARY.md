# 有机锂中间体稳定性预测：从分子机理到反应器推荐

> 12 页 PPT 叙事线：6 个发现，从科学到工程的完整闭环。每页底部附参考文献。

---

## Page 1: 背景与数据集

### 问题
有机锂中间体在流动化学中广泛使用，但热不稳定 [1]。选错反应器 → 中间体分解 → 产率崩塌。

### 目标
**输入 SMILES + 温度 → 推荐 flash / flow / batch 反应器**

### 数据集
- **75 个有机锂中间体**，从 12 篇文献中提取（主要来源：[1–6]）
- **2357 个 yield-vs-tR 数据点** → 动力学拟合 → 40 个 Arrhenius 参数（26 个 tier-1，≥3 温度）
- **42 个化合物有 t_max（最优停留时间）数据**，140 个 (compound, T) 数据点
- **类别分布**：ArLi (45), oxiranylLi (12), carbanion (6), alkylLi (5), 其他 (7)
- **溶剂**：THF / THF-hexane 4:1

### 动力学模型
```
yield(tR) = y_max × (1 - exp(-k_f × tR)) × exp(-k_d × tR)
                     ─────────────────────    ──────────────
                        中间体生成（升）         中间体分解（降）
t_max = ln(k_f/k_d) / (k_f - k_d)   ← 产率峰值位置 = 最优停留时间
```

### 图表
→ 分子结构总览图 (`ppt_fig_molecules_classified.png`)

> [1] De Gennaro 2014, *Lithium Compounds in Organic Synthesis*, Ch.18; [2–6] Nagaki et al. 2007–2019

---

## Page 2: Finding 1 — 机制异质性（跨类别预测失败）

### 核心结论
**不同结构类别的有机锂走完全不同的分解机制，不能用一个模型通吃。**

### 全局 Ea 模型
**Ea = +152.6×q_C − 338.0×d_LiC + 73.1×%Vbur + 700.6**
- LOO-R² = 0.694, MAE = 5.6 kJ/mol, p < 0.001 (n=24)
- 物理意义：电子效应(q_C) + 键强度(d_LiC) + 空间位阻(%Vbur)

### LOCO 验证（训练 2 类 → 预测第 3 类）

| 排除类别 | Ea R² | t½ R² |
|---|---|---|
| m+p ArLi | −0.21 | +0.08 |
| o-ArLi | −0.04 | −3.93 |
| oxiranylLi | **−12.9** | **−10.0** |

R² < 0 = 比直接猜均值还差 → **跨类别预测完全失败**。

### 三种分解机制
- **m+p ArLi**: THF 溶剂裂解（α-H 转移到碳负离子）
- **o-ArLi**: 苯炔消除（C-Br/C-I 断裂 + 环化）
- **oxiranylLi**: 环氧开环（C-O 键断裂 + Li 迁移）

### 图表
→ `finding1_LOCO.png`

> [14] Collum 2007, *Angew. Chem.* — "substrate-dependent mechanisms may be the rule, not the exception"

---

## Page 3: Finding 2 — Ea-lnA 补偿效应

### 核心结论
**Ea 可以从基态描述符预测，但 lnA（∝ ΔS‡）不可以。两者高度相关（r = 0.94），相互抵消导致 t½ 预测失败。**

### Ea-lnA 补偿
| Ea 范围 | lnA | 等效 ΔS‡ | 机制 |
|---|---|---|---|
| 低 Ea（快分解） | 低 lnA | 负 ΔS‡（有序 TS） | 缔合型（环开裂、溶剂参与） |
| 高 Ea（慢分解） | 高 lnA | 正 ΔS‡（无序 TS） | 解离型（苯炔碎裂） |

→ 高 Ea 不一定意味着更稳定，因为高 lnA 会部分补偿。

### 聚集态的贡献 [14]

有机锂在 THF 中以二聚体/四聚体存在 [14,15]。实验 Ea 是**表观活化能**：
```
Ea_obs = Ea_intrinsic(单体) + ΔH_deaggregation + ΔH_solvent_reorganization
```

Ramachandran [7] 的定量数据：

| 状态 | LiMeCl 势垒 | 变化 |
|---|---|---|
| 气相单体 | 24.6 kJ/mol | 基准 |
| 气相二聚体 | 48.2 kJ/mol | **+23.7**（聚集态升高） |
| 溶剂化二聚体·3THF | 17.2 kJ/mol | **-31.1**（溶剂化降低） |

**聚集态和溶剂化效应方向相反、量级接近** → Ea-lnA 补偿的物理本质。

### 后果
- Ea 模型 R²=0.69 → 不能翻译为 t½ 排序（ρ ≈ 0）
- lnA（∝ΔS‡）受聚集态/溶剂动力学主导 → 基态描述符无法捕捉

### 图表
→ `finding2_EalnA_compensation.png`

> [14] Collum 2007; [7] Ramachandran 2010, *J. Phys. Chem. A*; [15] Seebach 1988; [17] Lucht & Collum 1999

---

## Page 4: Finding 3 — 描述符筛选全景

### 核心结论
**28 个描述符 × 6 个计算层级穷举筛选，最廉价的 xTB（秒级）反而最好。**

### 计算层级总览

| 层级 | 方法 | 软件 | 成本 | 描述符数 |
|---|---|---|---|---|
| **Level 0** | 经验参数 | 查表 | 0 | 4 (σ, Es, δ_ortho, δ_benzyne) |
| **Level 1** | GFN2-xTB [8] | tblite + morfeus [9] | **~秒** | 11 |
| **Level 2** | HF/def2-SVP | Psi4 [10] | ~分钟 | 2 |
| **Level 3a** | M06-2X/def2-SVP [7] | Psi4 + Multiwfn [11] | ~10分钟 | 2 |
| **Level 3b** | ADCH (Hirshfeld) | Multiwfn [11] | ~10分钟 | 1 |
| **Level 3c** | QTAIM BCP | Multiwfn [11] | ~20分钟 | 3 |
| **Level 4** | M06-2X TS | ORCA [12] | ~小时-天 | 仅3个化合物 |
| **Sterimol** | 取代基形状 | morfeus [9] | ~秒 | 3 (B1, B5, L) |
| **RDKit** | 分子体积 | RDKit | ~秒 | 1 (mol_volume) |

### xTB 描述符详情

| 描述符 | 物理意义 |
|---|---|
| q(C_ipso) | C-Li 碳上 Mulliken 电荷 → 碳负离子稳定性 |
| d(Li-C) | Li-C 键长 → 键强度 |
| BDE(Li-C) | 键离解能 → 热力学稳定性 |
| %V_bur | Li 周围埋藏体积 → 空间位阻/溶剂可及性 |
| HOMO/LUMO/η | 前线轨道 → 电子反应性 |
| Gsolv | THF 溶剂化自由能 → 溶剂效应 |
| fukui_f⁻(C) | Fukui 函数 → 局部亲核性 |
| Sterimol B1/B5/L | 取代基三维形状 |

### 失败的描述符

| 描述符 | 层级 | 失败原因 |
|---|---|---|
| HF/6-31+G* Mulliken | HF+弥散基 | Li 电荷变负（弥散函数伪影） |
| M06-2X Mulliken | DFT | LOO-R²=0.41（不如 xTB 0.69） |
| ADCH (Hirshfeld) | DFT | LOO-R²=0.40 |
| M06-2X+PCM(THF) | DFT+溶剂 | o-I-ArLi SCF 不收敛 |
| QTAIM ρ(BCP) | DFT+QTAIM | r=0.16 with Ea（无区分度） |
| QTAIM \|V\|/G | DFT+QTAIM | 全部 0.89–0.94（无区分度） |
| pKa(R-H) | 文献 | r=0.04（热力学 ≠ 动力学） |

> [8] Bannwarth 2019 (GFN2-xTB); [10] Smith 2020 (Psi4); [11] Lu 2012 (Multiwfn); [16] Verkhov 2025 (QTAIM/ELF)

---

## Page 5: Finding 3 续 — 为什么 xTB 优于 DFT

### 描述符层级对比

| 方法 | q(C) 来源 | 全局 Ea LOO-R² | 成本 |
|---|---|---|---|
| **xTB (GFN2)** | Mulliken | **0.694** | **秒** |
| HF/def2-SVP | Mulliken | 0.55 | 分钟 |
| M06-2X/def2-SVP | Mulliken | 0.41 | 10 分钟 |
| M06-2X + ADCH | Hirshfeld | 0.40 | 10 分钟 |
| QTAIM | ρ(BCP) | ~0 | 20 分钟 |

### 物理解释
预测瓶颈不在"基态电子结构精度"，而在"溶液相动力学复杂性"——聚集态、溶剂配位、动态熵效应 [14]。更精确的气相电子结构反而引入了与溶液行为不相关的噪声。

### 最终选中的描述符

选中 q(C), d(LiC), Sterimol L 用于 t_max 预测。三者互相关性低（r < 0.3），分别编码**电子效应、键强度、空间保护**——物理意义正交。

### 图表
→ `ppt_fig1_full_correlation_matrix.png`（28 描述符全相关矩阵，绿色框=选中特征）

> [14] Collum 2007 — 溶液相复杂性远超气相精度差异; [8] Bannwarth 2019

---

## Page 6: Finding 4 — 气相 TS 计算不可行

### 核心结论
**从第一性原理精确计算 Ea 在当前不可行。气相 TS 高估 3.8×，加隐式溶剂仅修正 23%。**

### 3 个代表体系 (M06-2X/def2-SVP, ORCA [12])

| 体系 | 机制 | 气相 ΔH‡ | +CPCM | 实验 Ea |
|---|---|---|---|---|
| oxiranylLi | 环开裂 | 134.7 | 111.6 | 35.6 kJ/mol |
| PhLi+THF | α-H 转移 | ~96 | — | 36.5 kJ/mol |
| o-BrPhLi | 苯炔消除 | **无势垒** | — | 72.1 kJ/mol |

### 溶剂化效应分解（oxiranylLi）

```
气相单体 ΔH‡ = 134.7 kJ/mol
  ↓ CPCM(THF): −23.1 kJ/mol (23% of gap)
111.6 kJ/mol
  ↓ 显式 Li·(THF)₂₋₃: 文献报道 3THF 降低 31 kJ/mol [7]
~80 kJ/mol (estimated)
  ↓ 聚集态 + 溶剂动力学: 方向取决于机制 [14]
实验 Ea = 35.6 kJ/mol
```

*Implicit solvation partially corrects the enthalpy barrier but fails to capture entropy contributions, suggesting the importance of explicit solvent interactions and aggregation dynamics.*

要做对需要：显式溶剂 + 聚集态采样 + AIMD → 每个化合物天-周级计算 → **不可扩展**

### 图表
→ `finding4_5_TS_entropy.png` panel (a)

> [7] Ramachandran 2010; [12] Neese 2022 (ORCA); [14] Collum 2007 — "只考虑 TS 端溶剂效应 is complete nonsense"

---

## Page 7: Finding 5 — ΔS‡ 揭示机制多样性

### 核心结论
**实验 ΔS‡ 的符号直接区分缔合型 vs 解离型机制，与 LOCO 验证的跨类失败一致。**

### 三系统 ΔS‡ 对比

| 体系 | 实验 ΔS‡ [J/(mol·K)] | 机制类型 |
|---|---|---|
| oxiranylLi | −121 | 缔合型 TS（溶剂参与决速步） |
| PhLi | −125 | 缔合型 TS（THF 失去平动自由度） |
| o-BrPhLi | **+77** | 解离型 TS（碎裂产生平动熵） |

### ΔS‡ 的多重贡献 [14]
ΔS‡ 是多个效应的叠加，不可简单归因于单一因素：
- **解聚**：释放片段 → 正贡献
- **溶剂重组**：THF 分子被组织到 TS → 负贡献
- **TS 构象受限**：键断裂时的几何约束 → 负贡献
- **混合聚集体效应** [14, Section 3.14]：方向不定

### CPCM 对 ΔS‡ 完全无效
| | 气相 | CPCM(THF) | 实验 |
|---|---|---|---|
| oxiranylLi ΔS‡ | +1.0 | +1.1 | **−121** |

→ 122 J/(mol·K) 的差距 = 解聚 + 溶剂重组 + TS 有序化，静态计算原理上无法捕捉。

### 图表
→ `finding4_5_TS_entropy.png` panel (b)

> [14] Collum 2007; [15] Seebach 1988; [17] Lucht & Collum 1999; [18] Reich 2013

---

## Page 8: 类别特异 Ea 模型（科学层成果）

### Ea 预测模型汇总

#### m+p ArLi (n=12): 共轭稳定碳负离子
| 模型 | 目标 | LOO-R² | p |
|---|---|---|---|
| ω + BDE + B1 → Ea | Ea | **0.755** | <0.001 |
| BDE + H_BCP + vol → log(t½) | t½@-40°C | **0.608** | <0.005 |

#### o-ArLi (n=6): 螯合 + 苯炔消除
| 模型 | 目标 | LOO-R² | p |
|---|---|---|---|
| **σ + B5 → Ea** | Ea | **0.914** | 0.004 |

#### oxiranylLi (n=8): 开环机制
| 模型 | 目标 | LOO-R² | p |
|---|---|---|---|
| **BDE + vol → Ea** | Ea | **0.921** | <0.001 |
| **Gsolv + fukui → log(t½)** | t½@-40°C | **0.939** | <0.001 |

### 验证
- 所有模型通过置换检验 (1000 shuffles, p < 0.005)
- 前线轨道稳健性：去除 HOMO/η 后 R² 降 ~0.1 但模型仍有意义 → η 提供真实信息

### 科学意义
*Activation parameters (Ea, lnA) provide mechanistic insight into the decomposition process.* Ea 模型揭示了不同类别的分解机制由不同物理因素主导：
- m+p ArLi: 亲电性(ω) + 键强度(BDE) + 最小宽度(B1)
- o-ArLi: Hammett 电子效应(σ) + 最大宽度(B5)
- oxiranylLi: 键强度(BDE) + 分子体积(vol) 或 溶剂化(Gsolv)

> [1] De Gennaro 2014; [13] Hansch 1991 (Hammett σ)

---

## Page 9: Finding 6 — 反应器推荐工具（工程层成果）

### 核心结论
**绕过 Ea-lnA 补偿，直接预测 t_max → flash/flow/batch 分类，89.3% 准确率。**

### 为什么用 t_max 而不是 t½
| | Ea/lnA → t½ | 直接预测 t_max |
|---|---|---|
| 需要预测 | Ea + lnA (两个参数) | t_max (一个可观测量) |
| Ea-lnA 补偿 | 致命问题 | **绕过** |
| 数据量 | 26 化合物 | **140 数据点** |
| 工程意义 | 间接 | **= 最优停留时间** |

### 模型
```
log₁₀(t_max/s) = 10.86·q(C) − 13.09·d(Li-C) + 0.48·L + 336/T + 21.93
```

### 分类边界
| 反应器 | t_max 范围 | 对应设备 |
|---|---|---|
| **flash** | < 0.1 s | 微混合器 (T-mixer, ~ms 混合) |
| **flow** | 0.1 – 60 s | 管式反应器 |
| **batch** | > 60 s | 常规烧瓶操作 |

### 性能
- LOO-CV 分类准确率 = **89.3%** (125/140)
- 数据: 42 化合物 × 多温度 = 140 个数据点
- 温度范围: −78 ~ +25°C（边界不随温度变化，是设备物理限制）

### 图表
→ `ppt_fig2_model_performance.png`（log-log parity + 特征重要性排名）

### 特征物理意义
| 描述符 | 重要性 | 系数 | 物理意义 |
|---|---|---|---|
| q(C_ipso) | 0.413 | + | C 上正电荷越多 → C-Li 键越稳定 → t_max 越长 |
| d(Li-C) | 0.298 | − | 键越长 → 越弱 → 分解越快 → t_max 越短 |
| Sterimol L | 0.174 | + | 取代基越长 → 空间保护 → t_max 越长 |
| 1/T | 0.115 | + | 温度越低 → Arrhenius 减速 → t_max 越长 |

> [8] Bannwarth 2019 (GFN2-xTB); [9] Falivene 2016 (Sterimol)

---

## Page 10: 建模方法论与自我审视

### 建模流程

```
         ┌──────────────┐
         │ SMILES 输入   │
         └──────┬───────┘
                ↓
    ┌───────────────────────┐
    │ xTB 几何优化 (GFN2)   │ ← 秒级
    │ • 气相单体几何         │
    └───────────┬───────────┘
                ↓
   ┌────────────┼────────────┐
   ↓            ↓            ↓
电子描述符   键/结构描述符  溶剂/体积
q_C, HOMO    d_LiC, BDE    Gsolv, vol
ω, η         %Vbur, B1/B5  fukui
   └────────────┼────────────┘
                ↓
    ┌───────────────────────┐
    │ + 温度 (1/T)          │
    └───────────┬───────────┘
                ↓
    ┌───────────────────────┐
    │ log₁₀(t_max) 回归     │
    │ → flash/flow/batch    │
    └───────────────────────┘
```

### 已验证的判断 ✓

| 判断 | 证据 | 文献支撑 |
|---|---|---|
| 类别特异建模 | LOCO R² < -12 | Collum [14] |
| xTB 优于 DFT | R²: 0.69 > 0.41 > ~0 | 瓶颈在溶液相复杂性 [14] |
| CPCM 不足 | 仅修正 23%，ΔS‡ 修正 0% | Ramachandran [7] |
| ΔS‡ 区分机制 | 缔合(−121) vs 解离(+77) | Collum [14] Table 1 |
| t_max > t½ | 分类准确率 89% vs 76% | 绕过 Ea-lnA 补偿 |

### 待解决问题 ✗

**问题 1: 聚集态未纳入**
- 描述符基于气相单体 → 聚集态信息缺失
- 部分缓解：类别内聚集态相似

**问题 2: ΔS‡ 多重贡献**
- ΔS‡ = ΔS_deagg(+) + ΔS_solvent(−) + ΔS_conform(−)
- 这是 lnA 不可预测的根本原因

**问题 3: batch 类数据缺失**
- 训练集全部来自流动化学文献 → 无 batch 化合物
- "稳定"化合物只有 censored data (t_max > tR_window)

> [14] Collum 2007; [7] Ramachandran 2010

---

## Page 11: 有机锂聚集态——溶液中的隐藏变量

### 聚集态基本规律 [14,15,17]

| 有机锂 | 烃类溶剂 | Et₂O | THF |
|---|---|---|---|
| MeLi | 四聚体 | 四聚体 | — |
| n-BuLi | **六聚体** | 四聚体 | **二聚体** |
| s-BuLi | — | 二聚体 | 二聚体 |
| t-BuLi | 四聚体 | 二聚体 | **单体** |
| PhLi | — | 二聚体 | **二聚体** |
| LDA | — | — | 二溶剂化二聚体 (A₂S₂) |

规律：溶剂配位越强 → 聚集度越低；位阻越大 → 聚集度越低

### Collum 速率方程 [14]
```
kobs = k × [A₂S₂]^a × [S]^b

a = 1/2 → 单体路径（解聚）
a = 1   → 二聚体路径（不解聚）  ← >60% 的 LDA 反应!
b = 0   → 不需要额外溶剂       ← 60% 的速率方程
```

### 对我们模型的影响
- 我们的 75 个中间体在 THF/hexane 4:1 中的聚集态**未知**
- 模型隐式假设类别内聚集行为相似 → 类别内有效、跨类失败
- 这可能是 LOCO 验证失败的**部分**原因（不仅是机制不同，聚集态也不同）

> [14] Collum 2007; [15] Seebach 1988; [17] Lucht & Collum 1999; [18] Reich 2013; [19] Harrison-Marchand 2013

---

## Page 12: 总结 — 从分子到工程的完整闭环

### 三层结构

```
科学层 (WHY)                 解释层 (WHY NOT ab initio)    工程层 (HOW TO USE)

Finding 1: 机制异质性          Finding 4: TS 高估 3.8×        Finding 6: t_max 工具
  LOCO R² < -12                 CPCM 仅修正 23%               89.3% 准确率
                                                              flash/flow/batch
Finding 2: Ea-lnA 补偿        Finding 5: ΔS‡ 区分机制
  r=0.94, lnA 不可预测          缔合(-121) vs 解离(+77)

Finding 3: xTB > DFT
  28 描述符, 6 层级
```

### 论文核心论述

*Activation parameters (Ea, lnA) provide mechanistic insight into the decomposition process, whereas t_max directly determines optimal operating conditions in flow systems. The combination of mechanistic analysis and data-driven prediction forms a complete framework from molecular understanding to process design.*

### 当前局限
- 数据量（42 化合物）是主要瓶颈 → 更多实验数据 >> 更高级计算方法
- "batch" 类化合物在训练集中缺失（来自流动化学文献）
- 聚集态信息未显式纳入（类别内相似假设部分有效）
- The poor predictability of lnA *suggests* that entropy contributions are more sensitive to factors not captured by ground-state descriptors, such as aggregation and solvent dynamics.

### 潜在改进方向
- 扩充数据集（特别是杂环 ArLi、batch 稳定性数据）
- 聚集态描述符（NMR 表征 + 计算）
- Censored data 利用（formation_only 的 100 个数据点 → 生存分析）

> [1] De Gennaro 2014; [14] Collum 2007; [7] Ramachandran 2010

---

## 参考文献

[1] De Gennaro, L.; Fanelli, F.; Luisi, R. *Lithium Compounds in Organic Synthesis*, Wiley, 2014, Ch.18. — 有机锂分解动力学综述。

[2] Nagaki, A. et al. *J. Am. Chem. Soc.* **2014**, 136, 12245. — ArLi 热稳定性。

[3] Nagaki, A. et al. *Angew. Chem. Int. Ed.* **2019**, 58, 4027. — oxiranylLi 动力学。

[4] Nagaki, A. et al. *J. Flow Chem.* **2008**. — o-CO₂R-ArLi Arrhenius。

[5] Musci, P. et al. *React. Chem. Eng.* **2020**, 5, 935. — 卡宾类动力学。

[6] Stanetty, P.; Mihovilovic, M. *J. Org. Chem.* **1997**, 62, 1514. — 有机锂半衰期。

[7] Ramachandran, B. et al. *J. Phys. Chem. A* **2010**, 114, 8423. — M06-2X 基准；聚集态+24, 溶剂化-31 kJ/mol。

[8] Bannwarth, C. et al. *J. Chem. Theory Comput.* **2019**, 15, 1652. — GFN2-xTB。

[9] Falivene, L. et al. *Organometallics* **2016**, 35, 2286. — %Vbur, Sterimol。

[10] Smith, D. G. A. et al. *J. Chem. Phys.* **2020**, 152, 184108. — Psi4。

[11] Lu, T.; Chen, F. *J. Comput. Chem.* **2012**, 33, 580. — Multiwfn。

[12] Neese, F. *WIREs Comput. Mol. Sci.* **2022**, 12, e1606. — ORCA。

[13] Hansch, C. et al. *Chem. Rev.* **1991**, 91, 165. — Hammett σ。

[14] Collum, D. B. et al. *Angew. Chem. Int. Ed.* **2007**, 46, 3002. — 有机锂溶液动力学经典综述。

[15] Seebach, D. *Angew. Chem. Int. Ed. Engl.* **1988**, 27, 1624. — 聚集态晶体学。

[16] Verkhov, V. A. et al. *J. Chem. Phys.* **2025**, 162, 044114. — QTAIM/ELF。

[17] Lucht, B. L.; Collum, D. B. *Acc. Chem. Res.* **1999**, 32, 1035. — 溶液结构。

[18] Reich, H. J. *Chem. Rev.* **2013**, 113, 7130. — 聚集态与反应性。

[19] Harrison-Marchand, A.; Mongin, F. *Chem. Rev.* **2013**, 113, 7470. — 混合聚集体。
