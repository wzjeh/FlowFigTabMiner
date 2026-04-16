# 有机锂中间体分解动力学预测：模型与描述符筛选全景

> 本文档汇总所有建模尝试、描述符对比和结论，用于制作 PPT。

---

## 1. 问题定义

### 目标
从有机锂中间体的分子结构（SMILES）预测其在 THF 溶液中的分解动力学参数：
- 活化能 Ea (kJ/mol)
- 半衰期 t½ (s)（在指定温度下）

### 应用场景
流动化学反应器选型：flash mixing (<1 s) / flow reactor (1–1000 s) / batch (>1000 s)

### 数据集
- **75 个有机锂中间体**，从 12 篇文献中提取
- **40 个有 Arrhenius 参数**（Ea + lnA），其中 26 个为可靠数据（≥3 个温度拟合，R² > 0.95）
- **类别分布**：ArLi (45), oxiranylLi (12), carbanion (6), alkylLi (5), 其他 (7)
- **溶剂**：THF / THF-hexane 4:1（所有数据统一）

---

## 2. 描述符计算方法总览

我们在四个理论层级上计算了描述符：

| 层级 | 方法 | 软件 | 成本/化合物 | 描述符数量 |
|---|---|---|---|---|
| **Level 0** | 文献经验参数 | 查表 | 0 | 3 (σ, Es, δ) |
| **Level 1** | GFN2-xTB 半经验 | tblite + morfeus | ~秒 | 12 |
| **Level 2** | HF/def2-SVP | Psi4 | ~分钟 | 3 |
| **Level 3** | M06-2X/def2-SVP | Psi4 + Multiwfn | ~10分钟 | 5 |
| **Level 4** | M06-2X/def2-SVP TS | ORCA | ~小时-天 | 3 (仅3个化合物) |

### Level 1: xTB 描述符（最终采用）

| 描述符 | 物理意义 | 计算方法 |
|---|---|---|
| q(C_ipso) | C-Li 碳上的 Mulliken 电荷 → 碳负离子稳定性 | GFN2-xTB |
| d(Li-C) | Li-C 键长 → 键强度 | xTB 优化几何 |
| %V_bur | Li 周围的埋藏体积 → 空间位阻 | morfeus (r=3.5 Å) |
| BDE(Li-C) | Li-C 键离解能 → 热力学稳定性 | E(R·) + E(Li·) - E(RLi) |
| xtb_omega | 亲电性指数 μ²/(2η) | HOMO/LUMO |
| xtb_eta | 化学硬度 (LUMO-HOMO)/2 | HOMO/LUMO |
| HOMO | 最高占据轨道能量 | GFN2-xTB |
| Gsolv | THF 中溶剂化自由能 | xTB ALPB(THF) |
| fukui_f⁻(C) | C_ipso 的亲电 Fukui 函数 → 局部亲核性 | xTB (N vs N-1 电子) |
| Sterimol B1/B5/L | 取代基最小宽度/最大宽度/长度 | morfeus |
| mol_volume | 分子体积 → 溶剂重组熵的代理 | RDKit |

### Level 0: 经验参数（仅 ArLi 适用）

| 参数 | 意义 |
|---|---|
| σ (Hammett) | 取代基电子效应 |
| Es (Taft) | 取代基位阻效应 |
| δ_ortho | 邻位标志 (0/1) |

---

## 3. 尝试过的所有描述符和模型

### 3.1 经验 LFER 模型（σ + Es + δ）

来源：De Gennaro 2014 综述中的 Hammett-Taft 四参数模型。

| 模型 | 数据 | 结果 |
|---|---|---|
| σ + Es + δ_ortho + δ_benzyne → Ea | 文献 12 个 ArLi | LOO-R² = **-0.26** (失败) |
| σ + B5 → Ea (o-ArLi only, n=6) | 本工作 | LOO-R² = **0.914** (成功) |

**结论**：旧的四参数 LFER 在我们的新数据集上失败，因为它假设所有 ArLi 走同一机制。但 Hammett σ 对 o-ArLi 子类仍然有效。

### 3.2 xTB 全局模型（跨类别）

| 模型 | n | LOO-R² | MAE | p (置换检验) |
|---|---|---|---|---|
| **q_C + d_LiC + %Vbur → Ea** | 26 | **0.694** | 5.6 kJ/mol | <0.001 |
| q_C + d_LiC → Ea | 26 | 0.52 | — | — |
| BDE + vol → Ea | 26 | 0.45 | — | — |

最佳全局模型：**Ea = +152.6×q_C − 338.0×d_LiC + 73.1×%Vbur + 700.6**
物理意义：电子效应(q_C) + 键强度(d_LiC) + 空间位阻(%Vbur)

**局限**：全局模型的 Ea 预测合理，但 t½ 排序（ρ ≈ 0 at -40°C）因为 Ea-lnA 补偿效应 + 指数放大导致失效。

### 3.3 类别特异模型

#### m+p ArLi (n=12): 共轭稳定碳负离子

| 模型 | 目标 | LOO-R² | p |
|---|---|---|---|
| ω + BDE + B1 → Ea | Ea | **0.755** | <0.001 |
| BDE + H_BCP + vol → log(t½) | t½@-40°C | **0.608** | <0.005 |
| η + H_BCP + vol → log(t½) | t½@-40°C | 0.716 (含 η) | 0.001 |

#### o-ArLi (n=6): 螯合 + 苯炔消除

| 模型 | 目标 | LOO-R² | p |
|---|---|---|---|
| **σ + B5 → Ea** | Ea | **0.914** | 0.004 |
| HOMO + η → log(t½) | t½@-40°C | 0.974 (含前线轨道) | 0.003 |
| vol + BDE → log(t½) | t½@-40°C | 0.573 (无前线轨道) | — |

#### oxiranylLi (n=8): 开环机制

| 模型 | 目标 | LOO-R² | p |
|---|---|---|---|
| **BDE + vol → Ea** | Ea | **0.921** | <0.001 |
| **Gsolv + fukui → log(t½)** | t½@-40°C | **0.939** | <0.001 |

### 3.4 失败的描述符尝试

| 描述符 | 理论层级 | 失败原因 |
|---|---|---|
| HF/6-31+G* Mulliken charge | HF + 弥散基 | Li 电荷变为负值（弥散函数导致电荷分配失物理意义） |
| HF/def2-SVP Mulliken charge | HF | LOO-R² = 0.55（不如 xTB 的 0.69） |
| M06-2X/def2-SVP Mulliken charge | DFT | LOO-R² = 0.41（更差） |
| M06-2X/def2-SVP ADCH charge (Hirshfeld) | DFT + Multiwfn | LOO-R² = 0.40 |
| M06-2X/def2-SVPD + PCM(THF) charge | DFT + 溶剂 | o-I-ArLi SCF 不收敛 |
| QTAIM ρ(BCP) | DFT + QTAIM | r = 0.16 with Ea（无区分度） |
| QTAIM \|V\|/G ratio | DFT + QTAIM | 所有值 0.89–0.94（无区分度） |
| pKa(R-H) estimated | 文献 | r = 0.04 with Ea（热力学 ≠ 动力学） |
| Δq(C) 过程描述符 | xTB | r = -0.53，但不优于静态 q_C |
| ΔG‡ 作为目标 | — | 等价于预测 t½，无额外信息 |

### 3.5 描述符层级对比总结

| 方法 | 代表描述符 | 全局 Ea LOO-R² | 成本 | 结论 |
|---|---|---|---|---|
| **xTB (GFN2)** | q_C, d_LiC, %Vbur | **0.694** | 秒 | **最佳性价比** |
| HF/def2-SVP | Mulliken charge | 0.55 | 分钟 | 更贵但更差 |
| M06-2X/def2-SVP | Mulliken charge | 0.41 | 10 分钟 | 最贵且最差 |
| M06-2X + ADCH | Hirshfeld charge | 0.40 | 10 分钟 | 同上 |
| QTAIM | ρ(BCP) | ~0 | 20 分钟 | 完全失败 |

**关键发现：更昂贵的理论层级并不带来更好的描述符。** xTB 的半经验方法虽然近似，但对有机锂体系的相对趋势捕捉得最好。DFT 的 Mulliken 电荷对基组更敏感（噪声更大），反而降低了预测能力。

---

## 4. 验证策略

### 4.1 Leave-One-Out Cross-Validation (LOO-CV)
所有模型均通过 LOO-CV 评估。

### 4.2 置换检验 (Permutation Test)
1000 次随机打乱 y 值，计算 p 值。所有报告模型 **p < 0.005**，确认非统计巧合。

### 4.3 Leave-One-Class-Out (LOCO) — 核心验证

用全局模型 (q_C + d_LiC + %Vbur) 训练 2 个类别 → 预测第 3 个类别：

| 被排除的类别 | Ea R² | t½@-40°C R² |
|---|---|---|
| m+p ArLi | **-0.21** | +0.08 |
| o-ArLi | **-0.04** | **-3.93** |
| oxiranylLi | **-12.9** | **-10.0** |

**R² < 0 意味着预测比"直接用均值"更差。**

**结论：跨类别预测完全失败 → 不同类别的有机锂走完全不同的分解机制 → 必须分类别建模。**

### 4.4 前线轨道描述符稳健性检验

HOMO 和 η（化学硬度）可能编码了 ΔG‡ 信息（因为 HOMO ∝ IP → ΔG‡）。测试去除后影响：

| m+p ArLi t½ 模型 | LOO-R² | 含 HOMO/η？ |
|---|---|---|
| η + H_BCP + vol | 0.716 | 是 |
| **BDE + H_BCP + vol** | **0.608** | **否** |

去除 η 后 R² 降 0.11 但模型仍有意义 → η 提供了真实物理信息，非纯编码。

---

## 5. Ea-lnA 补偿效应

### 问题
Ea 和 lnA 高度相关（r = 0.96）：

| Ea (kJ/mol) | lnA | 等效 ΔS‡ |
|---|---|---|
| 低 Ea（快速分解） | 低 lnA（小 A） | 负 ΔS‡（有序 TS） |
| 高 Ea（慢速分解） | 高 lnA（大 A） | 正 ΔS‡（无序 TS） |

这意味着：**高 Ea 不一定意味着更稳定**，因为高 lnA 会部分补偿。

### 后果
- Ea 模型 (R²=0.69) 不能直接翻译为 t½ 排序（ρ ≈ 0）
- 必须单独建模 t½（或同时预测 Ea 和 lnA）
- lnA（∝ΔS‡）取决于过渡态结构和溶剂化动力学 → 无法从基态描述符完全预测

### 物理解释
- 负 ΔS‡：缔合型 TS（溶剂参与，如 PhLi + THF → TS）
- 正 ΔS‡：解离型 TS（碎裂，如 o-BrPhLi → 苯炔 + LiBr）
- **不同类别的有机锂有不同符号的 ΔS‡ → 机制本质不同**

---

## 6. 过渡态计算验证

### 目的
用 DFT 过渡态计算验证"为什么基态描述符比精确 TS 计算更实用"。

### 方法
M06-2X/def2-SVP, ORCA 6.1.1, NEB-TS → OptTS + Freq, 298.15 K

### 三个代表体系

| 体系 | 机制 | 计算方法 |
|---|---|---|
| oxiranylLi | 环氧开环 (单分子) | NEB-TS → OptTS + Freq ✓ |
| PhLi + THF | THF α-H 转移到 PhLi (双分子) | 松弛 PES scan |
| o-BrPhLi | 苯炔消除 (碎裂) | NEB-TS |

### 结果

| 体系 | 气相 ΔH‡ | +CPCM(THF) ΔH‡ | 实验 Ea | 气/实 | CPCM/实 |
|---|---|---|---|---|---|
| oxiranylLi | 134.7 kJ/mol | **111.6 kJ/mol** | 35.6 | 4.1× | **3.4×** |
| PhLi+THF | ~96 kJ/mol | — | 36.5 | 2.6× | — |
| o-BrPhLi | **无势垒** (吸热) | — | 72.1 | — | — |

### ΔS‡ 对比（最关键的发现）

| | 气相计算 | CPCM(THF) | 实验 |
|---|---|---|---|
| oxiranylLi ΔS‡ | +1.0 J/(mol·K) | **+1.1** | **-121** |

**CPCM 对 ΔS‡ 完全无效（0% 改善）。** 这证明 122 J/(mol·K) 的熵差距完全来自溶剂重组/聚集态效应，只有分子动力学才能捕捉。

### 三种机制的 ΔS‡ 特征

| 体系 | 实验 ΔS‡ | 意义 |
|---|---|---|
| oxiranylLi | −121 J/(mol·K) | 缔合型 TS：溶剂在 TS 中被组织/消耗 |
| PhLi | −125 J/(mol·K) | 缔合型 TS：THF 失去平动自由度 |
| o-BrPhLi | **+77** J/(mol·K) | 解离型 TS：碎裂产生平动熵 |

### 溶剂化效应分解（oxiranylLi）

```
气相 ΔH‡ = 134.7 kJ/mol
  ↓ + 隐式溶剂 CPCM(THF) — 介电稳定
111.6 kJ/mol (−23.1 kJ/mol, 回收了 23% 的差距)
  ↓ + 显式 Li·(THF)₂₋₃ 配位 — 未计算
~80 kJ/mol (估计再降 ~30 kJ/mol)
  ↓ + 聚集态 + 溶剂动力学 — 需要 AIMD
~50 kJ/mol (估计再降 ~30 kJ/mol)
  ↓
实验 ΔH‡ = 33.1 kJ/mol
```

**结论：即使加入隐式溶剂模型，仍高估 3.4×。溶剂不仅是"介电背景"，而是通过配位变化、聚集态、熵效应直接参与反应。**

---

## 7. 核心结论

### 7.1 机制异质性
不同类别的有机锂走完全不同的分解机制：
- **oxiranylLi**：环开裂（缔合型 TS，ΔS‡ < 0）
- **m+p ArLi (PhLi)**：THF 溶剂裂解（缔合型 TS，ΔS‡ < 0）
- **o-ArLi (o-BrPhLi)**：苯炔消除（解离型 TS，ΔS‡ > 0）

**统计证据**：LOCO R² < -12 证明跨类别预测完全失败。

### 7.2 描述符层级悖论
更昂贵的 DFT 描述符并不比廉价的 xTB 描述符更好：
- xTB Mulliken charge → LOO-R² = 0.69
- M06-2X Mulliken charge → LOO-R² = 0.41
- QTAIM ρ(BCP) → r ≈ 0

原因：预测模型的瓶颈不在"基态描述精度"，而在"无法捕捉过渡态和溶剂效应"。xTB 的近似反而减少了过拟合噪声。

### 7.3 TS 计算不可行
- 气相 TS 高估 2.6–4.1×
- +CPCM(THF) 仅修正 23%，ΔS‡ 修正 0%
- 苯炔消除在气相无过渡态
- 精确 TS 需要微溶剂化 + AIMD（天/化合物），无法扩展到 75 个化合物

### 7.4 推荐策略
**类别特异的基态描述符 QSPR 是唯一可行路线**：
- 秒级计算成本（xTB）
- 类别内 LOO-R² = 0.61–0.94
- 描述符（Gsolv, mol_volume）隐式编码了 TS 计算无法捕捉的溶剂效应
- 已部署为预测工具：SMILES → Ea + t½ + 反应器推荐

### 7.5 当前局限
- 数据量小（n=6–26 per class）是主要瓶颈
- lnA（∝ΔS‡）无法从基态描述符完全预测 → t½ 精度受限
- 对新结构类别（如杂环 ArLi）需要新的实验数据

---

## 8. 所有模型性能汇总表

| 模型 | 类别 | n | 目标 | 描述符 | LOO-R² | p | 备注 |
|---|---|---|---|---|---|---|---|
| 全局 3-param | All | 26 | Ea | q_C + d_LiC + %Vbur | 0.694 | <0.001 | 最佳全局 |
| LFER 4-param | ArLi | 12 | Ea | σ + Es + δ_ortho + δ_benzyne | -0.26 | — | 失败 |
| m+p Ea | m+p ArLi | 12 | Ea | ω + BDE + B1 | 0.755 | <0.001 | |
| m+p t½ (robust) | m+p ArLi | 12 | t½@-40°C | BDE + H_BCP + vol | 0.608 | <0.005 | 无 HOMO/η |
| m+p t½ (with η) | m+p ArLi | 12 | t½@-40°C | η + H_BCP + vol | 0.716 | 0.001 | 含 η |
| o-ArLi Ea | o-ArLi | 6 | Ea | σ + B5 | 0.914 | 0.004 | Hammett 有效 |
| o-ArLi t½ | o-ArLi | 6 | t½@-40°C | HOMO + η | 0.974 | 0.003 | n 小,慎用 |
| oxiranylLi Ea | oxiranylLi | 8 | Ea | BDE + vol | 0.921 | <0.001 | |
| oxiranylLi t½ | oxiranylLi | 8 | t½@-40°C | Gsolv + fukui | 0.939 | <0.001 | 最佳 t½ |
| HF charge model | All | 26 | Ea | HF q_C + d_LiC + %Vbur | 0.55 | — | 不如 xTB |
| M06-2X charge model | All | 26 | Ea | DFT q_C + d_LiC + %Vbur | 0.41 | — | 不如 xTB |
| QTAIM model | All | 26 | Ea | ρ_BCP + d_LiC + %Vbur | ~0 | — | 完全失败 |
| 气相 TS | oxiranylLi | 1 | ΔH‡ | M06-2X NEB-TS | — | — | 4.1× 高估 |
| CPCM TS | oxiranylLi | 1 | ΔH‡ | M06-2X + CPCM | — | — | 3.4× 高估 |

---

## 9. PPT 建议结构

1. **背景**：有机锂中间体在流动化学中的重要性 → 需要稳定性预测
2. **数据集**：75 个中间体，40 Arrhenius 参数，5 个结构类别
3. **描述符计算**：xTB → HF → M06-2X → QTAIM 四个层级
4. **关键发现 1**：更贵的描述符不更好（xTB R²=0.69 > DFT R²=0.41）
5. **关键发现 2**：LOCO 证明跨类别预测失败 → 机制异质性
6. **关键发现 3**：Ea-lnA 补偿 → 类别特异模型
7. **类别模型**：o-ArLi R²=0.91, oxiranylLi R²=0.92/0.94
8. **TS 计算验证**：三个体系，气相高估 2.6-4.1×，CPCM 仅修正 23%
9. **ΔS‡ 揭示机制**：缔合 vs 解离，CPCM 对熵无效
10. **结论**：基态描述符 QSPR 是唯一可行且有效的预测路线
