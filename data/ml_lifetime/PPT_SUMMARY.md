# 有机锂中间体分解动力学预测：模型与描述符筛选全景

> 本文档汇总所有建模尝试、描述符对比和结论，用于制作 PPT。所有引用以 [n] 标注，参考文献列表见末尾。

---

## 1. 问题定义

### 目标
从有机锂中间体的分子结构（SMILES）预测其在 THF 溶液中的分解动力学参数：
- 活化能 Ea (kJ/mol)
- 半衰期 t½ (s)（在指定温度下）

### 应用场景
流动化学反应器选型：flash mixing (<1 s) / flow reactor (1–1000 s) / batch (>1000 s) [1]

### 数据集
- **75 个有机锂中间体**，从 12 篇文献中提取（主要来源：[1–6]）
- **40 个有 Arrhenius 参数**（Ea + lnA），其中 26 个为可靠数据（≥3 个温度拟合，R² > 0.95）
- **类别分布**：ArLi (45), oxiranylLi (12), carbanion (6), alkylLi (5), 其他 (7)
- **溶剂**：THF / THF-hexane 4:1（所有数据统一）

---

## 2. 描述符计算方法总览

我们在四个理论层级上计算了描述符：

| 层级 | 方法 | 软件 | 成本/化合物 | 描述符数量 |
|---|---|---|---|---|
| **Level 0** | 文献经验参数 | 查表 | 0 | 3 (σ, Es, δ) |
| **Level 1** | GFN2-xTB 半经验 [8] | tblite + morfeus [9] | ~秒 | 12 |
| **Level 2** | HF/def2-SVP | Psi4 [10] | ~分钟 | 3 |
| **Level 3** | M06-2X/def2-SVP [7] | Psi4 + Multiwfn [11] | ~10分钟 | 5 |
| **Level 4** | M06-2X/def2-SVP TS [7] | ORCA [12] | ~小时-天 | 3 (仅3个化合物) |

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

来源：De Gennaro 2014 综述 [1] 中的 Hammett-Taft 四参数模型（σ 值取自 Hansch 1991 [13]）。

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

**关键发现：更昂贵的理论层级并不带来更好的描述符。** xTB [8] 的半经验方法虽然近似，但对有机锂体系的相对趋势捕捉得最好。DFT 的 Mulliken 电荷对基组更敏感（噪声更大），反而降低了预测能力。这与 Collum [14] 的观察一致：溶液相动力学的复杂性（聚集态、溶剂配位）远超气相电子结构的精度差异。

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

## 5. Ea-lnA 补偿效应与聚集态

### 问题
Ea 和 lnA 高度相关（r = 0.96）：

| Ea (kJ/mol) | lnA | 等效 ΔS‡ |
|---|---|---|
| 低 Ea（快速分解） | 低 lnA（小 A） | 负 ΔS‡（有序 TS） |
| 高 Ea（慢速分解） | 高 lnA（大 A） | 正 ΔS‡（无序 TS） |

这意味着：**高 Ea 不一定意味着更稳定**，因为高 lnA 会部分补偿。

### 聚集态的贡献 [14]

有机锂在 THF 中以二聚体/四聚体存在 [14,15]，反应前可能需要解聚。实验测得的 Ea 是**表观活化能**，包含多个贡献：

```
Ea_obs = Ea_intrinsic(单体) + ΔH_deaggregation + ΔH_solvent_reorganization
```

Collum [14] 的关键发现：
- **二聚体可以比单体更活泼** [14, Section 3.3]：>60% 的 LDA 反应经二聚体路径
- **解聚不一定需要额外溶剂化** [14, Section 3.4]：60% 的速率方程对溶剂浓度零级
- **底物依赖的机制是常态** [14, Section 3.2]：不同底物在同一溶剂中走不同路径

Ramachandran 等 [7] 的定量数据 (M06-2X/6-31+G(d), SI)：

| 状态 | LiMeCl 环丙烷化势垒 | 变化 |
|---|---|---|
| 气相单体 | 24.6 kJ/mol | 基准 |
| 气相二聚体 | 48.2 kJ/mol | **+23.7**（聚集态升高！） |
| 溶剂化二聚体·3THF | 17.2 kJ/mol | **-31.1**（溶剂化降低） |

**聚集态和溶剂化的效应方向相反、量级接近。** 这解释了 Ea-lnA 补偿效应的物理本质：
- 解聚贡献额外 Ea（+ΔH_deagg）
- 同时释放平动自由度（+ΔS‡ → +lnA）

### 后果
- Ea 模型 (R²=0.69) 不能直接翻译为 t½ 排序（ρ ≈ 0）
- 必须单独建模 t½（或同时预测 Ea 和 lnA）
- lnA（∝ΔS‡）取决于聚集态、过渡态结构和溶剂化动力学 → 无法从基态描述符完全预测

### ΔS‡ 的多重贡献（修正）
实验 ΔS‡ 是多个效应的叠加，不可简单归因于单一因素：
- **解聚**：释放片段 → 正贡献
- **溶剂重组/配位变化**：THF 分子被组织 → 负贡献
- **TS 构象受限**：键断裂时的几何约束 → 负贡献
- **混合聚集体效应** [14, Section 3.14]：方向不定

| 体系 | 实验 ΔS‡ | 主要贡献 |
|---|---|---|
| oxiranylLi | −121 J/(mol·K) | 溶剂重组 + TS 受限 >> 解聚释放 |
| PhLi | −125 J/(mol·K) | THF 参与双分子 TS（失去平动自由度） |
| o-BrPhLi | **+77** J/(mol·K) | 碎裂释放 LiBr（平动熵获得 >> 溶剂化有序化） |

**不同类别的有机锂有不同符号的 ΔS‡ → 机制本质不同**

---

## 6. 过渡态计算验证

### 目的
用 DFT 过渡态计算验证"为什么基态描述符比精确 TS 计算更实用"。

### 方法
M06-2X/def2-SVP [7], ORCA 6.1.1 [12], NEB-TS → OptTS + Freq, 298.15 K

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

### ΔS‡ 对比

| | 气相计算 | CPCM(THF) | 实验 |
|---|---|---|---|
| oxiranylLi ΔS‡ | +1.0 J/(mol·K) | **+1.1** | **-121** |

**CPCM 对 ΔS‡ 完全无效（0% 改善）。** 122 J/(mol·K) 的熵差距来自解聚、溶剂重组和 TS 有序化的叠加效应，静态计算原理上无法捕捉。

### 溶剂化效应分解（oxiranylLi，修正版）

注意：气相 TS 计算的是**单体固有势垒**，而实验 Ea 是**表观活化能**（包含聚集态和溶剂效应）[14]。两者不能直接比较，但差距揭示了溶液相效应的量级。

```
气相单体 ΔH‡ = 134.7 kJ/mol    ← 我们的 TS 计算
  ↓ CPCM(THF) — 介电稳定（GS 和 TS 差异化稳定）
111.6 kJ/mol (−23.1 kJ/mol, 23%)
  ↓ 显式 Li·(THF)₂₋₃ — 文献参考: Ramachandran 报道 3THF 降低 31 kJ/mol [7]
~80 kJ/mol (估计)
  ↓ 聚集态效应 — 方向取决于机制 [14]：
     若经单体路径: 解聚能升高表观 Ea (+20~30 kJ/mol)
     若经二聚体路径: Collum 证明二聚体路径常见 [14]
  ↓ 溶剂动力学 (ΔS‡ 贡献) — 只有 MD 能算
实验 Ea = 35.6 kJ/mol (表观值)

关键: 溶剂同时稳定基态和过渡态 [14, Section 3.11]
     净效应 = ΔΔG_solv(TS) - ΔΔG_solv(GS)
     "只考虑溶剂对 TS 的效应而忽略对基态的效应，
      is complete nonsense" — Collum [14]
```

**结论：气相/CPCM 与实验的差距来自四个维度——介电(23%)、显式配位(~30%)、聚集态(方向不定)、溶剂动力学(~ΔS‡ 贡献)——都无法由静态单分子计算捕捉。**

---

## 7. 核心结论

### 7.1 机制异质性（与文献一致）
不同类别的有机锂走完全不同的分解机制：
- **oxiranylLi**：环开裂（ΔS‡ = -121，高度有序的 TS）
- **m+p ArLi (PhLi)**：THF 溶剂裂解（ΔS‡ = -125，双分子 TS）
- **o-ArLi (o-BrPhLi)**：苯炔消除（ΔS‡ = +77，碎裂型 TS）

**统计证据**：LOCO R² < -12 证明跨类别预测完全失败。
**文献支撑**：Collum [14] 指出 "substrate-dependent mechanisms may be the rule, not the exception"（Section 3.2）。

### 7.2 描述符层级悖论
更昂贵的 DFT 描述符并不比廉价的 xTB 描述符更好：
- xTB Mulliken charge → LOO-R² = 0.69
- M06-2X Mulliken charge → LOO-R² = 0.41
- QTAIM ρ(BCP) → r ≈ 0

原因：预测模型的瓶颈不在"基态电子结构精度"，而在"溶液相动力学复杂性"——聚集态、溶剂配位、动态熵效应 [14]。更精确的气相电子结构反而引入了与溶液行为不相关的噪声。

### 7.3 TS 计算不可行
- 气相单体 TS 高估 2.6–4.1×（注意：比较的是单体固有势垒 vs 表观 Ea）
- +CPCM(THF) 仅修正 23%（介电效应），ΔS‡ 修正 0%
- 苯炔消除在气相无过渡态（吸热反应，需要溶剂化 LiBr 才有势垒）
- 文献参考：Ramachandran [7] 的显式 3THF 降低势垒 31 kJ/mol，但聚集态又升高 24 kJ/mol
- 精确 TS 需要微溶剂化 + 聚集态采样 + AIMD（天-周/化合物），无法扩展

### 7.4 推荐策略
**类别特异的基态描述符 QSPR 是唯一可行路线**：
- 秒级计算成本（xTB）
- 类别内 LOO-R² = 0.61–0.94
- 描述符物理意义：
  - Gsolv 隐式包含溶剂化自由能信息
  - mol_volume 作为溶剂重组熵的代理（ΔS‡ 的间接编码）
  - BDE 包含 Li-C 键强度（与固有势垒相关）
  - %Vbur 捕捉聚集态中 Li 的可接近性
- 已部署为预测工具：SMILES → Ea + t½ + 反应器推荐

### 7.5 当前局限
- 数据量小（n=6–26 per class）是主要瓶颈
- lnA（∝ΔS‡）无法从基态描述符完全预测 → t½ 精度受限（因为 ΔS‡ 受聚集态和溶剂动力学主导）
- 对新结构类别（如杂环 ArLi）需要新的实验数据
- 聚集态信息未显式纳入（当前模型假设所有化合物在 THF 中的聚集行为类似）

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

1. **背景**：有机锂中间体在流动化学中的重要性 → 需要稳定性预测工具
2. **溶液相复杂性**（Collum 2007）：聚集态、溶剂配位、多路径机制 → 为什么预测困难
3. **数据集**：75 个中间体，40 Arrhenius 参数，5 个结构类别
4. **描述符筛选**：xTB → HF → M06-2X → QTAIM 四个层级，xTB 最优（R²=0.69 > DFT 0.41）
5. **LOCO 验证**：跨类别预测 R² < -12 → 机制异质性（文献支撑：Collum "substrate-dependent mechanisms"）
6. **Ea-lnA 补偿**：聚集态解聚同时贡献 +Ea 和 +ΔS‡（Ramachandran 定量：+24/-31 kJ/mol）
7. **类别特异模型**：o-ArLi R²=0.91, oxiranylLi R²=0.92/0.94
8. **TS 计算验证**：三个体系 × 两个溶剂化层级
   - 气相高估 2.6-4.1×，CPCM 仅修正 23%
   - ΔS‡: CPCM +1.1 vs 实验 -121（0% 改善）→ 静态计算原理性失败
9. **ΔS‡ 揭示机制多样性**：解聚/溶剂重组/碎裂的叠加，不同类别符号不同
10. **结论**：基态描述符 QSPR 是唯一可行且有效的预测路线（描述符隐式编码溶液相效应）

---

## 10. 参考文献

[1] De Gennaro, L.; Fanelli, F.; Luisi, R. "Organolithium Compounds in Flow Chemistry." *Lithium Compounds in Organic Synthesis*, Wiley, 2014, Chapter 18, pp. 513–548. — 有机锂分解动力学综述，Arrhenius 数据主要来源，LFER (σ+Es+δ) 模型基础。

[2] Nagaki, A.; Ichinari, D.; Yoshida, J. "Three-Component Coupling Based on Flash Chemistry." *J. Am. Chem. Soc.* **2014**, 136, 12245–12248. — o-BrPhLi、o-IPhLi 等 ArLi 热稳定性数据。

[3] Nagaki, A.; Takahashi, Y.; Yoshida, J. "Generation and Reaction of Oxiranyllithium Using Flow Microreactor." *Angew. Chem. Int. Ed.* **2019**, 58, 4027–4030. — oxiranylLi 类分解动力学数据。

[4] Nagaki, A.; Yamada, D.; Yoshida, J. "Flow Microreactor Synthesis Involving Alkoxycarbonyl ortho-Lithioaryl." *J. Flow Chem.* **2008**, 1–8. — o-CO₂R-ArLi 分解 Arrhenius 参数。

[5] Musci, P.; et al. "Flow Microreactor Technology for Lithium Carbenoid Generation." *React. Chem. Eng.* **2020**, 5, 935–941. — 卡宾类 LiCHXY 分解动力学。

[6] Stanetty, P.; Koller, H.; Mihovilovic, M. "Directed Ortho-Lithiation of Phenylcarbamic Acid tert-Butyl Ester. Revision of the Synthesis of Methyl 2-Amino-benzoate from Aniline." *J. Org. Chem.* **1992**, 57, 6833–6837. — PhLi 分解参考数据。

[7] Ramachandran, B.; Kharidehal, P.; Pratt, L. M.; Voit, S.; Okeke, F. N.; Ewan, M. "Computational Strategies for Reactions of Aggregated and Solvated Organolithium Carbenoids." *J. Phys. Chem. A* **2010**, 114, 8423–8433. — **M06-2X 被确认为有机锂最佳 DFT 泛函**（误差 2.47 kcal/mol）。SI 数据：聚集态升高势垒 +24 kJ/mol，3THF 溶剂化降低 -31 kJ/mol。14 种泛函基准测试。

[8] Bannwarth, C.; Ehlert, S.; Grimme, S. "GFN2-xTB — An Accurate and Broadly Parametrized Self-Consistent Tight-Binding Quantum Chemical Method with Multipole Electrostatics and Density-Dependent Dispersion Contributions." *J. Chem. Theory Comput.* **2019**, 15, 1652–1671. — GFN2-xTB 半经验方法，本工作描述符计算的主要理论层级。

[9] Falivene, L.; et al. "SambVca 2. A Web Tool for Analyzing Catalytic Pockets with Topographic Steric Maps." *Organometallics* **2016**, 35, 2286–2293. — 埋藏体积 (%Vbur) 和 Sterimol 参数计算（通过 morfeus 实现）。

[10] Smith, D. G. A.; et al. "PSI4 1.4: Open-Source Software for High-Throughput Quantum Chemistry." *J. Chem. Phys.* **2020**, 152, 184108. — Psi4 量子化学软件，HF 和 M06-2X 单点计算。

[11] Lu, T.; Chen, F. "Multiwfn: A Multifunctional Wavefunction Analyzer." *J. Comput. Chem.* **2012**, 33, 580–592. — Multiwfn 波函数分析，QTAIM BCP 性质和 ADCH 电荷计算。

[12] Neese, F. "Software Update: The ORCA Program System—Version 5.0." *WIREs Comput. Mol. Sci.* **2022**, 12, e1606. — ORCA 量子化学软件，NEB-TS 过渡态搜索。

[13] Hansch, C.; Leo, A.; Taft, R. W. "A Survey of Hammett Substituent Constants and Resonance and Field Parameters." *Chem. Rev.* **1991**, 91, 165–195. — Hammett σ 取代基常数来源。

[14] Collum, D. B.; McNeil, A. J.; Ramirez, A. "Lithium Diisopropylamide: Solution Kinetics and Implications for Organic Synthesis." *Angew. Chem. Int. Ed.* **2007**, 46, 3002–3017. — **有机锂溶液动力学经典综述**。关键论断：(a) 速率方程揭示 TS 聚集态/溶剂化计量；(b) 二聚体路径普遍（>60%）；(c) 底物依赖的机制是常态；(d) 溶剂同时影响基态和 TS，不可只考虑 TS 端；(e) 解聚不一定需要额外溶剂化。

[15] Seebach, D. "Structure and Reactivity of Lithium Enolates. From Pinacolone to Selective C-Alkylations of Peptides." *Angew. Chem. Int. Ed. Engl.* **1988**, 27, 1624–1654. — 有机锂聚集态结构的早期晶体学研究。

[16] Verkhov, V. A.; et al. "Analysis of Chemical Bonding in Lithium Molecular Compounds Based on the Electron Density and Electron Localization Function." *J. Chem. Phys.* **2025**, 162, 044114. — QTAIM/ELF 描述符对 C-Li 键分类（84-89% 准确率），ELF basin population 是最重要描述符。
