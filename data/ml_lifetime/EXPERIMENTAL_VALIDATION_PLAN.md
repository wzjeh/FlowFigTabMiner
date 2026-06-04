# 有机锂中间体稳定性 HYBRID v2 模型 — 4 底物流动化学验证方案

**日期**: 2026-04-26
**目标**: 用 4 个底物的流动化学动力学数据验证 HYBRID v2 模型 (LOO-R² 中位 0.973, 反应器分类 -20~25°C 区 85%)
**装置**: 已有 0°C flow 反应器 (φ250 μm 微通道)
**作者**: Zhao
**版本**: v2 (替代旧 Hammett 文档)

---

## 1. 背景与目标

### 1.1 模型概述
我们建立了 4 类有机锂中间体（p-ArLi, m-ArLi, o-ArLi, oxiranylLi）的类特异 Arrhenius 参数预测模型：
- **训练集**: 30 个 Tier-C 化合物
- **描述符池**: 14 个单分子描述符 + 5 个聚集态描述符 (ΔE_dim, ΔH_dim, ΔG_dim, ΔS_dim, Δ%V_bur)
- **筛选规则**: k=3 + 互斥物理类别 + 类内 Pearson \|r\| < 0.7（ArLi 类）
- **R² 中位**: 0.973
- **反应器分类准确率**: -20~25°C 区 85%, 0~25°C 区 87%

### 1.2 验证目标
1. **测试模型外推能力**: 在数据集中数据稀疏的化合物
2. **验证 LFER 在极端电子条件下的有效性**: 极端电子贫和电子富区
3. **跨类机制对比**: 同时测试 p-ArLi、m-ArLi、oxiranylLi 三类
4. **修正模型**: 实验数据回流加入训练集，重新训练以提升预测精度

---

## 2. 流动反应装置

### 2.1 装置配置（按用户提供示意图）

```
                                                     0°C 恒温反应区
                                                     ───────────────
[Substrate]                                         ╭─────────────────╮
0.10 M in THF                                       │                 │
6 mL/min ────╮                                      │                 │
              ├── T-junction φ250μm ──── L cm ──╮  │                 │
n-BuLi       ╯                                   │  │                 │
0.42 M in hexane                                  │  │                 │
1.5 mL/min                                        │  │                 │
                                                  │  │                 │
                                                  ▼  │                 │
                                                T-junction φ250μm     │
                                                  ▼                   │
                                            ─── 100 cm ───            │
                                            φ250μm                    │
MeOH 0.60 M ─────────────────────────────╮  │                         │
in THF, 3 mL/min                          ╰──┘                         │
                                                                       │
                                                  ▼                   │
                                              到 NH4Cl 收集瓶          │
                                                                       │
                                                  ╰─────────────────╯
```

### 2.2 标准操作参数

| 参数 | 数值 |
|---|---|
| 微通道内径 | φ250 μm |
| 反应温度 | 0 °C |
| 底物浓度 | 0.10 M in THF |
| n-BuLi 浓度 | 0.42 M in hexane |
| MeOH 淬灭剂浓度 | 0.60 M in THF |
| 底物流速 | 6.0 mL/min |
| n-BuLi 流速 | 1.5 mL/min |
| MeOH 流速 | 3.0 mL/min |
| 反应区总流速 | 7.5 mL/min |
| 淬灭后总流速 | 10.5 mL/min |
| **化学计量** | **n-BuLi/底物 = 1.05 当量** |
| 稳定时间 | 30 秒 |
| 取样时间 | 30 秒 |
| 后处理 | NH4Cl 饱和水溶液 |

### 2.3 反应管长 L (cm) 与残留时间 tR (s) 对照

通道横截面积: A = π × (250/2)² × 10⁻⁶ mm² = 0.04909 mm² = 4.909 × 10⁻⁴ cm²
单 cm 体积: V/cm = 4.909 × 10⁻⁴ mL/cm
反应区流速: 7.5 mL/min = 0.125 mL/s

**tR (s) = 0.00393 × L (cm)**

| L (cm) | tR (s) | 取样体积 (μL) |
|---|---|---|
| 1 | 0.0039 | 0.49 |
| 5 | 0.020 | 2.45 |
| 10 | 0.039 | 4.91 |
| 25 | 0.098 | 12.27 |
| 50 | 0.196 | 24.55 |
| 100 | 0.393 | 49.09 |
| 250 | 0.982 | 122.7 |
| 500 | 1.964 | 245.5 |
| 1000 | 3.927 | 490.9 |

---

## 3. 推荐底物（4 个）

### 3.1 概览

| # | 底物 | 中间体 SMILES | 类 | DB 状态 | 预测 t_max @ 0°C | L 范围 (cm) | 优先级 |
|---|---|---|---|---|---|---|---|
| **1** | **3,5-二溴硝基苯** | `[Li]c1cc(Br)cc([N+](=O)[O-])c1` | m-ArLi | NEW (m-Br/m-NO₂ 已有) | 5-50 ms ⚡ | 1, 5, 10, 25 | ⭐⭐ |
| **2** | **4-溴氟苯** | `[Li]c1ccc(F)cc1` | p-ArLi | 1 datapoint @ 24°C | 1-3 s | 50, 100, 250, 500 | ⭐⭐⭐ |
| **3** | **1-溴-3,5-二甲基苯** | `[Li]c1cc(C)cc(C)c1` | m-ArLi | 1 datapoint @ 0°C | 0.05-0.2 s | 5, 25, 50, 100 | ⭐⭐⭐ |
| **4** | **(E)-反式二苯环氧乙烷** | `[Li][C@@]1(c2ccccc2)O[C@@H]1c1ccccc1` | oxiranylLi | 5 T (modeling 中) | 0.1-1 s | 25, 50, 100, 250 | ⭐⭐ |

---

### 3.2 底物 #1: 3,5-二溴硝基苯 (1,3-Dibromo-5-nitrobenzene)

#### 3.2.1 化学信息
- **化学式**: C₆H₃Br₂NO₂
- **CAS**: 6311-60-2
- **分子量**: 279.91 g/mol
- **SMILES (起始物)**: `O=[N+]([O-])c1cc(Br)cc(Br)c1`
- **SMILES (中间体)**: `[Li]c1cc(Br)cc([N+](=O)[O-])c1`
- **商品来源**: Sigma-Aldrich, TCI 等
- **预计成本**: ~¥200/g

#### 3.2.2 反应方程
```
1,3-Dibromo-5-nitrobenzene + n-BuLi (1.05 eq.)
        ↓ THF, 0°C, < 50 ms
1-Lithio-3-bromo-5-nitrobenzene
        ↓ MeOH (quench)
1-Bromo-3-nitrobenzene (主产物)
```

#### 3.2.3 推荐流动条件
| 参数 | 数值 |
|---|---|
| 底物 | 1,3-Dibromo-5-nitrobenzene 0.10 M in dry THF |
| 锂试剂 | n-BuLi 0.42 M in hexane |
| 流速 | 标准（6.0/1.5/3.0 mL/min）|
| 反应温度 | 0 °C |
| **L 取值** | **1, 5, 10, 25 cm**（tR = 4, 20, 39, 98 ms）|
| 淬灭 | MeOH（标准）|
| 后处理 | NH₄Cl 水溶液 |
| 分析 | GC 或 GC-MS（监测 1-Br-3-NO₂-苯 产率）|

#### 3.2.4 ⚠️ 安全/技术警告
- **NO₂ 与 n-BuLi 风险**: 长 tR (>100 ms) 下 n-Bu 可能进攻 NO₂ 氧
- **必须先做 L=1 cm 测试**: 看 GC 是否出现脱 Br 副产物
- **若有副反应**: 改用 t-BuLi 或降低 T 至 -40°C
- **底物溶解度**: 在 THF 中较低 (~0.1 M 接近上限)，需充分搅拌

#### 3.2.5 模型预测（基于 m-ArLi LFER 外推）
| 量 | 预测值 | 物理理由 |
|---|---|---|
| σ_meta_total | +1.10 (Br: +0.39, NO₂: +0.71) | 极端电子贫 |
| Ea_d | **25-35 kJ/mol** | 外推自 m-NO2 (Ea=51) 和 m-Br (Ea=69) |
| lnA_d | 8-12 | 类似 m-CN |
| t_max @ 0°C | **5-50 ms** | flash chemistry 区 |
| 推荐反应器 | flash | tR < 0.1 s |

#### 3.2.6 验证价值
- **测试 m-ArLi LFER 在 σ = +1.10 的外推**（DB 中最大 σ = +0.71）
- **完全 NEW 化合物**: 数据集和模型都未见过
- **机制清洁**: meta 取代不导致 benzyne 或螯合

---

### 3.3 底物 #2: 4-溴氟苯 (4-Bromofluorobenzene)

#### 3.3.1 化学信息
- **化学式**: C₆H₄BrF
- **CAS**: 460-00-4
- **分子量**: 175.00 g/mol
- **SMILES (起始物)**: `Fc1ccc(Br)cc1`
- **SMILES (中间体)**: `[Li]c1ccc(F)cc1`
- **商品来源**: Sigma-Aldrich, TCI（廉价）
- **预计成本**: ~¥50/g

#### 3.3.2 反应方程
```
4-Bromofluorobenzene + n-BuLi (1.05 eq.)
        ↓ THF, 0°C, ~1 s
4-Lithiofluorobenzene (p-F-苯基锂)
        ↓ MeOH (quench)
Fluorobenzene (主产物)
```

#### 3.3.3 推荐流动条件
| 参数 | 数值 |
|---|---|
| 底物 | 4-Bromofluorobenzene 0.10 M in dry THF |
| 锂试剂 | n-BuLi 0.42 M in hexane |
| 流速 | 标准 |
| 反应温度 | 0 °C |
| **L 取值** | **50, 100, 250, 500, 1000 cm**（tR = 0.2, 0.4, 1.0, 2.0, 3.9 s）|
| 淬灭 | MeOH |
| 分析 | GC（监测 fluorobenzene 产率）|

#### 3.3.4 ⚠️ 注意事项
- **F 完全惰性**: 不会与 n-BuLi 交换或反应（C-F 键能 ~115 kcal/mol）
- **简单清洁反应**: 此底物是模型最鲁棒的验证

#### 3.3.5 模型预测
| 量 | 预测值 | 物理理由 |
|---|---|---|
| σ_para(F) | +0.06 | 极弱 EW（诱导 + π 反馈相消）|
| Ea_d | **55-65 kJ/mol** | 介于 p-CN (Ea=28.8, σ=0.71) 和 p-anisyl (Ea=79.6, σ=-0.27) |
| lnA_d | 18-25 | 中等 p-ArLi |
| t_max @ 0°C | **1-3 s** | flow 区 |
| 推荐反应器 | flow | 0.1 < t_max < 60 s |

#### 3.3.6 验证价值
- **填补 p-ArLi 中性 σ ≈ 0 数据空白**
- **DB 中 1 个 datapoint @ 24°C** → 添加 0°C 系列即可拟合 Arrhenius
- **完全鲁棒**: 无副反应风险，最适合校准模型

---

### 3.4 底物 #3: 1-溴-3,5-二甲基苯 (m-ArLi 电子富推荐)

#### 3.4.1 化学信息
- **化学式**: C₈H₉Br
- **CAS**: 556-96-7
- **分子量**: 185.06 g/mol
- **SMILES (起始物)**: `Cc1cc(Br)cc(C)c1`
- **SMILES (中间体)**: `[Li]c1cc(C)cc(C)c1`
- **商品来源**: Sigma-Aldrich, TCI
- **预计成本**: ~¥100/g

#### 3.4.2 反应方程
```
1-Bromo-3,5-dimethylbenzene + n-BuLi (1.05 eq.)
        ↓ THF, 0°C, ~0.1 s
1-Lithio-3,5-dimethylbenzene
        ↓ MeOH (quench)
1,3-Dimethylbenzene (m-xylene, 主产物)
```

#### 3.4.3 推荐流动条件
| 参数 | 数值 |
|---|---|
| 底物 | 1-Br-3,5-Me₂-苯 0.10 M in dry THF |
| 锂试剂 | n-BuLi 0.42 M in hexane |
| 流速 | 标准 |
| 反应温度 | 0 °C |
| **L 取值** | **5, 25, 50, 100, 250 cm**（tR = 20, 98, 196, 393, 982 ms）|
| 淬灭 | MeOH |
| 分析 | GC（监测 m-xylene 产率，BP = 139°C 易分离）|

#### 3.4.4 ⚠️ 注意事项
- **Me 基对 n-BuLi 完全惰性**: 不存在 sp³ 脱质子风险（位阻 + 弱酸性）
- **GC 分析**: m-xylene BP = 139°C, 与 THF (66°C) 容易分离
- **稳定性**: 此化合物预测最稳定，可作为流动装置的"长 tR 校准"

#### 3.4.5 模型预测
| 量 | 预测值 | 物理理由 |
|---|---|---|
| σ_meta_total | -0.14 (2 × σ_m(CH₃) = -0.07) | 弱 EDG（甲基弱给电子）|
| Ea_d | **70-90 kJ/mol** | 外推到电子富 m-ArLi |
| lnA_d | 18-25 | 类似中等 m-ArLi |
| t_max @ 0°C | **0.05-0.2 s** | flow 区（短时端） |
| 推荐反应器 | flow | flash/flow 边界 |

#### 3.4.6 验证价值
- **DB 中 m-ArLi 类全部 σ_meta > 0**（5 化合物）
- **本底物 σ = -0.14 → 引入第一个电子富 m-ArLi**
- **σ 跨度从 0.88 单位扩展到 1.27 单位**（+44%）
- **DB 中 1 datapoint @ 0°C** → 添加多温度系列可加入 modeling

---

### 3.5 底物 #4: (E)-反式二苯环氧乙烷 → 反式-Li-2,3-二苯-环氧乙烷 (oxiranylLi 推荐)

#### 3.5.1 化学信息
- **化学式**: C₁₄H₁₂O
- **CAS**: 1439-07-2
- **分子量**: 196.24 g/mol
- **SMILES (起始物)**: `[H][C@@]1(c2ccccc2)O[C@@H]1c1ccccc1` (trans)
- **SMILES (中间体)**: `[Li][C@@]1(c2ccccc2)O[C@@H]1c1ccccc1`
- **商品来源**: Sigma-Aldrich, TCI（trans 异构体）
- **预计成本**: ~¥300/g

#### 3.5.2 反应方程
```
trans-Stilbene oxide + s-BuLi (1.05 eq.)   ⚠️ 用 s-BuLi 而非 n-BuLi
        ↓ THF, 0°C, ~0.5 s
trans-2-Lithio-2,3-diphenyl-oxirane
        ↓ MeOH (quench)
trans-Stilbene oxide (回收) + 部分开环产物 (deoxybenzoin?)
```

#### 3.5.3 推荐流动条件（注意修改）
| 参数 | 数值 |
|---|---|
| 底物 | (E)-Stilbene oxide 0.10 M in dry THF |
| 锂试剂 | **s-BuLi 1.3 M in cyclohexane** ⚠️ 修改 |
| s-BuLi 流速 | **0.48 mL/min** ⚠️ 修改（保持 1.05 当量）|
| 流速 | 6.0 / **0.48** / 3.0 mL/min |
| 反应温度 | 0 °C |
| **L 取值** | **25, 50, 100, 250 cm**（tR = 0.1, 0.2, 0.4, 1.0 s）|
| 淬灭 | MeOH |
| 分析 | HPLC 或 GC-MS（监测原料消耗 vs deoxybenzoin 形成）|

#### 3.5.4 ⚠️ 重要修改说明
- **使用 s-BuLi 而非 n-BuLi**: oxiranyl-Li 是通过 α-脱质子，s-BuLi 比 n-BuLi 快 ~10×
- **流速调整**: 0.42 M × 1.5 mL/min = 0.63 mmol/min (n-BuLi)；改 1.3 M × 0.48 mL/min = 0.62 mmol/min (s-BuLi) → 保持 1.05 当量
- **替代方案**: 若 s-BuLi 不可得，可用 n-BuLi/TMEDA (1:1)
- **0°C 风险**: oxiranyl-Li 通常 -78°C 制备；0°C 下生命短，但本底物 (双苯) 稳定性较好

#### 3.5.5 模型预测
| 量 | 预测值（来自现有 modeling）|
|---|---|
| Ea_d | ~70 kJ/mol（实验拟合值）|
| lnA_d | ~22 |
| t_max @ 0°C | **0.1-1 s** |
| 推荐反应器 | flow |

#### 3.5.6 验证价值
- **DB 中已有 5 T 数据 + 在 modeling set**
- **加 4 个新 T 点 → 改善 oxiranylLi 类的整体 R²（现 R²_med = 0.987-0.998）**
- **测试 oxiranylLi 在不同 σ 的稳定性预测**

---

## 4. 实验执行计划

### 4.1 时间安排

| 阶段 | 内容 | 预计时间 |
|---|---|---|
| 准备 | 试剂分装、装置调试、空白 | 0.5 天 |
| #2 (p-F) | 5 个 L × 重复测试 | 1 天 |
| #3 (3,5-Me₂) | 5 个 L × 重复测试 | 1 天 |
| #1 (3,5-Br/NO₂) | 4 个 L × 重复测试 | 1 天 |
| #4 (Stilbene oxide) | 4 个 L × 重复测试 | 1 天 |
| 数据分析 | GC 定量, Arrhenius 拟合 | 0.5 天 |
| **合计** |  | **5 天** |

### 4.2 取样协议（按图所示）
1. 设定 L 长度，调整流速到目标值
2. **30 秒稳定** (达到稳态浓度分布)
3. **30 秒取样** (收集 ~5 mL 反应混合物到 NH₄Cl 水溶液)
4. 有机相用 EtOAc 萃取
5. 有机相加内标（如十二烷或正辛烷）
6. GC/HPLC 定量产物 yield (%)

### 4.3 数据处理

每个底物得到 (L, tR, yield) 数据点表，输入到全局 5 参数 Arrhenius 模型：
$$\text{yield}(t_R, T) = y_{\max} \cdot (1 - e^{-k_f t_R}) \cdot e^{-k_d t_R}$$

由于 T 固定 (0°C)，简化为：
$$\text{yield}(t_R) = y_{\max} \cdot (1 - e^{-k_f t_R}) \cdot e^{-k_d t_R}$$

可拟合 3 个参数 (y_max, k_f, k_d)。

如需 Arrhenius，需要在多个 T 重复（建议 -40°C, 0°C, +20°C 三点）— 但这需要冷却装置，目前装置仅 0°C。

### 4.4 模型对比

实验后构造 parity plot：
- X 轴: 模型预测 t_max @ 0°C
- Y 轴: 实验测量 t_max @ 0°C
- 4 个点 (4 个底物)
- 理想: 接近 y=x 直线，误差 < 30%

---

## 5. 预期结果与解读

### 5.1 预期 yield-vs-tR 曲线

每个底物预计有钟形曲线：
- 短 tR: yield 上升（生成 > 分解）
- t_max: 最高 yield
- 长 tR: yield 下降（分解主导）

### 5.2 模型验证结果矩阵

| 底物 | 模型预测 Ea_d | 实验结果 | 解读 |
|---|---|---|---|
| #1 3,5-Br/NO₂-Li | 25-35 kJ/mol | TBD | 测试 m-ArLi LFER 极端 EW 外推 |
| #2 p-F-Li | 55-65 kJ/mol | TBD | 校准 p-ArLi 中性 σ 区 |
| #3 3,5-Me₂-Li | 70-90 kJ/mol | TBD | 测试 m-ArLi LFER 电子富外推 |
| #4 Stilbene oxide-Li | ~70 kJ/mol | TBD | oxiranyl 模型校准 |

### 5.3 可能的发现

1. **如果 #1 实验值与模型预测一致** (Δ < 10 kJ/mol)
   → m-ArLi LFER 在 σ=+1.10 仍线性 → 强支持模型外推能力

2. **如果 #2 实验值符合 p-ArLi 中位** (~60 kJ/mol)
   → 验证 p-ArLi 模型在 σ ≈ 0 的内插准确度

3. **如果 #3 实验值远高于预测** (e.g., > 100 kJ/mol)
   → m-ArLi LFER 在电子富区可能弯曲，需要二次项或类内分组

4. **如果 #4 加入后 oxiranyl 类总 R² 提升** → 模型稳健性增强

---

## 6. 模型反馈与迭代

### 6.1 数据回流
所有实验数据 (T, tR, yield) 加入到 `clean_organolithium_unified.csv` 中：
```
paper_id = "experimental_validation_2026"
data_source_type = "in-house_flow"
T1_C = 0
tR1_s = ...
yield_pct = ...
intermediate_smiles_canonical = (post-Li SMILES)
analysis_subset = "kd_clean"
```

### 6.2 模型更新流程
1. 重新跑 Step 3 全局 Arrhenius 拟合（45 → 49 化合物）
2. 重新跑 Step 6 Tier 分类
3. 重新跑 Step 7c.10 HYBRID v2 筛选
4. 重新跑 Step 8 LOO 反应器分类
5. 比较 R² 和准确率改善

### 6.3 论文/PPT 数据
实验后图表用于：
- Figure: parity plot (4 底物)
- Figure: 4 个底物的 yield-vs-tR 曲线
- Table: 实验 vs 模型预测 Ea_d, lnA_d
- Discussion: 验证 Reich 2013 / Collum 2007 框架在新底物上的适用性

---

## 7. 试剂清单

| 试剂 | CAS | 用量估算 | 来源 |
|---|---|---|---|
| 1,3-Dibromo-5-nitrobenzene | 6311-60-2 | 1 g | Aldrich/TCI |
| 4-Bromofluorobenzene | 460-00-4 | 5 g | Aldrich/TCI |
| 1-Bromo-3,5-dimethylbenzene | 556-96-7 | 2 g | Aldrich/TCI |
| (E)-Stilbene oxide | 1439-07-2 | 1 g | Aldrich/TCI |
| n-BuLi (2.5 M in hexane) | 109-72-8 | 50 mL | 标准 |
| s-BuLi (1.3 M in cyclohexane) | 598-30-1 | 30 mL | 标准（用于 #4）|
| MeOH (anhydrous) | 67-56-1 | 200 mL | Aldrich |
| THF (super-dry) | 109-99-9 | 1 L | Aldrich (over molecular sieves) |
| NH₄Cl (饱和水溶液) | 12125-02-9 | 500 mL | 标准 |
| EtOAc (萃取) | 141-78-6 | 1 L | 标准 |
| 内标 (n-Decane 或 dodecane) | 95-94-3 | 5 mL | Aldrich |

---

## 8. 安全注意事项

1. **n-BuLi 和 s-BuLi**: 强空气敏感，遇水放热剧烈反应。在 Schlenk 系统下处理。
2. **3,5-Dibromonitrobenzene + n-BuLi @ 0°C**: NO₂ 可能与 n-Bu 反应。**首次实验用最短 L 测试**。
3. **THF 必须新蒸或分子筛干燥**: 含水会消耗 BuLi。
4. **流动装置**: 防止压力积累；最大背压 < 2 bar。
5. **淬灭瓶**: 用大体积 NH₄Cl 水溶液，立刻搅拌。

---

## 9. 装置示意图

```
                     [底物]                  [n-BuLi]
                  0.10 M, THF              0.42 M, hexane
                   6 mL/min                 1.5 mL/min
                       │                         │
                       │                         │
                       └────┬────────────────────┘
                            │
                     ┌──────┴──────┐
                     │ T-junction  │
                     │  φ250 μm    │
                     └──────┬──────┘
                            │
                            │   (反应区, 0°C)
                            │     L cm 长度可变
                            │     管径 φ250 μm
                            │     7.5 mL/min 流速
                            │
                     ┌──────┴──────┐         [MeOH]
                     │ T-junction  │←──────  0.60 M, THF
                     │  φ250 μm    │         3 mL/min
                     └──────┬──────┘
                            │
                            │   后处理段
                            │   100 cm, φ250 μm
                            │
                     ┌──────┴──────┐
                     │  收集瓶     │
                     │  NH4Cl(aq)  │
                     └─────────────┘
```

---

## 10. 数据收集模板

| Substrate | T (°C) | L (cm) | tR (s) | Yield (%) | t_max_fitted (s) | Ea_d_fitted | Notes |
|---|---|---|---|---|---|---|---|
| 3,5-Br/NO₂-Li | 0 | 1 | 0.0039 | __ | __ | __ | __ |
| 3,5-Br/NO₂-Li | 0 | 5 | 0.020 | __ | __ | __ | __ |
| 3,5-Br/NO₂-Li | 0 | 10 | 0.039 | __ | __ | __ | __ |
| 3,5-Br/NO₂-Li | 0 | 25 | 0.098 | __ | __ | __ | __ |
| p-F-Li | 0 | 50 | 0.196 | __ | __ | __ | __ |
| p-F-Li | 0 | 100 | 0.393 | __ | __ | __ | __ |
| p-F-Li | 0 | 250 | 0.982 | __ | __ | __ | __ |
| p-F-Li | 0 | 500 | 1.964 | __ | __ | __ | __ |
| p-F-Li | 0 | 1000 | 3.927 | __ | __ | __ | __ |
| 3,5-Me₂-Li | 0 | 5 | 0.020 | __ | __ | __ | __ |
| 3,5-Me₂-Li | 0 | 25 | 0.098 | __ | __ | __ | __ |
| 3,5-Me₂-Li | 0 | 50 | 0.196 | __ | __ | __ | __ |
| 3,5-Me₂-Li | 0 | 100 | 0.393 | __ | __ | __ | __ |
| 3,5-Me₂-Li | 0 | 250 | 0.982 | __ | __ | __ | __ |
| Stilbene-Li | 0 | 25 | 0.098 | __ | __ | __ | __ |
| Stilbene-Li | 0 | 50 | 0.196 | __ | __ | __ | __ |
| Stilbene-Li | 0 | 100 | 0.393 | __ | __ | __ | __ |
| Stilbene-Li | 0 | 250 | 0.982 | __ | __ | __ | __ |

---

**文档结束**
**Last updated: 2026-04-26**
**对应模型版本: HYBRID v2 (changelog v4.5)**
