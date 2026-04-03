# Organolithium Intermediate Lifetime Prediction from Flow Chemistry Data Mining

## 一、项目来龙去脉

### 1.1 起点：FlowFigTabMiner 数据集

FlowFigTabMiner 是一个从流动化学文献 PDF 中自动提取图表数据的 pipeline。在处理有机锂文献时，我们积累了大量 **yield-vs-residence time (tR) 热图** 数据——这些热图记录了不同停留时间和温度条件下的反应收率。

### 1.2 关键发现：热图中隐藏着中间体寿命信息

流动化学热图的 yield-vs-tR 曲线呈**山峰形状**：
- **上升段**：tR 太短，反应不完全，yield 低
- **峰值**：最优 tR，yield 最高
- **下降段**：tR 太长，中间体分解，yield 下降

下降段直接编码了有机锂中间体的**热稳定性**。通过拟合竞争动力学模型，可以提取半衰期 (t₁/₂) 和分解活化能 (Ea)。

### 1.3 数据集构建

| 步骤 | 说明 | 脚本 |
|------|------|------|
| TF-ID 图表提取 | Florence-2 从 PDF 中裁剪图表 | `src/pipeline/main.py` |
| VLM 数据提取 | Claude/Gemini 从热图中提取 (tR, T, yield) | `scripts/parse_vlm_tr_output.py` |
| 上下文富化 | 10 级富化：v10 回填、SMILES、五位模型、tR 步骤分类 | `scripts/enrich_vlm_context.py` |
| 最终数据集 | 1,650 条数据，27 列，21 篇论文 | `organolithium_tr_subdataset_vlm_enriched.csv` |

### 1.4 ML Lifetime Pipeline

| Phase | 说明 | 输入 | 输出 |
|-------|------|------|------|
| **Phase A** | 从 yield-tR 衰减曲线提取 t₁/₂ | 1,183 个 tR1 数据点 | 39 个 (中间体, T) → t₁/₂ |
| **Phase B** | Arrhenius 拟合 → Ea | 39 个 (中间体, T, k_d) | **11 个中间体** 的 Ea |
| **Phase C** | QSPR: 分子描述符 → Ea | 11 个中间体 + 25 个描述符 | Q² = 0.445 |
| **电子效应分析** | Hammett σ、分解机制、位置效应 | 11 个中间体 | 4 张分析图 |
| **ORD 扩充** | 从 43K batch 反应推断 1,638 个 ArLi | ORD 数据库 | 分类器 (64% 准确率) |

---

## 二、核心结论

### 2.1 五个数量级的稳定性差异

11 个有机锂中间体在 -40°C 的半衰期跨越 **82,560 倍**（1.1 ms → 89.3 s）。

| 排名 | 中间体 | t₁/₂@-40°C | Ea (kJ/mol) | 反应器推荐 |
|------|--------|------------|-------------|-----------|
| 1 | p-lithiobenzonitrile | 89.3 s | 6.4 | **Batch** 兼容 |
| 2 | m-lithiobenzonitrile | 66.9 s | 4.7 | **Batch** 兼容 |
| 3 | oxiranyllithium | 35.2 s | 43.4 | 标准 Flow |
| 4 | tBu 4-(lithio)benzoate | 26.0 s | 21.0 | 标准 Flow |
| 5 | phenyllithium | 22.4 s | 36.5 | 标准 Flow |
| 6 | tBu o-(lithio)benzoate | 17.5 s | 6.5 | 标准 Flow |
| 7 | CHLi(I)(Cl) | 8.6 s | 25.0 | 标准 Flow |
| 8 | CHLi(I)(F) | 793 ms | 19.9 | **Flash** chemistry |
| 9 | (Br,Li)-benzene | 67 ms | 82.2 | **Flash** chemistry |
| 10 | LiCH₂F | 22 ms | 51.0 | **Flash** chemistry |
| 11 | (I,Li)-benzene | 1.1 ms | 79.5 | **Flash** chemistry |

### 2.2 Hammett σ 与 Ea 的负相关（r = -0.91）

排除 benzyne 消除 outlier 后，ArLi 子集（5 个中间体）中：

**Ea = -49.4σ + 36.0 kJ/mol** (r = -0.91, p = 0.031)

化学意义：**吸电子基 (EWG) 越强，分解活化能越低，但中间体越稳定**。这是因为 EWG 同时降低了 Ea 和 pre-exponential factor (ln_A)，净效果是分解速率降低。

### 2.3 Ea-ln(A) 焓熵补偿效应（r = 0.99）

**ln(A) = 0.628·Ea - 5.11** (r = 0.992, p < 10⁻⁶)

这是经典的 enthalpy-entropy compensation：高 Ea 的中间体也有高 ln(A)（高频率因子），意味着虽然能垒高，但一旦越过能垒后的分解非常快。

等动力学温度 (isokinetic temperature) = **-82°C (192 K)**。在这个温度以下，所有中间体的分解速率趋同；在这个温度以上，稳定性差异急剧放大。

### 2.4 三种分解机制

| 机制 | Ea 范围 | 特征 | 代表中间体 |
|------|---------|------|-----------|
| **共轭稳定** | 4.7-21.0 | 低 Ea + 低 ln_A → 慢分解 | p/m-CN-ArLi, ester-ArLi |
| **Benzyne 消除** | 79-82 | 极高 Ea + 极高 ln_A → 温度极敏感 | (Br,Li)-, (I,Li)-benzene |
| **α-消除** | 11-51 | 中等 Ea，受卤素类型影响 | LiCH₂F, CHLi(I)(F), CHLi(I)(Cl) |

### 2.5 位置效应 (o/m/p-CN)

| 位置 | t₁/₂@20°C |
|------|-----------|
| ortho | 114 s |
| para | 45 s |
| meta | 41 s |

ortho-CN 比 para/meta 稳定约 2.7 倍，可能源于 **Li···N 分子内螯合**效应。

### 2.6 验证：与文献已知 lifetime 一致

| 中间体 | 文献 lifetime | 提取的 t₁/₂ | 比值 |
|--------|-------------|------------|------|
| fluoromethyllithium | 13 ms (-60°C) | 125 ms | 9.65× |
| iodofluoromethyllithium | 82 ms (-40°C) | 810 ms | 9.88× |

系统偏移 ~10×（因为模型同时拟合了形成+分解+猝灭），但**两个已知中间体的相对比值完美保留**：6.5× (提取) vs 6.3× (文献)。

---

## 三、明确的算法

### 3.1 竞争动力学模型（Phase A）

```
yield(tR) = y_max × (1 - exp(-k_f·tR)) × exp(-k_d·tR)
```

- k_f: 形成速率常数
- k_d: 分解速率常数
- t₁/₂ = ln(2) / k_d

对每个 (中间体, T) 对用 scipy.optimize.curve_fit 拟合。

### 3.2 Arrhenius 拟合（Phase B）

```
ln(k_d) = ln(A) - Ea/(R·T)
```

对每个中间体在 2-8 个温度点做线性回归。

### 3.3 Hammett 回归（电子效应分析）

```
Ea = -49.4·σ + 36.0   (r = -0.91, 仅 ArLi 子集)
```

### 3.4 反应器推荐规则

```
if t₁/₂(-40°C) > 60s:   → Batch compatible
if 1s < t₁/₂ < 60s:     → Standard flow reactor
if t₁/₂ < 1s:           → Flash chemistry microreactor
```

---

## 四、能做什么预测

### 4.1 对已知结构类型的中间体

给定一个新的 ArLi 中间体：
1. 如果取代基的 Hammett σ 已知 → 用 Ea = -49.4σ + 36.0 预测 Ea
2. 用 Ea-ln(A) compensation（ln(A) = 0.628·Ea - 5.11）预测 ln(A)
3. 用 Arrhenius 公式预测任意温度下的 t₁/₂
4. 根据 t₁/₂ 推荐反应器类型

**适用范围**：ArLi 中间体，取代基为 EWG（CN, ester, CF₃ 等）且不存在邻位卤素。

### 4.2 对新结构类型

Hammett 回归不适用于 sp3 carbenoid 或 benzyne 消除路径。但分解机制分类可以预测：
- 邻位卤素 → benzyne 消除 → flash chemistry 必须
- sp3 + 多卤素 → α-消除 → 需要 flow

### 4.3 预测的局限性

- 仅 11 个中间体，统计功效有限
- Hammett 回归仅在排除 benzyne 后 n=5
- 系统偏移 ~10× 需要校正因子
- 不适用于完全不同的反应类型（如 directed metalation, deprotonation）

---

## 五、湿实验验证方案

### 5.1 实验目的

验证两个核心预测：
1. **Hammett 回归的预测能力**：选择模型未见过的 ArLi，用 σ 预测 Ea，与实验对比
2. **反应器推荐规则的实用性**：模型推荐 "batch compatible" 的中间体确实能在 batch 中高 yield

### 5.2 候选中间体（模型外推）

选择 3 个模型**未训练**的 ArLi 中间体，涵盖不同的预测区间：

| # | 候选底物 | ArLi 中间体 | σ | 预测 Ea | 预测 t₁/₂@-40°C | 预测反应器 |
|---|---------|------------|---|---------|-----------------|-----------|
| 1 | 4-bromobenzotrifluoride | p-CF₃-phenyllithium | +0.54 | **9.3 kJ/mol** | ~75 s | Batch |
| 2 | 4-bromoacetophenone | p-COCH₃-phenyllithium | +0.50 | **11.3 kJ/mol** | ~55 s | Batch |
| 3 | 4-bromobiphenyl | p-Ph-phenyllithium | -0.01 | **36.5 kJ/mol** | ~22 s | Flow |

中间体 1（p-CF₃）是最强 EWG，预测最稳定。中间体 3 接近 PhLi（σ≈0），预测中等稳定。

### 5.3 实验方法：flow microreactor 中的 yield-vs-tR 扫描

**设备**：T-shaped micromixer + microtube reactor (ID = 250-500 μm)，与 Nagaki 组的设置一致。

**Step 1: Br/Li 交换**
```
ArBr + n-BuLi (or s-BuLi) ──[R1, tR1, T1]──→ ArLi (intermediate)
```

**Step 2: 猝灭**（用 MeOH 或 D₂O 作为猝灭剂，简化分析）
```
ArLi + MeOH ──[R2]──→ ArH + LiOMe
```

**变量**：
- **tR1**: 0.01, 0.03, 0.1, 0.3, 1.0, 3.16, 10.0, 31.6 s（8 个水平，对数等间距）
- **T1**: -78, -60, -40, -20, 0 °C（5 个温度）
- 总计：8 × 5 = **40 个实验点** per 中间体

**分析**：GC 或 GC-MS 测 yield（ArH vs ArBr 消失比例）

**条件**：
- ArBr 浓度：0.10 M in THF
- n-BuLi：0.42 M in hexane
- equiv_orgLi: 1.1 equiv
- Flow rate: 调节以实现目标 tR1

### 5.4 数据处理

对每个中间体的 40 个 (tR, T, yield) 数据点：
1. 用 Phase A 的竞争动力学模型拟合 → k_f, k_d, t₁/₂ per T
2. 用 Phase B 的 Arrhenius 拟合 → Ea_exp
3. 与 Hammett 预测值比较：Ea_pred vs Ea_exp

### 5.5 预期结果与成功标准

| 指标 | 成功标准 |
|------|---------|
| Ea_pred vs Ea_exp 偏差 | < 10 kJ/mol (对 3 个中间体均成立) |
| t₁/₂ 排序一致性 | 实验排序与 σ 排序一致 (p-CF₃ > p-COCH₃ > p-Ph) |
| 反应器推荐正确性 | 模型推荐 "batch" 的中间体在 batch 中 yield > 80% |

### 5.6 对照实验

同时在 **batch** 中重复上述 3 个反应：
- 条件：-78°C, THF, 30 min, 与 MeOH 猝灭
- 如果 yield 与 flow 最优条件接近 → 确认 "batch compatible" 预测正确
- 如果 yield 显著低于 flow → 说明中间体在 batch 时间尺度已分解

### 5.7 实验时间估计

- 每个中间体 40 个点 × 5 min/point = 3.3 h flow 实验
- 3 个中间体 + 3 个 batch 对照 = ~2 天实验
- GC 分析 ~1 天
- 总计：**3-4 个工作日**

### 5.8 如果实验成功，发表角度

**Title**: "Quantitative Prediction of Organolithium Intermediate Lifetime from Hammett Constants: Data Mining Meets Flow Chemistry"

**Story**: 
1. 从 21 篇文献的热图中自动提取了 11 个有机锂中间体的 Ea 和 t₁/₂
2. 发现 Ea 与 Hammett σ 强相关 (r=-0.91)
3. 用这个相关性预测了 3 个新中间体的 t₁/₂
4. 湿实验验证预测正确
5. 建立了一个"输入取代基 → 输出反应器类型"的实用工具

---

## 六、文件清单

### 脚本
| 文件 | 功能 |
|------|------|
| `scripts/enrich_vlm_context.py` | 10 级数据富化 (v10 回填 → SMILES → 五位模型 → tR 分类) |
| `scripts/ml_lifetime/phase_a_extract_halflife.py` | 竞争动力学拟合 → t₁/₂ |
| `scripts/ml_lifetime/phase_b_arrhenius.py` | Arrhenius 拟合 → Ea |
| `scripts/ml_lifetime/phase_c_qspr.py` | QSPR: 分子描述符 → Ea |
| `scripts/ml_lifetime/analyze_electronic_effects.py` | Hammett + 机制分类 + 位置效应 |
| `scripts/ml_lifetime/plot_orgli_diversity_comparison.py` | 数据集多样性对比图 |
| `scripts/ml_lifetime/augment_qspr_from_ord.py` | ORD batch 数据扩充 |

### 数据
| 文件 | 内容 |
|------|------|
| `data/ml_lifetime/phase_a_halflives.csv` | 39 个 (中间体, T, t₁/₂, k_f, k_d) |
| `data/ml_lifetime/phase_b_arrhenius.csv` | 11 个 (中间体, Ea, ln_A, t₁/₂ at -78/-40/0/25°C) |
| `data/ml_lifetime/electronic_analysis.csv` | 11 个 (中间体, σ, Ea, 机制, 反应器推荐) |
| `data/ml_lifetime/ord_inferred_intermediates.csv` | 1,638 个 ORD ArLi (SMILES + 描述符 + 稳定性标签) |

### 图表
| 文件 | 内容 |
|------|------|
| `data/ml_lifetime/phase_a_curves/*.png` | 12 个中间体的衰减曲线拟合 |
| `data/ml_lifetime/phase_b_arrhenius_plot.png` | Arrhenius 图 (ln(k_d) vs 1/T) |
| `data/ml_lifetime/analysis_figures/stability_ranking_reactor_zones.png` | 稳定性排名 + 反应器推荐 |
| `data/ml_lifetime/analysis_figures/hammett_and_hybridization.png` | Hammett σ vs Ea + sp2/sp3 对比 |
| `data/ml_lifetime/analysis_figures/ea_lna_compensation.png` | 焓熵补偿效应 |
| `data/ml_lifetime/analysis_figures/position_effect_cn.png` | o/m/p-CN 位置效应 |
| `data/final_output/dataset_comparison/orgli_diversity_comparison.png` | 数据集多样性对比 (Flow vs ORD vs USPTO) |
