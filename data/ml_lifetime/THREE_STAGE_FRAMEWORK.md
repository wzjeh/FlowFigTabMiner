# 有机锂中间体寿命预测：三阶段验证框架

## 概述

本框架描述从文献数据挖掘到实验验证的完整逻辑链。每个阶段的数据来源、方法和验证结果明确标注。

---

## Stage 1: 建模 — 从热图提取动力学参数并建立结构-活性关系

### 做了什么

1. **热图数据提取**：用 VLM（Gemini/Claude）从 14 篇流动化学文献的 yield-vs-tR 热图中提取数据点
2. **动力学拟合**：对每个（中间体, 温度）组合拟合竞争动力学模型，得到分解速率常数 k_d
3. **Arrhenius 拟合**：对每个中间体的多温度 k_d 做 ln(k_d) vs 1/T 线性回归，得到 Ea 和 ln(A)
4. **结构-活性关联**：
   - Hammett: Ea = -48.5σ + 36.1（间位/对位 ArLi，r = -0.979, n = 4）
   - Taft: log(t½) = -1.13·Es + 0.04（邻位酯 ArLi，r = -0.969, n = 4）

### 数据来源

| 数据 | 来源 | 数量 |
|------|------|------|
| yield-tR-T 数据点 | 14 篇论文的热图，VLM 提取 + Zhao 手动 tR 矫正 | 1,470 行 |
| Hammett σ 值 | Hansch, Leo & Taft (1991) 标准文献值 | 4 个间/对位 ArLi |
| Taft Es 值 | Taft (1956) 标准文献值 | 4 个邻位酯 ArLi |
| YOLO 独立验证 | 微目标检测模型独立提取的 yield 值 | 737 行，99.6% 一致 |

### 关键输出

- 14 个中间体的 Arrhenius 参数（Ea, ln_A）
- 半衰期跨越近 4 个数量级（3.3 ms ~ 51.9 s @ -40°C）
- Hammett/Taft 两参数框架：间/对位用 σ，邻位用 Es

### 文件

- `data/final_output/organolithium_tr_subdataset_vlm_enriched.csv` — 原始数据
- `data/ml_lifetime/electronic_analysis.csv` — 14 个中间体的 Ea, ln_A, σ, Es
- `data/ml_lifetime/phase_b_arrhenius.csv` — Arrhenius 拟合结果

---

## Stage 2: 条件外推验证 — 用已有参数预测不同条件下的结果

### 做了什么

取 Stage 1 中已经拟合好 Arrhenius 参数的 4 个邻位酯中间体（Me/Et/iPr/tBu），将其动力学参数从热图条件（flow, -70~20°C, 0.003~100s tR）**外推**到间歇条件（batch, -78°C, 600s），计算中间体在 600s 后的存活分数，与同一篇论文中 scope 表格报告的间歇产率对比。

### 数据来源

| 数据 | 来源 | 说明 |
|------|------|------|
| Ea, ln_A（4 个邻位酯） | Stage 1 的 Arrhenius 拟合 | 来自 Nagaki 2008/2010 热图 |
| 间歇产率（实际值） | Nagaki 2010 酯论文 (DOI: 10.1002/chem.201000876) 的 scope 表格 | **同一篇论文**的 Table 1/2/4/5/7 |
| scope 表格数据提取 | FlowFigTabMiner pipeline → final.json → VLM 裁定 | 1,267 行 scope 数据集 |

**重要澄清**：间歇 600s 数据**不是来自 ORD 数据库**，而是来自 Nagaki 2010 论文本身。这篇论文同时包含：
- 热图（flow 条件下的 tR-T 产率扫描）→ 我们提取了 Arrhenius 参数
- scope 表格（batch 条件下 -78°C, 600s 的底物筛选）→ 我们用来验证外推

### 验证结果

| 酯基 R | Ea (kJ/mol) | t½(-78°C) 预测 | 存活率 @600s 预测 | 间歇产率（实际） | 判定 |
|--------|-------------|---------------|-----------------|----------------|------|
| **tBu** | 26.8 | 765.5 s | **58.1%** | **61%** | 几乎精确 |
| iPr | 38.0 | 256.6 s | 19.8% | 12% | 方向正确，略高估 |
| Et | 41.3 | 116.4 s | 2.8% | 0% | 正确预测失败 |
| Me | 36.5 | 23.6 s | 0.0% | 0% | 正确预测失败 |

### 这一步验证了什么

- 我们从 flow 热图提取的 Arrhenius 参数（Ea, ln_A）在**外推到不同温度和不同反应器类型**时仍然定量有效
- tBu 预测 58.1% vs 实际 61%，误差仅 3 个百分点
- Et/Me 正确预测为"在间歇条件下完全分解"
- **这是对动力学参数本身的验证，不是对新底物的预测**

### 这一步没有做什么

- 没有预测新底物的 Ea（4 个酯都是 Stage 1 中已经测量过的中间体）
- 没有使用 Hammett/Taft 公式来预测（直接用了每个中间体独立拟合的 Ea/ln_A）

### 文件

- `data/final_output/organolithium_scope_table_dataset.csv` — scope 表格数据集
- `data/ml_lifetime/analysis_figures/scope_validation.png` — 外推验证图

---

## Stage 3: 新底物预测 — 用 Hammett σ 预测未测量中间体的动力学参数

### 要做什么

用 Stage 1 建立的 Hammett 回归方程，预测 3 个**从未出现在训练数据中的** ArLi 中间体的 Ea、t½ 和反应器推荐，然后通过 flow microreactor 实验验证。

### 数据来源

| 数据 | 来源 | 说明 |
|------|------|------|
| Hammett σ 值 | Hansch, Leo & Taft (1991) 文献 | 标准物化常数 |
| Ea 预测公式 | Stage 1: Ea = -48.5σ + 36.1 | 4 个间/对位 ArLi 回归 |
| ln_A 预测公式 | Stage 1: ln(A) = 0.628·Ea - 4.63 | 焓熵补偿关系 |
| 实验 Ea（待测） | 湿实验：flow microreactor, 8 tR × 5 T = 40 点/底物 | 每个底物独立拟合 |

### 预测目标（更新后）

| 底物 | 中间体 | σ | 预测 Ea | 预测 t½(-40°C) | 反应器推荐 |
|------|--------|---|---------|---------------|-----------|
| 4-bromobenzotrifluoride | p-CF₃-PhLi | +0.54 | 9.9 kJ/mol | ~75 s | batch compatible |
| 1-bromo-4-fluorobenzene | p-F-PhLi | +0.06 | 33.2 kJ/mol | ~25 s | flow microreactor |
| 4-bromotoluene | p-CH₃-PhLi | -0.17 | 44.3 kJ/mol | ~10 s | flow microreactor |

### 成功标准

| 指标 | 基本成功 | 强成功 |
|------|---------|--------|
| Ea 偏差 | \|Ea_pred - Ea_exp\| < 15 kJ/mol | < 10 kJ/mol |
| 排序一致 | Ea(CF₃) < Ea(F) < Ea(CH₃) | 排序一致 + 单调 |
| t½ 数量级 | 预测值与实验值在 ×0.3 ~ ×3 范围内 | ×0.5 ~ ×2 |
| 反应器推荐 | CF₃ batch yield > 70% | > 80% |

### 实验方案

见 `EXPERIMENTAL_VALIDATION_PLAN.md`（方案细节：3 底物 × 40 点，n-BuLi 做 Br/Li 交换，MeOH 猝灭，GC 分析）

### 状态

**待实验验证**。Stage 2 的成功（外推定量吻合）为 Stage 3 提供了信心基础。

---

## 三阶段逻辑关系

```
Stage 1 (建模)          Stage 2 (条件外推验证)       Stage 3 (新底物预测)
                       
14 热图 → Arrhenius     同一批底物                    新底物
          参数          不同条件 (-78°C batch)         (p-CF₃, p-F, p-CH₃)
            ↓                    ↓                         ↓
     Hammett/Taft       Ea/ln_A 直接外推              Hammett σ → Ea → t½
     关联建立            → 预测存活率                   → 预测反应器类型
            ↓                    ↓                         ↓
     r=-0.979           tBu: 58% vs 61%               待实验验证
     r=-0.969           Et/Me: 0% vs 0%
                               ✅
```

**为什么 Stage 2 必须在 Stage 3 之前**：如果 Arrhenius 参数在温度外推时就不准确（Stage 2 失败），那么用 Hammett 预测新底物的 Ea 再做外推（Stage 3）就更不可信。Stage 2 证明了参数的外推可靠性，为 Stage 3 的预测提供了定量信心。

---

## 数据流总览

```
                    ┌─────────────────────────────┐
                    │  14 篇论文的热图（flow 数据）  │
                    │  来源：VLM 提取 + Zhao tR 矫正 │
                    └──────────┬──────────────────┘
                               │
                    ┌──────────▼──────────────────┐
           Stage 1  │  竞争动力学 → Arrhenius 拟合   │
                    │  14 中间体: Ea, ln_A, t½      │
                    │  Hammett σ / Taft Es 关联     │
                    └──────────┬──────────────────┘
                               │
              ┌────────────────┼────────────────┐
              │                                 │
   ┌──────────▼────────────┐     ┌──────────────▼───────────┐
   │  Stage 2: 条件外推      │     │  Stage 3: 新底物预测       │
   │                        │     │                          │
   │  同一中间体 (4 邻位酯)   │     │  Hammett σ → Ea_pred     │
   │  Ea/ln_A → k(-78°C)   │     │  → ln_A_pred → t½_pred  │
   │  → 存活率 @600s        │     │  → 反应器推荐             │
   │                        │     │                          │
   │  验证数据来源：          │     │  验证数据来源：            │
   │  Nagaki 2010 同篇论文    │     │  湿实验（待执行）          │
   │  scope 表格 batch 产率  │     │  3 底物 × 40 点           │
   │                        │     │                          │
   │  结果：tBu 58% vs 61%  │     │  状态：⬜ 待验证           │
   │        ✅ 定量吻合       │     │                          │
   └────────────────────────┘     └──────────────────────────┘
```
