# ArLi LFER 模型变更历史

记录模型从初版到当前版本的每一次重大变更，包括**为什么改**、**改了什么**、**结果如何**。

---

## v1.0 — 独立 Hammett + Taft 模型（论文初版）

**日期**: 2026-03 初  
**数据量**: 4+4 = 8 个点（分两组）

**模型**:
- Hammett (仅 para/meta, n=4): `Ea = -48.5σ + 36.1`
- Taft (仅 ortho ester, n=4): `log(t½) = -1.13Es + 0.04`
- 桥接: 焓熵补偿关系 `lnA = 0.627Ea - 4.60`，将 Ea → t½

**问题**:
1. 每组只有 4 个点，无法交叉验证
2. Hammett 和 Taft 是独立的，无法比较 ortho vs para
3. 焓熵补偿关系可能是统计假象（Krug et al. 1976）
4. p-CH₃ 外推预测 t½ = 0.5 s — 严重不合理（教授指出 p-CH₃ 应与 PhLi 相当）

---

## v2.0 — 统一三参数模型 (σ + Es + δ_ortho)

**日期**: 2026-03 中  
**数据量**: 8 个 ArLi（排除 benzyne 类 o-I, o-Br）  
**动机**: 解决 v1.0 的外推问题和补偿关系依赖

**模型**:
```
Ea  = -48.6σ + 8.0Es + 25.6δ_ortho + 36.1  (R²=0.970, Q²=0.817)
lnA = -24.2σ + 6.7Es + 15.6δ_ortho + 15.4  (R²=0.980, Q²=0.925)
```

**改进**:
- 合并了 Hammett 和 Taft 为统一方程
- 发现了 Li···O 螯合效应 (δ_ortho, +25.6 kJ/mol on Ea)
- 分别建模 Ea 和 lnA，不再依赖补偿关系
- p-CH₃ 预测修正为 t½ ≈ 20 s（合理）

**遗留问题**:
- 8 个点、6 个参数（Ea 3 + lnA 3），自由度仅 2
- LOOCV Q²(t½) = -0.8（PhLi、o-CO₂ᵗBu 留出后崩溃）
- 排除了 benzyne 类（o-I, o-Br），损失了 2 个数据点

---

## v3.0-A — 四参数模型，Encoding A（δ_ortho + δ_benzyne 叠加）

**日期**: 2026-04-05  
**数据量**: 10 个 ArLi（纳入 benzyne）  
**动机**: 纳入 benzyne 类以增加数据点和覆盖范围

**编码方式 (Encoding A)**:
- ortho ester: δ_ortho=1, δ_benzyne=0
- benzyne (o-I, o-Br): δ_ortho=1, **δ_benzyne=1**（叠加在 δ_ortho 上）
- 其他: δ_ortho=0, δ_benzyne=0

**模型**:
```
Ea  = -48.01σ + 8.00Es + 25.53δ_ortho + 18.91δ_benzyne + 35.89
lnA = -24.16σ + 6.72Es + 15.55δ_ortho + 14.15δ_benzyne + 15.41
```

**问题**: δ_benzyne 系数 (18.91) 不直观——实际上 benzyne 的总效应 = δ_ortho(25.53) + δ_benzyne(18.91) = **44.44** kJ/mol，但从系数上看不出来。而且 benzyne 消除和 Li···O 螯合是完全不同的机制，叠加 δ_ortho 没有物理意义。

---

## v3.0-B — 四参数模型，Encoding B（δ_ortho 和 δ_benzyne 互斥）⭐ 当前版本

**日期**: 2026-04-06  
**数据量**: 10 个 ArLi  
**动机**: Encoding A 的系数不反映真实物理效应；benzyne 消除与 Li···O 螯合是独立机制，不应叠加

**编码方式 (Encoding B)**:
- ortho ester: δ_ortho=1, δ_benzyne=0
- benzyne (o-I, o-Br): δ_ortho=**0**, δ_benzyne=1（互斥）
- 其他: δ_ortho=0, δ_benzyne=0

**模型**:
```
Ea  = -48.01σ + 8.00Es + 25.53δ_ortho + 44.44δ_benzyne + 35.89  (R²=0.980, Q²_Ea=0.879)
lnA = -24.17σ + 6.72Es + 15.55δ_ortho + 29.70δ_benzyne + 15.41  (R²=0.993, Q²_lnA=0.975)

t½(T) = ln(2) / exp(lnA - Ea/(RT))
```

**为什么改**: Encoding A 和 B 数学上完全等价（预测值相同），但 B 的系数直接对应物理效应：
- δ_benzyne = 44.44 = benzyne 消除使 Ea 增加的绝对量（vs 基线 ArLi）
- δ_ortho = 25.53 = Li···O 螯合使 Ea 增加的绝对量
- 两者互斥，不需要心算叠加

**关键指标**:
- 训练集 t½ 预测: 8/10 在 3× 以内（benzyne 2 个在 ~4×）
- LOOCV Q²(log t½) = 0.310
- 验证: o-CO₂ᵗBu 在 -78°C 预测 survival 58%, 实验 batch yield 61%

---

## 评估记录: Method B — ML 动力学代理模型

**日期**: 2026-04-06  
**动机**: 14 个底物有 ~700 条原始热图数据点(tR, T, yield)，能否用 ML 直接建模 yield = f(σ, Es, tR, T) 来绕过 Arrhenius 拟合？

**方法**: GaussianProcessRegressor + GradientBoostingRegressor，LOSO (Leave-One-Substrate-Out) 交叉验证

**结果**:
- 质量过滤后（R²<0.3 的曲线移除, PhLi T>30°C 移除）: 276 点, 10 底物
- GBR LOSO R² = 0.293
- GP LOSO R² ≈ 0.1

**结论**: 与 Method A (LFER) 的 LOOCV Q²=0.310 相当。瓶颈是**底物数量（10个）**，不是数据点数量。276 个数据点对预测新底物没有帮助，因为 LOSO 本质上还是 10-fold CV。

**决策**: 保留 Method A (LFER) 作为最终模型。Method B 脚本保留在 `method_b_kinetic_surrogate.py` 供参考。

---

## δ_benzyne 去留评估

**日期**: 2026-04-06  
**问题**: 只有 2 个 benzyne 数据点 (o-I, o-Br)，保留 δ_benzyne 是否过拟合？

**实验**: 去掉 δ_benzyne，用三参数模型 (σ + Es + δ_ortho) 在 10 个点上拟合

**结果**: Q²(log t½) 从 **+0.310 降至 -0.788**（完全崩溃）

**原因**: o-I 和 o-Br 的 Ea (60-65 kJ/mol) 远高于三参数模型能解释的范围，强行拟合导致所有其他系数被扭曲。

**决策**: 保留 δ_benzyne。虽然只有 2 个点，但它们对回归的稳定性至关重要。

---

## 验证预测值修正

**日期**: 2026-04-07  
**问题**: 旧文档中 validation 预测值有误

| 底物 | 旧值 | 新值 (当前模型) | 差异原因 |
|------|------|----------------|---------|
| p-CF₃ Ea | 9.9 kJ/mol | 10.0 kJ/mol | 基本一致 |
| p-CF₃ t½(-40°C) | 23 s | **11.2 s** | 旧值可能来自 v1.0 补偿关系外推 |
| p-CF₃ 结论 | batch-compatible | **NOT batch-compatible** (-78°C survival≈0%) | |
| p-CH₃ Ea | 44.3 kJ/mol | 44.1 kJ/mol | 基本一致 |
| p-CH₃ t½(-40°C) | 500 ms | **17.2 s** (差34倍!) | 旧值来自 v1.0 补偿关系外推 |
| p-CH₃ 结论 | flash chemistry | **flow + batch-compatible** (-78°C survival=75%) | |

**物理解释**: Ea 在 v1.0 和 v3.0 中接近，但 t½ 差异巨大，因为 v1.0 用焓熵补偿关系 `lnA = 0.627Ea - 4.60` 来推 lnA，这个关系在 σ<0 区域外推不可靠。v3.0 分别建模 Ea 和 lnA，给出更合理的 p-CH₃ lnA=19.5（vs v1.0 补偿推出的 ~23）。

---

## 图表更新记录

**日期**: 2026-04-07 ~ 2026-04-09  
**文件**: `regenerate_summary_figure.py` → `analysis_figures/final_results_summary.png`

| 变更 | 原因 |
|------|------|
| Panel (b): σ-vs-Ea 带箭头 → Ea parity plot | Zhao 反馈原图有莫名其妙的竖线，看不懂（4个 ortho ester σ 全=0.45，竖线堆叠） |
| Panel (b)(c): 去掉 LOOCV 空心标记 | Zhao 反馈空心方块/三角形看起来奇怪，先不展示 LOOCV |
| Panel (c): 不同 marker 形状 → 统一圆形 | 方块/三角形在 log 轴上视觉混乱 |
| 标签加灰色引导线 | 中间区域点密集，标签容易遮挡数据点 |
| Panel (b) 中间区域标签偏移加大 | PhLi(Ea=36.54) 和 o-CO₂Me(Ea=36.53) 几乎重叠 |
| Validation: 4个底物 → 2个 (p-CF₃, p-CH₃) | Zhao 要求精简，选覆盖 σ 范围最大的两个 |
| Validation 金色星号加入 panel (b)(c) | Zhao 要求将预测值也显示在 parity plot 上 |
