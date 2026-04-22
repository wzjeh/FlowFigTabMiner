# 全局 Arrhenius 方法：从 SMILES 到反应器推荐

## 方法概述

### 核心思想

不再逐温度单独拟合 k_f 和 k_d，而是将一个化合物在所有温度下的 yield-vs-tR 数据**同时拟合**到 5 个参数：

```
yield(tR, T) = y_max × (1 - exp(-k_f(T)·tR)) × exp(-k_d(T)·tR)

k_f(T) = A_f × exp(-Ea_f / RT)    ← 生成速率的 Arrhenius
k_d(T) = A_d × exp(-Ea_d / RT)    ← 分解速率的 Arrhenius

5 个参数: Ea_f, lnA_f, Ea_d, lnA_d, y_max
```

### 为什么比逐温度拟合更好

逐温度拟合的问题：当 k_f >> k_d 时（高温），yield 曲线对 k_f 不敏感——k_f = 100 和 k_f = 100000 给出几乎相同的曲线。导致 k_f 的 Arrhenius 拟合不可靠（出现负 Ea_f）。

全局拟合的优势：低温数据约束 k_f（低温下 k_f 小，yield 上升慢，k_f 可精确拟合），高温数据约束 k_d（衰减明显），**互相补充**。

---

## 拟合结果

### 数据

- 45 个化合物成功全局拟合（R² > 0.3，Ea_f > 0，Ea_d > 0）
- 其中 41 个 R² > 0.6（可靠）
- R² 中位数 = 0.879

### 参数范围

| 参数 | 范围 | 含义 |
|---|---|---|
| Ea_f | 0.1 ~ 72.1 kJ/mol | 生成活化能（全部 > 0，物理合理）|
| Ea_d | 4.4 ~ 149.2 kJ/mol | 分解活化能 |
| lnA_f | -0.2 ~ 50.0 | 生成前指因子 |
| lnA_d | -9.7 ~ 50.0 | 分解前指因子 |

### 数据文件

- `global_arrhenius.csv`: 45 化合物的 5 参数 + R²

---

## 类别特异描述符模型

### 分类

基于 SMILES 结构（不是名字）分类：
- **o-ArLi**: `[Li]c1ccccc1X` (Li 和 X 在苯环 1,2 位)
- **m-ArLi**: `[Li]c1cccc(X)c1` (1,3 位)
- **p-ArLi**: `[Li]c1ccc(X)cc1` (1,4 位)
- **oxiranylLi**: 含三元环氧
- **hetero-ArLi**: 含杂原子的芳香环
- **other**: 其他

### 各类别描述符 → 4 参数的 LOO-R²

| 类别 | n | Ea_f | Ea_d | lnA_f | lnA_d |
|---|---|---|---|---|---|
| **oxiranylLi** | 8 | **0.854** (η+B1+L) | **0.969** (%Vbur+Gsolv+B1) | **0.850** (q_C+d_LiC+fukui) | **0.930** (%Vbur+Gsolv+B5) |
| **m-ArLi** | 7 | **0.682** (BDE+B1+sub_X) | **0.836** (fukui+sub_X+sub_mw) | **0.941** (η+B1+vol) | **0.970** (Gsolv+vol+dipole) |
| **o-ArLi** | 8 | 0.369 (HOMO+vol+sub_mw) | **0.880** (d_LiC+η+sub_mw) | 0.507 (HOMO+vol+sub_mw) | **0.914** (d_LiC+B5+dipole) |
| **p-ArLi** (去p-BrPhLi) | 7 | N/A (用均值) | **0.642** (fukui+vol+dipole) | **0.845** (q_C+B1+L) | **0.982** (d_LiC+BDE+B5) |

### 关键发现

- **oxiranylLi 和 m-ArLi**: 4 个参数全部可预测 (R² > 0.68)
- **o-ArLi**: Ea_d + lnA_d 可预测 (R² > 0.88), Ea_f 较弱
- **p-ArLi**: 去掉 p-BrPhLi 离群值后 Ea_d + lnA 可预测
- **Ea_f 是最难预测的参数**: 对所有 ArLi 类都较弱，因为 Ea_f 受底物 C-X 键和实验条件影响

---

## 端到端验证

### 方法

LOO-CV: 对每个化合物，用同类别其他化合物训练描述符→4 参数模型，预测该化合物的 4 参数，在各温度计算 t_max，分类 flash/flow/batch。

### t_max 反应器分类准确率

| 温度 | 准确率 | 分布 (flash/flow/batch) |
|---|---|---|
| -78°C | 67% | 多为 flow/batch |
| -40°C | 65% | flash 和 flow 混合 |
| **0°C** | **95%** | 多为 flash |
| **25°C** | **95%** | 几乎全 flash |
| **总体** | **80%** | |

### t_max 量级精度

| 宽容度 | 总体准确率 |
|---|---|
| 3 倍以内 (±0.5 order) | 37% |
| 量级以内 (±1.0 order) | 47% |
| 30 倍以内 (±1.5 order) | 69% |

### 对比旧方法

| | 旧方法 (直接线性) | 新方法 (全局 Arrhenius) |
|---|---|---|
| 原理 | 3 desc + 1/T → log(t_max) | 全局 Arrhenius → 4 params → t_max |
| 分类数 | flash/flow **2 类** | flash/flow/batch **3 类** |
| 准确率 | 89% (2 类) | **80% (3 类), 0°C 95%** |
| batch 覆盖 | ✗ 无 | ✓ 有 |
| 温度外推 | 仅训练范围 | Arrhenius 物理外推 |
| t½ 预测 | ✗ 不能 | ✓ 能 (R²=0.60 @0°C) |
| 物理可解释 | 弱 (经验回归) | 强 (Ea_f, Ea_d 有物理意义) |

---

## 反应器推荐边界

| 反应器 | t_max 范围 | 设备 |
|---|---|---|
| **flash** | < 0.1 s | 微混合器 (T-mixer) |
| **flow** | 0.1 – 60 s | 管式反应器 |
| **batch** | > 60 s | 常规烧瓶 |

边界基于反应器设备的物理限制，不随温度变化。

---

## 使用方法

### 已知化合物 (45 个，查表)

```python
# 从 global_arrhenius.csv 查 Ea_f, lnA_f, Ea_d, lnA_d
# 计算任意温度的 t_max 和 t½:
T_K = T_celsius + 273.15
k_f = exp(lnA_f - Ea_f / (R * T_K))
k_d = exp(lnA_d - Ea_d / (R * T_K))
t_max = ln(k_f / k_d) / (k_f - k_d)
t_half = ln(2) / k_d
```

### 新化合物 (descriptor → 4 params)

1. 判断结构类别 (o-/m-/p-ArLi, oxiranylLi 等)
2. 用 xTB 计算描述符 (q_C, d_LiC, BDE, %Vbur, Gsolv, HOMO, η, fukui, B1, B5, L, vol, dipole)
3. 用类别对应的描述符模型预测 Ea_f, Ea_d, lnA_f, lnA_d
4. 计算 t_max(T) 和 t½(T)
5. 分类: flash (<0.1s) / flow (0.1-60s) / batch (>60s)

---

## 局限性

1. **数据量**: 每类只有 7-8 个化合物，限制了模型精度
2. **Ea_f 预测**: 对所有 ArLi 类较弱 (R² < 0.7)，因为 Ea_f 受底物 C-X 键影响
3. **低温精度**: -78°C 和 -40°C 的分类准确率 (~65%) 低于 0°C (~95%)
4. **p-ArLi**: p-BrPhLi 为离群值 (Ea_d=149)，需要更多 p-ArLi 数据
5. **batch 验证不足**: 训练集中 batch 样本来自 formation_only 的间接推断

---

## 参考文献

- Ramachandran 2010, J. Phys. Chem. A — M06-2X 基准, 聚集态效应
- Collum 2007, Angew. Chem. — 有机锂溶液动力学框架
- Bannwarth 2019, J. Chem. Theory Comput. — GFN2-xTB 方法
- De Gennaro 2014, Lithium Compounds in Organic Synthesis — 数据来源
