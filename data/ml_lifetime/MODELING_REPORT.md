# 有机锂中间体稳定性预测：从数据到模型的完整技术报告

## 1. 数据来源与处理

### 1.1 原始数据

**来源**: 20篇流动化学论文的 yield-vs-tR 热图，经 VLM 提取 + Zhao 手动 tR 矫正。

**文件**: `clean_organolithium_unified_descriptors.csv`
- 清理后: 2,357 行 × 50 列，75 个唯一中间体
- 溶剂: **全部 THF 基** (THF, THF/hexane, THF/Et₂O 等)
- analysis_subset: kd_clean(1827), kd_valid(329), kd_and_trapping(122), scope(62), k2_trapping(17)

### 1.2 Arrhenius 参数提取

**Phase A**: 对每个 (中间体, 温度) 拟合竞争动力学模型:
```
yield(tR) = ymax × (1 - exp(-k_f × tR)) × exp(-k_d × tR)
```
提取 k_d（分解速率常数）。

**Phase B**: 对每个中间体的多温度 k_d 做 Arrhenius 回归:
```
ln(k_d) = ln(A) - Ea/(R×T)
```
提取 Ea（活化能）和 ln(A)（指前因子）。

**文件**: `phase_b_arrhenius.csv` — 40 个底物
- 质量分层: Tier 1 (24个, n_T≥3, r²≥0.85), Tier 2_2pt (12个), Tier 2_low_r² (2个), 已排除(2个)
- 已排除: (E)-β-styryllithium (负 Ea)

### 1.3 可靠建模数据集

**筛选标准**: has_arrhenius=True + quality_tier≠tier2_2pt + n_datapoints>10

**结果**: 26 个底物
- ArLi: 16, oxiranylLi: 6, benzylLi: 2, perfluoroalkylLi: 1, alkylLi: 1
- Ea 范围: 2.5 – 64.9 kJ/mol
- ln(A) 范围: -2.2 – 33.5

**⚠️ 问题 1**: 排除 2-pt fits 是必要的（r²=1.0 无法验证 Arrhenius 线性），但丢掉了 12 个底物。含 2-pt 的 37 个子集上，模型 LOO-R² 从 0.69 降到 0.46。

**⚠️ 问题 2**: 11 个有 kd_clean 数据但无 Arrhenius 拟合的中间体——phase_a 管道未能提取 k_d。经验证，这些中间体在实验 tR 范围内确实没有分解（yield 单调上升，无衰减）。

---

## 2. 描述符计算

### 2.1 已有描述符 (fill_empirical_descriptors.py + fill_dft_descriptors.py)

| 描述符 | 方法 | 填充率 | 来源 |
|---|---|---|---|
| sigma_hammett | 文献查表 (Hansch 1991) | 52% (仅 ArLi) | Hammett σ |
| Es_taft | 文献查表 | 100% (非 ortho-ester=0) | Taft 空间参数 |
| delta_ortho, delta_benzyne | 规则编码 | 100% | 二元指标 |
| dft_charge_Li, dft_charge_C_ipso | GFN2-xTB Mulliken charge | 100% | tblite + scipy L-BFGS-B |
| dft_HOMO_eV, dft_LUMO_eV | GFN2-xTB 轨道能 | 100% | 同上 |
| dft_LiC_bond_A | xTB 优化后几何 | 100% | 同上 |
| dft_LiC_BDE_kJ | xTB fragment SP | 100% | E(RLi) - E(R·) - E(Li·) |
| dft_dipole_D | xTB 偶极矩 | 100% | tblite |
| dft_Gsolv_kJ | xTB ALPB(THF) | 100% | xtb CLI |

### 2.2 新增描述符 (fill_new_descriptors.py)

| 描述符 | 方法 | 填充率 |
|---|---|---|
| HOMO_LUMO_gap_eV | LUMO - HOMO | 100% |
| sterimol_B1/B5/L | morfeus, C_ipso→Li 方向 | 100% |
| buried_vol_Li | morfeus, Li 周围 r=3.5Å | 100% |
| mol_volume | RDKit ComputeMolVolume | 100% |
| fukui_f_minus_C | xtb CLI, q(N) - q(N-1) | 100% |

### 2.3 描述符验证

**xTB 值范围合理**:
- Li-C bond: 1.84 – 2.02 Å (文献 1.9-2.2 Å ✓)
- charge_Li: +0.37 ~ +0.61 (正值 ✓)
- charge_C_ipso: -0.31 ~ +0.02
- BDE: 361 – 610 kJ/mol

**σ 值 spot-check**: 9 个已知 ArLi 全部匹配文献值 ✓

**⚠️ 问题 3**: xTB Mulliken charge 的**空间分辨率有限** — 对 p-CN、p-NO₂ 等远程共轭效应捕捉不足。C_ipso 的 charge 几乎不受 para 取代基影响 (p-CN: -0.224 vs 普通 ArLi: -0.231, 差仅 0.007)。而 Hammett σ 能清楚区分 (CN: 0.66 vs 0.37-0.45)。

---

## 3. 描述符筛选

### 3.1 单描述符相关性 (n=26, 与 Ea)

| 描述符 | r | LOO-R² | 备注 |
|---|---|---|---|
| **sterimol_B1** | +0.57 | **+0.23** | 单描述符最强 |
| dft_charge_C_ipso | +0.54 | -0.25 | 正值无 LOO 意义 |
| sigma_hammett | -0.46 | +0.02 (n=14) | 仅 ArLi |
| dft_charge_Li | +0.45 | -0.03 | |
| dft_LiC_BDE_kJ | +0.42 | +0.01 | |
| dft_dipole_D | -0.30 | -0.03 | |
| dft_HOMO_eV | -0.05 | -0.14 | 几乎无用 |
| **fukui_f_minus_C** | **-0.04** | **-0.31** | **完全无用** |

### 3.2 多描述符组合搜索

**穷举 C(17,3) = 680 个 3-feature 组合，LOO-CV:**

| 排名 | 组合 | LOO-R² | MAE (kJ/mol) |
|---|---|---|---|
| **1** | **q(C_ipso) + d(Li-C) + %Vbur** | **0.692** | **6.0** |
| 2 | q(C_ipso) + d(Li-C) + dipole | 0.650 | 6.6 |
| 3 | q(C_ipso) + d(Li-C) + δ_ortho | 0.617 | 6.6 |
| 4 | q(C_ipso) + d(Li-C) + δ_benzyne | 0.577 | 7.3 |

**4-feature 组合全部低于最佳 3-feature** (最高 0.44) → 26 个点下 4 参数过拟合。

### 3.3 尝试过但不 work 的描述符

| 尝试 | 为什么不 work |
|---|---|
| Fukui f⁻(C_ipso) | RLi 是强碱/强亲核体，主导因素是 static charge 而非 electronic response |
| HF/6-31+G* Mulliken charge | 弥散函数让 Mulliken charge 物理无意义 (Li 变负电荷) |
| HF/def2-SVP Mulliken charge | 虽然合理但 r(Ea)=0.34 < xTB(0.54)，LOO-R²=0.55 < xTB(0.69) |
| Hammett σ (加入模型) | 仅 14 个有 σ，加为第 4 参数过拟合；σ 和 q_C 共线 (r=0.76) |
| pKa(R-H) 估计值 | pKa 是热力学量 (ΔG_acid)，Ea 是动力学量 (ΔG‡_decomp)，跨类别不相关 (r=0.04) |
| 分类别建模 | ArLi (n=16) 单独: LOO-R² = -3.1; 每类 n 太少 |

---

## 4. 最终模型

### 4.1 模型方程

```
Ea (kJ/mol) = +152.6 × q(C_ipso) - 338.0 × d(Li-C, Å) + 73.1 × %V_bur + 700.6
```

所有描述符均基于 GFN2-xTB 优化几何计算 (q_C 和 d_LiC 来自 tblite single-point, %V_bur 来自 morfeus 在 xTB 优化构型上)。

### 4.2 性能

| 指标 | 值 |
|---|---|
| 训练集 | 26 个底物 (Tier 1 + 2 个 low-r²) |
| LOO-R² | 0.694 |
| MAE | 5.6 kJ/mol |
| Spearman ρ (Ea 排序) | 0.708 |

### 4.3 物理解释

| 描述符 | 系数 | 物理意义 |
|---|---|---|
| q(C_ipso) | +157 | C 上正电荷越多 → 碳负离子越稳定 → Ea↑ |
| d(Li-C) | -325 | Li-C 键越长 → 越离子性 → 不同分解路径 → Ea↓ |
| %V_bur(Li) | +115 | Li 周围位阻越大 → 被攻击越难 → Ea↑ |

### 4.4 Outlier 分析

| 底物 | |Ea 残差| | 原因 |
|---|---|---|
| o-I-ArLi | 18.8 kJ/mol | benzyne 消除 (不同机理) |
| p-CO₂tBu-ArLi | 14.0 | 特殊 steric stabilization |
| benzylLi (2个) | ~15 | sp3 benzylic 稳定化 |
| o-CO₂tBu-ArLi | 10.8 | ortho chelation (Li···O) |

### 4.5 关键限制

**⚠️ 问题 4**: 模型对远程电子效应不敏感。p-CN-ArLi (Ea=4.5) 被预测为 ~38 kJ/mol — 因为 xTB charge 无法区分 CN 的 π 共轭效应。这是 C_ipso 局部电荷描述符的固有盲区。

**⚠️ 问题 5**: 跨类别模型的成功是"用 oxiranylLi 的高 Ea 撑起了 R²"。ArLi 子集 (n=16) 单独 LOO-R² = -3.1 — 完全失败。模型本质上是类别分离器而非类内预测器。

---

## 5. 从 Ea 到 Lifetime (t½) 的转换

### 5.1 方法

```
ln(A) = slope_class × Ea + intercept_class    (per-class compensation)
k_d = exp(ln(A) - Ea/(R×T))
t½ = ln(2) / k_d
```

Per-class compensation:
| 类别 | slope | intercept | r(Ea,lnA) |
|---|---|---|---|
| ArLi | 0.680 | -7.93 | 0.893 |
| oxiranylLi | 0.548 | -2.85 | 0.976 |
| benzylLi | 0.569 | -3.62 | 1.000 (n=2) |
| 全局 | 0.586 | -4.10 | 0.959 |

### 5.2 t½ 预测的验证 (LOO)

| 温度 | t½ Spearman ρ | 10× 以内 | Reactor 准确率 |
|---|---|---|---|
| +25°C | **+0.41** | 46% | **88%** |
| 0°C | +0.33 | 54% | **73%** |
| -40°C | -0.14 | 54% | 35% |
| -78°C | -0.21 | 54% | 50% |

### 5.3 为什么高温更准

等动力学温度 T_iso ≈ -82°C。在 T >> T_iso 时，Ea 差异主导 t½ 排序；在 T ≈ T_iso 时，所有中间体趋同。

### 5.4 为什么 Ea LOO-R²=0.69 但 t½ 排序 ρ≈0 (at -40°C)

**⚠️ 问题 6 (核心)**:

Ea 的 6 kJ/mol MAE 经 Arrhenius 指数放大:
```
在 -40°C: exp(6 / (8.314e-3 × 233)) = exp(3.1) = 22×
```
加上 lnA compensation 的残差 (std=2.5):
```
exp(2.5) = 12×
```
两者叠加可导致 t½ 预测偏差 >100×。

**lnA 不可从基态描述符预测**: lnA ∝ ΔS‡ (活化熵)，取决于过渡态结构 + 聚集态平衡 (Curtin-Hammett)，不是基态分子性质。

---

## 6. 溶剂效应

### 6.1 现有数据的溶剂情况

所有 26 个训练底物在 **THF 基**溶剂中。模型隐含 "THF 条件"。

### 6.2 文献溶剂校正数据

来源: Stanetty 1997, Degennaro 2014

| RLi | THF Ea | Et₂O Ea | DME Ea | 溶剂效应 |
|---|---|---|---|---|
| n-BuLi | 92.0 | 79.9 | 91.4 | THF ≈ DME >> Et₂O |
| t-BuLi | 52.2 | 53.6 | 22.4 | Et₂O ≈ THF >> DME |

**⚠️ 问题 7**: 溶剂效应因底物而异，不能用统一校正。n-BuLi 在 Et₂O 中更稳定，但 t-BuLi 在 DME 中反而不稳定。

---

## 7. 预测工具

### 7.1 实现

**文件**: `predict_stability.py`

```
python predict_stability.py '[Li]c1ccc(F)cc1' THF -40
```

Pipeline:
1. SMILES → RDKit 3D → xTB 优化 → 提取 q(C_ipso), d(Li-C)
2. RDKit 几何 → morfeus %V_bur
3. Ea = 157.3×q_C - 325.1×d + 114.9×Vbur + 669.6
4. lnA = slope_class × Ea + intercept_class
5. t½ = ln(2) / exp(lnA - Ea/RT)
6. Reactor: flash(<1s) / flow(1-60s) / batch(>60s)

### 7.2 工具的诚实能力

| 输出 | 可靠性 | 依据 |
|---|---|---|
| **Ea 预测** | **中等** (MAE=6 kJ/mol) | LOO-R²=0.69 |
| **Ea 排序** | **较好** (ρ=0.71) | 26 个底物验证 |
| t½ 绝对值 | **差** (~10-100× 不确定性) | 指数放大 + compensation 误差 |
| t½ 排序 @25°C | **可用** (ρ=0.41) | LOO 验证 |
| t½ 排序 @-40°C | **不可靠** (ρ≈0) | LOO 验证 |
| **Reactor @25°C** | **好** (88%) | LOO 验证 |
| **Reactor @0°C** | **可用** (73%) | LOO 验证 |
| Reactor @-40°C | 不可靠 (35%) | LOO 验证 |

### 7.3 已知失败模式

1. **p-CN, p-NO₂ 等强 EWG para-ArLi**: Ea 高估 ~7-33 kJ/mol
2. **benzylLi (sp3)**: 训练点仅 2 个，外推可能给出负 Ea
3. **benzyne precursor (o-I, o-Br)**: 不同机理，Ea 偏差 ~19 kJ/mol
4. **非 THF 溶剂**: 未验证，需要文献校正

---

## 8. Scope 数据与合成 yield

### 8.1 Scope 数据集

`scope_synthesis_dataset.csv`: 201 行, 30 个中间体, 45 种 electrophile

### 8.2 关键发现

在 flash chemistry 条件 (tR < 100ms) 下，**几乎所有中间体 survival ≈ 100%**。Scope yield 的差异主要来自 **electrophile trapping 效率**:

| Electrophile 类型 | 平均 yield |
|---|---|
| protonation (ROH) | 86% |
| silylation (TMSCl) | 82% |
| aldehyde | 78% |
| ketone | 73% |
| Weinreb amide | 68% |
| alkylation (MeI) | 67% |

### 8.3 Trapping 动力学

kd_and_trapping 数据 (122 行) 中提取了 5 个杂环 ArLi 对 octafluorocyclopentene 的 k_trapping。k_trap 也遵循 Arrhenius (Ea_trap = 17-57 kJ/mol)。

---

## 9. 尝试过但未成功的改进方向

| 方向 | 结果 | 结论 |
|---|---|---|
| HF/6-31+G* charge | Mulliken 因弥散函数不稳定 | 弃用 |
| HF/def2-SVP charge | r(Ea)=0.34 < xTB(0.54) | 不如 xTB |
| 加入 σ | 仅 14 个有 σ，过拟合 | 不加入 |
| 分类别建模 | 每类 n 太少 | 统一模型更好 |
| pKa 作为描述符 | 热力学 ≠ 动力学 (r=0.04) | 不相关 |
| 直接预测 log(t½) | LOO-R²=0.18 (best) | 不如经由 Ea |
| Hierarchical Bayesian | t½ CrI 覆盖 100% 但极宽 | 不实用 |
| TS 搜索 (xtb scan) | xtb optimizer ARM64 bug + 1D scan 不充分 | 需 ORCA |
| 外部数据 (USPTO/ORD) | USPTO 无 yield, ORD 无温度 | 不能用 |

---

## 10. 自审：未解决的问题

### 10.1 数据层面
1. **样本量不足**: 26 个训练底物, 3 个参数, 有效自由度 23 — 统计上勉强
2. **类别不平衡**: ArLi(16) 主导，oxiranylLi(6) 次之，其他类仅 1-2 个
3. **Ea 值可能有误**: 新数据集的 Ea 与旧数据集差异显著 (如 p-CO₂tBu: 旧15.0 → 新53.0)
4. **溶剂单一**: 全部 THF, 无法验证溶剂校正

### 10.2 描述符层面
5. **q(C_ipso) 对共轭效应不敏感**: 这是半经验方法 (xTB) + Mulliken population analysis 的固有缺陷。NPA charge 可能解决但需要 NBO 程序
6. ~~**%V_bur 用 UFF 几何**~~ → **已修复**: 所有描述符 (q_C, d_LiC, %V_bur, Sterimol) 现在统一基于 GFN2-xTB 优化几何
7. **3D 构象依赖**: 仅用单构象 (RDKit ETKDGv3 seed=42 → xTB 优化), 未做多构象搜索

### 10.3 模型层面
8. **跨类别 vs 类内**: 模型 R²=0.69 主要来自类间差异, ArLi 内部 LOO-R²=-3.1
9. **Ea→t½ 的指数放大**: 6 kJ/mol Ea 误差 → 22× t½ 误差 (at -40°C)
10. **lnA 不可预测**: 活化熵 ΔS‡ 取决于过渡态和聚集态, 无法从基态描述符得到
11. **Compensation 是统计假象风险**: Ea 和 lnA 从同一组 Arrhenius 回归提取, 误差数学耦合 (Krug et al. 1976)

### 10.4 工具层面
12. **无 confidence interval**: 输出单一 t½ 值, 没有不确定性范围
13. **benzylLi 外推**: 训练仅 2 个点, 可能预测负 Ea
14. **溶剂校正缺乏**: 文献数据显示溶剂效应因底物而异, 无法统一校正

---

## 11. 文件清单

| 文件 | 内容 |
|---|---|
| `clean_organolithium_unified_descriptors.csv` | 主数据集 (2357行×50列) |
| `intermediates_master.csv` | 浓缩表 (75行×52列) |
| `phase_b_arrhenius.csv` | Arrhenius 参数 (40行) |
| `fill_empirical_descriptors.py` | σ, Es, δ 计算 |
| `fill_dft_descriptors.py` | xTB 描述符计算 |
| `fill_new_descriptors.py` | Sterimol, Fukui, volume |
| `model_dft_lfer.py` | Stage 1 DFT-LFER 基线 |
| `model_hierarchical_bayesian.py` | Stage 2 贝叶斯模型 |
| `stage3_comparison.py` | Stage 3 对比 |
| `predict_stability.py` | 预测工具 |
| `scope_synthesis_dataset.csv` | Scope 合成数据 (201行) |
| `literature_halflives_degennaro2014.csv` | 文献 t½ 数据 |
| `hf_def2svp_cache.json` | HF/def2-SVP 计算缓存 |
| `xtb_cache.json` | xTB 计算缓存 |
