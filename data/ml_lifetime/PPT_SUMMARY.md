# PPT Presentation Outline — Organolithium (有機リチウム) Stability Prediction

> A slide-by-slide blueprint for the ml_lifetime project presentation.
> Technical terms are given in **English (日本語)** style for bilingual delivery.
> Figures are placed as `[FIG: path]` placeholders.

---

## Slide 1 — Title

**Title**: Predicting the thermal stability of organolithium (有機リチウム) intermediates — from flow-chemistry (フロー化学) kinetic data to reactor-selection tools

**Sub-title**: A class-specific linear free-energy model (類別線形自由エネルギー関係モデル) built on 14 xTB / geometric descriptors

**Key deliverables**:
- Global Arrhenius (アレニウス) fit across 45 compounds
- Class-specific descriptor (記述子) models for 4 mechanism classes
- 81 % accuracy for flash / flow / batch reactor recommendation

---

## Slide 2 — Background & Motivation (背景と動機)

**Problem**: Organolithium intermediates decompose via attack on THF solvent (tetrahydrofuran, テトラヒドロフラン) or intramolecular pathways. Stability varies by more than 8 orders of magnitude across substrates.

**Gap**: No general framework exists to predict, from molecular structure alone, whether a new aryllithium (アリールリチウム) or oxiranyllithium (オキシラニルリチウム) needs flash, flow, or batch chemistry.

**Goal**: From SMILES + temperature → recommend reactor type in seconds.

`[FIG: analysis_figures/ppt_fig_molecules_base.png]` — substrate gallery (基質群)

---

## Slide 3 — Dataset & Data Flow (データセット)

**Raw data**: `clean_organolithium_unified.csv` — 2,609 (tR, T, yield) triplets from 26 papers, 75 unique intermediates (中間体).

**Global Arrhenius dataset**: 45 compounds with ≥3 temperatures, fitted to a 5-parameter model (5パラメータモデル):

$$\mathrm{yield}(t_R, T) = y_{\max} \cdot (1 - e^{-k_f t_R}) \cdot e^{-k_d t_R}$$

where k_f and k_d are both Arrhenius functions of temperature (温度).

- 5 parameters per compound: Ea_f, lnA_f, Ea_d, lnA_d, y_max
- Median R²_global = 0.879

---

## Slide 4 — Model Tier Classification (層別化)

**Diagnostic** (診断): first-order (一次) decay vs. stretched decay (ストレッチ減衰) via AIC comparison.

| Tier | Criterion | n | Usage |
|---|---|---|---|
| A | ΔAIC > 10, β < 0.8 | 6 | Excluded (非一次動力学) |
| B | 2 < ΔAIC < 10 | 5 | Excluded |
| **C** | first-order sufficient | **30** | **Primary modeling set** |

`[FIG: analysis_figures/ppt_fig_model_comparison.png]` — tier distribution

---

## Slide 5 — Four Mechanism Classes (4つの機構クラス)

After structure-based classification (構造分類) of the 30 Tier-C compounds:

| Class | n | Mechanism | Example |
|---|---|---|---|
| **p-ArLi** | 9 | THF deprotonation (THF 脱プロトン化) | p-bromophenyllithium |
| **m-ArLi** | 5 | THF deprotonation | m-cyanophenyllithium |
| **o-ArLi** | 8 | Chelation (キレート化) / benzyne (ベンザイン) elimination | o-iodophenyllithium |
| **oxiranylLi** | 7 | Ring-opening (開環) | 2-naphthyl-oxiranyllithium |

`[FIG: analysis_figures/class_p_ArLi.png]` — p-ArLi structures
`[FIG: analysis_figures/class_m_ArLi.png]` — m-ArLi structures
`[FIG: analysis_figures/class_o_ArLi.png]` — o-ArLi structures
`[FIG: analysis_figures/class_oxiranylLi.png]` — oxiranylLi structures

---

## Slide 6 — Global Arrhenius Model vs. Raw Data (大域アレニウスモデル vs. 生データ)

Per-class yield-vs-tR curves (収率 - 滞留時間曲線) with the fitted 5-parameter model overlaid on raw experimental data, colored by temperature (温度による色分け).

`[FIG: analysis_figures/yield_curves_p_ArLi.png]`
`[FIG: analysis_figures/yield_curves_m_ArLi.png]`
`[FIG: analysis_figures/yield_curves_o_ArLi.png]`
`[FIG: analysis_figures/yield_curves_oxiranylLi.png]`

**Key observation** (主要観察): oxiranylLi shows the cleanest Arrhenius behavior (median R² > 0.95); p-bromophenyllithium is an outlier with Ea = 149 kJ/mol.

---

## Slide 7 — Descriptor Pool: 4 Computational Levels (記述子プール: 4計算レベル)

We exhaustively (網羅的に) explored ~34 descriptors across 4–5 computational levels (計算レベル):

| Level | Method | n descriptors | Outcome |
|---|---|---|---|
| **L1a** | Empirical (経験値) — σ_Hammett | 1 | ✅ Retained |
| **L1b** | Geometry (幾何) — Sterimol, %V_bur, mol_volume | 5 | ✅ Retained |
| **L1c** | GFN2-xTB (拡張密結合 — semi-empirical QM, 半経験的量子化学) | ~10 | ✅ Retained |
| **L2** | HF/def2-SVP (ハートリー・フォック法) | 6 | ❌ LOO-R² = 0.55 (< L1) |
| **L3** | M06-2X/def2-SVP (DFT, 密度汎関数理論) | 6 | ❌ LOO-R² = 0.41 |
| **L4** | QTAIM / ADCH at DFT level (波動関数解析) | 9 | ❌ ρ(BCP): r = 0.16 |
| **L5** | M06-2X + PCM(THF) (溶媒モデル) | — | ⚠ SCF crash for o-I series |

**Surprising finding** (意外な発見): Higher-level QM did NOT improve prediction — **xTB is the sweet spot** (xTB が最適).

---

## Slide 8 — Selected 14 Descriptors: Meaning (採用14記述子とその意味)

After class-specific LOO cross-validation (一つ抜き交差検証, Leave-One-Out CV), 14 descriptors from L1 were retained.

### Electronic (電子的) — 6 descriptors

| Short | Full meaning |
|---|---|
| **q(C_ipso)** | Mulliken charge (マリケン電荷) on the ipso-C bonded to Li. More positive = more ionic C-Li bond (イオン性の高い C-Li 結合). |
| **HOMO** | Highest Occupied Molecular Orbital energy, eV (最高被占軌道エネルギー). Lower = more stable carbanion (安定な炭素アニオン). |
| **η (gap)** | Chemical hardness (化学的硬度), = (LUMO − HOMO)/2. High η = unreactive (低反応性). |
| **fukui f⁻** | Electrophilic Fukui function (電子受容フクイ関数) at C_ipso. Measures local electron-donating ability (局所的電子供与能). |
| **σ_Hammett** | Hammett substituent constant (ハメット置換基定数). Classical electronic effect scale (古典的電子効果指標). |
| **BDE(Li-C)** | Li-C bond dissociation energy, kJ/mol (リチウム - 炭素結合解離エネルギー). |

### Steric (立体的) — 5 descriptors

| Short | Full meaning |
|---|---|
| **Sterimol B1** | Minimum width perpendicular to C→Li axis, Å (C→Li 軸に垂直な最小幅). |
| **Sterimol B5** | Maximum width perpendicular to C→Li axis, Å (C→Li 軸に垂直な最大幅). |
| **Sterimol L** | Length along C→Li axis, Å (置換基長). |
| **%V_bur** | Buried volume (埋没体積率) around Li, r = 3.5 Å. |
| **mol_volume** | Molecular volume, Å³ (分子体積). |

### Bonding (結合) — 1 descriptor

| Short | Full meaning |
|---|---|
| **d(Li-C)** | Optimized Li-C bond length, Å (最適化 Li-C 結合長). |

### Solvation / Polarity (溶媒和 / 極性) — 2 descriptors

| Short | Full meaning |
|---|---|
| **Gsolv(THF)** | Solvation free energy (溶媒和自由エネルギー) in THF, kJ/mol. Computed via ALPB implicit solvation model (陰溶媒和モデル). |
| **dipole** | Molecular dipole moment, Debye (双極子モーメント). |

---

## Slide 9 — Pearson Correlation of 14 Descriptors (14記述子間のピアソン相関)

Descriptors ordered by physical category (物理類別). Strong within-category correlation (高い族内相関) confirms the grouping.

`[FIG: analysis_figures/descriptor_correlation_heatmap.png]`

**Strong collinearities** (強共線性, |r| > 0.8):
- %V_bur ↔ BDE(Li-C) r = +0.79 — bulky substituents → stronger C-Li bond (大きな置換基 → 強い C-Li 結合)
- mol_volume ↔ Sterimol B5 r = +0.80 — both measure molecular size (両者とも分子サイズを反映)
- %V_bur ↔ dipole r = −0.75 — large steric → less polar (大きな立体障害 → 低い極性)

**Cross-category weak correlations** (類別間の弱相関): Solvation (Gsolv, dipole) is largely orthogonal to Electronic and Steric — **independent information for lnA prediction** (lnA 予測のための独立情報).

---

## Slide 10 — Class-Specific Models: Ea (Activation Enthalpy, 活性化エンタルピー)

Each class has its own best descriptor combination via exhaustive LOO search (網羅的 LOO 探索).

| Class | Ea_d formula | LOO-R² |
|---|---|---|
| **p-ArLi** | Ea_d = −8429·fukui + 2.47·vol − 4.30·dipole − 1227 | 0.642 |
| **m-ArLi** | Ea_d = −83.2·dipole − 13.8·B5 − 33.1·σ + 890 | **0.985** |
| **o-ArLi** | Ea_d = +1.11·Gsolv − 3.73·B5 + 0.15·vol + 120 | **0.987** |
| **oxiranylLi** | Ea_d = +278·HOMO − 505·fukui + 42.0·dipole + 2249 | **0.998** |

**Pattern** (パターン): Ea models consistently combine **Electronic + Steric** descriptors (電子 + 立体記述子) — reflecting C-Li bond reorganization cost (C-Li 結合再配列のエネルギー).

---

## Slide 11 — Class-Specific Models: lnA (Pre-exponential Factor, 前指数因子)

| Class | lnA_d formula | LOO-R² |
|---|---|---|
| **p-ArLi** | lnA_d = −3401·d_LiC + 2.94·BDE + 24.1·B5 + 5206 | **0.982** |
| **m-ArLi** | lnA_d = **−0.958·Gsolv** − 6.16·η − 46.1 | **0.998** |
| **o-ArLi** | lnA_d = −138·d_LiC **− 2.97·Gsolv** − 0.80·vol + 231 | 0.944 |
| **oxiranylLi** | lnA_d = −282·fukui + 16.3·B1 + 7.17·B5 − 67.8 | **0.995** |

**Pattern** (パターン): lnA models consistently involve **Gsolv(THF) or polarity-related descriptors** (溶媒和または極性関連記述子) — reflecting THF coordination changes in the transition state (遷移状態での THF 配位変化).

---

## Slide 12 — Physical Interpretation (物理的解釈)

Combining Slides 10 & 11 yields the central conclusion:

$$\boxed{\;E_a \;(\mathrm{enthalpy},\,エンタルピー) \;\leftarrow\; \text{Electronic} + \text{Steric}\;}$$
$$\boxed{\;\ln A \;(\mathrm{entropy},\,エントロピー) \;\leftarrow\; \text{Solvation} + \text{Polarity}\;}$$

**Theoretical support** (理論的裏付け):
- **Eyring transition state theory** (アイリング遷移状態理論, Eyring 1935): Ea ≈ ΔH‡, lnA ∝ ΔS‡
- **Reich 2013** (*Chem. Rev.* 113, 7130): ΔS‡ = −32.5 eu for n-BuLi + Ph₃CH in THF, explicitly attributed to "extra solvation required to dissociate a tetramer to dimers" (4量体から2量体への解離に必要な余分な溶媒和)
- **Collum 2007** (*Angew. Chem. Int. Ed.* 46, 3002): solvation (溶媒和) and aggregation (会合) are first-order determinants of organolithium reactivity (有機リチウム反応性の一次決定因子)

`[FIG: analysis_figures/descriptor_mechanism_heatmap.png]` — visual summary of descriptor-parameter mapping

---

## Slide 13 — Reactor Recommendation Tool (反応器推奨ツール)

**Workflow** (ワークフロー):
1. Input (入力): SMILES + temperature T (°C)
2. Compute 14 descriptors (記述子計算) from xTB-optimized geometry (数秒 / 化合物)
3. Apply class-specific formulas → Ea_d, lnA_d, y_max
4. Compute k_d(T), then t_max = argmax yield(tR)
5. Classify (分類):
   - t_max < 0.1 s → **flash chemistry** (フラッシュ化学 — microsecond mixing, マイクロ秒混合)
   - 0.1 s < t_max < 60 s → **flow chemistry** (フロー化学)
   - t_max > 60 s → **batch** (バッチ反応)

---

## Slide 14 — Validation Performance (検証性能)

**Leave-One-Out cross-validation** (一つ抜き交差検証) on the Tier-C modeling set:

| Temperature range | Accuracy (正答率) | n |
|---|---|---|
| **−20 to +25 °C (recommended)** | **81 %** | 121 / 150 |
| 0 to +25 °C | 78 % | 78 / 100 |
| −78 to +25 °C (full) | 78 % | 156 / 200 |

**Per-class accuracy @ −20 to +25 °C** (クラス別正答率):
- m-ArLi: **100 %** (30/30)
- oxiranylLi: **89 %** (32/36)
- p-ArLi: 79 % (33/42)
- o-ArLi: 62 % (26/42)

`[FIG: analysis_figures/step7b_parity_plots.png]` — parity plots (予測 vs. 実測)

---

## Slide 15 — Case Study: p-Cyanophenyllithium (p-シアノフェニルリチウム)

From the predicted Ea_d / lnA_d:
- At −78 °C: t_max ≈ 100 s → **batch compatible** (バッチ可)
- At −20 °C: t_max ≈ 1 s → **flow required** (フロー要求)
- At 0 °C: t_max < 0.1 s → **flash required** (フラッシュ要求)

Experimental cross-check (実験的照合): matches published flow chemistry conditions for this compound.

`[FIG: analysis_figures/yield_curves_p_ArLi.png]` — highlight panel (h), p-cyanophenyllithium

---

## Slide 16 — Limitations & Outlook (制限事項と展望)

**Limitations** (制限事項):
- Only 30 Tier-C compounds — more data needed (データ拡充が必要)
- o-ArLi accuracy (62 %) — benzyne and chelation pathways need more samples (ベンザインおよびキレート機構のサンプル不足)
- p-ArLi Ea_f unpredictable — replaced by class mean (予測不能のためクラス平均で代用)
- Tier A (n = 6) compounds follow stretched decay — require kinetic models beyond first-order (一次モデルを超える動力学モデルが必要)

**Outlook** (展望):
- Expand dataset with more electron-rich aryllithiums (電子豊富アリールリチウムの追加)
- Explicit THF coordination numbers via AIMD (分子動力学) to improve lnA predictions
- Extend to heteroaryllithiums (ヘテロアリールリチウム, pyridyl, furyl, etc.)

---

## Slide 17 — Conclusion (結論)

1. **Global Arrhenius fitting** (大域アレニウス拟合) on flow-chemistry yield data provides clean Ea and lnA for 30 compounds.

2. **Exhaustive descriptor screening** (網羅的記述子スクリーニング) across 4 QM levels identifies 14 informative descriptors — all at GFN2-xTB level; higher DFT adds noise (高次 DFT は改善せず).

3. **Ea is driven by Electronic + Steric descriptors** (活性化エネルギーは電子・立体記述子に依存); **lnA is driven by Solvation descriptors** (前指数因子は溶媒和記述子に依存) — supporting the Eyring / Reich framework.

4. **Reactor recommendation accuracy** (反応器推薦精度): **81 % at −20 to +25 °C** from SMILES + T input only.

5. Tool ready for organic chemists planning new flow-chemistry campaigns (フロー化学実験計画に応用可能).

---

## Appendix A — Descriptor Pool Inventory (付録 A: 記述子一覧)

Available as table: `analysis_figures/descriptor_inventory.csv`
34 descriptors × 5 computational levels × 4 physical categories.

## Appendix B — Fitted Model Parameters (付録 B: 拟合モデルパラメータ)

Available as table: `analysis_figures/class_fitted_models.csv`
16 (class × parameter) linear models with coefficients, intercepts, LOO-R².

## Appendix C — Caption Files (付録 C: 図表キャプション)

- `analysis_figures/yield_curves_captions.md` — Markdown captions
- `analysis_figures/yield_curves_captions.tex` — LaTeX captions

---

## Figure File Reference (図ファイル一覧)

| Slide | File | Purpose |
|---|---|---|
| 2 | `ppt_fig_molecules_base.png` | Substrate gallery |
| 4 | `ppt_fig_model_comparison.png` | Tier distribution |
| 5 | `class_{p,m,o}_ArLi.png`, `class_oxiranylLi.png` | Structures per class |
| 6 | `yield_curves_{p,m,o}_ArLi.png`, `yield_curves_oxiranylLi.png` | Raw + fit |
| 9 | `descriptor_correlation_heatmap.png` | Pearson matrix |
| 12 | `descriptor_mechanism_heatmap.png` | Ea / lnA descriptor usage |
| 14 | `step7b_parity_plots.png` | Model parity |

---

**Last updated**: 2026-04-23 (notebook 38 cells, accuracy 81 %)
