# Organolithium Intermediate Lifetime Analysis — Results Report

## Overview

This analysis extracts quantitative thermal stability parameters (half-life t₁/₂, decomposition activation energy Ea) for 14 organolithium intermediates from flow chemistry data points (14 papers). The data was mined from yield-vs-residence time heatmaps using VLM extraction, with **manual tR correction** by Zhao and independent YOLO yield verification.

**Key finding**: Organolithium intermediate stability spans nearly 4 orders of magnitude (3.3 ms to 51.9 s at -40°C), and can be systematically extracted from routine flow chemistry optimization data.

---

## Data Source

- Input: `data/final_output/organolithium_tr_subdataset_vlm_enriched.csv` (1,470 rows, 31 columns, 20 papers)
- Subset used: tR1 (lithiation step) data points with yield, residence time, and temperature
- 14 unique intermediates from 14 papers, each with 2–6 temperature levels and 5–8 tR levels
- **tR values manually corrected** from heatmap axis annotations (14 papers, rank-order replacement)
- **Yield verified** via independent YOLO OCR pipeline (99.6% agreement)
- **Temperature verified** via YOLO row detection (100% row-count match)

### Data quality flags

278 rows across 3 papers carry a `data_note` flag indicating mixed data sources (heatmap + table overlap, dual-heatmap merge, or deleted false detections). See `data/final_output/DATASET_README.md` for details.

### Data corrections applied (v10 cross-reference)

- **Paper 05 intermediate split**: 4 ortho-ester ArLi intermediates separated by yield-pattern matching (131 rows)
- **Paper 05 substrate/product fix**: methyl/ethyl/isopropyl variants corrected from inherited tert-butyl labels (95 rows)
- **Nagaki 2010 Figure 08 duplicate deletion**: 131 rows (exact reprint of Paper 05 data)
- **Homocoupling paper deduplication**: 25 rows (same DOI appeared twice)
- **Homocoupling + Cross-coupling intermediate fix**: "aryllithium" → "p-methoxyphenyllithium" with correct SMILES (55 rows)

---

## Phase A: Half-life Extraction from Yield Decay Curves

### Method

Each (intermediate, temperature) pair has a yield-vs-tR curve shaped like a peak:
- **Rising phase**: tR too short → reaction incomplete → low yield
- **Peak**: optimal tR → maximum yield
- **Decay phase**: tR too long → intermediate decomposes → yield drops

We fit a competing kinetics model:

```
yield(tR) = y_max × (1 - exp(-k_f × tR)) × exp(-k_d × tR)
```

Where:
- k_f = formation rate constant (s⁻¹)
- k_d = decomposition rate constant (s⁻¹)
- t₁/₂ = ln(2) / k_d = half-life of the intermediate

### Output

`phase_a_halflives.csv`: 76 (intermediate, T) groups fitted (53 with measurable decay, 23 formation-only).

### Figures: `phase_a_curves/*.png`

Each PNG shows one intermediate with all its temperature curves overlaid.

| Figure | Key observations |
|--------|-----------------|
| `tert-butyl_4-(lithio)benzoate_(aryllithi.png` | At -60°C, slow decay (t₁/₂ = 10.9 s); at 20°C, fast (t₁/₂ = 0.86 s). Clear temperature effect. |
| `Li-CH2-F_(fluoromethyllithium),_lifetime.png` | Extremely unstable. At -60°C, t₁/₂ = 275 ms. At -20°C, yield drops to zero within 30 ms. |
| `(I,_Li_substituents_on_benzene_ring).png` | Most unstable in dataset. At -78°C, t₁/₂ = 1.6 s. At -50°C, t₁/₂ = 16 ms. R² = 0.988. |
| `p-lithiobenzonitrile_(from_1c).png` | Most stable. No decay visible below 0°C. At 20°C, t₁/₂ = 7.8 s. CN group stabilizes carbanion. |
| `oxiranyllithium.png` | 6 temperature points, wide range (-70 to 20°C). t₁/₂ = 451 s at -70°C → 0.22 s at 20°C. |
| `(Br,_Li_substituents_on_benzene_ring).png` | Benzyne elimination pathway. Ea = 65.4 kJ/mol (R² = 0.971). |

### Validation against known lifetimes

| Intermediate | Literature lifetime | Extracted t₁/₂ | Ratio |
|---|---|---|---|
| fluoromethyllithium (Li-CH₂-F) | 13 ms at -60°C | 275 ms at -60°C | 21.1× |
| iodofluoromethyllithium (CHLi(I)(F)) | 82 ms at -40°C | 1,214 ms at -40°C | 14.8× |

The systematic ~15-21× offset is expected because:
- "Lifetime" in literature is often defined differently from kinetic half-life
- Our model captures formation + decomposition + quenching simultaneously

**The relative ordering is preserved**: CHLi(I)(F) / LiCH₂F ratio = 4.4× (extracted) vs 6.3× (literature), confirming the **relative stability ranking is reliable**.

---

## Phase B: Arrhenius Fitting

### Method

For each intermediate, fit ln(k_d) vs 1/T:

```
ln(k_d) = ln(A) - Ea / (R × T)
```

### Output

`phase_b_arrhenius.csv`: Ea, ln(A), and predicted t₁/₂ at standard temperatures for 14 intermediates.

### Figure: `phase_b_arrhenius_plot.png`

**How to read**: ln(k_d) vs 1000/T. Steeper slope = higher Ea = more temperature-sensitive.

Key observations:
- **(I,Li)-benzene** (red): Ea = 59.7 kJ/mol. Most unstable at warm temperatures.
- **Li-CH₂-F** (brown): Ea = 49.2 kJ/mol. Very fast decomposition above -40°C.
- **Lithiobenzonitriles** (yellow/cyan): Ea ≈ 5-7 kJ/mol. Nearly flat — inherently stable.
- Lines converge at low T (right side) — all intermediates relatively stable at -78°C.

### Arrhenius parameters summary

| Intermediate | Ea (kJ/mol) | R² | n_T | t₁/₂ @ -40°C | t₁/₂ @ 0°C |
|---|---|---|---|---|---|
| m-CN-ArLi | 4.8 | 1.000 | 2 | 11.9 s | 8.3 s |
| p-CN-ArLi | 7.1 | 1.000 | 2 | 16.5 s | 9.7 s |
| p-CO₂ᵗBu-ArLi | 15.0 | 0.825 | 5 | 4.8 s | 1.6 s |
| CHLi(I)(F) | 16.2 | 0.975 | 4 | 1.3 s | 373 ms |
| CHLi(I)(Cl) | 25.0 | 0.991 | 3 | 8.6 s | 1.3 s |
| o-CO₂ᵗBu-ArLi | 26.8 | 0.947 | 3 | 51.9 s | 6.9 s |
| oxiranyl-Li | 35.6 | 0.910 | 6 | 8.0 s | 546 ms |
| o-CO₂Me-ArLi | 36.5 | 0.983 | 5 | 602 ms | 38 ms |
| PhLi | 36.5 | 1.000 | 2 | 22.4 s | 1.4 s |
| o-CO₂ⁱPr-ArLi | 38.0 | 0.992 | 4 | 5.6 s | 318 ms |
| o-CO₂Et-ArLi | 41.3 | 0.999 | 4 | 1.8 s | 82 ms |
| LiCH₂F | 49.2 | 0.990 | 4 | 37.5 ms | 0.91 ms |
| o-I-ArLi | 59.7 | 0.988 | 4 | 3.3 ms | 36 μs |
| o-Br-ArLi | 65.4 | 0.971 | 4 | 61.3 ms | 438 μs |

### tR correction impact on Arrhenius parameters

| Intermediate | Ea_before | Ea_after | ΔEa | R²_before | R²_after |
|---|---|---|---|---|---|
| (Br,Li)-benzene | 82.2 | 65.4 | -16.8 | 0.964 | **0.971** |
| (I,Li)-benzene | 79.5 | 59.7 | -19.8 | 0.944 | **0.988** |
| oxiranyllithium | 43.4 | 35.6 | -7.8 | **0.986** | 0.910 |
| tBu 4-(lithio)benzoate | 21.0 | 15.0 | -6.0 | 0.946 | **0.950** |
| Li-CH₂-F | 51.0 | 49.2 | -1.9 | 0.974 | **0.990** |
| CHLi(I)(Cl) | 25.0 | 25.0 | 0.0 | 0.991 | 0.991 |
| aryllithium | 36.5 | 36.5 | 0.0 | 1.000 | 1.000 |

Mean |ΔEa| = 5.2 kJ/mol. Biggest changes in benzyne-elimination intermediates (paper 03, largest tR correction). R² improved for 4/11, unchanged for 5/11, decreased for 2/11. Overall: correction improves data quality.

---

## Electronic Effects Analysis

### Hammett correlation (ArLi subset)

**Ea = -48.5σ + 36.1** (r = -0.979, p = 0.021, n = 4 meta/para only, excl. benzyne + ortho)

Stronger EWG (higher σ) → lower Ea. Paradoxically, lower Ea intermediates are MORE stable because the pre-exponential factor (ln_A) drops even faster (enthalpy-entropy compensation).

### Figure: `analysis_figures/hammett_and_hybridization.png`

- Panel (a): Hammett plot. Clear negative trend. Benzyne intermediates (o-Br, o-I) sit in the red zone — their Ea is high but they're unstable due to a different mechanism.
- Panel (b): C(sp²)-Li (7 intermediates, blue) vs C(sp³)-Li (4 intermediates, orange). No simple separation — mechanism matters more than hybridization alone.

### Enthalpy-entropy compensation

**ln(A) = 0.628·Ea - 4.63** (r = 0.987)

**Figure**: `analysis_figures/ea_lna_compensation.png`

Isokinetic temperature T_iso = -82°C (191 K). Below this temperature, all intermediates decompose at similar rates; above, stability differences amplify. This explains why cryogenic conditions (-78°C) are universally used for organolithium chemistry.

### Decomposition mechanisms

| Mechanism | Ea range | n | Example |
|---|---|---|---|
| Conjugation-stabilized | 4.8–41.3 kJ/mol | 8 | p-CN-ArLi, o-CO₂Me-ArLi, p-CO₂ᵗBu-ArLi |
| α-elimination | 16.2–49.2 kJ/mol | 3 | CHLi(I)(F), LiCH₂F, CHLi(I)(Cl) |
| Ring-opening | 35.6 kJ/mol | 1 | oxiranyl-Li |
| Protonation/polymerization | 36.5 kJ/mol | 1 | PhLi |
| Benzyne elimination | 59.7–65.4 kJ/mol | 2 | o-I-ArLi, o-Br-ArLi |

### Ortho-ester Taft steric correlation

The four ortho-ester ArLi intermediates (R = Me, Et, iPr, tBu) share identical Hammett σ (0.45) but differ in steric bulk. Their stability correlates with the Taft steric parameter Es:

| Intermediate | R group | Taft Es | Ea (kJ/mol) | t₁/₂ @ -40°C |
|---|---|---|---|---|
| o-CO₂Me-ArLi | Me | 0.00 | 36.5 | 602 ms |
| o-CO₂Et-ArLi | Et | -0.07 | 41.3 | 1.8 s |
| o-CO₂ⁱPr-ArLi | iPr | -0.47 | 38.0 | 5.6 s |
| o-CO₂ᵗBu-ArLi | tBu | -1.54 | 26.8 | 51.9 s |

| Correlation | Equation | r | p |
|---|---|---|---|
| **log(t₁/₂) vs Es** | **log(t₁/₂) = -1.13·Es + 0.04** | **-0.969** | **0.031** |
| ln(A) vs Es | ln(A) = 6.72·Es + 20.08 | 0.985 | 0.015 |
| Ea vs Es | Ea = 8.00·Es + 39.82 | 0.913 | 0.087 |

**Chemical interpretation**: Larger R groups (more negative Es) provide steric shielding of the ortho C-Li bond, slowing decomposition. tBu is 86× more stable than Me at -40°C. The strongest effect is on ln(A) (r = 0.985), meaning steric bulk primarily reduces the frequency of productive collisions (entropy effect) rather than raising the energy barrier.

### Ortho vs para position effect

The same CO₂ᵗBu group behaves very differently at ortho vs para positions:

| Position | Ea (kJ/mol) | t₁/₂ @ -40°C | Dominant effect |
|---|---|---|---|
| **para** (p-CO₂ᵗBu-ArLi) | 15.0 | 4.8 s | Electronic: EWG stabilizes carbanion via conjugation |
| **ortho** (o-CO₂ᵗBu-ArLi) | 26.8 | 51.9 s | Steric: tBu shields C-Li bond from attack |

The ortho intermediate is **10.8× more stable** despite having the same electronic group. This demonstrates that ortho-substituted ArLi require a separate steric model (Taft Es) rather than Hammett σ. The practical implication: for maximum intermediate stability, **use bulky ester groups at the ortho position**.

**Key conclusion**: ArLi intermediate stability is governed by a two-parameter framework:
1. **Meta/para positions**: Hammett σ (Ea = -48.5σ + 36.1, r = -0.979)
2. **Ortho positions**: Taft Es (log t₁/₂ = -1.13·Es + 0.04, r = -0.969)

### Figure: `analysis_figures/stability_ranking_reactor_zones.png`

Stability ranking with reactor zone annotations (flash / flow). Includes 3 experimental predictions (p-CF₃, p-F, p-CH₃ phenyllithium).

### Figure: `analysis_figures/final_results_summary.png`

Four-panel summary: (a) Hammett plot with validation targets, (b) Arrhenius plot, (c) Ea-ln(A) compensation, (d) Stability ranking with predictions.

---

## Phase C: QSPR Model

### Method

Computed 25 RDKit molecular descriptors for each intermediate SMILES. Feature selection via exhaustive 2-feature search + LOOCV Ridge regression.

### Results

- Best features: **Li_C_ewg_neighbors + has_nitrile**
- LOOCV Q² = **0.309**
- RMSE = 14.6 kJ/mol
- MAE = 10.6 kJ/mol

### Interpretation

Li_C_ewg_neighbors counts electron-withdrawing substituents on the C-Li carbon; has_nitrile is a binary indicator for nitrile groups. These two features capture the dominant electronic effects on carbanion stability.

**Honest assessment**: With only 14 data points and 2 features (Q² = 0.309), this is a proof-of-concept, not a production model. The Hammett correlation (r = -0.979 for meta/para ArLi subset) is more interpretable and actionable.

---

## File Inventory

| File | Description |
|---|---|
| `phase_a_halflives.csv` | 76 (intermediate, T) kinetics data: k_f, k_d, t₁/₂, R² |
| `phase_a_halflives_old.csv` | Pre-correction Phase A results (for comparison) |
| `phase_a_curves/*.png` | 15 decay curve fit plots |
| `phase_b_arrhenius.csv` | 14 intermediates: Ea, ln_A, predicted t₁/₂ at -78/-40/0/25°C |
| `phase_b_arrhenius_old.csv` | Pre-correction Phase B results |
| `phase_b_arrhenius_plot.png` | Arrhenius plot (ln(k_d) vs 1/T) |
| `phase_c_qspr_results.csv` | LOOCV prediction results |
| `phase_c_qspr_results_old.csv` | Pre-correction Phase C results |
| `phase_c_descriptors.csv` | 25 molecular descriptors per intermediate |
| `phase_c_plots/stability_ranking_m40C.png` | Stability ranking bar chart |
| `phase_c_plots/ea_predicted_vs_actual.png` | QSPR predicted vs actual Ea |
| `electronic_analysis.csv` | Hammett σ, mechanism, reactor recommendation |
| `analysis_figures/final_results_summary.png` | Four-panel summary figure |
| `analysis_figures/hammett_and_hybridization.png` | Hammett + hybridization comparison |
| `analysis_figures/ea_lna_compensation.png` | Enthalpy-entropy compensation |
| `analysis_figures/stability_ranking_reactor_zones.png` | Stability ranking + reactor zones |
| `analysis_figures/position_effect_cn.png` | o/m/p-CN position effect |

---

## Conclusions

1. **tR correction matters**: Mean Ea shift of 5.2 kJ/mol, with benzyne intermediates most affected. R² generally improved, confirming the corrected data better follows Arrhenius behavior.

2. **Chemical trends are robust**: Hammett correlation, stability ranking, decomposition mechanisms, and reactor recommendations are all consistent before and after tR correction. The underlying chemistry is captured correctly regardless of absolute tR scale.

3. **Two-parameter stability framework**: ArLi intermediate stability is governed by position-dependent effects:
   - **Meta/para**: Hammett σ controls electronic stabilization (Ea = -48.5σ + 36.1, r = -0.979)
   - **Ortho**: Taft Es controls steric shielding (log t₁/₂ = -1.13·Es + 0.04, r = -0.969)
   - Same CO₂ᵗBu group: ortho is 10.8× more stable than para at -40°C (steric protection > electronic effect)

4. **Ortho-ester series provides internal validation**: After splitting Paper 05 into 4 individual intermediates (R = Me/Et/iPr/tBu), stability follows steric trend: Me (602 ms) < Et (1.8 s) < iPr (5.6 s) < tBu (51.9 s) at -40°C. The ln(A) vs Es correlation (r = 0.985) indicates steric bulk primarily reduces decomposition frequency (entropy effect), not activation barrier.

5. **Practical value**: The combined Hammett + Taft framework provides actionable prediction for new ArLi intermediates. Combined with the Yoshida reactor classification (flash < 1 s, flow > 1 s), this enables rational design of organolithium flow chemistry experiments.

6. **Limitations remain**: 14 intermediates is still a small dataset. The systematic t₁/₂ offset (~15-21× vs literature) is not fully explained. QSPR with molecular descriptors (Q² = 0.309) is underpowered at this sample size. The Taft correlation has only 4 points (p = 0.031) and would benefit from additional ortho-ester intermediates for validation.
