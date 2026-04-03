# Organolithium Intermediate Lifetime Prediction via ML

## Motivation

Flow chemistry data encodes a hidden treasure: the **lifetime of unstable organolithium intermediates**. When yield is plotted against residence time (tR) at various temperatures, the resulting curves show:

1. **Rising phase** (tR too short → reaction incomplete → low yield)
2. **Peak** (optimal tR → maximum yield)
3. **Decay phase** (tR too long → intermediate decomposes → yield drops)

The decay phase directly encodes the intermediate's thermal stability. By fitting kinetic models to these curves, we can extract **half-lives (t₁/₂)** and **decomposition activation energies (Ea)** — quantities that are expensive to measure experimentally but critical for reaction optimization.

## Dataset

Source: `data/final_output/organolithium_tr_subdataset_vlm_enriched.csv`
- 1449 data points from 14 papers
- 1183 tR1 (lithiation step) points across 11 unique intermediates
- Each intermediate has 4–8 temperature levels × 6–9 tR levels
- All species have validated SMILES (RDKit-parseable)
- 2 intermediates have literature-reported lifetimes (validation set)

### Intermediates with decay data

| Intermediate | SMILES | Points | T levels | Decay visible? |
|---|---|---|---|---|
| tBu 4-(lithio)benzoate | `CC(C)(C)OC(=O)c1ccc([Li])cc1` | 495 | 6 | Strong (≥-60°C) |
| oxiranyllithium | `[Li]C1CO1` | 212 | 8 | Strong (≥-50°C) |
| tBu o-(lithio)benzoate | `CC(C)(C)OC(=O)c1ccccc1[Li]` | 131 | 6 | Moderate (≥0°C) |
| p-lithiobenzonitrile | `N#Cc1ccc([Li])cc1` | 36 | 6 | Moderate (≥0°C) |
| o-lithiobenzonitrile | `N#Cc1ccccc1[Li]` | 36 | 6 | Weak (≥0°C) |
| m-lithiobenzonitrile | `N#Cc1cccc([Li])c1` | 36 | 6 | Weak (≥20°C) |
| carbolithiation ArLi | `Brc1ccccc1[Li]` | 32 | 4 | Strong (≥-60°C) |
| I,Li-benzene | `Ic1ccccc1[Li]` | 29 | 4 | Very strong (all T) |
| iodofluoromethyllithium | `[Li]C(F)I` | 28 | 4 | Very strong (all T) |
| fluoromethyllithium | `[Li]CF` | 27 | 4 | Very strong (all T) |
| aryllithium (borylated) | `[Li]c1ccccc1` | 15 | 2 | Moderate |

### Validation data (from literature, example1 paper)

| Intermediate | Known lifetime | Temperature |
|---|---|---|
| fluoromethyllithium (Li-CH2-F) | 13 ms | -60°C |
| iodofluoromethyllithium (CHLi(I)(F)) | 82 ms | -40°C |

---

## Phase A: Half-life extraction from yield-vs-tR curves

### Kinetic model

The yield curve reflects two competing processes:

```
Substrate + OrgLi  --k_f-->  ArLi (intermediate)     [formation]
ArLi               --k_d-->  Decomposition products   [decay]
```

For a plug-flow reactor with residence time tR:

```
yield(tR) = (k_f / (k_d - k_f)) * (exp(-k_f * tR) - exp(-k_d * tR))
```

When k_f >> k_d (fast formation, slow decay — most systems):
```
yield(tR) ≈ (1 - exp(-k_f * tR)) * exp(-k_d * tR)
```

Fit parameters: k_f (formation rate), k_d (decomposition rate), y_max (scale factor)

### Output

For each (intermediate, T):
- `k_f`: formation rate constant (s⁻¹)
- `k_d`: decomposition rate constant (s⁻¹)
- `t_half`: ln(2) / k_d (seconds)
- `t_peak`: tR at maximum yield
- `y_max`: maximum achievable yield (%)
- `fit_r2`: goodness of fit

Output file: `data/ml_lifetime/phase_a_halflives.csv`

### Implementation

Script: `scripts/ml_lifetime/phase_a_extract_halflife.py`

---

## Phase B: Arrhenius fitting → Ea per intermediate

### Model

For each intermediate, fit k_d(T) to the Arrhenius equation:

```
ln(k_d) = ln(A) - Ea / (R * T)
```

Where:
- Ea = activation energy of decomposition (kJ/mol)
- A = pre-exponential factor (s⁻¹)
- R = 8.314 J/(mol·K)
- T in Kelvin

### Output

For each intermediate:
- `Ea_decomp_kJ_mol`: activation energy
- `ln_A`: log of pre-exponential factor
- `T_range_K`: temperature range used for fitting
- `n_temperatures`: number of temperature points
- `arrhenius_r2`: goodness of fit

Output file: `data/ml_lifetime/phase_b_arrhenius.csv`

### Implementation

Script: `scripts/ml_lifetime/phase_b_arrhenius.py`

---

## Phase C: QSPR model — predict lifetime from structure

### Molecular descriptors

From intermediate SMILES, compute:
- **Fingerprints**: Morgan (radius=2, 1024-bit), MACCS keys
- **Physicochemical**: MW, LogP, TPSA, HBA/HBD, rotatable bonds
- **Electronic**: partial charges on Li, C-Li bond environment
- **Structural**: ring count, aromatic atoms, heteroatom count, functional group flags (CN, CO, ester, halide)

### Model candidates

Given small dataset (11 intermediates → ~11 Ea values):
1. **Linear regression** with 3-5 selected descriptors (interpretable)
2. **Random Forest** with leave-one-out CV
3. **Gaussian Process** regression (uncertainty quantification)
4. **Augmented training**: add literature organolithium lifetime data (target: 30+ molecules)

### Validation strategy

- Leave-one-out cross-validation (LOOCV) across 11 intermediates
- External validation: predict lifetime of fluoromethyllithium and iodofluoromethyllithium, compare to known values (13 ms and 82 ms)
- Chemical interpretability: do extracted Ea values make chemical sense? (e.g., electron-withdrawing groups → more stable ArLi)

### Output

- Trained model + feature importances
- Predicted vs actual Ea scatter plot
- Per-intermediate lifetime predictions at standard temperatures (-78°C, -40°C, 0°C, 25°C)

Output file: `data/ml_lifetime/phase_c_qspr_results.csv`

### Implementation

Script: `scripts/ml_lifetime/phase_c_qspr.py`

---

## Directory structure

```
scripts/ml_lifetime/
  phase_a_extract_halflife.py
  phase_b_arrhenius.py
  phase_c_qspr.py

data/ml_lifetime/
  phase_a_halflives.csv
  phase_b_arrhenius.csv
  phase_c_qspr_results.csv
```

---

## Key risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| yield decay not purely first-order | Ea estimates biased | Use competing kinetics model; flag poor fits |
| Only 11 intermediates for QSPR | Overfitting | LOOCV + augment with literature data |
| Mixing effects confound kinetics | k_f artificially low | Use papers with known good micromixing (Nagaki) |
| Some curves lack clear decay | Cannot extract t₁/₂ | Skip; report confidence per intermediate |
