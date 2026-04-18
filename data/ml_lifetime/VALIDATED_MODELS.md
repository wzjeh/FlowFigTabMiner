# Validated Models for Organolithium Decomposition Kinetics

## Core Finding

**No universal descriptor model can predict organolithium decomposition across structural classes.**
Leave-One-Class-Out (LOCO) validation: R² < 0 for all cross-class predictions (as low as -12.9).
This reveals **mechanistic heterogeneity** — different classes decompose via distinct pathways
requiring class-specific descriptor models.

---

## Validation Strategy

1. **LOO-CV**: Leave-one-out cross-validation for all models
2. **Permutation test**: 1000 random shuffles, all models p < 0.005
3. **LOCO**: Leave-one-class-out proves cross-class generalization fails
4. **Descriptor robustness**: Removing frontier orbital descriptors (HOMO/η) to check for ΔG‡ encoding

---

## Global Model (baseline)

### Ea prediction (n=26)
- **q(C_ipso) + d(Li-C) + %V_bur → Ea**
- Ea = +152.6×q_C - 338.0×d_LiC + 73.1×%Vbur + 700.6
- LOO-R² = 0.694, MAE = 5.6 kJ/mol, **p = 0.000**
- Physical: electronic + bonding + steric
- **Limitation**: t½ ranking ρ ≈ 0 at -40°C due to Ea-lnA compensation + exponential amplification

---

## Class-Specific Models (Primary Results)

### m+p ArLi (n=12): Conjugation-stabilized carbanions

**Ea prediction:**
- **xtb_ω + BDE(Li-C) + Sterimol_B1 → Ea**
- LOO-R² = 0.755, p = 0.000
- Physical: electrophilicity + bond strength + minimum steric width

**t½@-40°C prediction (recommended, no HOMO/η):**
- **BDE(Li-C) + H(BCP) + mol_volume → log(t½)**
- LOO-R² = 0.608, p < 0.005
- Physical: bond strength + bond energy density + molecular size (→ ΔS‡)
- No frontier orbital descriptors — robust against ΔG‡ encoding

**t½@-40°C prediction (with HOMO/η, higher but cautionary):**
- xtb_η + H(BCP) + mol_volume → log(t½)
- LOO-R² = 0.716, ρ = 0.902, p = 0.001
- Comparable performance without η (R²=0.61) suggests η provides genuine information, not pure ΔG‡ encoding

**t½@-78°C prediction:**
- HOMO + Gsolv + mol_volume → log(t½)
- LOO-R² = 0.766, ρ = 0.895, p = 0.002
- Without HOMO: R² drops below 0 — HOMO critical at this temperature

### ortho-ArLi (n=6): Chelation + benzyne elimination

**Ea prediction (no HOMO/η, fully robust):**
- **σ_hammett + Sterimol_B5 → Ea**
- LOO-R² = 0.914, p = 0.004
- Physical: Hammett electronic effect + maximum steric width
- Captures both ester chelation (σ=0.45, varying B5) and benzyne (σ=0.23-0.35, small B5)

**t½@-40°C prediction (with frontier orbitals):**
- HOMO + xtb_η → log(t½)
- LOO-R² = 0.974, ρ = 1.000, p = 0.003
- **Caution**: n=6, may partially encode ΔG‡

**t½@-40°C prediction (without HOMO/η, robust):**
- mol_volume + BDE(Li-C) → log(t½)
- LOO-R² = 0.573
- Lower but more trustworthy

### oxiranylLi (n=8): Ring-opening mechanism

**Ea prediction:**
- **BDE(Li-C) + mol_volume → Ea**
- LOO-R² = 0.921, MAE = 4.0 kJ/mol, p = 0.000
- Physical: C-Li bond strength + molecular size (ring strain proxy)

**t½@-40°C prediction:**
- **Gsolv(THF) + fukui_f⁻(C_ipso) → log(t½)**
- LOO-R² = 0.939, ρ = 0.905, p = 0.000
- Physical: solvation free energy + local nucleophilicity
- No frontier orbital descriptors — fully robust

---

## LOCO Validation (Cross-Class Generalization)

Using global model (q_C + d_LiC + %Vbur), train on 2 classes → predict 3rd:

| Left out | Ea R² | t½@-40 R² |
|---|---|---|
| m+p ArLi | -0.21 | +0.08 |
| o-ArLi | -0.04 | -3.93 |
| oxiranylLi | **-12.9** | -10.0 |

**Conclusion**: Universal model completely fails across classes → class-specific modeling is necessary.

---

## Descriptor Robustness: With vs Without HOMO/η

For m+p ArLi t½@-40°C:

| Model | LOO-R² | Contains HOMO/η? |
|---|---|---|
| η + H_BCP + vol | 0.716 | ⚠ Yes |
| **BDE + H_BCP + vol** | **0.608** | ✓ No |
| Gsolv + H_BCP + vol | 0.440 | ✓ No |

**Removing η reduces R² by ~0.11 but model remains meaningful** → η provides genuine physical information, not pure ΔG‡ encoding.

---

## Descriptor Computation Methods

All descriptors computed at GFN2-xTB level on xTB-optimized geometries:

| Descriptor | Method | Software |
|---|---|---|
| q(C_ipso) | Mulliken charge on C bonded to Li | tblite |
| d(Li-C) | Optimized Li-C bond length (Å) | tblite + scipy |
| %V_bur | Buried volume around Li (r=3.5Å) | morfeus (xTB geom) |
| mol_volume | Molecular volume (ų) | RDKit |
| xtb_ω | Electrophilicity index μ²/(2η) | tblite HOMO/LUMO |
| xtb_η | Chemical hardness (LUMO-HOMO)/2 | tblite |
| BDE | E(R·)+E(Li·)-E(RLi) (kJ/mol) | xtb CLI (--uhf 1) |
| Sterimol B1/B5 | Min/max width at C→Li direction | morfeus (xTB geom) |
| Gsolv | Solvation free energy in THF | xtb ALPB(THF) |
| H(BCP) | Energy density at bond critical point | Multiwfn (M06-2X/def2-SVP fchk) |
| fukui_f⁻(C) | Electrophilic Fukui at C_ipso | xtb (N vs N-1 electrons) |
| σ_hammett | Hammett substituent constant | Literature (Hansch 1991) |
| HOMO | Highest occupied MO energy (eV) | tblite |

## Reactor Classification Tool: SMILES + T → flash/flow/batch

### Model

**log₁₀(t_max/s) = +10.86×q_C − 13.09×d_LiC + 0.477×L + 336×(1/T) + 21.93**

Then classify:
- t_max < 0.1 s → **flash** (microsecond mixing required)
- 0.1 s < t_max < 100 s → **flow** (standard flow reactor)
- t_max > 100 s → **batch** (stable, conventional operation)

### Performance

- **LOO classification accuracy: 89.3%** (125/140, 42 compounds × multi-temperature)
- Data: 140 (compound, T) data points with exact t_max from competing kinetics fits
- Temperature range: −78 to +25°C

| Temperature | Accuracy | n |
|---|---|---|
| −78°C | 91% | 11 |
| −58°C | 87% | 15 |
| −48°C | 88% | 17 |
| −40°C | 100% | 7 |
| −28°C | 89% | 18 |
| 0°C | 85% | 27 |
| 20–25°C | 100% | 14 |

### Confusion Matrix

| | Predicted flash | Predicted flow |
|---|---|---|
| **Actual flash** | 108 | 3 |
| **Actual flow** | 12 | 17 |

- Flash precision: 90% (108/120)
- Flow recall: 59% (17/29) — conservative, tends to recommend flash when flow would work
- False negatives (predict flash when actually flow): 12 cases → user uses faster mixing than needed (safe)
- False positives (predict flow when actually flash): 3 cases → user may lose yield (risky)

### Descriptor Physical Interpretation

| Descriptor | Coefficient | Meaning |
|---|---|---|
| q(C_ipso) | +10.86 | More positive C charge → longer t_max (more stable C-Li) |
| d(Li-C) | −13.09 | Longer Li-C bond → shorter t_max (weaker bond, faster decomposition) |
| Sterimol L | +0.477 | Longer substituent → longer t_max (steric protection) |
| 1/T | +336 | Lower temperature → longer t_max (Arrhenius effect) |

### Comparison with t½-based approach

| | t½ route | t_max route |
|---|---|---|
| Data points | 26 compounds | **140 data points** |
| Approach | Predict Ea → compensate lnA → compute t½ | **Directly predict log(t_max)** |
| Ea-lnA problem | Yes (compensation flattens t½) | **No (bypassed)** |
| Overall accuracy | 76% | **89%** |
| −40°C accuracy | 69% | **100%** |

### Usage

```python
# 1. Compute descriptors from SMILES (xTB, seconds)
q_C = xtb_mulliken_charge_on_C_ipso
d_LiC = xtb_optimized_LiC_bond_length  # Angstrom
L = sterimol_L_along_CLi_axis          # Angstrom

# 2. Input temperature
T_K = T_celsius + 273.15

# 3. Predict t_max
log_tmax = 10.86*q_C - 13.09*d_LiC + 0.477*L + 336/T_K + 21.93
t_max = 10**log_tmax  # seconds

# 4. Recommend reactor
if t_max < 0.1:    reactor = "flash"
elif t_max < 100:   reactor = "flow"
else:               reactor = "batch"
```

---

## Alternative Descriptors Tested (Not Improving Models)

| Descriptor | Level | Result |
|---|---|---|
| HF/def2-SVP Mulliken charge | HF | LOO-R²=0.55 (< xTB 0.69) |
| M06-2X/def2-SVP Mulliken charge | DFT | LOO-R²=0.41 |
| M06-2X/def2-SVP ADCH charge | DFT(Hirshfeld) | LOO-R²=0.40 |
| M06-2X/def2-SVPD + PCM(THF) | DFT+solvent | o-I-ArLi SCF crashed |
| QTAIM ρ(BCP) | DFT/QTAIM | r=0.16, LOO-R²=-35 |
| QTAIM |V|/G ratio | DFT/QTAIM | All values 0.89-0.94, no discrimination |
| pKa(R-H) estimated | Literature | r=0.04 with Ea (thermodynamic ≠ kinetic) |
| Δq(C) process descriptor | xTB | r=-0.53 with Ea but no improvement over static q_C |

---

## Dataset

- Main dataset: `clean_organolithium_unified_descriptors.csv` (2357 rows, 75 intermediates)
- Master table: `intermediates_master.csv` (75 rows, all descriptors)
- Arrhenius parameters: `phase_b_arrhenius.csv` (40 entries)
- Reliable modeling set: 26 (excl. 2-pt) or 37 (incl. 2-pt)
- Class distribution: m+p ArLi (12), o-ArLi (6), oxiranylLi (8), hetero-ArLi (4), other (7)
