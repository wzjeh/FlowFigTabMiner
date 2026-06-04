# C2 (remote-EWG ArLi) decomposition-kinetics model — HOMO+Gsolv (v7)

Found by physics-informed closed-loop search (propose → veto → reject → next), 2026-05-26.

## Model
```
Ea_d   = −25.30·HOMO − 2.18·Gsolv − 367.6      (LOO R² = 0.58)
ln A_d = −7.02·HOMO  − 1.05·Gsolv − 131.4      (LOO R² = 0.68)
t½(T)  = ln2 / exp(lnA_d − Ea_d/RT)            R = 8.314e−3 kJ/mol·K
```
Descriptors (GFN2-xTB, same pipeline as `compute_virtual_arli.py`):
- `HOMO`  = HOMO energy (eV) of the ArLi monomer
- `Gsolv` = E(ALPB-THF) − E(vacuum), kJ/mol

## Training set
12 real in-house C2 aryllithiums (m/p-CN, m/p-NO₂, m/p-CO₂R esters). Decomposition
Ea_d/lnA_d from global Arrhenius fits (`v60_classified_substrates.csv`).

## Why HOMO instead of Hammett σ (key insight)
σ_total **averages out** the resonance vs. inductive differences among CN/NO₂/ester/CF₃,
so a σ-only or σ+Gsolv model fails (σ-only LOO<0; σ+Gsolv gives a non-physical σ sign
inversion and extrapolates 3,5-diCN to t½≈5 days). The DFT **HOMO directly quantifies how
deeply each EWG stabilizes the carbanion**, unifying the 5 EWG sub-mechanisms into one
continuous, chemically meaningful axis. Coefficient sign is correct (deeper HOMO → higher Ea_d).

## External validation (virtual screening veto)
39 virtual C2 ArLi (σ 0.37→1.78, HOMO −10.7→−8.8; `virtual_arli_descriptors.csv` +
`virtual_c2_extended.csv`):
- Ea_p ∈ [5,120] kJ/mol: **100%**
- t½ in reasonable domain: **95% (37/39)**
- 3,5-diCN (σ=1.12): t½ = 567 s (reasonable; σ+Gsolv gave 5 days)
- Only 2 extreme points over-stable: 3,5-diNO₂ (σ=1.42), 3,4,5-triCF₃ (σ=1.40)

## Applicability domain
Remote-EWG aryllithiums, **σ ∈ [0.37, ~1.3]**, HOMO ∈ [−9.7, −8.8] eV. Beyond σ≈1.4
(multiple strong EWG) the model extrapolates over-stable — use with caution.

## Limitation (honest)
Both in-house blind substrates (4-Br-FC₆H₄ = C1, 5-Br-2-F-CN = C2) are **stable / do not
decompose within the experimental t_R window**, so observable yield is insensitive to Ea_d.
→ The decomposition Ea_d cannot be truth-validated with current experiments; support is
virtual-extrapolation veto + literature-Arrhenius consistency only. A substrate that
visibly decomposes within t_R is needed for a real Ea_d blind test.
On 5-Br-2-F-CN observable yield: v7 MAE=12.0 pp vs v6.2-anchor 7.65 pp — but this底物
不分解, so it does NOT discriminate the decomposition models.

## Scripts
- `compute_virtual_arli.py` — generate Br/Li-exchange-derived ArLi + fast GFN2-xTB descriptors
- `compute_c2_extended.py`  — extended remote-EWG C2 library
