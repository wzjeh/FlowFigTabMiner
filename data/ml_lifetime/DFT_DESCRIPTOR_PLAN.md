# DFT Descriptor Plan for Organolithium Stability Prediction

## Goal

Use quantum-chemical (DFT / semi-empirical xtb) calculations to compute electronic and structural descriptors for each organolithium intermediate (ArLi, vinylLi, carbenoid, etc.). These descriptors serve as ML features alongside the existing empirical parameters (Hammett σ, Taft Es, δ_ortho, δ_benzyne) to predict decomposition half-life (t½) and activation energy (Ea).

The empirical LFER model (v3.0-B) achieves R² ≈ 0.97 with 4 parameters, but is limited to intermediates with known Hammett/Taft values. DFT descriptors are **computable for any structure from SMILES alone**, enabling:
1. Prediction for novel intermediates without empirical parameters
2. Physical interpretability — connect stability to electronic structure
3. Potential replacement of categorical flags (δ_ortho, δ_benzyne) with continuous variables

---

## Descriptors

### 1. `dft_charge_Li` (float) — Lithium atom partial charge

- **Method**: NBO (Natural Bond Orbital) analysis or Mulliken population
- **Unit**: elementary charge (e)
- **Significance**: Measures the ionic character of the Li–C bond. Higher positive charge on Li indicates more ionic bonding, which correlates with higher reactivity and lower stability. For ArLi species, electron-withdrawing substituents increase Li charge.

### 2. `dft_charge_C_ipso` (float) — ipso-Carbon partial charge

- **Method**: NBO or Mulliken population
- **Unit**: elementary charge (e)
- **Significance**: The carbon directly bonded to Li carries the carbanion character. More negative charge = stronger nucleophilicity. Combined with `dft_charge_Li`, the charge difference (ΔQ = q_Li - q_C) quantifies bond polarity and predicts electrophilic quenching rates (k2).

### 3. `dft_HOMO_eV` (float) — Highest Occupied Molecular Orbital energy

- **Method**: Kohn-Sham orbital from DFT or GFN2-xTB
- **Unit**: eV
- **Significance**: The HOMO of ArLi is predominantly the Li–C σ-bond / carbanion lone pair. Higher (less negative) HOMO = stronger nucleophile = faster quenching but also more susceptible to side reactions. Directly relates to Hammett σ via Koopman's theorem.

### 4. `dft_LUMO_eV` (float) — Lowest Unoccupied Molecular Orbital energy

- **Method**: Kohn-Sham orbital from DFT or GFN2-xTB
- **Unit**: eV
- **Significance**: Low LUMO in aryllithiums indicates electron-deficient rings prone to nucleophilic addition/benzyne formation. The HOMO–LUMO gap correlates with kinetic stability — smaller gap = more reactive.

### 5. `dft_LiC_bond_A` (float) — Li–C bond length

- **Method**: Geometry optimization
- **Unit**: Å (angstrom)
- **Significance**: Longer Li–C bonds indicate weaker binding and lower activation barriers for decomposition. ortho-Substituted ArLi with chelating groups (OMe, NMe₂) show shortened Li–C bonds due to intramolecular coordination, explaining the δ_ortho stabilization effect.

### 6. `dft_LiC_BDE_kJ` (float) — Li–C bond dissociation energy

- **Method**: E(ArLi) → E(Ar·) + E(Li·), with ZPE correction
- **Unit**: kJ/mol
- **Significance**: The most direct measure of thermodynamic stability. Higher BDE = harder to break = more stable intermediate. Expected to correlate strongly with experimental Ea from Arrhenius fits. This descriptor could potentially replace the entire LFER (σ + Es + δ) with a single physically meaningful number.

### 7. `dft_wiberg_LiC` (float) — Wiberg bond index for Li–C

- **Method**: NBO analysis
- **Unit**: dimensionless (0–1 range for Li–C)
- **Significance**: Measures covalent bond order. Pure ionic bond → 0, pure covalent → 1. For ArLi, typical values are 0.3–0.6. Lower Wiberg index = more ionic character = easier heterolytic dissociation. Captures substituent effects on bonding character that charge alone misses.

### 8. `dft_dipole_D` (float) — Molecular dipole moment

- **Method**: From converged wavefunction
- **Unit**: Debye (D)
- **Significance**: Large dipole moments indicate significant charge separation (ionic Li–C bond). Solvent stabilization of the dipole affects decomposition pathway — high dipole moments in THF lead to greater solvation stabilization, modifying the effective decomposition barrier.

### 9. `dft_Gsolv_kJ` (float) — Solvation free energy in THF

- **Method**: ALPB or GBSA implicit solvation model (xtb) or SMD/PCM (DFT)
- **Unit**: kJ/mol
- **Significance**: All organolithium reactions occur in THF solution. Solvation stabilizes the charge-separated Li⁺···C⁻ character. More negative ΔG_solv = more stabilized in solution = potentially different stability ranking than gas phase. Critical for comparing ArLi with different polarity profiles.

### 10. `dft_method` (str) — Computation method identifier

- **Values**: e.g., `"xtb-GFN2"`, `"B3LYP/6-31G*"`, `"ωB97X-D/def2-SVP"`
- **Significance**: Tracks provenance for reproducibility. Different methods may give systematically different values; this column ensures consistent comparison within a method level.

---

## Computation Strategy

### Phase 1: Fast screening with xtb (GFN2-xTB)

- **Tool**: Grimme's xtb program (open-source, fast)
- **Input**: SMILES → 3D structure (RDKit `AllChem.EmbedMolecule`) → xtb optimization
- **Available descriptors**: charges (Mulliken), HOMO/LUMO, bond lengths, dipole, ΔG_solv (ALPB/THF)
- **Speed**: ~seconds per molecule
- **Limitation**: No NBO analysis (no Wiberg index), approximate BDE

### Phase 2: DFT validation on subset

- **Tool**: ORCA or Gaussian
- **Level**: B3LYP/6-31G* or ωB97X-D/def2-SVP (common for organolithiums in literature)
- **Additional**: NBO analysis for Wiberg index and NBO charges, accurate BDE
- **Speed**: ~minutes to hours per molecule
- **Apply to**: All unique intermediates in `analysis_subset ∈ {kd_clean, kd_valid}`

### Phase 3: Correlation and selection

- Compare xtb vs DFT values — if rank-order is preserved, xtb suffices for screening
- Identify which DFT descriptors are most predictive via LASSO / feature importance
- Build hybrid model: empirical LFER + top DFT descriptors

---

## Workflow

```
SMILES (from dataset)
    │
    ▼
RDKit: 3D embedding + MMFF optimization
    │
    ▼
xtb: GFN2-xTB geometry optimization (gas phase)
    │
    ├──→ Charges, HOMO, LUMO, dipole, bond lengths
    │
    ▼
xtb: ALPB solvation (THF)
    │
    ├──→ ΔG_solv
    │
    ▼
(Optional) ORCA/Gaussian: DFT single-point + NBO
    │
    ├──→ NBO charges, Wiberg index, BDE
    │
    ▼
Populate dataset columns
```

---

## Relationship to Existing Descriptors

| Empirical | DFT equivalent | Advantage of DFT |
|-----------|---------------|-------------------|
| σ (Hammett) | dft_charge_C_ipso, dft_HOMO_eV | Continuous, applicable to non-Hammett substituents |
| Es (Taft) | dft_LiC_bond_A | Captures actual geometry, not tabulated values |
| δ_ortho | dft_LiC_bond_A (shortening) | Continuous measure of chelation strength |
| δ_benzyne | dft_LUMO_eV (low = benzyne-prone) | Quantitative orbital energy vs binary flag |
| (none) | dft_LiC_BDE_kJ | Direct thermodynamic stability measure |
| (none) | dft_Gsolv_kJ | Solvent effect on stability |
| (none) | dft_wiberg_LiC | Bond covalency measure |

---

## Intermediates to Compute (~25 unique structures)

Each unique `intermediate_smiles` in the dataset needs one computation. Approximate list:
- PhLi, o-MeOPhLi, p-MeOPhLi, m-MeOPhLi
- o-FPhLi, p-FPhLi, m-FPhLi
- o-CF₃PhLi, p-CF₃PhLi
- p-NO₂PhLi, m-NO₂PhLi
- o-BrPhLi (benzyne precursor)
- 2,6-dimethoxyphenyllithium
- Heteroaryl: pyridyl, thienyl, benzo[b]thiophenyl
- Vinyl, carbenoid (CHLiIF, CHLiClF)
- Functional alkyllithiums (from Nagaki 2019)

Total: ~25 unique SMILES → ~25 xtb jobs (minutes) + ~25 DFT jobs (hours).
