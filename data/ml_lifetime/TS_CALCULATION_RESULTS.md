# Transition State Calculations

## Method
- Level: M06-2X/def2-SVP (recommended for organolithium, Ramachandran 2010)
- Software: ORCA 6.1.1
- Reactant: full Opt + Freq
- TS search: NEB-TS (8 images) → OptTS + Freq
- Temperature: 298.15 K
- Phase: gas phase (no solvent model)

---

## oxiranylLi Ring-Opening ([Li]C1CO1)

### Reaction
```
[Li]-CH₂-CH₂-O (3-ring) → Li-O-CH=CH₂ (open-chain lithium vinyl alkoxide)
```

### TS Confirmation
- **Imaginary frequency: -576.2 cm⁻¹** (1 and only 1 → confirmed saddle point)
- Mode: C-O ring-opening stretch + Li migration from C toward O

### Energetics (M06-2X/def2-SVP, gas phase)

| Property | Reactant | TS | Δ‡ |
|---|---|---|---|
| E (Eh) | -160.501309 | -160.445642 | +146.2 kJ/mol |
| H (Eh) | -160.448928 | -160.397629 | **+134.7 kJ/mol** |
| G (Eh) | -160.479253 | -160.428067 | **+134.4 kJ/mol** |
| S [J/(mol·K)] | 267.1 | 268.0 | **+1.0 J/(mol·K)** |
| -TΔS‡ | — | — | -0.3 kJ/mol |

### Comparison with Experiment

| | Computed (gas, M06-2X) | Experimental (THF solution) |
|---|---|---|
| **ΔH‡** | **134.7 kJ/mol** | Ea = 35.6 kJ/mol |
| **ΔS‡** | **+1.0 J/(mol·K)** | **~−109 J/(mol·K)** |
| **ΔG‡ (298K)** | 134.4 kJ/mol | ~61 kJ/mol |

### Key Insight: ΔS‡ discrepancy reveals solvent involvement

The gas-phase TS has ΔS‡ ≈ 0 (entropy-neutral intramolecular process).
The experimental ΔS‡ ≈ −109 J/(mol·K) (highly ordered TS).

**This 110 J/(mol·K) gap proves that the real decomposition involves significant solvent reorganization** — not a simple intramolecular ring opening. The solvent (THF) must participate in the rate-determining step, either through:
- Li coordination changes (desolvation/resolvation)
- Solvent-assisted proton transfer
- Aggregation state changes

This explains why **mol_volume** (a solution-phase proxy for solvation shell size) correlates with t½ in our models — it captures the solvent reorganization entropy that cannot be predicted from gas-phase TS calculations alone.

### CPCM(THF) Solvation Test

Added implicit solvation (CPCM, THF dielectric ε=7.52) to both reactant and TS.
TS confirmed: imaginary frequency -585.8 cm⁻¹ (vs gas -576.2 cm⁻¹).

| Property | Gas | CPCM(THF) | Experiment |
|---|---|---|---|
| **ΔH‡ (kJ/mol)** | 134.7 | **111.6** | **33.1** |
| **ΔG‡ (kJ/mol)** | 134.4 | **111.3** | **69.2** |
| **ΔS‡ [J/(mol·K)]** | +1.0 | **+1.1** | **−121** |
| ΔH‡/ΔH‡_exp | 4.1× | **3.4×** | 1.0× |

**CPCM recovered only 23% of the enthalpy gap (23.1 of 101.6 kJ/mol).**

Key findings:
1. **ΔS‡ completely unchanged** (+1.1 vs +1.0) — CPCM only shifts the PES, not the vibrational frequencies or entropy. The 122 J/(mol·K) entropy gap remains entirely unexplained.
2. **Dielectric accounts for ~23% of barrier reduction** — bulk polarization stabilizes the polar TS somewhat, but is far from sufficient.
3. **Remaining 77% gap requires**: explicit Li-THF coordination changes, aggregation state effects, and solvent reorganization dynamics — none of which are captured by continuum solvation models.

### Why computed barrier is 3–4× higher than experiment

1. **Dielectric only partial** (~23%): CPCM reduces barrier by 23 kJ/mol, but 79 kJ/mol gap remains
2. **No explicit solvation**: Li is coordinated by 2-3 THF molecules; coordination changes in TS lower the barrier further
3. **No aggregation**: Real oxiranylLi may decompose from a specific aggregate state (Curtin-Hammett)
4. **Different mechanism**: Experimental ΔS‡ = −121 J/(mol·K) implies bimolecular (solvent-assisted) rate-determining step, not simple intramolecular ring opening
5. **Entropy invisible to static calculations**: Even microsolvation + CPCM would not recover the solution-phase ΔS‡ without explicit dynamics

---

## Reactant Thermochemistry (all 3 molecules)

| Molecule | E (Eh) | H (Eh) | G (Eh) | S [J/(mol·K)] | Low freq (<200 cm⁻¹) |
|---|---|---|---|---|---|
| PhLi | -238.855727 | -238.758771 | -238.794964 | 318.7 | 2 (127, 134) |
| oxiranylLi | -160.501309 | -160.448928 | -160.479253 | 267.1 | 0 |
| o-CO₂Me-ArLi | -466.523487 | -466.377824 | -466.422429 | 392.8 | 5 (81, 119, 162, 189, 195) |

### Correlation: computed S(reactant) vs experimental lnA

| Molecule | S_computed [J/(mol·K)] | Experimental lnA | mol_volume (Å³) |
|---|---|---|---|
| oxiranylLi | 267 | 15.9 | 80.8 |
| PhLi | 319 | 15.4 | ~110 |
| o-CO₂Me-ArLi | 393 | 19.0 | 160.2 |

Trend: larger S(reactant) correlates with larger mol_volume (as expected).
However, S(reactant) does NOT directly correlate with experimental lnA (which reflects ΔS‡, not S).

---

## o-BrPhLi Benzyne Elimination ([Li]c1ccccc1Br)

### Reaction
```
o-Br-C₆H₄-Li → benzyne (C₆H₄) + LiBr
```

### NEB Result (M06-2X/def2-SVP, gas phase)

NEB-TS with 8 images, converged after 32 iterations.

**Result: No saddle point found — path is monotonically increasing (endothermic reaction).**

| Image | Distance (Å) | E (Eh) | dE (kcal/mol) |
|---|---|---|---|
| 0 (reactant) | 0.000 | -2812.17096 | 0.00 |
| 1 | 0.316 | -2812.16626 | +2.95 |
| 2 | 0.636 | -2812.15695 | +8.79 |
| 3 | 0.964 | -2812.14910 | +13.71 |
| 4 | 1.316 | -2812.14304 | +17.52 |
| 5 | 1.708 | -2812.13845 | +20.40 |
| 6 | 2.174 | -2812.13526 | +22.40 |
| 7 | 2.741 | -2812.13402 | +23.18 |
| 8 | 3.295 | -2812.12416 | +29.37 |
| 9 (CI) | 3.841 | -2812.02781 | +89.83 |

### Key Insight: Endothermic fragmentation requires solvation to create a barrier

Gas-phase benzyne elimination is strongly endothermic (+97–376 kJ/mol depending on fragment separation). No saddle point exists on the monotonically rising path. The experimental barrier (Ea = 72.1 kJ/mol) only appears because:

1. **LiBr solvation**: In THF solution, LiBr is stabilized by ion pair solvation → reaction becomes less endothermic
2. **Benzyne trapping**: Rapid electrophilic trapping prevents reverse reaction
3. **Aggregation effects**: Experimental ΔS‡ = +77 J/(mol·K) (positive, consistent with fragmentation)

The positive experimental ΔS‡ contrasts sharply with the other two systems (both ~-120 J/(mol·K)), confirming fundamentally different mechanisms.

---

## PhLi + THF Solvent Cleavage ([Li]c1ccccc1 + THF)

### Reaction
```
PhLi·(THF) → Ph-H + lithium enolate (THF α-H transfer to carbanion)
```

### Relaxed Scan (M06-2X/def2-SVP, gas phase)

Scan coordinate: d(H_α(THF)···C_ipso(Ph)) from 2.913 → 1.100 Å, 10 steps.
Complex: 25 atoms (PhLi + 1 explicit THF, Li-O = 1.877 Å).
Steps 1–8 converged; step 9 optimization failed (max cycles exceeded).

| Step | d(H···C) (Å) | E (Eh) | dE (kJ/mol) |
|---|---|---|---|
| 1 | 2.913 | -471.059442 | 0.0 |
| 2 | 2.712 | -471.059014 | +1.1 |
| 3 | 2.510 | -471.057810 | +4.3 |
| 4 | 2.309 | -471.056446 | +7.9 |
| 5 | 2.107 | -471.053219 | +16.3 |
| 6 | 1.906 | -471.047603 | +31.1 |
| 7 | 1.704 | -471.038160 | +55.9 |
| 8 | 1.503 | -471.023076 | **+95.5** |
| 9 | 1.302 | -471.027504 | +83.9 ⚠ unconverged |

### Barrier estimate

Step 9 (unconverged) is lower than step 8 → barrier maximum lies between d = 1.50 and 1.30 Å.
**Approximate 1D scan barrier: ~96 kJ/mol** (upper bound; true TS would be lower).

### Comparison with Experiment

| | Computed (gas, 1 THF) | Experimental (THF solution) |
|---|---|---|
| **Barrier** | ~96 kJ/mol (scan) | Ea = 36.5 kJ/mol |
| **Ratio** | **2.6×** | — |
| **ΔS‡** | N/A (no freq) | **−125 J/(mol·K)** |

The 2.6× overestimation (vs 3.8× for oxiranylLi) is because one explicit THF molecule is already included, partially capturing the bimolecular nature. However, bulk solvation (coordination shell, dielectric stabilization) is still missing.

The experimental ΔS‡ = −125 J/(mol·K) confirms bimolecular rate-determining step — THF loses translational freedom upon association with PhLi in the TS.

---

## Three-System Comparison

### Experimental Eyring Parameters (from Arrhenius fits)

| System | Mechanism | Ea (kJ/mol) | lnA | ΔH‡ (kJ/mol) | ΔS‡ [J/(mol·K)] | ΔG‡₂₉₈ (kJ/mol) |
|---|---|---|---|---|---|---|
| oxiranylLi | Ring-opening | 35.6 | 15.9 | 33.1 | **−121** | 69.2 |
| PhLi | THF cleavage | 36.5 | 15.4 | 34.1 | **−125** | 71.4 |
| o-BrPhLi | Benzyne elim. | 72.1 | 39.8 | 69.7 | **+77** | 46.6 |

### Gas-Phase vs CPCM vs Solution Comparison

| System | Gas ΔH‡ | CPCM ΔH‡ | Exp ΔH‡ | Gas/Exp | CPCM/Exp | Gas ΔS‡ | Exp ΔS‡ |
|---|---|---|---|---|---|---|---|
| oxiranylLi | 134.7 | **111.6** | 33.1 | 4.1× | **3.4×** | +1.0 | −121 |
| PhLi+THF | ~96 (scan) | — | 34.1 | 2.8× | — | N/A | −125 |
| o-BrPhLi | No barrier | — | 69.7 | — | — | — | +77 |

**CPCM(THF) only recovered 23% of the gap for oxiranylLi.**

### Solvation Effect Decomposition (oxiranylLi)

```
Gas-phase ΔH‡ = 134.7 kJ/mol
                                    ↓ CPCM (dielectric ε=7.52)
+ Bulk dielectric           → 111.6 kJ/mol  (−23.1, 23% of gap)
                                    ↓ 未计算
+ Explicit Li·(THF)₂₋₃     → ~80?  kJ/mol  (estimated ~30 kJ/mol)
+ Aggregation/dynamics      → ~50?  kJ/mol  (estimated ~30 kJ/mol)
                                    ↓
Experimental ΔH‡            =  33.1 kJ/mol

ΔS‡: gas +1.0 → CPCM +1.1 → exp −121 J/(mol·K)
     CPCM 对 ΔS‡ 完全无效（连 1% 都没改善）
```

### Mechanistic Conclusions

1. **ΔS‡ sign reveals mechanism type**:
   - Negative ΔS‡ (oxiranylLi, PhLi): associative TS — solvent molecules are organized/consumed in rate-determining step
   - Positive ΔS‡ (o-BrPhLi): dissociative TS — fragmentation generates translational entropy

2. **Implicit solvation (CPCM) is grossly insufficient**: only 23% of enthalpy gap recovered, 0% of entropy gap. The dominant effects are:
   - Explicit Li-THF coordination changes in TS (~30% of gap, estimated)
   - Aggregation state (Curtin-Hammett) + solvent reorganization dynamics (~50% of gap)
   - These require microsolvation + ab initio MD, which cost days per compound

3. **Benzyne elimination has no gas-phase barrier** because it is endothermic. Solvation of the LiBr fragment creates the observed barrier. This class fundamentally cannot be modeled by gas-phase or implicit-solvation TS calculations.

4. **Justification for descriptor-based modeling**: TS calculations at every affordable solvation level (gas, CPCM) fail quantitatively. Ground-state descriptor QSPR models (xTB level, seconds per compound) implicitly encode solution-phase effects through descriptors like Gsolv (solvation free energy) and mol_volume (solvation shell size proxy), achieving LOO-R² = 0.69–0.94 for class-specific Ea and t½ prediction.

---

## Files

| File | Content |
|---|---|
| `ts_calc/phli_opt.out` | PhLi M06-2X/def2-SVP Opt+Freq |
| `ts_calc/oxiranyl_opt.out` | oxiranylLi Opt+Freq (reactant) |
| `ts_calc/oco2me_opt.out` | o-CO₂Me-ArLi Opt+Freq |
| `ts_calc/oxiranyl_scan2.out` | Relaxed scan C-O (12 steps) |
| `ts_calc/oxiranyl_neb2.out` | NEB-TS (8 images, 28 iterations) |
| `ts_calc/oxiranyl_optts.out` | OptTS + Freq (TS confirmed) |
| `ts_calc/oxiranyl_neb2_NEB-TS_converged.xyz` | TS geometry |
| `ts_calc/obr_opt.out` | o-BrPhLi Opt+Freq (reactant) |
| `ts_calc/obr_neb.out` | NEB-TS benzyne (32 iter, no barrier) |
| `ts_calc/phli_thf_opt.out` | PhLi·THF complex Opt+Freq |
| `ts_calc/phli_thf_scan.out` | Relaxed scan H-transfer (8/10 steps) |
| `ts_calc/oxiranyl_opt_cpcm.out` | oxiranylLi reactant + CPCM(THF) |
| `ts_calc/oxiranyl_optts_cpcm.out` | oxiranylLi TS + CPCM(THF) |
