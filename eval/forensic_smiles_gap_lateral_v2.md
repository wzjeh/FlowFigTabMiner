# SMILES-gap forensic — lateral_v2

Corpus: `@/tmp/lateral_done_names.txt` (12 papers)
Total gap records (non-hollow, has identity, no product_smiles): **455**

| Bucket | Count | % | Meaning |
|---|---|---|---|
| POOL_HAS_IT | 5 | 1.1% | pool resolves it but record empty → resolve/stale bug |
| STRUCT_EXISTS_UNLINKED | 26 | 5.7% | labelled, paper HAS structures, label not linked |
| NO_STRUCT_SOURCE | 44 | 9.7% | labelled, paper has NO structures (MolNexTR/none drawn) |
| NAME_ONLY_STRUCT | 272 | 59.8% | name only, paper HAS structures → name-link miss (fixable) |
| NAME_ONLY_NO_STRUCT | 108 | 23.7% | name only, no structures → PubChem territory |

### Unique-compound view (26 distinct (paper,label,name); de-dupes figure multi-point inflation)

| Bucket | Unique | % |
|---|---|---|
| POOL_HAS_IT | 1 | 3.8% |
| STRUCT_EXISTS_UNLINKED | 5 | 19.2% |
| NO_STRUCT_SOURCE | 7 | 26.9% |
| NAME_ONLY_STRUCT | 9 | 34.6% |
| NAME_ONLY_NO_STRUCT | 4 | 15.4% |

## Linking-gap deterministic fixability

Of the 298 linking-gap records (14 unique compounds), how many could a DETERMINISTIC fix (wildcard-`*` / label-is-SMILES / table-harvest) resolve vs HARD (figure fixed-product / descriptive name → needs scheme-pool or semantic linking):

| Fixability | Records | % | Unique | % |
|---|---|---|---|---|
| **Deterministic** | 6 | 2.0% | 1 | 7.1% |
| **HARD_no_anchor** | 292 | 98.0% | 13 | 92.9% |

Fix breakdown (records / unique):
- FIX_label_is_smiles: 0 / 0
- FIX_name_is_smiles: 0 / 0
- FIX_table_harvest: 6 / 1
- HARD_no_anchor: 292 / 13

## Samples (up to 3 per bucket)

**POOL_HAS_IT** (5)
- `Hafner et al. 2016 - A simple ` Table (page 3) — Γable 2. HPLC Analysis of the Reaction at Differen emperatures | label='2a' name='2-trifluoromethyl-5-fluoroph' struct_in_paper=0

**STRUCT_EXISTS_UNLINKED** (26)
- `Asai et al. 2012 - Practical s` Table (page 10) — Table 4. Synthesis of unsymmetrical diarylethenes from octafluorocyclo- pentene | label='4bc' name='None' struct_in_paper=8
- `Asai et al. 2012 - Practical s` Table (page 8) — Table 1. The Br/Li exchange reaction of 1a or 1c (1 equiv) with nBuLi d by the r | label='3c' name='None' struct_in_paper=8
- `Djukanovic et al. 2021 - Conti` Table (page 2) — Table 2. Optimization of the acylation temperature continuous flow. | label='3aa' name='ketone' struct_in_paper=2

**NO_STRUCT_SOURCE** (44)
- `Kupracz and Kirschning 2013 - ` Table (page 2) — Table 1. Wurtz coupling under batch condition 1. n-BuLi, THF, temp. | label='5' name='None' struct_in_paper=0
- `Thaisrivongs et al. 2016 - Usi` Table (page 2) — hle ) Eect of Various Mixers on the Flow Depro | label='13' name='adduct 13' struct_in_paper=0
- `Thaisrivongs et al. 2016 - Usi` Table (page 4) — e 3. Eect of Sto try on the | label='13' name='product 13' struct_in_paper=0

**NAME_ONLY_STRUCT** (272)
- `Asai et al. 2012 - Practical s` Figure (page 4) — This figure likely shows the effect of a reaction parameter (implied by the x-ax | label=None name='diarylethene' struct_in_paper=8
- `Asai et al. 2012 - Practical s` Figure (page 7) — This figure likely shows the optimization of reaction conditions for the synthes | label=None name='photochromic diarylethene' struct_in_paper=8
- `carbolithiation-of-conjugated-` Figure (page 3) — This figure shows the yield of different products (3a+4a, 5+6, 2) as a function | label='3a+4a' name='allenylsilane products' struct_in_paper=6

**NAME_ONLY_NO_STRUCT** (108)
- `feasibility-study-on-continuou` Figure (page 3) — This figure is a picture of the flow microreactor system used for the anionic po | label=None name='polystyrene' struct_in_paper=0
- `feasibility-study-on-continuou` Figure (page 4) — This figure shows the yield of poly(styrene) as a function of an unspecified exp | label=None name='poly(styrene)' struct_in_paper=0
- `molecular-weight-distribution-` Figure (page 2) — This figure compares the mixability of V-shape and T-shape mixers, likely in the | label=None name='polystyrene' struct_in_paper=0

## Gap records per paper

-  100  three-component-coupling-based-on-flash-chemistry-carbolithi
-   94  Asai et al. 2012 - Practical synthesis of photochromic diary
-   92  carbolithiation-of-conjugated-enynes-with-aryllithiums-in-mi
-   53  molecular-weight-distribution-of-polymers-produced-by-anioni
-   51  feasibility-study-on-continuous-flow-controlled-living-anion
-   27  ncomms1264
-   17  Thaisrivongs et al. 2016 - Using flow to outpace fast proton
-   12  Djukanovic et al. 2021 - Continuous flow acylation of (heter
-    5  Hafner et al. 2016 - A simple scale-up strategy for organoli
-    4  Kupracz and Kirschning 2013 - Multiple organolithium generat