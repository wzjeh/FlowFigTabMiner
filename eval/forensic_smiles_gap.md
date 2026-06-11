# SMILES-gap forensic

Corpus: `data/input/Clean organolithium` (20 papers)
Total gap records (non-hollow, has identity, no product_smiles): **710**

| Bucket | Count | % | Meaning |
|---|---|---|---|
| POOL_HAS_IT | 0 | 0.0% | pool resolves it but record empty → resolve/stale bug |
| STRUCT_EXISTS_UNLINKED | 249 | 35.1% | labelled, paper HAS structures, label not linked |
| NO_STRUCT_SOURCE | 95 | 13.4% | labelled, paper has NO structures (MolNexTR/none drawn) |
| NAME_ONLY_STRUCT | 214 | 30.1% | name only, paper HAS structures → name-link miss (fixable) |
| NAME_ONLY_NO_STRUCT | 152 | 21.4% | name only, no structures → PubChem territory |

### Unique-compound view (35 distinct (paper,label,name); de-dupes figure multi-point inflation)

| Bucket | Unique | % |
|---|---|---|
| POOL_HAS_IT | 0 | 0.0% |
| STRUCT_EXISTS_UNLINKED | 14 | 40.0% |
| NO_STRUCT_SOURCE | 2 | 5.7% |
| NAME_ONLY_STRUCT | 15 | 42.9% |
| NAME_ONLY_NO_STRUCT | 4 | 11.4% |

## Samples (up to 3 per bucket)

**POOL_HAS_IT** (0)
- (none)

**STRUCT_EXISTS_UNLINKED** (249)
- `Kim et al. 2011 - A flow-micro` Figure (page 2) — Investigation of residence time effects in reactor R2 on the yield of iodine-lit | label='3' name='o-pentanoyl-substituted prot' struct_in_paper=36
- `Kim et al. 2011 - A flow-micro` Figure (page 2) — Optimization of residence time in R2 for iodine-lithium exchange and methanol tr | label='3' name='protonated product 3' struct_in_paper=36
- `Nagaki et al. 2009 - Generatio` Figure (page 2, instance 0) | label='2' name='1-trimethylsilyl-1-triphenyl' struct_in_paper=12

**NO_STRUCT_SOURCE** (95)
- `Nagaki et al. 2008 - Aryllithi` Figure (page 2) — Scatter plot showing experimental data points of temperature (T) and residence t | label=None name='3' struct_in_paper=0
- `Nagaki et al. 2008 - Aryllithi` Figure (page 2) — Scatter plot showing the relationship between residence time in reactor R1 (tR) | label='3' name='product 3' struct_in_paper=0

**NAME_ONLY_STRUCT** (214)
- `Nagaki et al. 2007 - Integrate` Table (page 2) — exchange of p-dibromobenzene able 1. Br-Li reaction with by reaction with MeOH w | label=None name='g-Bromobutylbenzene' struct_in_paper=47
- `Nagaki et al. 2009 - Generatio` Table (page 2) — reactions of Table 1. Generation and r N-Bus-aziridinyllithiums W1 lectrophiles | label='*C1(c2ccccc2)CN1CCCC' name='None' struct_in_paper=8
- `Nagaki et al. 2010 - Generatio` Figure (page 3) — Scatter plot showing the relationship between residence time in reactor R1 and t | label=None name='benzonitrile derivative' struct_in_paper=4

**NAME_ONLY_NO_STRUCT** (152)
- `Nagaki et al. 2010 - A flow mi` Figure (page 5) — Scatter plot showing distribution of retention times (tR) for reaction products | label=None name='alkoxycarbonyl-substituted a' struct_in_paper=0
- `Nagaki et al. 2010 - A flow mi` Figure (page 6) — Scatter plot showing distribution of experimental data points across residence t | label=None name='alkoxycarbonyl-substituted p' struct_in_paper=0
- `Nagaki et al. 2010 - A flow mi` Figure (page 9) — Scatter plot showing distribution of residence times (tR) for successful generat | label=None name='electrophile-trapped product' struct_in_paper=0

## Gap records per paper

-  297  Nagaki et al. 2010 - Generation and reactions of oxiranyllit
-  127  Nagaki et al. 2010 - A flow microreactor system enables orga
-   95  Nagaki et al. 2008 - Aryllithium compounds bearing alkoxycar
-   45  Nagaki et al. 2009 - Generation and reactions of α-silyloxir
-   36  Nagaki et al. 2010 - Generation and reaction of cyano-substi
-   35  Kim et al. 2011 - A flow-microreactor approach to protecting
-   27  Nagaki et al. 2009 - Generations and reactions ofN-(t-butyls
-   25  Nagaki et al. 2011 - Homocoupling of aryl halides in flow - 
-   14  Nagaki et al. 2009 - Nitro-substituted aryl lithium compound
-    5  Nagaki et al. 2009 - Synthesis of unsymmetrically substitute
-    3  Nagaki et al. 2007 - Integrated micro flow synthesis based o
-    1  Nagaki et al. 2011 - Perfluoroalkylation in flow microreacto