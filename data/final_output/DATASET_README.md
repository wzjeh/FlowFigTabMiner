# Organolithium tR Subdataset (VLM-Enriched)

**File**: `organolithium_tr_subdataset_vlm_enriched.csv`
**Rows**: 1470 | **Columns**: 31 | **Papers**: 20

## Column Definitions

### Paper Metadata
| Column | Type | Description |
|--------|------|-------------|
| `paper` | str | Paper title or identifier |
| `paper_doi` | str | DOI (1443 non-null) |
| `paper_year` | float | Publication year (1620 non-null) |

### Reaction Conditions
| Column | Type | Description |
|--------|------|-------------|
| `tR_step` | str | Which residence time step this row describes: `tR1` or `tR2` |
| `tR1_s` | float | **Corrected** first-step residence time in seconds (1408 non-null). Manually corrected from heatmap axis annotations by Zhao. |
| `tR2_s` | float | **Corrected** second-step residence time in seconds (218 non-null). Used when `tR_step=tR2`. |
| `T1_C` | float | First-step temperature in Celsius (1376 non-null). From VLM Y-axis extraction, verified against YOLO row detection. |
| `T2_C` | float | Second-step temperature in Celsius (214 non-null). Used when `tR_step=tR2`. |
| `yield_pct` | float | Product yield percentage (1626 non-null). From VLM extraction, verified 99.6% match with YOLO OCR. |
| `reaction_class` | str | Reaction type, e.g. "C-C coupling" (1620 non-null) |
| `solvent` | str | Reaction solvent (1467 non-null) |
| `reactor_type` | str | Reactor type, e.g. "flow microreactor" (1329 non-null) |

### Chemical Species
| Column | Type | Description |
|--------|------|-------------|
| `substrate1` | str | First substrate name (1387 non-null) |
| `organolithium_reagent` | str | Organolithium reagent name, e.g. "PhLi" (1571 non-null) |
| `intermediate` | str | Organolithium intermediate name (1626 non-null) |
| `substrate2_electrophile` | str | Electrophile name (1215 non-null) |
| `electrophile_type` | str | Electrophile category: "synthetic" etc. (1571 non-null) |
| `product_final` | str | Final product name (1409 non-null) |

### Concentrations
| Column | Type | Description |
|--------|------|-------------|
| `conc_substrate_M` | float | Substrate concentration in mol/L (1112 non-null) |
| `conc_orgLi_M` | float | Organolithium concentration in mol/L (1075 non-null) |
| `equiv_orgLi` | float | Equivalents of organolithium (872 non-null) |
| `conc_electrophile_M` | float | Electrophile concentration in mol/L (302 non-null) |

### SMILES
| Column | Type | Description |
|--------|------|-------------|
| `substrate1_smiles` | str | Substrate 1 SMILES (1387 non-null) |
| `organolithium_smiles` | str | Organolithium reagent SMILES (1571 non-null) |
| `intermediate_smiles` | str | Intermediate SMILES (1378 non-null) |
| `substrate2_smiles` | str | Electrophile SMILES (1090 non-null) |
| `product_smiles` | str | Product SMILES (1191 non-null) |

### QC / Provenance Columns
| Column | Type | Description |
|--------|------|-------------|
| `vlm_tR1_s_original` | float | VLM's original tR1 value before manual correction (1408 non-null). Useful for auditing the correction mapping. |
| `vlm_tR2_s_original` | float | VLM's original tR2 value before manual correction (218 non-null). |
| `yield_yolo` | float | YOLO pipeline-extracted yield for cross-validation (737 non-null). Independent OCR from micro-YOLO data_value detection. |
| `data_note` | str | Data quality flag (278 flagged rows). See below. |

## Data Quality Notes

All 14 papers' tR values have been manually corrected by Zhao from heatmap axis annotations (log10 exponents).

### Correction Method
- **Rank-order replacement**: For each paper, sort both the VLM-extracted unique tR values and Zhao's corrected values, then replace positionally (smallest to smallest).
- **Exception**: Nagaki 2016 exponents give minutes, converted via x60 to seconds.

### Flagged Rows (`data_note`)
| Flag | Rows | Explanation |
|------|------|-------------|
| `存疑:已删除24行误检,双热图源(04+10)` | 188 | Oxiranyl paper. 24 rows with tR=0.0316/0.1 were deleted (VLM false detection from heatmap 04). Remaining 188 rows corrected from 7 true columns. |
| `存疑:双热图合并,部分tR原始值重复` | 55 | example1 paper. Data from 2 heatmaps (15, 16) merged; VLM read one column as tR=0.3 vs 0.316 (same physical position). Merged then corrected. |
| `存疑:热图+表格混合来源` | 30 | Angew/Nagaki 2019. Heatmap-sourced rows, tR corrected. |
| `存疑:表格数据混入,tR未矫正` | 5 | Angew/Nagaki 2019. Table-sourced rows (tR=2.0, 3.4ms), tR NOT corrected. |

### Intermediate Corrections
| Correction | Rows affected | Explanation |
|------------|--------------|-------------|
| Paper 05 (Nagaki 2008 ortho-ester): split 4 intermediates | 131 | VLM labeled all 4 heatmaps (R=methyl/ethyl/isopropyl/tert-butyl) as "tert-butyl o-(lithio)benzoate." Split into: methyl/ethyl/isopropyl/tert-butyl o-(lithio)benzoate with correct SMILES. |
| Paper 05 substrate1/product correction | 95 (3×~30) | After splitting, methyl/ethyl/isopropyl intermediates still had substrate1="tert-butyl o-bromobenzoate" and product="tert-butyl benzoate". Corrected to respective methyl/ethyl/isopropyl variants with matching SMILES. |
| Nagaki 2010 Figure 08: deleted duplicate data | 131 deleted | Figure 08 (page 9) is an exact reprint of Paper 05's 4 heatmaps (yield values identical). These were mislabeled as "tert-butyl 4-(lithio)benzoate" (para). Deleted to avoid double-counting. |
| Homocoupling paper deduplication | 25 deleted | Same paper (DOI: 10.3762/bjoc.7.122) appeared twice under different names. Deleted "Homocoupling of aryl halides in flow-..." version, kept "Nagaki et al. 2011 - Homocoupling..." version. |
| Homocoupling + Cross-coupling intermediate fix | 55 (25+30) | VLM labeled both papers' intermediate as generic "aryllithium" with PhLi SMILES. Both use p-bromoanisole as substrate, so intermediate is p-methoxyphenyllithium (COc1ccc([Li])cc1). Cross-coupling paper also missing substrate1, now filled. |

### Verification Results
- **Temperature (Y-axis)**: All 11 matched papers' VLM temperatures perfectly match YOLO row detection (100% row count match).
- **Yield**: VLM vs YOLO yield agreement is 99.6% (734/737 exact match). 3 outliers are YOLO OCR errors.
- **tR (X-axis)**: Manually corrected by Zhao; original VLM values preserved in `vlm_tR1_s_original` / `vlm_tR2_s_original`.

## Backup
Pre-correction backup: `organolithium_tr_subdataset_vlm_enriched_backup_pre_correction.csv` (1650 rows, original VLM tR values).
