# Organolithium Scope Table Dataset

**File**: `organolithium_scope_table_dataset.csv`
**Rows**: 1,267 (deduplicated) | **Columns**: 24 | **Papers**: 97

## Source

Extracted from `final.json` files produced by the VLM adjudication step (Step 5) of the FlowFigTabMiner pipeline. Each final.json contains table-sourced reaction entries with structured fields. Entries were filtered for organolithium-relevant reactions (keyword: "lithium", "lithio", "BuLi", "PhLi", "organolithium", etc. in reactant names, notes, or reaction class).

## Deduplication

Many papers have overlapping data from two extraction paths:
- **TATR + OCR pipeline** (source = `page_X_table_Y_extracted.csv`)
- **VLM direct reading** (source = `Table N`)

Duplicates were removed by matching on (paper_doi, reactant1_name, reactant2_name, yield_pct, temperature_C, residence_time_s). VLM "Table N" entries were preferred when duplicates existed (generally more accurate on table body content).

Original pre-dedup count: 1,983 rows. Removed: 716 duplicates.

## Column Definitions

| Column | Type | Non-null | Description |
|--------|------|----------|-------------|
| `paper` | str | 1267 | Paper name/identifier |
| `paper_doi` | str | 1117 | DOI |
| `source` | str | 1267 | Extraction source (page table CSV or VLM Table N) |
| `entry_number` | str | 557 | Entry/row number in original table |
| `reactant1_name` | str | 1063 | First reactant name |
| `reactant1_smiles` | str | 81 | First reactant SMILES |
| `reactant2_name` | str | 1061 | Second reactant name |
| `reactant2_smiles` | str | 69 | Second reactant SMILES |
| `product_name` | str | 921 | Product name |
| `product_smiles` | str | 117 | Product SMILES |
| `product_label` | str | 723 | Product label (e.g., "3a", "4b") |
| `yield_pct` | float | 890 | Yield percentage |
| `yield_type` | str | 956 | Yield type: "isolated", "GC", "NMR", etc. |
| `conversion_pct` | float | 208 | Conversion percentage |
| `selectivity_pct` | float | 35 | Selectivity percentage |
| `ee_pct` | float | 39 | Enantiomeric excess |
| `stoichiometry` | str | 509 | Reagent equivalents/stoichiometry |
| `reaction_class` | str | 1236 | Reaction type (nucleophilic addition, C-C coupling, etc.) |
| `temperature_C` | float | 1026 | Reaction temperature in Celsius |
| `residence_time_s` | float | 806 | Residence time in seconds |
| `solvent` | str | 1096 | Reaction solvent |
| `reactor_type` | str | 1031 | Reactor type |
| `catalyst` | str | 439 | Catalyst |
| `notes` | str | 946 | VLM notes (conditions, context) |

## Key Statistics

| Metric | Value |
|--------|-------|
| Entries with yield | 890 (70.2%) |
| Entries with yield + tR | 578 (45.6%) |
| Flow entries | 1,150 (mean yield 65.2%) |
| Batch entries | 117 (mean yield 42.1%) |
| SMILES coverage (reactant1) | 81 (6.4%) |
| SMILES coverage (product) | 117 (9.2%) |

## Reaction Class Distribution

| Class | Count |
|-------|-------|
| nucleophilic addition | 385 |
| halogen-metal exchange | 306 |
| C-C coupling | 223 |
| directed metalation | 92 |
| anionic polymerization | 57 |
| alkylation | 52 |
| carbolithiation | 47 |
| reduction | 33 |

## Overlap with Heatmap Kinetics Dataset

8 papers appear in both this scope dataset and the heatmap kinetics dataset (`organolithium_tr_subdataset_vlm_enriched.csv`). The scope data provides complementary information: different substrates, electrophiles, or batch-vs-flow comparisons for the same intermediates characterized kinetically.

Key cross-reference finding: Arrhenius extrapolation from heatmap kinetics predicts batch scope yields quantitatively (tBu ester: 58.1% predicted vs 61% actual). See `data/ml_lifetime/ANALYSIS_REPORT.md` for details.

## Data Quality Notes

- SMILES coverage is low (6-9%) because VLM adjudication prioritizes name-based extraction. SMILES are only present when the VLM explicitly recognized molecular structures.
- Yield values are as reported in papers (isolated, GC, NMR). No normalization applied.
- Residence time units were converted to seconds where possible. Some entries may have unconverted units in the `notes` field.
- 4 paper-name duplicates were normalized (different pipeline runs assigned different names to the same DOI).
