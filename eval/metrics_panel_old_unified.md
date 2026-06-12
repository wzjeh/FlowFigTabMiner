# Metrics panel — old_unified_csv

- **Layer**: `+manual_or_vlm`  _(raw_pipeline / +post / +manual_or_vlm — do not compare across layers naively)_
- **Input**: `data/ml_lifetime/clean_organolithium_unified.csv`
- **Records**: 2609  |  **Papers**: 26
- **is_hollow**: 0 (0.0%)  |  **non-hollow**: 2609  ⚠️ no is_hollow field in this dataset

## Field coverage (% non-null)

| Group | Field | All records | Non-hollow |
|---|---|---|---|
| Identity | product_smiles | 100.0% (2609) | 100.0% |
| Identity | reactant_smiles | 96.2% (2511) | 96.2% |
| Identity | product_name | 96.4% (2516) | 96.4% |
| Outcome | yield | 98.7% (2576) | 98.7% |
| Outcome | conversion | 0.0% (0) | 0.0% |
| Outcome | selectivity | 0.0% (0) | 0.0% |
| Outcome | ee | 0.0% (0) | 0.0% |
| Conditions | temperature | 89.7% (2339) | 89.7% |
| Conditions | residence_time | 90.2% (2353) | 90.2% |
| Conditions | solvent | 100.0% (2609) | 100.0% |
| Conditions | reactor_type | 0.0% (0) | 0.0% |
| Conditions | flow_rate | 0.0% (0) | 0.0% |
| Class | reaction_class | 100.0% (2609) | 100.0% |

## SMILES sources (records with product_smiles)

| Source | Count |
|---|---|
| inline_or_unknown | 2609 |

## Cost / latency

_(no timing.json found for these basenames)_
- Token cost: N/A (not persisted historically; capture on re-run from per_source logs)
