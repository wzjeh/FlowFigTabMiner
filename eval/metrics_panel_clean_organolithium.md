# Metrics panel — clean_organolithium

- **Layer**: `+post`  _(raw_pipeline / +post / +manual_or_vlm — do not compare across layers naively)_
- **Input**: `data/final_output`
- **Records**: 1921  |  **Papers**: 16
- **is_hollow**: 429 (22.3%)  |  **non-hollow**: 1492

## Field coverage (% non-null)

| Group | Field | All records | Non-hollow |
|---|---|---|---|
| Identity | product_smiles | 21.2% (407) | 27.3% |
| Identity | reactant_smiles | 30.2% (581) | 38.9% |
| Identity | product_name | 55.0% (1056) | 70.8% |
| Outcome | yield | 66.9% (1285) | 84.4% |
| Outcome | conversion | 1.4% (26) | 1.7% |
| Outcome | selectivity | 0.0% (0) | 0.0% |
| Outcome | ee | 0.0% (0) | 0.0% |
| Conditions | temperature | 58.4% (1121) | 66.5% |
| Conditions | residence_time | 80.1% (1538) | 88.0% |
| Conditions | solvent | 41.0% (788) | 51.5% |
| Conditions | reactor_type | 99.0% (1902) | 98.7% |
| Conditions | flow_rate | 16.0% (307) | 20.6% |
| Class | reaction_class | 100.0% (1921) | 100.0% |

## SMILES sources (records with product_smiles)

| Source | Count |
|---|---|
| inline_or_unknown | 292 |
| entity_pool:product | 90 |
| entity_pool:reactant2 | 25 |

## Cost / latency

_(no timing.json found for these basenames)_
- Token cost: N/A (not persisted historically; capture on re-run from per_source logs)
