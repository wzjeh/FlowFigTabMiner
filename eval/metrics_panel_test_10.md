# Metrics panel — test_10

- **Layer**: `+post`  _(raw_pipeline / +post / +manual_or_vlm — do not compare across layers naively)_
- **Input**: `data/final_output`
- **Records**: 270  |  **Papers**: 8
- **is_hollow**: 64 (23.7%)  |  **non-hollow**: 206

## Field coverage (% non-null)

| Group | Field | All records | Non-hollow |
|---|---|---|---|
| Identity | product_smiles | 19.6% (53) | 25.7% |
| Identity | reactant_smiles | 31.5% (85) | 41.3% |
| Identity | product_name | 45.2% (122) | 59.2% |
| Outcome | yield | 45.2% (122) | 59.2% |
| Outcome | conversion | 8.5% (23) | 11.2% |
| Outcome | selectivity | 0.0% (0) | 0.0% |
| Outcome | ee | 0.0% (0) | 0.0% |
| Conditions | temperature | 90.7% (245) | 87.9% |
| Conditions | residence_time | 41.1% (111) | 53.9% |
| Conditions | solvent | 95.2% (257) | 93.7% |
| Conditions | reactor_type | 73.0% (197) | 95.6% |
| Conditions | flow_rate | 7.8% (21) | 10.2% |
| Class | reaction_class | 100.0% (270) | 100.0% |

## SMILES sources (records with product_smiles)

| Source | Count |
|---|---|
| inline_or_unknown | 53 |

## Cost / latency

- Papers with timing: 10
- Total wall time: 3201.7 s  |  mean/paper: 320.2 s
- Stage totals: filter=1.6s, tfid=49.0s, figure=115.1s, table=2491.2s, scheme=227.4s, local_vars=0.0s, assembly=270.4s, post=47.2s
- Token cost: N/A (not persisted historically; capture on re-run from per_source logs)
