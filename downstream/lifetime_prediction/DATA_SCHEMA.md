# Lifetime Dataset Schema

These fields should be added in downstream exports, not forced into the main
pipeline unless truly necessary.

## Core Fields

- `paper_basename`
- `paper_doi`
- `paper_year`
- `source_table_or_figure`
- `source_type`
- `reaction_class`
- `ml_tier`
- `structure_resolution_status`
- `structure_resolution_source`

## Intermediate Fields

- `intermediate_name`
- `intermediate_smiles`
- `intermediate_label_original`
- `intermediate_class`
- `generation_mode`
- `trapping_mode`
- `fate`

## Condition Fields

- `temperature_C`
- `residence_time_s`
- `flow_rate_mL_min`
- `flow_rate_stream1_mL_min`
- `flow_rate_stream2_mL_min`
- `solvent`
- `additive`
- `catalyst`
- `pressure_bar`
- `reactor_type`
- `batch_or_flow`

## Lifetime-Oriented Fields

- `stability_note`
- `lifetime_evidence_text`
- `lifetime_evidence_source`
- `lifetime_value_s`
- `lifetime_lower_bound_s`
- `lifetime_upper_bound_s`
- `lifetime_bucket`
- `stability_class`
- `lifetime_confidence`

## Suggested First Label Sets

### `stability_class`

- `stable`
- `moderately_sensitive`
- `highly_unstable`
- `must_trap_immediately`
- `unknown`

### `lifetime_bucket`

- `<1 s`
- `1-10 s`
- `10-300 s`
- `>300 s`
- `qualitative_only`
- `unknown`

## Notes

Do not require all fields from the start.
For the first modeling iteration, the most important fields are:

- intermediate identity / class
- temperature
- solvent
- residence time
- batch vs flow context
- stability / lifetime proxy label
