# Lifetime Prediction Workspace

This folder is intentionally separate from the main extraction pipeline.
The extraction architecture is treated as stable. Work here should focus on
task-specific cleaning, labeling, dataset curation, and modeling for
organolithium intermediate lifetime / stability prediction.

## Goal

The end goal is not to jump straight to continuous-value lifetime regression.
The safer path is:

1. Build an intermediate-centric dataset.
2. Start with simpler labels.
3. Validate those labels by manual review.
4. Only then move toward finer-grained lifetime prediction.

## Recommended Stage Order

### Stage 1: Stability Classification

Start with easier targets such as:

- `stable`
- `moderately_sensitive`
- `highly_unstable`
- `must_trap_immediately`

This stage best matches the current dataset, because many papers provide
qualitative stability language even when they do not provide exact lifetimes.

### Stage 2: Lifetime Bucket Prediction

Convert evidence into coarse time buckets such as:

- `<1 s`
- `1-10 s`
- `10-300 s`
- `>300 s`
- `qualitative_only`

This is likely the best first modeling task that still points toward the final
lifetime objective.

### Stage 3: Continuous Lifetime Prediction

Only after Stage 2 is stable should we attempt regression on a numeric value
like `lifetime_value_s`.

This stage should use only high-confidence records with explicit time evidence
or strong proxy evidence.

## Why This Order

Current extraction outputs are stronger on:

- temperature
- solvent
- residence time
- reactor / batch-vs-flow context
- qualitative instability language

They are weaker on:

- full intermediate structures for every record
- exact numeric lifetime labels

So the highest-value path is to use the data where it is already strongest.

## Data Scope

This downstream task should cover all organolithium intermediates, not just
aryllithiums.

Important classes to keep separate when possible:

- aryllithium
- vinyllithium
- alkyllithium
- heteroaryllithium
- oxiranyllithium
- ynolate / lithium ynolate
- carbamoyl anion
- dianion / related highly reactive lithium intermediates

## Immediate Next Tasks

1. Build an intermediate-centric export from the current normalized records.
2. Add lifetime-oriented fields.
3. Create a manual review set for `A` and `B` tier records.
4. Define a first classification label set.
5. Train a simple baseline classifier before attempting regression.

See `TASK_STAGES.md` and `DATA_SCHEMA.md` in this folder.
