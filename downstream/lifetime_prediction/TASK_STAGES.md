# Task Stages

## Stage 0: Freeze Main Pipeline

Do not modify the main extraction architecture unless a real blocker appears.
All lifetime work should happen downstream of the existing `_normalized.json`
outputs.

## Stage 1: Intermediate-Centric Dataset Build

Transform reaction-centric records into intermediate-centric records.

Each row should answer:

- what intermediate is being generated
- how it is generated
- under what conditions it exists
- what evidence suggests it is stable or unstable

## Stage 2: Label Construction

First labels should be simple and robust:

- `stability_class`
- `needs_flow`
- `needs_low_temperature`
- `needs_ultrashort_residence_time`

Then move to:

- `lifetime_bucket`

Only later:

- `lifetime_value_s`

## Stage 3: Manual Review

Prioritize review of:

- `A` tier records first
- then `B` tier records with strong notes / context
- then edge cases used to define label rules

Suggested first review size:

- 100 to 200 records

## Stage 4: Baseline Models

Train simple baselines first:

- logistic regression
- gradient boosting
- small tree-based models

Do not start with large deep models.

## Stage 5: Error Analysis

Check failures by:

- intermediate class
- temperature regime
- solvent family
- source type (`table`, `figure`, `scheme`)
- resolution quality (`original` vs `resolved`)

## Stage 6: Regression Only After Classification Stabilizes

Only attempt exact lifetime regression after:

- bucket labels are trustworthy
- key intermediate classes have enough samples
- manual review confirms label consistency
