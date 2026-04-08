# Proof scenario: Gradual relational drift

- Name: `gradual_drift`
- Description: Cross-signal structure drifts steadily before any single sensor crosses a hard threshold.
- Expected behavior: Neraium warning appears before threshold trigger with smooth risk progression.
- Neraium first warning cycle: `31`
- Threshold first trigger cycle: `80`
- Lead cycles (threshold - Neraium): `49`
- Progression interpretable: `True`
- Quality statuses observed: `DATA_QUALITY_LIMITED`

## Compact proof statement
Gradual relational drift: Neraium warning at cycle 31, threshold first trigger at cycle 80, lead=49.

## Caveats
- This scenario is deterministic synthetic evidence, not a universal benchmark.
- Neraium outputs are read-only decision support for human operators.
