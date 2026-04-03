# Proof scenario: Stable baseline

- Name: `stable_baseline`
- Description: System remains in normal operating envelope with slow correlated movement.
- Expected behavior: Neraium remains below warning threshold and threshold baseline never fires.
- Neraium first warning cycle: `None`
- Threshold first trigger cycle: `None`
- Lead cycles (threshold - Neraium): `None`
- Progression interpretable: `True`
- Quality statuses observed: `DATA_QUALITY_LIMITED`

## Compact proof statement
Stable baseline: Neraium warning at cycle None, threshold first trigger at cycle None, lead=None.

## Caveats
- This scenario is deterministic synthetic evidence, not a universal benchmark.
- Neraium outputs are read-only decision support for human operators.
