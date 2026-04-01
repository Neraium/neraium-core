# Proof scenario: Progressive degradation to critical

- Name: `progressive_critical`
- Description: Long, compounding degradation eventually crosses hard operational thresholds.
- Expected behavior: Neraium warning appears materially earlier than threshold baseline.
- Neraium first warning cycle: `26`
- Threshold first trigger cycle: `60`
- Lead cycles (threshold - Neraium): `34`
- Progression interpretable: `True`
- Quality statuses observed: `DATA_QUALITY_LIMITED`

## Compact proof statement
Progressive degradation to critical: Neraium warning at cycle 26, threshold first trigger at cycle 60, lead=34.

## Caveats
- This scenario is deterministic synthetic evidence, not a universal benchmark.
- Neraium outputs are read-only decision support for human operators.
