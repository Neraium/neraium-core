# Proof scenario: Progressive degradation to critical

- Name: `progressive_critical`
- Description: Long, compounding degradation eventually crosses hard operational thresholds.
- Expected behavior: Neraium warning appears materially earlier than threshold baseline.
- Neraium first warning cycle: `25`
- Threshold first trigger cycle: `60`
- Lead cycles (threshold - Neraium): `35`
- Progression interpretable: `True`
- Quality statuses observed: `DATA_QUALITY_LIMITED`

## Compact proof statement
Progressive degradation to critical: Neraium warning at cycle 25, threshold first trigger at cycle 60, lead=35.

## Caveats
- This scenario is deterministic synthetic evidence, not a universal benchmark.
- Neraium outputs are read-only decision support for human operators.
