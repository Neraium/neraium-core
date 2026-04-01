# Proof scenario: Missing / messy data

- Name: `messy_data`
- Description: Dropouts and partial windows stress data quality handling and uncertainty reporting.
- Expected behavior: Engine remains explicit about uncertainty and avoids unsafe confidence jumps.
- Neraium first warning cycle: `26`
- Threshold first trigger cycle: `None`
- Lead cycles (threshold - Neraium): `None`
- Progression interpretable: `True`
- Quality statuses observed: `DATA_QUALITY_LIMITED, FRAME_DROPPED, TEMPORAL_IRREGULARITY`

## Compact proof statement
Missing / messy data: Neraium warning at cycle 26, threshold first trigger at cycle None, lead=None.

## Caveats
- This scenario is deterministic synthetic evidence, not a universal benchmark.
- Neraium outputs are read-only decision support for human operators.
