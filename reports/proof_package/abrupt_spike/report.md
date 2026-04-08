# Proof scenario: Abrupt disturbance / spike

- Name: `abrupt_spike`
- Description: Short-lived transient perturbation should not permanently escalate systemic risk.
- Expected behavior: Signal may blip but should recover; no sustained escalation.
- Neraium first warning cycle: `44`
- Threshold first trigger cycle: `None`
- Lead cycles (threshold - Neraium): `None`
- Progression interpretable: `True`
- Quality statuses observed: `DATA_QUALITY_LIMITED`

## Compact proof statement
Abrupt disturbance / spike: Neraium warning at cycle 44, threshold first trigger at cycle None, lead=None.

## Caveats
- This scenario is deterministic synthetic evidence, not a universal benchmark.
- Neraium outputs are read-only decision support for human operators.
