# Neraium canonical proof package summary

This package compares Neraium instability signaling against a transparent threshold-style baseline.

| Scenario | Neraium warning | Threshold trigger | Lead cycles | Interpretable |
|---|---:|---:|---:|---|
| stable_baseline | None | None | None | yes |
| gradual_drift | 31 | 80 | 49 | yes |
| abrupt_spike | 44 | None | None | yes |
| progressive_critical | 25 | 60 | 35 | yes |
| messy_data | 26 | None | None | no |

## Founder-facing takeaway
Neraium surfaces instability through structural drift/composite progression before hard per-sensor limits in the drift and progressive critical scenarios, while remaining disciplined in stable and noisy scenarios.

## Notable caveats
- Threshold baseline is intentionally simple and transparent.
- Evidence is deterministic and scenario-driven, not a universal performance guarantee.
- System remains read-only and non-actuating.
