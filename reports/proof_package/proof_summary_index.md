# Neraium canonical proof package summary

This package compares Neraium instability signaling against a transparent threshold-style baseline.

| Scenario | Neraium warning | Threshold trigger | Lead cycles | Interpretable |
|---|---:|---:|---:|---|
| stable_baseline | None | None | None | yes |
| gradual_drift | 33 | 80 | 47 | yes |
| abrupt_spike | 50 | None | None | yes |
| progressive_critical | 26 | 60 | 34 | yes |
| messy_data | 26 | None | None | yes |

## Founder-facing takeaway
Neraium surfaces instability through structural drift/composite progression before hard per-sensor limits in the drift and progressive critical scenarios, while remaining disciplined in stable and noisy scenarios.

## Notable caveats
- Threshold baseline is intentionally simple and transparent.
- Evidence is deterministic and scenario-driven, not a universal performance guarantee.
- System remains read-only and non-actuating.
