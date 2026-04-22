# Neraium proof package one-glance artifact

## What happened
Five deterministic scenarios compare structural instability signaling vs conventional per-signal thresholds.

## Timing evidence
| Scenario | Neraium warning | Threshold trigger | Lead cycles | Interpretable |
|---|---:|---:|---:|---|
| stable_baseline | None | None | None | yes |
| gradual_drift | 31 | 80 | 49 | yes |
| abrupt_spike | 44 | None | None | yes |
| progressive_critical | 25 | 60 | 35 | yes |
| messy_data | 26 | None | None | no |

## Why it matters
In progressive drift/degradation cases, threshold crossings arrive after relational instability is already visible. Neraium gives operators lead time for investigation before critical hard-limit alarms.

## Safe product posture
Read-only analytics. Human-in-the-loop. No automated actuation claims.
