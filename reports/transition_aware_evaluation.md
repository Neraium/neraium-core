# Transition-aware Failure Detection Evaluation (Shock-Term Refinement)

## Files changed
- `neraium_core/alignment.py`
- `tests/test_transition_pressure.py`
- `scripts/evaluate_transition_awareness.py`
- `reports/transition_aware_evaluation.json`
- `reports/transition_aware_evaluation.md`

## Commands run
- `pytest -q tests/test_transition_pressure.py tests/test_scoring.py`
- `PYTHONPATH=. python scripts/evaluate_transition_awareness.py`

## Test result summary
- `9 passed in 9.99s`

## Shock-term formula update
Transition pressure now combines:
1. Existing recency/EMA kinetic pressure (continuous transition signal).
2. **Discrete shock detector** (event channel, separate from EMA), using:
   - single-step correlation jump (`corr_jump_raw`),
   - graph edge-flip jump (`graph_edge_flip_rate`),
   - spectral-radius jump (`spectral_jump_raw`).

A shock event triggers only when change exceeds recent-history variance thresholds, and requires a correlation jump plus one additional corroborating jump (edge or spectral). When triggered, a short-lived boost is applied for at most two steps and then decays via refractory logic.

## Same-harness comparison (legacy vs transition-aware, current)

| Scenario | Variant | first WATCH | first ALERT | alert flips | WATCH/ALERT time | ALERT time | onset pressure spike | late chronic pressure |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| steady_bad | legacy | 91 | 91 | 1 | 129 | 129 | 0.000 | 0.000 |
| steady_bad | transition_aware | 91 | 91 | 1 | 129 | 129 | 1.728 | 0.002 |
| active_break | legacy | 91 | 91 | 1 | 99 | 99 | 0.000 | 0.000 |
| active_break | transition_aware | 91 | 91 | 1 | 99 | 99 | 0.021 | 0.770 |
| regime_shift | legacy | 82 | 82 | 1 | 108 | 108 | 0.000 | 0.000 |
| regime_shift | transition_aware | 82 | 82 | 1 | 108 | 108 | 2.951 | 0.002 |
| noisy_oscillatory | legacy | 70 | 71 | 9 | 102 | 83 | 0.000 | 0.000 |
| noisy_oscillatory | transition_aware | 71 | 71 | 11 | 115 | 87 | 0.056 | 0.718 |
| sharp_break | legacy | 96 | 96 | 1 | 94 | 94 | 0.000 | 0.000 |
| sharp_break | transition_aware | 96 | 96 | 1 | 94 | 94 | 1.726 | 0.002 |

## Before/after comparison on the same harness (previous formula vs shock-term formula)

| Scenario | Metric | Before | After | Δ |
|---|---|---:|---:|---:|
| steady_bad | onset_pressure_spike | 1.7285 | 1.7285 | +0.0000 |
| steady_bad | late_chronic_pressure | 0.0016 | 0.0016 | +0.0000 |
| active_break | onset_pressure_spike | 0.0209 | 0.0209 | +0.0000 |
| active_break | late_chronic_pressure | 0.7697 | 0.7697 | +0.0000 |
| regime_shift | onset_pressure_spike | 1.8249 | 2.9512 | +1.1263 |
| regime_shift | late_chronic_pressure | 0.0016 | 0.0016 | +0.0000 |
| noisy_oscillatory | onset_pressure_spike | 0.0545 | 0.0555 | +0.0010 |
| noisy_oscillatory | late_chronic_pressure | 0.7182 | 0.7182 | -0.0000 |

## Outcome
- **Discrete breaks are now captured strongly** (`sharp_break` onset spike `1.726`; regime-shift onset increased).
- **Noisy oscillation does not create sustained shock streaks** (covered by new test with max shock streak <=2).
- **Active-break onset in this harness remains modest**; this path still behaves more like continuous transition than discrete shock in the current scenario shape.

## Recommendation
**REVISE**: shock event detection now works for true discontinuities, but active-break harness onset is still not clearly separated and noisy dwell remains above legacy.
