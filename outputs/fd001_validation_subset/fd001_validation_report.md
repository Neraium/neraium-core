# FD001 subset validation report

## Scope
- Runner: `run_fd001_demo.py`
- Units tested: 1 and 2
- Cycles per unit: 120
- Input used: `outputs/fd001_validation_subset/fd001_subset_generated.txt` (FD001-shaped local subset because `test_FD001.txt` is not bundled in this repo).

## What looks correct
- Warmup gating is sensible: decisions are unavailable early and become available at cycle 12 for both units.
- Core output sections are emitted each cycle (`attribution`, `regime_memory`, `risk_assessment`, `operator_guidance`, `causal_analysis`, `decision`).
- Decision action text is stable after warmup (0 action flips for both units).
- A stable causal hypothesis appears early and then persists (mainly `Localized structural drift centered on coherence_loss_score.`).

## What looks weak or unstable
- Decision confidence does not consistently rise with degradation progression (unit 2 declines from 0.4919 to 0.3902).
- Risk trend labels are somewhat noisy (`flat/increasing/decreasing` flips), especially for unit 2.
- Attribution and causal outputs are weakly aligned semantically: top attribution is sensor-centric (e.g., `s21`) while causal text is metric-centric (`coherence_loss_score`/`regime_distance`).

## Per-unit answers to requested questions
| unit | first risk rise | first stable causal hypothesis | first non-fallback decision | confidence increase with degradation? | attribution vs causal aligned? | noisy flips? |
|---|---:|---:|---:|---|---|---|
| 1 | 12 | 12 | 12 | Mixed (small net increase +0.0158, but negative cycle-confidence correlation -0.2349) | 0/109 direct matches | actions:0, risk_level:0, risk_trend:4 |
| 2 | 12 | 13 | 12 | No (net -0.1017, negative cycle-confidence correlation -0.2383) | 0/109 direct matches | actions:0, risk_level:2, risk_trend:6 |

## Usefulness for early warning / decision support
- **Moderately useful for early warning**, because the platform exits fallback quickly and keeps guidance stable.
- **Weaker for high-confidence decision support** on this run, due to confidence drift and attribution↔causal mismatch.

## Top 3 tuning opportunities
1. Tie decision confidence more tightly to sustained degradation evidence so confidence does not sag late in monotonic runs.
2. Smooth risk trend-state transitions to reduce short-horizon flip noise (`flat/increasing/decreasing`).
3. Improve causal-attribution linkage (e.g., map metric-level causal hypotheses back to dominant physical sensors/drivers).
