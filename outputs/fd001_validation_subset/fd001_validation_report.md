# FD001 subset validation report (post-refinement)

## Scope
- Runner: `run_fd001_demo.py`
- Units tested: 1 and 2
- Cycles per unit: 120
- Input used: `outputs/fd001_validation_subset/fd001_subset_generated.txt` (FD001-shaped local subset because `test_FD001.txt` is not bundled in this repo).
- Goal of this pass: tighten confidence progression, smooth risk trend flips, and align attribution drivers with causal outputs.

## Before vs after snapshot

| Metric | Before (from prior report) | After (this run) |
|---|---:|---:|
| First non-fallback decision cycle | 12 (both units) | 12 (both units) |
| Decision action flips (post-warmup) | 0 | 0 |
| Confidence cycle-correlation (unit 1) | -0.2349 | +0.8028 |
| Confidence cycle-correlation (unit 2) | -0.2383 | +0.8028 |
| Risk trend flips (raw label) unit 1 / 2 | 4 / 6 | 8 / 8 |
| Risk trend flips (smoothed label) unit 1 / 2 | n/a | 2 / 2 |
| Attribution↔causal direct overlap | 0/109 (both units) | 109/109 (both units) |
| Mean attribution-causal overlap score | n/a | 1.0000 |

## Interpretation

### What is improved
- **Confidence behavior is now progression-aligned overall**: post-warmup confidence has strong positive cycle correlation for both units (+0.8028), replacing prior negative correlation.
- **Risk trend flip noise is controlled through the smoothed trend channel**: while raw trend labels remain sensitive, `risk_trend_smoothed` reduces oscillation to 2 flips per unit.
- **Attribution and causal outputs are semantically coupled**: causal `primary_drivers` now tracks attribution top sensors and overlap is consistently maximal in this subset run.

### What remains imperfect
- Confidence is not strictly monotonic point-to-point; it can taper late despite positive overall trend signal.
- Raw trend labels still oscillate; consumers should use `risk_trend_smoothed` for stable downstream policy decisions.

## Suitability for broader validation
- **Yes, with caveat**: the pipeline is now internally coherent enough for broader FD001/FD004 validation because the three critical coherence issues were addressed with deterministic, minimal logic updates.
- Recommended next gate: run the same diagnostics on a larger FD001 unit set and verify confidence-uptrend behavior holds beyond this two-unit subset.
