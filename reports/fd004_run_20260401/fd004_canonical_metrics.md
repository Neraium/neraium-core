# FD004 Canonical Metric Report

## Aggregate
- units_total: 8
- structural_rank_compactness_mean: 0.3478
- decision_stability_mean: 0.1779
- early_signal_rate: 1.0
- confidence_threshold_hit_rate: 0.0
- action_flip_count_total: 388
- attribution_alignment_coupled_mean: 0.8625
- attribution_alignment_independent_mean: 0.0

## Metric definitions
### structural_rank_compactness_mean
- meaning: Average of SII geometry-layer coherence_score (effective covariance rank / feature count). Captures structural dimensional compactness, not decision stability.
- expected_range: [0, 1]
- better_direction: context-dependent
- caveat: Lower values can be normal for strongly coupled systems; do not read as decision incoherence.

### decision_stability_mean
- meaning: Mean per-unit stability score derived from action flips after warmup (1=no flips, 0=maximally unstable).
- expected_range: [0, 1]
- better_direction: higher
- caveat: Depends on warmup and action-availability windows.

### early_signal_rate
- meaning: Fraction of units with first canonical early-signal cycle at least lead_cycles_before_end before terminal cycle. Canonical early signal = decision available AND risk level in {medium, high} AND risk trend increasing.
- expected_range: [0, 1]
- better_direction: higher
- caveat: Sensitive to lead-cycle requirement and risk-trend smoothing.

### confidence_threshold_hit_rate
- meaning: Fraction of units that reach decision.confidence >= configured threshold at least once.
- expected_range: [0, 1]
- better_direction: higher
- caveat: Threshold-dependent; compare runs only with identical threshold.

### action_flip_count_total
- meaning: Total number of action label changes across all units after warmup.
- expected_range: [0, +∞)
- better_direction: lower
- caveat: Counts only when decision actions are available.

### attribution_alignment_coupled_mean
- meaning: Mean of causal_analysis.attribution_causal_overlap_score; uses causal primary drivers seeded from attribution top sensors.
- expected_range: [0, 1]
- better_direction: higher
- caveat: Partially attribution-coupled and not an independence test.

### attribution_alignment_independent_mean
- meaning: Mean overlap between attribution top sensors and independent dominant drivers from score components (out['dominant_drivers']).
- expected_range: [0, 1]
- better_direction: higher
- caveat: Still heuristic, but less self-referential than coupled overlap.
