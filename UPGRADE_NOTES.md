# Neraium Intelligence Stack — Upgrade Notes

This document summarizes the production-readiness upgrade pass: what changed, why it improves the engine, and what new outputs were added.

## 1. Causal Attribution Layer

**What changed**
- New module `neraium_core/causal_attribution.py` computes a ranked attribution of which signals contribute most to current structural instability.
- Attribution combines per-signal correlation drift contribution, causal (Granger-style) outbound strength, and structural importance. Purely observational; no interventions.

**Why it helps**
- Operators and evaluators can see *which* sensors or nodes are driving instability instead of only a global score.
- Supports root-cause analysis and triage without changing the read-only framing.

**New outputs in `process_frame`**
- `causal_attribution`: `{ "top_drivers": [...], "driver_scores": { sensor_name: score } }`
- `dominant_driver`: name of the top-ranked driver, or `None` when there are fewer than two valid signals.

---

## 2. Missing-Data Robustness

**What changed**
- `data_quality.py`: added `data_quality_summary()`, `should_use_degraded_analytics()`, and `impute_missing_simple()`.
- Data-quality gating is unchanged; when the gate fails but missingness is below a configurable ceiling (default 85%), the engine may still run analytics using simple column-mean imputation and returns output with **degraded confidence**.
- Confidence is explicitly reduced when using this fallback path.

**Why it helps**
- Partial signal dropout or stale sensors no longer force a complete loss of output.
- Results under degraded data are still interpretable (e.g. for monitoring) while confidence and data_quality_summary make the limitation explicit.

**New/updated outputs**
- `data_quality_summary`: compact dict with `gate_passed`, `missingness_rate`, `valid_signal_count`, `total_sensors`, `stale_sensor_count`, `flatlined_sensor_count`, `missing_sensor_count`, `statuses`, `sensor_coverage`, `variability_coverage`.
- `active_sensor_count`, `missing_sensor_count` at top level for quick checks.
- When the degraded path is used, evidence confidence is scaled down so the reported confidence score reflects uncertainty.

---

## 3. Adaptive Baseline Behavior

**What changed**
- **Rolling baseline**: The engine keeps an optional `_rolling_baseline_corr` that is updated only when the system is in `NOMINAL_STRUCTURE` and the composite instability score is below a threshold (0.85). Update uses exponential moving average (default α = 0.92) so the baseline adapts slowly and does not absorb instability.
- **Regime-specific baseline**: When a regime already has a stored baseline, it is updated with EMA (α = 0.88) so the regime baseline gradually adapts inside that regime.
- Drift is computed against the rolling baseline when available; otherwise the fixed initial window is used.

**Why it helps**
- Reduces false drift from slow, legitimate changes in stable operation.
- Regime distinctiveness is preserved because we do not update the rolling baseline during instability.

**New outputs**
- `baseline_mode`: `"fixed"` or `"rolling"` (or `None` before enough history).
- `regime_memory_state`: `{ "regime_name", "library_size", "baseline_count" }` for experiment and diagnostics.

---

## 4. Explicit Uncertainty / Confidence Stabilization

**What changed**
- Confidence is no longer based only on “active” components. It now incorporates:
  - **Data quality**: missingness, variability coverage, sensor coverage, and gate status (and degraded-path penalty).
  - **Classification stability**: fraction of recent interpreted states that match the most common state over a short window.
  - **Metric disagreement**: high variance across Tier-1 components slightly reduces confidence.
- The engine passes a **stabilized confidence score** (0–1) into the decision layer, which maps it to the existing categorical confidence (high/medium/low) for backward compatibility.

**Why it helps**
- Confidence better reflects both data quality and the stability of the classification over time, and is lower when metrics disagree or data is poor.

**New outputs**
- `confidence_score`: numeric [0, 1] confidence.
- `classification_stability`: (when present) stability of recent classifications, in [0, 1].

---

## 5. Decision Layer Separation

**What changed**
- **Regime shift** is clearly separated: returned when structure has moved (relational drift) but there is *no* strong directional/spectral (coupling) breakdown and persistence is bounded.
- **Coupling instability** depends more strongly on directional and spectral (interaction) breakdown; a lower “sustained” bar is used so persistent coupling breakdown is reported as `COUPLING_INSTABILITY_OBSERVED` before being classified as coherence-under-constraint.
- **Structural instability** depends more strongly on relational drift plus entropy and regime drift; it still requires sustained, multi-indicator confirmation.
- Entropy is now an explicit input to the interpret state; structural evidence combines motion, regime departure, and entropy.

**Why it helps**
- Clearer separation reduces confusion between “regime change” and “structural/coupling breakdown” and keeps operator-safe, observational language without operational directives.

---

## 6. Experiment-Friendly Analytics

**What changed**
- `process_frame` output was extended with fields that make evaluation and experimentation easier.

**New top-level fields**
- `classification_stability` (when computed)
- `data_quality_summary`
- `active_sensor_count`, `missing_sensor_count`
- `dominant_driver`
- `baseline_mode`, `regime_memory_state`
- `confidence_score`
- `causal_attribution`

Existing fields (e.g. `interpreted_state`, `structural_drift_score`, `regime_drift`, `experimental_analytics`) are unchanged. No interpreted states were removed.

---

## 7. Tests

**What changed**
- New test file `tests/test_upgrade_scenarios.py` with scenarios for:
  - Nominal operation (output shape and stability)
  - Regime shift under clean transition
  - Coupling instability presence
  - Structural instability (relational + persistence)
  - Missing data (degraded output)
  - Stale sensor behavior
  - Adaptive baseline (`baseline_mode`)
  - Causal attribution presence
  - Classification stability and confidence score

Existing tests (decision_layer, multinode persistence/hysteresis, single-sensor stream, scoring, etc.) remain and pass.

---

## Backward Compatibility

- All existing interpreted states are preserved.
- Existing `process_frame` keys are unchanged; new keys are additive.
- Decision layer still returns `confidence` as a categorical string; `confidence_score` is an additional numeric field.
- The engine remains read-only and observational; no operational directives were introduced.

## Files Touched (Summary)

- **New**: `neraium_core/causal_attribution.py`
- **Modified**: `neraium_core/alignment.py`, `neraium_core/data_quality.py`, `neraium_core/decision_layer.py`
- **New tests**: `tests/test_upgrade_scenarios.py`
