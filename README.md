# neraium-core

`neraium-core` is a deployable, **read-only structural instability instrumentation system**.  
It ingests multivariate telemetry, computes **Systemic Infrastructure Intelligence (SII)**, and returns operator-facing evidence outputs.

---

## How Neraium is different

Most tools optimize for **single sensors** or **component failure prediction**: thresholds, per-signal anomalies, or models trained on “normal” history. Neraium focuses on **systemic stability**, how signals **relate** to each other over time, so teams can see **structural drift and approaching instability** before many component-level alarms fire.

| Typical approach | Neraium |
|------------------|---------|
| “Is this sensor bad?” | “Is the **system’s structure** becoming unstable?” |
| Heavy reliance on past failures / big training sets | **Structural** signals from baseline vs recent **relationship geometry** |
| Often opaque ML scores | Deterministic, explainable **structural** metrics + layered operator signaling |

**Full customer-facing narrative (positioning vs predictive maintenance and AI monitoring):**  
→ **[docs/HOW_NERAIUM_IS_DIFFERENT.md](docs/HOW_NERAIUM_IS_DIFFERENT.md)**

---

## Deployment constraints

Neraium is intentionally constrained for current deployments:

- **Read-only analytics** over telemetry and CSV/streaming inputs  
- **Human-in-the-loop decision support only**  
- **No infrastructure control path**  
- **No automated actuation**  

---

## Current product scope

Neraium today is a **system stability instrumentation layer** for detecting structural degradation and instability.  
It does not write back into operational systems and does not execute control actions.

### Input

- Multivariate telemetry from API ingest  
- Batch CSV uploads  
- Time-ordered streaming-like updates via repeated ingest calls  
- Canonical raw industrial ingestion bridge (tabular rows or directory signal blocks): `docs/RAW_INGESTION.md`  

### Processing

- Systemic Infrastructure Intelligence (SII)  
- Structural relationship analysis over time (baseline vs recent windows)  
- Phase detection: `stable`, `drift`, `unstable`  

### Output

- `structural_drift_score`  
- `composite_instability`  
- `phase`  
- `trend`  
- `risk_level`  
- `operator_message`  
- `causal_analysis` (ranked hypotheses, counterfactual robustness, validation plan, value-of-information-ranked actions)  
- Proof artifacts / reports where available (for example FD004 summaries, CSV timelines, and plots)  

---

## Architecture (high level)

Operational systems / telemetry sources  
→ One-way data access (ingest only)  
→ Systemic Infrastructure Intelligence (read-only computation)  
→ Human operators and evidence outputs (API results, CSV, reports)  

---

## What SII does (technical)

Neraium implements **SII** as the statistical estimation of **evolving relational geometry** in complex systems.

- It is **not** a generic anomaly detector.  
- It is **not** a classical predictive-maintenance classifier.  
- It compares **baseline and recent sensor-relationship structure** to estimate stability and drift.  

Operator-facing output is **additive and heuristic**, layered on top of math outputs:

- `risk_level`: LOW / MEDIUM / HIGH  
- `trend`: STABLE / RISING / FALLING / UNKNOWN  
- `confidence`: normalized [0.0, 1.0] confidence proxy  
- `operator_message`: plain-language guidance  
- `structural_analysis_available`: whether relational analysis ran  
- `skipped_reason`: why relational analysis was skipped  

---

## Structural degradation detection (FD004 validation)

### Problem

Traditional monitoring often raises alarms only when a system is already close to failure.  
In FD004, the system shows **structural degradation before final failure**, so waiting for hard thresholds alone is too late.

### Approach

SII watches how sensors move **together** over time, not only whether one sensor value crosses a limit.  
That helps reveal **structural change** earlier in the degradation path.

### Results (from FD004)

- 100.0% of units reached **MEDIUM** before **HIGH**.  
- No direct **LOW → HIGH** jumps were observed.  
- Instability increases over time for the hero unit.  
- An early warning window exists before the critical **HIGH** state (average: 1 cycle).  

### Example output

- `reports/fd004_proof_summary.md`  
- `fd004_outputs_subset/hero_unit_timeseries.csv`  

### Interpretation

- Detects structural degradation early, before severe instability.  
- Reduces noisy alerts by avoiding abrupt risk jumps.  
- Matches expected failure progression (`LOW → MEDIUM → HIGH`).  
- Gives operators a short but usable warning window before critical state.  

---

## Mathematical implementation status

### Rigorous structural observables

- Sliding windows `X_t in R^(m x n)` with explicit `baseline_window`, `recent_window`, and `stride` controls.  
- Per-window normalization `z_i(t) = (x_i(t)-mu_i)/sigma_i` with zero-variance and missing-data guards.  
- First-class correlation geometry `R_t = corr(Z_t)`.  
- Baseline-relative structural drift `D_t = ||R_t - R_0||` (Frobenius norm).  
- Signal structural importance `I_i = mean_j |R_ij|`.  
- Graph reconstruction from thresholded correlation and graph observables (degree, density, clustering, connectivity, mean absolute connectivity).  
- Spectral stability observables (spectral radius, spectral gap, dominant mode eigenvector loading).  
- Early warning metrics from temporal signal behavior (per-signal variance and lag-1 autocorrelation, exposed as averaged indicators).  
- Interaction entropy over structural matrix magnitudes.  
- Subsystem-local instability via thresholded graph components and local spectral radius.  

### Proxy / inferential layers

- Directional lagged structure `C_ij = corr(x_i(t), x_j(t+1))` and derived causal energy/asymmetry/divergence are **proxy indicators**, not formal causal proof.  
- Regime awareness is currently a minimal scaffold using a signature vector `[mu_1..mu_n, sigma_1..sigma_n]` with nearest-signature lookup.  
- Forecasting is heuristic extrapolation based on instability trend and velocity (time-to-instability estimate), not a guaranteed failure-time predictor.  

---

## Signal evaluation (current)

Neraium includes a temporary, explicit **advisory recommendation layer** that interprets raw SII outputs for operator-facing signaling.

- It is **rule-based and deterministic**: no learned model, no opaque weighting.
- It evaluates recent `composite_instability`, `structural_drift_score`, and `phase` progression to produce a recommended next step.
- It suppresses noisy patterns (for example, sharp instability spike-and-drop behavior or unstable/stable oscillation).
- It returns structured advisory output (`signal_emitted`, `signal_strength`, `confidence`, `reason`, `phase`, `risk_level`) for clear downstream reporting.

## Structural attribution and operational recommendation upgrade

The SII platform upgrades from pure structural change detection to **structural attribution + advisory operational recommendation intelligence**.

Canonical top-level contract:
- `attribution`
- `regime_memory`
- `risk_assessment`
- `operator_guidance`
- `causal_analysis`
- `operational_recommendation` (canonical)

`operational_recommendation.recommendation_confidence` is bounded to `[0,1]` and computed from converging evidence across causal hypotheses, risk trend clarity, localization strength, and source convergence.

Example canonical recommendation payload:
```json
{
  "operational_recommendation": {
    "status": {"available": true, "advisory": true, "reason": "recommendation_available"},
    "recommended_action": "Inspect subsystem/cluster first: cluster_A.",
    "recommended_target": "cluster_A",
    "priority": 1,
    "recommendation_confidence": 0.74,
    "urgency": "medium",
    "rationale": "Recommendation available from converging structural evidence.",
    "supporting_evidence": [
      {"driver": "cluster_A", "score": 0.72}
    ],
    "operator_note": "Recommendations are advisory outputs intended to support, not replace, qualified operator judgment and site-specific procedures."
  }
}
```

Compatibility note:
- Legacy `decision` views remain available only through deprecated compatibility aliases and endpoints.

---

## Deprecated compatibility module note

`neraium_core.casual` is deprecated and retained only as a temporary compatibility shim.
All canonical runtime imports must use `neraium_core.causal`.

Hardening constraints for the shim:
- no causal logic is implemented in `casual.py`,
- only minimal re-exports are allowed,
- importing the shim emits a `DeprecationWarning`.

---

## Platform Output Structure

Neraium’s canonical platform output is organized into top-level sections. The
runtime canonical schema uses `operational_recommendation` as a **top-level**
field (not nested under `operator_guidance`).

- `attribution`
  - `top_drivers`: ranked structural contributors.
  - `group_contributions`: grouped structural contribution summary.

- `regime_memory`
  - `regime_name`: nearest known structural regime label.
  - `regime_distance`: distance to nearest regime signature.
  - `library_size` / `baseline_count`: regime memory depth and baseline confidence context.

- `risk_assessment`
  - `risk_level`: interpreted risk category from structural evidence.
  - `trend`: stability trajectory direction.
  - `latest_instability`: current composite instability.

- `causal_analysis` *(additive layer)*
  - `hypotheses`: competing causal hypotheses (`physical`, `sensor`, `systemic`) grounded in structural signals.
  - `top_hypothesis`: highest-ranked current explanation candidate.
  - `counterfactual`: lightweight robustness checks (`remove_top_driver`, `remove_relationship_change`, localization dependency).
  - `validation_plan`: confirm/falsify actions with expected true/false outcomes.
  - `recommended_sequence` / `best_next_action`: value-of-information-ranked execution order.
  - `status`: availability state (`ok`, `warmup`, `insufficient_evidence`).

- `operational_recommendation`
  - Canonical top recommendation payload for actioning current risk.

- `explanation_text`
  - Canonical operator-facing explanation summary string.

- `events`
  - Canonical derived event flags for downstream monitoring/alert routing.

- `aliases` *(deprecated compatibility only; optional)*
  - `response_recommendations` is deprecated.
  - Legacy decision/recommendation aliases are retained only for compatibility.
  - Canonical consumers should use `operational_recommendation`.

### Example: `causal_analysis`

```json
{
  "causal_analysis": {
    "hypotheses": [
      {
        "hypothesis_id": "hyp_physical_localized_degradation",
        "type": "physical",
        "confidence": 0.71
      }
    ],
    "top_hypothesis": {},
    "counterfactual": {
      "counterfactual_checks": [],
      "robustness": 0.67,
      "interpretation": "Top hypothesis has mixed robustness; targeted validation is required."
    },
    "validation_plan": [],
    "recommended_sequence": [],
    "best_next_action": {},
    "status": { "available": true, "reason": "ok" }
  }
}
```

> Compatibility note: `neraium_core/casual.py` is deprecated and will be removed in a future release. Use `neraium_core/causal.py`.

---
