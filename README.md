# neraium-core

`neraium-core` is a deployable, **read-only structural instability instrumentation system**.
It ingests multivariate telemetry, computes Systemic Infrastructure Intelligence (SII) metrics, and returns operator-facing evidence outputs.

Neraium is intentionally constrained for current deployments:
- **Read-only analytics** over telemetry and CSV/streaming inputs
- **Human-in-the-loop decision support only**
- **No infrastructure control path**
- **No automated actuation**

## Current Product Scope

Neraium today is a system stability instrumentation layer for detecting structural degradation and instability.
It does not write back into operational systems and does not execute control actions.

### Input
- Multivariate telemetry from API ingest
- Batch CSV uploads
- Time-ordered streaming-like updates via repeated ingest calls

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
- Proof artifacts / reports where available (for example FD004 summaries, CSV timelines, and plots)

## Current-State Architecture

Operational systems / telemetry sources  
→ One-way data access (ingest only)  
→ Systemic Infrastructure Intelligence (read-only computation)  
→ Human operators and evidence outputs (API results, CSV, reports)

## What the system does

Neraium implements **SII** as the statistical estimation of evolving relational geometry in complex systems.

- It is **not** a generic anomaly detector.
- It is **not** a classical predictive-maintenance classifier.
- It compares baseline and recent sensor-relationship structure to estimate stability and drift.

Operator-facing output is additive and heuristic, layered on top of math outputs:

- `risk_level`: LOW / MEDIUM / HIGH
- `trend`: STABLE / RISING / FALLING / UNKNOWN
- `confidence`: normalized [0.0, 1.0] confidence proxy
- `operator_message`: plain-language guidance
- `structural_analysis_available`: whether relational analysis ran
- `skipped_reason`: why relational analysis was skipped

## Structural Degradation Detection (FD004 Validation)

### Problem
Traditional monitoring often raises alarms only when a system is already close to failure.
In FD004, the system shows structural degradation before final failure, so waiting for hard thresholds is too late.

### Approach
SII watches how sensors move **together** over time, not just whether one sensor value crosses a limit.
This helps reveal structural change earlier in the degradation path.

### Results (from FD004)
- 100.0% of units reached **MEDIUM** before **HIGH**.
- No direct **LOW → HIGH** jumps were observed.
- Instability increases over time for the hero unit.
- An early warning window exists before the critical **HIGH** state (average: 1 cycle).

### Example Output
- `reports/fd004_proof_summary.md`
- `fd004_outputs_subset/hero_unit_timeseries.csv`

### Interpretation
- Detects structural degradation early, before severe instability.
- Reduces noisy alerts by avoiding abrupt risk jumps.
- Matches expected failure progression (`LOW → MEDIUM → HIGH`).
- Gives operators a short but usable warning window before critical state.

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

## Planned Future Extensions

Interpretive governance and formal assurance layers are planned for a later funded phase.
They are **not** part of the required definition of the currently working Neraium core platform.

Planned future work includes:
- Interpretive governance guardrails around operator interpretation and escalation pathways
- Formal schematic and assurance guardrails for higher-assurance deployments
- External assurance collaboration discussions with Dr. Chason Coelho and NCC Group / Adelard

These roadmap items are not current operational dependencies for running `neraium-core` today.

## How to run

```bash
python -m pip install -e .[dev]
uvicorn apps.api.main:app --host 0.0.0.0 --port 8000
```

## How to test

```bash
ruff check .
pytest
```

Environment variables:

- `NERAIUM_API_KEY` (optional)
- `NERAIUM_DB_PATH` (optional, default: `neraium.db`)

## API consistency

Result-bearing endpoints return a stable envelope:

```json
{
  "latest": {"...": "latest result or null"},
  "count": 1,
  "results": [{"...": "result list"}]
}
```

Endpoints:

- `POST /ingest`
- `POST /ingest/batch`
- `POST /ingest/csv`
- `GET /results/latest`
- `GET /results/recent?limit=100`

`/results/recent` is ordered newest-first. `limit` controls max returned rows.

Health endpoint:

```json
{
  "status": "ok|degraded",
  "version": "0.1.0",
  "auth_configured": false,
  "persistence_available": true,
  "latest_result_available": false
}
```

Validation semantics:

- `422` = schema validation failure (FastAPI/Pydantic)
- `400` = semantic/business validation failure (service/pipeline)

## Pilot/demo usage

1. Start API and send single-sensor or multi-sensor telemetry with `/ingest`.
2. Monitor `risk_level`, `trend`, `confidence`, and `operator_message` for operator briefings.
3. Use `structural_analysis_available` and `skipped_reason` to explain when full relational metrics are unavailable.
4. Use `/results/recent` for timeline review and `/results/latest` for current state dashboards.
5. Use `/reset` to restart pilot sessions while preserving deployment configuration.
