# neraium-core

Neraium core structural monitoring engine with a lightweight FastAPI service for telemetry ingest.

## SII framing (what this system is and is not)

Neraium implements **SII** as the statistical estimation of evolving relational geometry in complex systems.

- It is **not** a generic anomaly detector.
- It is **not** a classical predictive-maintenance classifier.
- It estimates structural stability by comparing rolling-window correlation geometry against a baseline structure.

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
- Operator-facing fields (`risk_level`, `action_state`, `operator_message`) are heuristic interpretation overlays.

## Run locally

```bash
python -m pip install -e .[dev]
uvicorn apps.api.main:app --host 0.0.0.0 --port 8000
```

## Configuration

Environment variables:

- `NERAIUM_API_KEY` (optional)
- `NERAIUM_DB_PATH` (optional, default: `neraium.db`)

## API and stability notes

Architecture remains stable:

- `neraium_core` = canonical package
- `alignment.py` engine = analytics
- `service.py` = orchestration
- `store.py` = persistence
- `apps/api/main.py` = FastAPI API surface

API semantics:

- `422` = schema validation failure (FastAPI/Pydantic)
- `400` = semantic/business validation failure (service/pipeline)

Interpretation fields remain additive and separate from structural metrics.
When fewer than 2 valid signals exist, relational metrics are skipped and surfaced as unavailable.
