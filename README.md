# Neraium Core

`neraium-core` provides a structural intelligence engine for telemetry drift detection, plus a small service layer for payload/CSV ingestion and demo-server integration.

## What the structural engine currently implements

`neraium_core.engine.StructuralEngine` currently provides:

- **Baseline formation** from early valid telemetry windows.
- **Mahalanobis drift** from baseline manifold.
- **Covariance (relational) drift** between baseline and recent windows.
- **Fused drift score** with configurable weighting.
- **Drift velocity** from score deltas.
- **Heuristic lead-time estimate** when drift is accelerating.

## Rigorous observables vs heuristics

### Rigorous observables

- Mean/covariance baseline estimation from complete vectors.
- Mahalanobis distance and covariance Frobenius delta.
- Derived fused drift and drift velocity.

### Heuristic/operator-facing outputs

- Alert states (`STABLE`, `WATCH`, `ALERT`).
- System health mapping (`0..100`).
- Lead-time estimate and confidence.
- Human-readable driver/impact/explanation fields.

## Repository structure

- `neraium_core/engine.py`: main structural engine implementation.
- `neraium_core/ingest.py`: payload + CSV normalization/parsing.
- `neraium_core/service.py`: service boundary for ingest + processing.
- `neraium_core/lead_time.py`: standalone lead-time detector model.
- `apps/demo/server.py`: demo server using service layer.
- `scripts/`: replay/plot/simulator scripts.
- `tests/`: package-level tests for engine/ingest/service behavior.

Legacy top-level modules (`run_engine.py`, `ingest.py`, `lead_time_engine.py`, `server.py`) remain as compatibility wrappers.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Run the demo

```bash
python server.py
```

or directly:

```bash
python apps/demo/server.py
```

## Run tests and lint

```bash
ruff check .
pytest
```

## FD004 replay script

```bash
python scripts/run_fd004_test.py --input /path/to/train_FD004.txt --output fd004_results.csv
```

You can also set `FD004_PATH` instead of passing `--input`.

## Current limitations

- Baseline assumptions are simple and stationarity-sensitive.
- Missing value handling currently drops incomplete vectors from covariance estimation.
- Lead-time remains heuristic and should not be treated as a formal reliability bound.
- Demo server is single-process and intentionally lightweight.

## Future SII roadmap

- Stronger regime-aware baseline adaptation.
- Probabilistic lead-time intervals calibrated on historical outcomes.
- Richer sensor quality propagation into confidence scoring.
- Cleaner API layer and persisted state management.
- Expanded validation datasets and scenario testing.
