# neraium-core

`neraium-core` is the analytical kernel for Neraium Structural Intelligence (SII): ingest telemetry, normalize it, and produce structural drift indicators for operational monitoring.

## What the structural engine currently implements

The `StructuralEngine` in `neraium_core.engine` currently computes:

- baseline formation from early clean frames
- Mahalanobis drift against baseline manifold
- covariance/relational drift between baseline and recent windows
- fused drift score using weighted Mahalanobis + covariance drift
- relational stability score and drift velocity
- heuristic lead-time-to-instability estimate
- operator-facing state (`STABLE`, `WATCH`, `ALERT`) and health score

## What remains heuristic

The following outputs are explicitly heuristic and should not be interpreted as physically rigorous predictions:

- lead-time estimate (`lead_time_hours`)
- confidence score derived from bounded stability/quality heuristics
- textual fields such as `structural_driver`, `predicted_impact`, and `explanation`
- threshold-based state bucketing and `system_health`

## Repository structure

- `neraium_core/engine.py` – structural analytics engine
- `neraium_core/ingest.py` – payload + CSV normalization/validation
- `neraium_core/lead_time.py` – lead-time detector model objects
- `neraium_core/service.py` – service boundary around ingestion + engine lifecycle
- `apps/demo/server.py` – demo HTTP server using service layer
- `scripts/` – replay/simulation/plot utility scripts
- `tests/` – pytest coverage for engine/ingest/service behavior
- `run_engine.py`, `ingest.py`, `lead_time_engine.py`, `server.py` – compatibility wrappers

## Install

```bash
python -m pip install --upgrade pip
pip install -e '.[dev]'
```

## Run the demo

```bash
python server.py
```

Or directly:

```bash
python -m apps.demo.server
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

You can also provide `FD004_INPUT` as an environment variable.

## Current limitations

- baseline quality depends on clean early telemetry windows
- NaN-containing frames are retained for continuity but excluded from strict baseline/covariance calculations
- lead-time and impact messaging are intentionally conservative heuristics
- no model persistence/checkpointing in this pass

## SII roadmap (near-term)

- calibrated uncertainty intervals from replay/backtest datasets
- richer quality metadata weighting in fused scoring
- multi-asset historical baselining and warm-start support
- clearer separation between rigorous observables and product-level risk language
