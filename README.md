# neraium-core

Neraium core is a structural monitoring engine plus a lightweight FastAPI service for telemetry ingest, persistence, and operator-facing interpretation.

## What the system does

Neraium implements **SII** as statistical estimation of evolving relational geometry in complex systems.

- It is **not** a generic anomaly detector.
- It is **not** a predictive-maintenance classifier.
- It estimates structural stability by comparing rolling-window correlation geometry against a baseline structure.

Operator-oriented output is additive to core metrics and includes:

- `risk_level`
- `trend`
- `confidence`
- `operator_message`
- `structural_analysis_available`
- `skipped_reason`

These fields are heuristic overlays and do not change the core math/engine architecture.

## API behavior

### Stable response envelope

Result-bearing endpoints use the same envelope:

```json
{
  "latest": {"...": "latest result or null"},
  "count": 1,
  "results": [{"...": "result objects"}]
}
```

Endpoints with this envelope:

- `POST /ingest`
- `POST /ingest/batch`
- `POST /ingest/csv`
- `GET /results/latest`
- `GET /results/recent?limit=N`

`/results/recent` returns newest-first ordering and applies a bounded limit server-side.

### Health endpoint

`GET /health` returns:

- `status`
- `version`
- `auth_configured`
- `persistence_available`
- `latest_result_available`

## Run locally

```bash
python -m pip install -e .[dev]
uvicorn apps.api.main:app --host 0.0.0.0 --port 8000
```

## Test and lint

```bash
ruff check .
pytest
```

Dev dependencies include `pytest` and `httpx` so FastAPI `TestClient` runs in standard environments without extra manual installs.

## Configuration

Environment variables:

- `NERAIUM_API_KEY` (optional)
- `NERAIUM_DB_PATH` (optional, default: `neraium.db`)

## Pilot/demo workflow

1. Start the API.
2. Check `GET /health` for runtime readiness flags.
3. Ingest one or more payloads via `/ingest`, `/ingest/batch`, or `/ingest/csv`.
4. Read operator-facing status from `risk_level`, `trend`, `confidence`, and `operator_message`.
5. Use `structural_analysis_available` and `skipped_reason` to understand when relational metrics are intentionally unavailable.
6. Review history through `/results/recent` for newest-first snapshots.
7. Use `/reset` to clear state between pilot scenarios.
