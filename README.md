# neraium-core

Neraium core structural monitoring engine with a lightweight FastAPI service for telemetry ingest.

## What this release adds

- Single-sensor ingest is supported without crashing the SII pipeline.
  - Relational/correlation-driven analytics are skipped until at least **2 valid signals** exist.
- Local SQLite persistence for ingested event metadata and result history.
- Basic API key auth for write endpoints.
- Heuristic operator-facing interpretation fields layered on top of raw structural metrics.
- Malformed client input now returns **HTTP 400** (not HTTP 500) on ingest endpoints.

## Run locally

```bash
python -m pip install -e .[dev]
uvicorn apps.api.main:app --host 0.0.0.0 --port 8000
```

## Configuration

Environment variables:

- `NERAIUM_API_KEY` (optional)
  - If set, write endpoints require `X-API-Key` to match.
  - If unset, auth is open for local development.
- `NERAIUM_DB_PATH` (optional, default: `neraium.db`)
  - SQLite file used for minimal persistence.

## API overview

### Health

```bash
curl http://localhost:8000/health
```

Returns service status, API version, whether a latest result exists, auth configuration status, and basic persistence availability.

### Ingest single frame (valid)

```bash
curl -X POST http://localhost:8000/ingest \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: changeme' \
  -d '{
    "timestamp": "2026-01-01T00:00:00+00:00",
    "site_id": "site-a",
    "asset_id": "asset-1",
    "sensor_values": {"pressure": 60.0, "flow": 125.0}
  }'
```

### Ingest single frame (invalid -> 400)

```bash
curl -X POST http://localhost:8000/ingest \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: changeme' \
  -d '{
    "timestamp": "not-a-timestamp",
    "site_id": "site-a",
    "asset_id": "asset-1",
    "sensor_values": {"pressure": 60.0}
  }'
```

### Ingest batch

```bash
curl -X POST http://localhost:8000/ingest/batch \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: changeme' \
  -d '{"items": [{"site_id": "site-a", "asset_id": "asset-1", "sensor_values": {"pressure": 60.0}}]}'
```

### Ingest CSV

```bash
curl -X POST http://localhost:8000/ingest/csv \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: changeme' \
  -d '{"csv_text": "timestamp,site_id,asset_id,pressure\n2026-01-01T00:00:00+00:00,site-a,asset-1,60.0"}'
```

### Latest result retrieval

```bash
curl http://localhost:8000/results/latest
```

### Recent results retrieval

```bash
curl 'http://localhost:8000/results/recent?limit=20'
```

### Reset

```bash
curl -X POST http://localhost:8000/reset -H 'X-API-Key: changeme'
```

## Persistence scope

Current persistence is intentionally minimal:

- Event metadata (`timestamp`, `site_id`, `asset_id`) and normalized payload JSON
- Latest structural result
- Result history (with persistence timestamp)

## Interpretation layer (heuristic, additive)

Responses preserve raw analytics and include additive operator-facing fields:

- `risk_level`: `LOW|MEDIUM|HIGH`
- `action_state`: `STABLE|WATCH|ALERT`
- `operator_message`
- `structural_analysis_available`: `true|false`
- `skipped_reason`: explanatory reason when structural relational metrics are unavailable

Legacy nested `interpretation` remains available and marked with `heuristic: true`.

Interpretation fields are **heuristics** for operator context and do **not** replace structural/analytical metrics.
When fewer than 2 valid signals are present, structural relational analysis is flagged unavailable.
