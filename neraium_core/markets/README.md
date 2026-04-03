# Neraium Markets

## Scope
Neraium Markets is **market data ingestion + structural analytics + governance-driven decision support**.

It is explicitly **not** trade execution.

## What this build now supports
- Massive historical U.S. equities fetch (REST aggregates).
- Local CSV caching for reproducible replay runs.
- Replay against cached data into existing structural + governance pipeline.
- Massive live websocket ingestion adapter with normalized event/bar schema.
- FastAPI endpoints for historical fetch, replay, and live monitoring.
- Minimal operator UI at `/`.
- Local SQLite persistence for fetch jobs, replay runs/results, live telemetry, and errors.

## Required environment variables
- `MASSIVE_API_KEY`
- `MASSIVE_REST_BASE_URL` (default: `https://api.massive.com`)
- `MASSIVE_WS_BASE_URL` (default: `wss://socket.massive.com/stocks`)
- `NERAIUM_MARKETS_CACHE_PATH` (default: `artifacts/neraium_markets/cache`)
- `NERAIUM_LIVE_DB_PATH` (default: `artifacts/neraium_markets/live.sqlite3`)
- `NERAIUM_LIVE_EVENT_RETENTION` (default: `5000`)
- `NERAIUM_LIVE_BAR_RETENTION` (default: `2000`)

## Setup
```bash
pip install -e .
```

## Fetch historical Massive data
```bash
curl -X POST http://127.0.0.1:8000/integrations/massive/historical/fetch \
  -H 'content-type: application/json' \
  -d '{
    "symbols": ["SPY", "QQQ", "AAPL", "NVDA", "MSFT"],
    "timeframe": "15m",
    "start_date": "2026-01-01",
    "end_date": "2026-03-31"
  }'
```

## Run replay from cached Massive data
```bash
python tools/run_massive_replay.py --symbols SPY,QQQ,AAPL,NVDA --timeframe 15m --start 2026-01-01 --end 2026-03-31
```

Or from API:
```bash
curl -X POST 'http://127.0.0.1:8000/run-replay?timeframe=15m&use_massive_cached_data=true'
```

## Start live streaming from Massive
```bash
curl -X POST http://127.0.0.1:8000/live/start \
  -H 'content-type: application/json' \
  -d '{"symbols": ["SPY", "QQQ", "AAPL", "NVDA"], "timeframe": "5m"}'
```

Check live status:
```bash
curl http://127.0.0.1:8000/live/status
```

### Warming up state
`connected_warming_up` means streaming is connected but not enough synchronized bar history exists yet for reliable signal emission.

## Run API
```bash
uvicorn neraium_core.markets.app.api:app --reload
```

## Run tests
```bash
pytest tests/markets/test_massive_integration.py tests/markets/test_live_and_api_massive.py -q
```
