# Neraium Markets

## Scope
Neraium Markets is **market data ingestion + structural analytics + governance-driven decision support** for stock-market operators.

It is explicitly **not** trade execution, order routing, portfolio management, or OMS.

## What the module now supports
- Historical Massive fetch and local cache datasets.
- Replay with emission governance discipline (cooldown, duplicate suppression, abstention handling).
- Live streaming with the same governed output discipline used by replay.
- Timeframe-correct live sessions (`1m`, `5m`, `15m`) with proper bar indexing.
- Warmup/readiness telemetry: bars collected/required, percent ready, readiness state.
- Operator-grade console at `/` with command center, live monitoring, replay controls, signal review, and integrations health.
- SQLite persistence for live emissions, suppressed outputs, replay outputs, session metadata, errors, and provider health checks.

## Live vs replay behavior
- **Shared discipline:** both paths use the same `SignalEmissionController` semantics (warmup, cooldown, duplicate filtering, hysteresis stabilization).
- **Replay:** deterministic pass over fixed historical windows.
- **Live:** event-driven updates with the same governance rules applied per frame.

## Warmup/readiness semantics
`/live/status` exposes:
- `bars_collected`
- `bars_required`
- `warmup_progress`
- `readiness_state`

Readiness states:
- `disconnected`
- `connecting`
- `reconnecting`
- `connected_no_data`
- `receiving_data_warming_up`
- `live_no_valid_signals`
- `live_ready`

## Provider health semantics (`/integrations/massive/status`)
The endpoint now distinguishes:
- config presence
- api key presence
- api key validity
- REST reachability
- websocket configured/dependency present
- recent fetch success
- recent live event and signal receipt
- latest operator-visible error

## History/review workflow
`/signals/history` supports filters for:
- `ticker`
- `session_type` (`live`/`replay`)
- `action_permission`
- `best_action`
- `start_at`, `end_at`
- `include_suppressed`

Returns rows + summary counters (`suppression_count`, `abstention_count`).

## Required environment variables
- `MASSIVE_API_KEY`
- `MASSIVE_REST_BASE_URL` (default: `https://api.massive.com`)
- `MASSIVE_WS_BASE_URL` (default: `wss://socket.massive.com/stocks`)
- `NERAIUM_MARKETS_CACHE_PATH` (default: `artifacts/neraium_markets/cache`)
- `NERAIUM_LIVE_DB_PATH` (default: `artifacts/neraium_markets/live.sqlite3`)
- `NERAIUM_LIVE_EVENT_RETENTION` (default: `5000`)
- `NERAIUM_LIVE_BAR_RETENTION` (default: `2000`)
- `NERAIUM_LIVE_WARMUP_BARS` (default: `30`)

## Run API
```bash
uvicorn neraium_core.markets.app.api:app --reload
```

## Local run flow
1. Start API.
2. Open `http://127.0.0.1:8000/`.
3. Use **Replay / Historical** to fetch and replay.
4. Use **Live Session** to start live stream (`1m`/`5m`/`15m`).
5. Use **Signals / History** for operator review filters.
6. Use **Integrations / Data** for provider diagnostics.

## Test commands
```bash
pytest tests/markets/test_live_and_api_massive.py tests/markets/test_massive_integration.py -q
```
