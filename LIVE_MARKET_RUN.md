# Live Stock Market Runner (Polling)

> **Safety mode:** this runner performs analytics/signals only and does **not** place trades.

## Supported provider

- `polygon` (REST polling, default)
- `alphavantage` (REST polling)
- `mock` (offline smoke testing)

## Required environment variables

For Polygon (recommended/default):

- `POLYGON_API_KEY` (required for `--provider polygon`)

For Alpha Vantage:

- `ALPHAVANTAGE_API_KEY` (required)

Optional defaults:

- `LIVE_DATA_PROVIDER` (default: `polygon`)
- `LIVE_POLL_INTERVAL` (default: `15` seconds)
- `ALPHAVANTAGE_INTERVAL` (default: `1min`)

## Setup (exact steps)

1. Open a terminal in the repo root.
2. Set the Polygon API key.
3. Run the live polling script with one or more comma-separated tickers.

## Command examples

### Windows PowerShell (Polygon, default live provider)

```powershell
$env:POLYGON_API_KEY="YOUR_POLYGON_KEY"
python run_live_stock_market.py --tickers AAPL,MSFT --provider polygon --poll-interval 15 --output-log logs/live_signals_polygon.csv
```

### Windows PowerShell (Alpha Vantage)

```powershell
$env:ALPHAVANTAGE_API_KEY="YOUR_KEY"
python run_live_stock_market.py --tickers AAPL,MSFT --provider alphavantage --poll-interval 20 --provider-interval 1min --output-log logs/live_signals.csv
```

### Windows PowerShell (offline/mock smoke run)

```powershell
python run_live_stock_market.py --tickers AAPL,MSFT --provider mock --poll-interval 1 --max-iterations 3 --output-log logs/live_signals_mock.csv
```

### Windows PowerShell (auto-fallback to mock if network/API fails)

```powershell
$env:POLYGON_API_KEY="YOUR_POLYGON_KEY"
python run_live_stock_market.py --tickers AAPL,MSFT --provider polygon --fallback-to-mock --poll-interval 5 --max-iterations 3
```

## Expected output format

Console line per ticker/bar:

```text
2026-04-02T12:34:56.123456+00:00 | AAPL | state=NORMAL | signal=BUY | drift=0.42 | instability=0.42 | health=91.6
```

Optional CSV (`--output-log`) columns:

- `timestamp`
- `ticker`
- `state`
- `trading_signal`
- `structural_drift_score`
- `latest_instability`
- `system_health`
- `evidence_confidence`

## Execution scope reminder

This flow is **analytics/signals only** and explicitly **not** brokerage trade execution.
