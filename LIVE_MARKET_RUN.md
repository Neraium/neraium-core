# Live Stock Market Runner (Polling)

> **Safety mode:** this runner performs analytics/signals only and does **not** place trades.

## Supported provider

- `massive` (REST polling, current brand alias for Polygon)
- `polygon` (REST polling, retained for backwards compatibility)
- `alphavantage` (REST polling)
- `mock` (offline smoke testing)

## Required environment variables

For Massive / Polygon (same backend, Massive is current brand):

- `MASSIVE_API_KEY` (preferred)
- `POLYGON_API_KEY` (backward-compatible alternative)

For Alpha Vantage:

- `ALPHAVANTAGE_API_KEY` (required)

Optional defaults:

- `LIVE_DATA_PROVIDER` (default: `polygon`; `massive` is also supported)
- `LIVE_POLL_INTERVAL` (default: `15` seconds)
- `ALPHAVANTAGE_INTERVAL` (default: `1min`)

## Setup (exact steps)

1. Open a terminal in the repo root.
2. Set the Massive API key (or Polygon for compatibility).
3. Run the live polling script with one or more comma-separated tickers.

## Command examples

### Git Bash (Massive provider, recommended)

```bash
export MASSIVE_API_KEY="YOUR_MASSIVE_KEY"
python run_live_stock_market.py --tickers AAPL,MSFT --provider massive --interval 15 --output logs/live_signals_massive.csv
```

### Windows PowerShell (Massive provider, recommended)

```powershell
$env:MASSIVE_API_KEY="YOUR_MASSIVE_KEY"
python run_live_stock_market.py --tickers AAPL,MSFT --provider massive --interval 15 --output logs/live_signals_massive.csv
```

### Windows PowerShell (Polygon name, backwards-compatible)

```powershell
$env:POLYGON_API_KEY="YOUR_POLYGON_KEY"
python run_live_stock_market.py --tickers AAPL,MSFT --provider polygon --interval 15 --output logs/live_signals_polygon.csv
```

### Windows PowerShell (Alpha Vantage)

```powershell
$env:ALPHAVANTAGE_API_KEY="YOUR_KEY"
python run_live_stock_market.py --tickers AAPL,MSFT --provider alphavantage --interval 20 --provider-interval 1min --output logs/live_signals.csv
```

### Windows PowerShell (offline/mock smoke run)

```powershell
python run_live_stock_market.py --tickers AAPL,MSFT --mock --interval 1 --max-iterations 3 --output logs/live_signals_mock.csv
```

### Windows PowerShell (auto-fallback to mock if network/API fails)

```powershell
$env:MASSIVE_API_KEY="YOUR_MASSIVE_KEY"
python run_live_stock_market.py --tickers AAPL,MSFT --provider massive --fallback-to-mock --interval 5 --max-iterations 3
```

## Expected output format

Console line per ticker/bar:

```text
2026-04-02T12:34:56.123456+00:00 | AAPL | state=NORMAL | signal=BUY | drift=0.42 | instability=0.42 | health=91.6
```

Optional CSV (`--output`) columns:

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

## Branding note

Massive is the current brand name for this market data integration. Existing Polygon naming (`--provider polygon`, `POLYGON_API_KEY`) remains supported for compatibility.
