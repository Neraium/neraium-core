# Live Stock Market Runner (Polling)

> **Safety mode:** this runner performs analytics/signals only and does **not** place trades.

## Supported provider

- `alphavantage` (REST polling)
- `mock` (offline smoke testing)

## Required environment variables

For Alpha Vantage:

- `ALPHAVANTAGE_API_KEY` (required)

Optional defaults:

- `LIVE_DATA_PROVIDER` (default: `alphavantage`)
- `LIVE_POLL_INTERVAL` (default: `15` seconds)
- `ALPHAVANTAGE_INTERVAL` (default: `1min`)

## Command examples

### Windows PowerShell (Alpha Vantage)

```powershell
$env:ALPHAVANTAGE_API_KEY="YOUR_KEY"
python run_live_stock_market.py --tickers AAPL,MSFT --provider alphavantage --poll-interval 20 --provider-interval 1min --output-log logs/live_signals.csv
```

### Windows PowerShell (offline/mock smoke run)

```powershell
python run_live_stock_market.py --tickers AAPL,MSFT --provider mock --poll-interval 1 --max-iterations 3 --output-log logs/live_signals_mock.csv
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
