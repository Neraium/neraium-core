# neraium-core

Universal structural analytics engine with a **non-executing intraday market signal workflow**.

## Day-trader usage (analytics only, no auto-trading)

### What this does
- Polls live or mock intraday bars.
- Normalizes bars into engine ingress frames.
- Runs structural analytics per ticker.
- Emits compact trader-facing signals with noise controls.
- Supports replay mode for paper-trading style validation.

### What this does NOT do
- No brokerage integration.
- No order placement.
- No automatic trading execution.
- No guarantee of returns.

## Quick setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pytest -q tests/test_run_live_stock_market.py tests/test_trading_signals_intraday.py
```

## Live run

Massive/Polygon:

```bash
export MASSIVE_API_KEY="YOUR_KEY"
python run_live_stock_market.py \
  --tickers AAPL,MSFT \
  --provider massive \
  --interval 15 \
  --min-confidence 0.50 \
  --cooldown-seconds 60 \
  --warmup-bars 8 \
  --output logs/live_signals.csv
```

Offline smoke test:

```bash
python run_live_stock_market.py --tickers AAPL,MSFT --mock --interval 1 --max-iterations 5 --output logs/live_signals_mock.csv
```

## Replay (paper-trading style validation)

Input CSV needs at least: `timestamp,ticker,open,high,low,close,volume`.

```bash
python run_live_stock_market.py \
  --tickers AAPL,MSFT \
  --replay-csv data/sample_market_data.csv \
  --min-confidence 0.50 \
  --cooldown-seconds 60 \
  --output logs/replay_signals.csv
```

## Trader-facing output contract

Each emitted row contains:
- `timestamp`
- `ticker`
- `state`
- `trading_signal`
- `confidence`
- `structural_drift_score`
- `latest_instability`
- `reason`
- `transition` (optional transition flag)
- `cooldown_remaining_seconds` (repeat-signal metadata)
- `emitted` (whether the row passed emission gates)

See `LIVE_MARKET_RUN.md` for full operational notes.
