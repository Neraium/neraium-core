# Neraium Markets

## What it does
Neraium Markets ingests synchronized market telemetry, builds multidimensional state vectors, measures structural drift, classifies regimes, and outputs confidence-weighted operator recommendations.

## What it does not do
- No order routing
- No broker/API execution
- No autonomous trading
- No black-box ML predictions in MVP v1

## Setup
```bash
pip install -e .
```

## Run ingestion + signal generation
```bash
python examples/markets/run_markets_pipeline.py
```

## Run API
```bash
uvicorn neraium_core.markets.app.api:app --reload
```

## Inspect evidence logs
Evidence file:
`artifacts/neraium_markets/evidence.jsonl`

Latest signals from API:
```bash
curl http://127.0.0.1:8000/signals/latest
```

## Run tests
```bash
pytest tests/markets -q
```
