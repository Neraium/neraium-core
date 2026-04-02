# Neraium Markets — Day 1

`neraium_markets` is a **read-only structural intelligence engine skeleton** for financial market data.

## What Day 1 does

- Loads market CSVs from `sample_data/`
- Normalizes and parses timestamps
- Validates each asset dataset
- Aligns all asset close series onto a common timestamp index (outer join)
- Prints merged dataset shape and first 10 rows
- Includes tests for loading, validation, and alignment

## What Day 1 does not do

- No trading execution
- No broker API integration
- No machine learning
- No dashboard or UI

## Project structure

```
neraium_markets/
  README.md
  requirements.txt
  main.py
  config.py
  sample_data/
    spy.csv ... us10y.csv
  neraium/
    __init__.py
    data_loader.py
    alignment.py
    validation.py
  tests/
    test_data_loader.py
    test_alignment.py
    test_validation.py
```

## Expected CSV schema

Each CSV must include:

- `timestamp`
- `open`
- `high`
- `low`
- `close`
- `volume`

Example header:

```csv
timestamp,open,high,low,close,volume
```

## Install

From project root (`/workspace/neraium-core`):

```bash
python -m pip install -r neraium_markets/requirements.txt
```

## Run

```bash
cd neraium_markets
python main.py
```

## Run tests

```bash
cd neraium_markets
pytest -q
```

## Notes

If real market CSVs are unavailable, synthetic CSVs are provided in `sample_data/` with daily timestamps and 40 rows per asset.
