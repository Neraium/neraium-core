# Neraium Markets — Day 2

`neraium_markets` is a **read-only structural intelligence engine skeleton** for financial market data.

## What Day 2 does

- Loads market CSVs from `sample_data/`
- Normalizes and parses timestamps
- Validates each asset dataset
- Aligns all asset close series onto a common timestamp index (outer join)
- Builds a first deterministic feature engine for Day 2
- Prints aligned and feature-table shapes
- Prints sample feature columns
- Saves engineered features to `output/features.csv`
- Includes tests for loading, validation, alignment, and Day 2 feature outputs

## Day 2 feature set

- Returns:
  - `*_ret_1d`, `*_ret_5d` for core equity, sector, and cross-asset symbols
- Volatility:
  - `spy_vol_10d`, `spy_vol_20d`
  - `qqq_vol_10d`, `qqq_vol_20d`
  - `iwm_vol_10d`, `iwm_vol_20d`
- Breadth:
  - `breadth_pct_above_20dma`
- Sector participation:
  - `sector_dispersion_1d`
  - `sector_concentration_top2`
- Cross-asset context:
  - `vix_ret_1d`, `dxy_ret_1d`, `gold_ret_1d`, `oil_ret_1d`, `us2y_ret_1d`, `us10y_ret_1d`
  - `rates_2s10s = us10y - us2y`
  - `risk_off_proxy` (deterministic composite)

## What this project still does not do

- No trading execution
- No broker API integration
- No machine learning
- No regime engine (yet)
- No dashboard or UI
- No API surface

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
    features.py
  tests/
    test_data_loader.py
    test_alignment.py
    test_validation.py
    test_features.py
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
