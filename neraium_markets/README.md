# Neraium Markets (Day 1)

Read-only market intelligence skeleton: load OHLCV CSVs, validate them, and align close prices on a common timestamp index for downstream use.

## Purpose

- Ingest one CSV per asset from `sample_data/`
- Validate schema, nulls, duplicates, sort order, and numeric closes
- Outer-join all assets on `timestamp` into a single merged frame (timestamp + one column per ticker symbol)

## What is not included (by design)

- Order execution, broker APIs, or trading logic
- Machine learning or derived alpha features
- Dashboards or HTTP APIs
- Feature engineering beyond loading, validation, and alignment

## CSV schema

Each file is named `{asset}.csv` (lowercase, e.g. `spy.csv`). Required columns:

| Column     | Description        |
|-----------|--------------------|
| timestamp | Date (parsed as datetime); unique per file; ascending |
| open      | Numeric            |
| high      | Numeric            |
| low       | Numeric            |
| close     | Numeric (used for alignment) |
| volume    | Numeric            |

## Install

From this directory (`neraium_markets/`):

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On Linux or macOS, activate with `source .venv/bin/activate`.

## Run

```bash
cd neraium_markets
python main.py
```

Loads all configured assets, validates, merges closes, prints shape and the first 10 rows. Exits with code 1 if validation fails.

## Regenerate sample data

Synthetic daily OHLCV (35 rows per asset) can be regenerated with:

```bash
cd neraium_markets
python tools/generate_sample_data.py
```

## Tests

```bash
cd neraium_markets
python -m pytest tests -q
```

## Layout

- `config.py` – asset list and column names (`PipelineConfig` via Pydantic)
- `main.py` – CLI entry: load, validate, align, print
- `neraium/data_loader.py` – CSV loading
- `neraium/validation.py` – checks + sample Pydantic row validation
- `neraium/alignment.py` – outer join on timestamp
- `neraium/schemas.py` – `OHLCVRow` model
- `sample_data/` – one CSV per asset
- `tests/` – pytest suite
