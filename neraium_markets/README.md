# Neraium Markets (Day 3)

Read-only market structure intelligence MVP.

## What Day 3 adds

Day 3 turns the Day 2 feature table into a **structural snapshot** that scores:

- **Correlation drift** (`corr_drift_score`): baseline-vs-recent correlation geometry change.
- **Lead-lag drift** (`lag_drift_score`): baseline-vs-recent best lag relationship change.
- **Sector entropy** (`sector_entropy`): normalized entropy of sector absolute-return shares.
- **Sector concentration** (`sector_concentration_score`): Herfindahl-like concentration of sector absolute-return shares.
- **Instability** (`instability_score`): deterministic weighted blend of drift, concentration, dispersion, inverse breadth, and risk-off proxy.
- **Coherence** (`coherence_score`): directional agreement across stress components.

All formulas are explicit and deterministic; no machine learning is used.

## What this MVP still does not include

- Trading execution
- Broker APIs
- Regime labels/classification (planned for Day 4)
- Dashboards or web APIs
- Machine learning models

## Pipeline flow

1. Load OHLCV CSVs from `sample_data/`
2. Validate data quality and schema
3. Align close prices on timestamp
4. Build Day 2 feature table
5. Build Day 3 structural snapshot
6. Print shape summaries + structural preview
7. Optionally write CSV outputs

## Run

From `neraium_markets/`:

```bash
python main.py
```

Optionally save outputs:

```bash
python main.py --save-output
```

## Outputs

With `--save-output`, files are written to:

- `output/features.csv`
- `output/structural_snapshot.csv`

## Tests

```bash
python -m pytest tests -q
```

Day 3 structural tests are in `tests/test_structure.py`.

## Layout

- `config.py` – asset list + Day 3 structural parameters
- `main.py` – Day 3 pipeline entrypoint
- `neraium/features.py` – Day 2 feature engineering
- `neraium/structure.py` – Day 3 structural drift/concentration/instability/coherence scoring
- `tests/test_structure.py` – Day 3 structural tests
