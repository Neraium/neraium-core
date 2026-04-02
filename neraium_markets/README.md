# Neraium Markets

Read-only market intelligence pipeline: load OHLCV CSVs, validate, align closes, engineer features, build a structural snapshot, then produce **regime-aware signals** with confidence, an interpretive gate, and action posture (Day 4).

## Purpose

- Ingest one CSV per asset from `sample_data/`
- Validate schema, nulls, duplicates, sort order, and numeric closes
- Outer-join all assets on `timestamp`
- Engineer returns, volatility, breadth, sector dispersion, and cross-asset context
- Compute structural scores (correlation drift, lag drift, sector entropy, instability, coherence)
- Classify **regime**, score **confidence**, apply an **interpretive gate** (abstain / avoid / wait), and emit **action posture** plus a text **explanation**

## What is not included (by design)

- Order execution, broker APIs, or trading logic
- Machine learning (rules-based regime and gate only)
- Dashboards or HTTP APIs

## Regimes (`regime_label`)

| Label | Meaning (intuition) |
|-------|---------------------|
| `stable_trend` | Low instability and contained correlation drift |
| `fragile_rally` | Instability rising while breadth weakens |
| `risk_off_transition` | High instability with strong risk-off cross-asset tilt |
| `high_volatility` | Elevated realized vol and elevated correlation drift |
| `unstable` | Low coherence with high drift (mixed, unreliable structure) |
| `false_calm` | Calm surface vol/drift but drift metrics inflecting up |
| `mean_reversion` | Default when no stronger pattern matches |

Rules use **percentile ranks** (`rank(method="first", pct=True)`) so thresholds stay defined even with ties.

## Confidence (`confidence_score`)

Scalar in **[0, 1]**:

`0.4 * coherence_score + 0.3 * (1 - instability_score) + 0.3 * signal_agreement`

where `signal_agreement` shrinks when coherence, stability, and breadth inputs disagree (higher row-wise dispersion → lower agreement). Missing values are filled with 0 before clamping.

## Interpretive gate (`gate_action`)

Internal column (used to shape posture): `proceed` | `no_action` | `avoid_market` | `wait`

- **no_action** — confidence &lt; 0.5
- **avoid_market** — top-decile instability with bottom-third coherence
- **wait** — risk-off proxy and SPY 1d return push in opposite directions (both non-trivial)
- **proceed** — otherwise

## Action posture (`action_posture`)

| Value | Typical use |
|-------|----------------|
| `watch` | No strong directional lean |
| `lean_long` | Stable trend with high confidence (and gate allows) |
| `lean_short` | Reserved for future bearish rules (not emitted by defaults) |
| `reduce_exposure` | Fragile rally or high volatility |
| `avoid_risk` | Risk-off transition, or gate `avoid_market` |
| `wait` | Unstable regime, gate `wait` / `no_action`, or conflict |

## CSV schema

Each file is named `{asset}.csv` (lowercase). Required columns:

| Column | Description |
|--------|----------------|
| timestamp | Parsed as datetime; unique per file; ascending |
| open, high, low, close, volume | Numeric |

## Install

From `neraium_markets/`:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On Linux or macOS: `source .venv/bin/activate`.

## Run (signal generation)

```bash
cd neraium_markets
python main.py
```

Loads data, validates, aligns, builds features and structural snapshot, runs regime → confidence → gate → posture, prints shapes, regime counts, last 10 signal rows, and writes **`output/signals.csv`**.

## Regenerate sample data

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

- `config.py` — assets, paths, columns, `REQUIRED_COLUMNS`, groups
- `main.py` — full pipeline through Day 4
- `neraium/data_loader.py` — CSV loading
- `neraium/validation.py` — checks + sample Pydantic row validation
- `neraium/alignment.py` — outer join on timestamp (uppercase symbols)
- `neraium/features.py` — Day 2 feature table
- `neraium/structural.py` — Day 3 structural snapshot scores
- `neraium/regime.py` — regime, confidence, gate, posture
- `neraium/signals.py` — `generate_signals`, optional CSV save
- `neraium/schemas.py` — `OHLCVRow`
- `sample_data/` — one CSV per asset
- `output/` — generated `signals.csv` (created on run)
- `tests/` — pytest suite
