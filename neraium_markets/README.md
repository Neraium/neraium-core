# Neraium Markets

Read-only market intelligence pipeline: load OHLCV CSVs, validate, align closes, engineer features, build a structural snapshot, then produce **regime-aware signals** with confidence, an interpretive gate, and action posture (Day 4).

## Pipeline overview

- Ingest one CSV per asset from `sample_data/`
- Validate schema, nulls, duplicates, sort order, and numeric closes
- Outer-join all assets on `timestamp`
- Engineer returns, volatility, breadth, sector dispersion, and cross-asset context
- Compute structural scores (correlation drift, lag drift, sector entropy, instability, coherence)
- Classify **regime**, score **confidence**, apply an **interpretive gate** (abstain / avoid / wait), and emit **action posture** plus a text **explanation**

## Structural snapshot (Day 3)

The feature table is augmented with scores including:

- **Correlation drift** (`corr_drift_score`): instability in rolling return-correlation geometry (SPY/QQQ–based in `neraium/structural.py`).
- **Lead-lag drift** (`lag_drift_score`): instability in lagged correlation structure.
- **Sector entropy** (`sector_entropy`): normalized dispersion of sector participation.
- **Sector concentration** (`sector_concentration_score`): concentration of sector moves (from feature `sector_concentration_top2` where applicable).
- **Instability** (`instability_score`): composite stress from vol, dispersion, drift, breadth churn.
- **Coherence** (`coherence_score`): cross-sectional agreement of core equity stress/read.

For an alternate, windowed baseline-vs-recent formulation (correlation Frobenius drift, lead–lag pairs, etc.), see `neraium/structure.py` and `tests/test_structure.py`.

All formulas are explicit and deterministic; regime classification uses rules and percentile ranks, not machine learning.

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

## What this MVP still does not include

- Order execution, broker APIs, or trading logic
- Machine learning models (rules-based regime and gate only)
- Dashboards or HTTP APIs

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

## Pipeline flow

1. Load OHLCV CSVs from `sample_data/`
2. Validate data quality and schema
3. Align close prices on timestamp
4. Build Day 2 feature table
5. Build Day 3 structural snapshot (`neraium/structural.py`)
6. Run regime → confidence → gate → posture (`neraium/regime.py`, `neraium/signals.py`)
7. Print shape summaries, regime counts, last 10 signal rows
8. Write `output/signals.csv`
9. Optionally write `output/features.csv` and `output/structural_snapshot.csv` (`--save-output`)

## Run

From `neraium_markets/`:

```bash
python main.py
```

Loads data, validates, aligns, builds features and structural snapshot, runs regime → confidence → gate → posture, prints shapes, regime counts, last 10 signal rows, and writes **`output/signals.csv`**.

Optional intermediate CSVs:

```bash
python main.py --save-output
```

## Regenerate sample data

Synthetic daily OHLCV (35 rows per asset) can be regenerated with:

```bash
python tools/generate_sample_data.py
```

## Outputs

- **Default:** `output/signals.csv`
- **With `--save-output`:** also `output/features.csv` and `output/structural_snapshot.csv`

## Tests

```bash
python -m pytest tests -q
```

Day 3 structural utilities are covered in `tests/test_structure.py` (imports from `neraium/structure.py`).

## Layout

- `config.py` — assets, paths, columns, `REQUIRED_COLUMNS`, groups, Day 3 structural parameters for `structure.py`
- `main.py` — full pipeline through Day 4
- `neraium/data_loader.py` — CSV loading
- `neraium/validation.py` — checks + sample Pydantic row validation
- `neraium/alignment.py` — outer join on timestamp (uppercase symbols)
- `neraium/features.py` — Day 2 feature table
- `neraium/structural.py` — structural snapshot used by regime/signals (Day 4 path)
- `neraium/structure.py` — alternate Day 3 structural scoring (tests)
- `neraium/regime.py` — regime, confidence, gate, posture
- `neraium/signals.py` — `generate_signals`, CSV save
- `neraium/schemas.py` — `OHLCVRow`
- `sample_data/` — one CSV per asset
- `output/` — generated CSVs (created on run)
- `tests/` — pytest suite
