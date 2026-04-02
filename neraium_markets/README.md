# Neraium Markets

Read-only market intelligence pipeline: load OHLCV CSVs, validate, align closes, engineer features, build a structural snapshot, produce **regime-aware signals**, run a deterministic **Day 5 validation layer** (forward outcomes, usefulness scoring, calibration, baselines), **Day 6 reliability analysis** (regime persistence, transitions, signal stability, false-positive diagnostics, filtered signals), **Day 7 multi-timeframe confirmation** (daily/1h/15m alignment, agreement scores, confidence adjustment, alignment-aware filtering), and **Day 8 cross-asset / market-wide state** (clustering, propagation, influence, synthesized market regime and posture).

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

## Pipeline flow (Days 1–8)

1. Load OHLCV CSVs from `sample_data/`
2. Validate data quality and schema
3. Align close prices on timestamp
4. Build Day 2 feature table
5. Build Day 3 structural snapshot (`neraium/structural.py`)
6. Classify regime, score confidence, apply interpretive gate, generate signals (`neraium/regime.py`, `neraium/signals.py`)
7. Compute forward returns and action usefulness (Day 5)
8. Confidence calibration and baseline comparison (Day 5)
9. **Day 6:** regime runs & persistence, transition matrix & transition-quality stats, signal stability, false-positive flags, filtered postures, filtered vs unfiltered comparison, reliability report
10. **Day 7:** run daily/1h/15m pipelines, align timeframe states on 15m timestamps, compute regime/action agreement, adjust confidence, apply alignment filter, compare aligned vs unaligned usefulness
11. **Day 8:** per-asset similarity & clustering, regime propagation & influence, market-wide state & posture, market vs asset usefulness comparison

## Run

From `neraium_markets/`:

```bash
python main.py
```

Runs the full pipeline through Day 8, prints validation, reliability, alignment, and market-structure summaries, and (with `--save-output`) writes CSV/JSON under `output/`.

## Regenerate sample data

Synthetic daily OHLCV (35 rows per asset) can be regenerated with:

```bash
python tools/generate_sample_data.py
```

## Outputs

With `--save-output`, Day 5/6 base artifacts are written as before, plus:

- **Day 7:** `output/timeframe_alignment.csv`, `output/alignment_comparison.csv`, `output/day7_alignment_summary.json`
- **Day 8:** `output/asset_similarity_matrix.csv`, `output/asset_clusters.csv`, `output/cluster_summary.csv`, `output/regime_propagation.csv`, `output/asset_influence_scores.csv`, `output/sector_influence_scores.csv`, `output/market_state.csv`, `output/market_vs_asset_comparison.csv`

## Tests

```bash
python -m pytest tests -q
```

Day 3 structural utilities are covered in `tests/test_structure.py` (imports from `neraium/structure.py`).

## Layout

- `config.py` — assets, paths, columns, `REQUIRED_COLUMNS`, groups, Day 3 structural parameters for `structure.py`
- `main.py` — full pipeline through Day 8 (Days 5–8 outputs on `--save-output`)
- `neraium/data_loader.py` — CSV loading
- `neraium/validation.py` — checks + sample Pydantic row validation
- `neraium/alignment.py` — outer join on timestamp (uppercase symbols)
- `neraium/features.py` — Day 2 feature table
- `neraium/structural.py` — structural snapshot used by regime/signals (Day 4 path)
- `neraium/structure.py` — alternate Day 3 structural scoring (tests)
- `neraium/regime.py` — regime, confidence, gate, posture
- `neraium/signals.py` — `generate_signals`, CSV save
- `neraium/transitions.py` — regime runs, persistence summary, transition matrix, transition usefulness
- `neraium/diagnostics.py` — signal stability (flip/jump flags, `stability_score`)
- `neraium/filtering.py` — false-positive pattern flags, `filtered_action_posture`, filtered vs unfiltered comparison
- `neraium/clustering.py` — asset similarity matrix, threshold clusters, cluster summaries
- `neraium/propagation.py` — regime propagation counts, asset and sector influence scores
- `neraium/market_state.py` — panel aggregates, `synthesize_market_state`, market action & explanation, market vs asset usefulness
- `neraium/schemas.py` — `OHLCVRow`
- `sample_data/` — one CSV per asset
- `output/` — generated CSVs (created on run)
- `tests/` — pytest suite


## Day 5 validation (first backtesting layer)

Day 5 adds *validation*, not trade execution. The goal is to test whether the Day 4 `regime_label -> action_posture` mapping is coherent, stable, and directionally useful versus naive alternatives.

### What Day 5 now does

1. **Forward outcome evaluation**
   - Computes `fwd_ret_1d`, `fwd_ret_5d`, `fwd_ret_10d` from `spy`.
2. **Action usefulness scoring (MVP heuristic)**
   - Writes `action_useful_1d`, `action_useful_5d`, `action_useful_10d` in `{-1, 0, 1}`.
   - `1` useful, `0` neutral, `-1` harmful.
3. **Confidence calibration summary**
   - Bins confidence into `[0.0-0.2, ..., 0.8-1.0]`.
   - Reports counts and average usefulness by bin, plus a monotonicity diagnostic.
4. **Baseline comparison**
   - Trend-only baseline (`spy_ret_5d`).
   - Volatility-only baseline (`spy_vol_10d`).
   - Breadth-only baseline (`breadth_pct_above_20dma`).
   - Compares avg usefulness, hit rate, non-neutral count, abstention rate.
5. **Validation report summary**
   - Aggregates regime/action counts, average confidence, usefulness by horizon/regime, calibration table, and baseline table.

### Important non-goals (still not included)

- **Not a trading backtester** (no fills/slippage/fees/position accounting).
- **Not execution-aware** (no order simulation, no broker integration).
- **Not a PnL engine yet** (usefulness is directional heuristic quality, not returns attribution).
- No ML / optimizer / dashboards in this Day 5 slice.

### Day 5 pipeline flow

From `main.py`:

1. load data
2. validate
3. align
4. build features
5. build structural snapshot
6. classify regime
7. compute confidence
8. apply interpretive gate
9. generate signals
10. compute forward returns
11. score action usefulness
12. evaluate confidence calibration
13. compare to baselines
14. build validation report

### Run Day 5 pipeline

```bash
python main.py
```

Optional output files:

```bash
python main.py --save-output
```

### Day 5 outputs

When `--save-output` is set:

- `output/signals_with_forward_returns.csv`
- `output/confidence_calibration.csv`
- `output/baseline_comparison.csv`
- `output/validation_summary.json`

---

## Day 6 reliability (persistence, transitions, false positives)

Day 6 is **offline reliability analysis**: it does not add execution, brokers, ML, portfolio optimization, or dashboards. The goal is to see which regimes last long enough to matter, which transitions line up with better or worse usefulness, where signals flip or jump, which rows look like false positives, and whether simple **post-signal filters** improve the usefulness proxy.

### Regime persistence

Consecutive rows with the same `regime_label` form a **run**. For each run we record length and position-in-run. Summaries per label include how many runs occurred, average/median/max length, and the share of **single-step** runs (length 1), which often indicates noise.

### Transition analysis

From `regime_label[t]` to `regime_label[t+1]` we build a **transition matrix** (counts and conditional probabilities `P(to | from)`). **Transition quality** joins each transition to average `action_useful_*` at time `t`, so you can see whether certain hand-offs (for example into `risk_off_transition`) align with higher or lower usefulness.

### Signal stability

`compute_signal_stability` adds flip/jump flags (regime change, posture change, large confidence step) and a **stability score** (rolling smoothness, higher is calmer). This highlights unstable periods even when the headline regime name is stable.

### False-positive diagnostics and filtering

Heuristic flags (documented in `neraium/filtering.py`) mark low persistence, confidence–coherence contradictions, unstable stability, quick action reversals, and a composite **likely false positive**. `apply_signal_filters` sets `filtered_action_posture` to `wait` when rules fire (suppression), and `signal_kept_flag` records whether the original posture was kept.

**Filtered signals** are not a live trading directive; they are an abstention-heavy view of the same pipeline to reduce weak or contradictory rows.

### Filtered vs unfiltered

`compare_filtered_vs_unfiltered` reports average usefulness by horizon, active (non-`wait`/`watch`) counts, abstention rate (share of neutral usefulness), and average confidence on kept rows, for both versions.

### Day 6 pipeline steps (main)

---

## Day 7 multi-timeframe alignment (confirmation logic only)

Day 7 adds **cross-timeframe confirmation**, not execution.

- **Timeframes:** daily, 1h, 15m.
- **Alignment rule:** each 15m row is matched to the most recent 1h and daily row at or before that timestamp.
- **Agreement layers:**
  - `regime_agreement_score` + `regime_alignment_label` (`strong_alignment`, `medium_alignment`, `weak_alignment`)
  - `action_agreement_score` + `action_alignment_label`
- **Adjusted confidence:** starts from 15m confidence, then applies transparent boosts/haircuts from regime/action agreement and higher-timeframe conflict.
- **Alignment filter:** suppresses aggressive lower-timeframe posture when higher-timeframe posture is defensive, or when alignment is weak with low adjusted confidence.
- **Comparison:** `unaligned` vs `alignment_filtered` summaries for usefulness, abstention, active-signal count, and average confidence.

### Day 7 non-goals (explicit)

- No broker APIs
- No live trading execution
- No ML
- No portfolio optimization
- No production trading infrastructure

This is a deterministic trust-calibration/confirmation layer for offline analysis.

14. compute regime runs  
15. summarize persistence  
16. build transition matrix  
17. compute signal stability  
18. identify false-positive patterns  
19. apply filters  
20. score filtered usefulness and compare  
21. build Day 6 reliability report  

### Run Day 6

```bash
python main.py
python main.py --save-output
```

### Day 6 outputs (`--save-output`)

- `output/regime_persistence.csv`
- `output/transition_matrix.csv`
- `output/transition_quality.csv`
- `output/filtered_signal_comparison.csv`
- `output/day6_reliability_summary.json`

The main run also prints: average regime run length, top transitions by count, count of likely false positives, the filtered vs unfiltered table, and whether mean usefulness improved after filtering.

### Day 6 non-goals (unchanged)

- Not execution, not broker APIs, not PnL optimization, not a production trading system.

---

## Day 8 cross-asset and market-wide state

Day 8 adds **system-level structural intelligence** on top of the single-stream SPY pipeline. It does **not** add execution, ML, portfolio optimization, or dashboards. It is **not** a production trading system and **not** optimized for PnL.

### Asset clustering

For each configured equity ETF (core + sectors), the pipeline **re-runs** the Day 4 regime stack on a per-asset blend of market structural scores and that asset’s volatility (so names can diverge). **Similarity** mixes return correlation, regime agreement rate, and confidence alignment. **Clusters** are connected components of the similarity graph above a fixed threshold (deterministic, no ML).

### Regime propagation

When an asset’s `regime_label` changes at date *t*, other assets are scanned in the next few rows for a regime change. Aggregated counts yield **propagation** pairs (source/target, regimes, counts, average delay). **Influence** scores favor sources that broadcast many follow-on changes with short delays; the same table is rolled up to **sectors** via `config.ASSET_TO_SECTOR`.

### Market-wide state

Per timestamp, cross-sectional **fractions** (risk-off, stable, etc.) and mean panel confidence/instability/coherence are merged onto the structural timeline. **Rules** (documented in `neraium/market_state.py`) assign `market_regime_label` (e.g. `market_stable`, `market_risk_off`, `market_fragmented`). **Market action posture** maps that label plus scores to `risk_on`, `cautious_risk_on`, `neutral`, `reduce_risk`, `defensive`, `wait`, with a short **market_explanation** string.

### Market vs asset usefulness

`compare_market_vs_asset_usefulness` contrasts **asset_level** (mean usefulness across the long panel) with **market_level** (usefulness of mapped market posture against SPY forward returns). This is a sanity check on whether the synthesized layer is informative.

### Day 8 outputs (`--save-output`)

- `output/asset_similarity_matrix.csv`
- `output/asset_clusters.csv`
- `output/cluster_summary.csv`
- `output/regime_propagation.csv`
- `output/asset_influence_scores.csv`
- `output/sector_influence_scores.csv`
- `output/market_state.csv`
- `output/market_vs_asset_comparison.csv`

### Run Day 8

```bash
python main.py
python main.py --save-output
```
