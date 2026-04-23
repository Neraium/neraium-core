# Neraium Markets

Read-only market intelligence pipeline: load OHLCV via a **connector / ingestion layer** (Day 13), validate, align closes, engineer features, build a structural snapshot, produce **regime-aware signals**, run a deterministic **Day 5 validation layer** (forward outcomes, usefulness scoring, calibration, baselines), **Day 6 reliability analysis** (regime persistence, transitions, signal stability, false-positive diagnostics, filtered signals), **Day 7 multi-timeframe confirmation** (daily/1h/15m alignment, agreement scores, confidence adjustment, alignment-aware filtering), **Day 8 cross-asset / market-wide state** (clustering, propagation, influence, synthesized market regime and posture), **Day 9 trajectory & path intelligence**, **Day 10 evidence / decision audit**, **Day 11 realtime operator monitoring** (polling hook, alerts, health, operator inbox — **not** execution), and **Day 14 local persistence** (pipeline state, snapshots, file cache, incremental runs — **not** distributed orchestration or cloud workflows).

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

Optional editable install (local packaging, same import layout as running from this folder):

```bash
pip install -e .
```

## Pipeline flow (Days 1–11)

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
12. **Day 9:** trajectory, scenario paths, warnings, path-adjusted actions
13. **Day 10:** decision records, evidence JSONL, review tables, decision memory
14. **Day 11 (optional):** realtime cycle, alert evaluation, suppression, health log, operator inbox (`--realtime`)
15. **Day 12:** configuration profiles, CLI entrypoints, run manifests, standardized output layout, safe runtime wrappers, fallbacks, local deployment hardening (**not** cloud infrastructure, **not** live execution)
16. **Day 13:** connector interface, normalization to a canonical schema, source-quality gates, ingestion orchestration, mock API-style fixtures (**not** broker APIs, **not** live HTTP, **not** cloud ingestion products)
17. **Day 14:** local pipeline state, input/output snapshots, file cache, incremental detection and windowing, resume summaries (**not** distributed processing, **not** cloud orchestration, **not** execution infrastructure)

## Run

From `neraium_markets/`:

**Preferred (Day 12 CLI)** — subcommand first, then flags:

```bash
python -m neraium.cli run-batch --profile dev
python -m neraium.cli run-realtime --profile prod_local --max-cycles 5
python -m neraium.cli run-day11-realtime --profile dev
python -m neraium.cli health-check --profile dev
python -m neraium.cli rebuild-outputs --profile prod_local
```

**Day 13 — sources and ingestion** (inspect connector, validate quality, preview ingest):

```bash
python -m neraium.cli inspect-sources --profile dev
python -m neraium.cli validate-source --profile dev
python -m neraium.cli ingest-preview --profile test --source-type mock_api
```

Optional: `--source-type csv` or `--source-type mock_api` overrides the profile default.

**Day 14 — state, cache, snapshots, incremental:**

```bash
python -m neraium.cli show-state --profile dev
python -m neraium.cli clear-cache --profile dev
python -m neraium.cli rebuild-from-snapshots --profile dev
python -m neraium.cli run-incremental --profile prod_local
```

Omitting the subcommand defaults to `run-batch` (same as `python -m neraium.cli run-batch`).

**Legacy `main.py`** — still supported; routes to the same CLI (subcommands) or legacy flags:

**Batch (default)** — full pipeline Days 1–10 print; evidence log appended each run:

```bash
python main.py
```

With artifact saves:

```bash
python main.py --save-output
```

**Realtime operator mode** — one monitoring cycle: latest snapshot, optional alerts vs prior snapshot under `output/latest/`, health + inbox (**advisory only; not execution**):

```bash
python main.py --realtime
# equivalent:
python main.py --mode realtime
```

Runs the full pipeline through Day 10 in both modes; batch prints validation through Day 10; realtime prints a short Day 11 summary and writes under `output/` (see **Day 12 output layout** below): `latest/realtime_snapshot.json`, `reviews/operator_inbox.csv`, append-only `alerts/alert_log.jsonl` and `health/health_log.jsonl`.

## Day 12: local deployment hardening

Day 12 is about **repeatability, configuration, and safe failure reporting** on your machine. It is **not** a production trading stack, **not** broker integration, **not** cloud deployment automation, and **not** live order execution.

### Configuration profiles

Profiles (`dev`, `test`, `prod_local`) are loaded by `neraium.settings.load_settings(profile)`. They set defaults for output directories, polling interval, max cycles, freshness and suppression windows, and whether to save batch artifacts. Override with environment variables:

| Variable | Purpose |
|----------|---------|
| `NERAIUM_PROFILE` | `dev` / `test` / `prod_local` |
| `NERAIUM_DATA_DIR` | Directory of `{asset}.csv` files (absolute or relative to project root) |
| `NERAIUM_OUTPUT_DIR` | Base output directory (absolute or relative to project root) |
| `NERAIUM_SAVE_OUTPUTS` | `true` / `false` — default batch artifact saving |
| `NERAIUM_SOURCE_TYPE` | `csv` or `mock_api` — overrides profile default |
| `NERAIUM_MOCK_API_FIXTURE_DIR` | Directory of `{asset}.json` fixtures for mock API connector |

`validate_settings()` rejects inconsistent values (e.g. missing data directory for `csv`, missing fixture directory for `mock_api`, invalid timeframes).

### Day 13: connectors, normalization, and source quality

This layer is **source flexibility and schema safety** for local / test data. It is **not** a broker connector, **not** order execution, and **not** a managed cloud ingestion service.

- **Connectors** (`neraium/connectors/`): `CSVConnector` reads `{asset}.csv` from `data_dir` (optional `data_dir/{timeframe}/` if present). `MockAPIConnector` loads `{asset}.json` fixtures (list of rows or `{"bars": [...]}`) and records simulated latency metadata — **no network calls**.
- **Registry** (`get_connector(source_type, settings)`): `csv` | `mock_api`.
- **Normalization** (`neraium/normalization.py`): maps aliases → canonical columns `timestamp`, `open`, `high`, `low`, `close`, `volume`; parses timestamps; sorts; drops duplicate timestamps.
- **Source quality** (`neraium/source_quality.py`): per-asset checks (null rates, duplicates, monotonicity, row count vs thresholds). Status `good` | `degraded` | `invalid`. Profiles set `reject_invalid_assets` and `source_quality_thresholds`.
- **Ingestion** (`neraium/ingestion.py`): connector → normalize → quality → accepted asset dict; `build_ingestion_summary(...)` for manifests and CLI.
- **Pipeline path**: `run_full_pipeline(..., settings=...)` uses ingestion when `settings` is passed (CLI batch/realtime). Legacy calls without `settings` still use direct CSV loading for tests/scripts.
- **Manifests**: batch manifests include `extra.ingestion_summary` and `source_metadata` when available.

**Profiles:** `dev` defaults to `csv`; `test` defaults to `mock_api` with fixtures under `tests/fixtures/mock_api/` (generated from `sample_data/` CSVs as JSON); `prod_local` uses `csv` with stricter quality thresholds and `reject_invalid_assets=True`.

### Day 14: persistence, snapshots, cache, and incremental runs

Day 14 adds **local, inspectable** persistence so repeated runs can **resume context**, **avoid redundant work** when nothing changed, and **trim inputs** for incremental updates. It is **not** a distributed compute framework, **not** cloud workflow orchestration, and **not** trading execution.

- **State** (`neraium/state_store.py`): `save_pipeline_state` / `load_pipeline_state` (JSON). Canonical files: `output/latest/pipeline_state.json` and a copy under `output/state/pipeline_state.json`. Fields include last successful `run_id`, `latest_timestamp_by_asset`, `source_type`, manifest path, optional cache key metadata.
- **Snapshots** (`neraium/snapshots.py`): per-run folders under `output/snapshots/{run_id}/inputs|outputs/` (CSV for frames, JSON for dicts). `load_latest_snapshot` picks the newest `outputs/{artifact}.csv` or `.json` across runs.
- **Cache** (`neraium/cache.py`): pickle files under `output/cache/` keyed by `make_cache_key` (SHA-256 of profile, artifact name, timeframe, fingerprint). Used to store the aligned **merged** frame after a successful batch run when `enable_cache` is on.
- **Incremental** (`neraium/incremental.py`): `detect_new_data` vs prior state; `filter_to_incremental_window` keeps new rows plus a lookback buffer; `merge_incremental_outputs` for stitched outputs; `build_resume_summary` for operator JSON (`output/latest/run_resume_summary.json`).
- **Batch CLI path**: ingests once, optionally **skips the full pipeline** if `enable_incremental` and there is **no new data** since the last run (prints a short Day 14 message). Otherwise runs `run_full_pipeline` with `preloaded_data` (full or filtered). Successful runs update pipeline state and extend manifests / `run_status.json` with `extra.day14` and a `day14` block on status when applicable.
- **Profiles:** `dev` — cache on, incremental off, snapshots off by default; `test` — cache off, snapshots on (deterministic tests); `prod_local` — cache on, incremental on, snapshots on.

### Output directory layout

Under the configured output base (default `output/`):

| Path | Role |
|------|------|
| `latest/run_status.json` | Single place to read last run outcome (success, manifest path, outputs, health summary; may include `day14` metadata) |
| `latest/pipeline_state.json` | Last persisted pipeline state (Day 14) |
| `latest/run_resume_summary.json` | Incremental / cache / snapshot operator summary (Day 14) |
| `state/pipeline_state.json` | Copy of pipeline state for retention |
| `snapshots/{run_id}/…` | Input/output snapshot artifacts (Day 14) |
| `cache/*.cache.pkl` | Pickled cached intermediates (e.g. merged prices) (Day 14) |
| `latest/realtime_snapshot.json` | Latest realtime snapshot JSON |
| `manifests/{run_id}.json` | Run manifest (profile, mode, settings snapshot, git commit if available, outputs) |
| `runs/` | Optional per-run retention (helpers in `neraium.runtime`) |
| `evidence/evidence_log.jsonl` | Batch evidence log (when emitted) |
| `alerts/alert_log.jsonl` | Alert JSONL |
| `health/health_log.jsonl` | Health JSONL |
| `reviews/operator_inbox.csv` | Operator inbox CSV |

### Run manifests and reproducibility

Each batch or realtime run writes a JSON manifest under `manifests/` with `run_id`, timestamp, profile, mode, assets/timeframes metadata, paths written, optional `git_commit`, embedded `settings_snapshot`, and success/failure.

### Safe failure and fallback

`neraium.runtime.safe_run_pipeline` wraps work that may raise: it returns a structured dict (`success`, `error_message`, `stage_failed`, `latest_available_output`, `health_status`) instead of crashing the process.

`fallback_to_last_good_output(output_dir)` surfaces the best-known path from the last good `run_status.json`, newest manifest, or `latest/realtime_snapshot.json` when a run fails.

Atomic JSON writes (`atomic_write_json`) reduce the chance of half-written status files.

## Regenerate sample data

Synthetic daily OHLCV (35 rows per asset) can be regenerated with:

```bash
python tools/generate_sample_data.py
```

## Outputs

With `--save-output`, Day 5/6 base artifacts are written as before, plus:

- **Day 7:** `output/timeframe_alignment.csv`, `output/alignment_comparison.csv`, `output/day7_alignment_summary.json`
- **Day 8:** `output/asset_similarity_matrix.csv`, `output/asset_clusters.csv`, `output/cluster_summary.csv`, `output/regime_propagation.csv`, `output/asset_influence_scores.csv`, `output/sector_influence_scores.csv`, `output/market_state.csv`, `output/market_vs_asset_comparison.csv`
- **Day 9–10:** `output/market_state_day9.csv`, scenario/path CSVs, decision CSVs, `output/evidence_log.jsonl`, etc. (see earlier Day 9/10 docs in this file)
- **Day 11 (realtime mode):** under `output/latest/`, `output/alerts/`, `output/health/`, `output/reviews/` (see Day 12 layout); legacy paths may still appear if `NERAIUM_OUTPUT_DIR` points at an older flat `output/` tree

## Tests

```bash
python -m pytest tests -q
```

Day 3 structural utilities are covered in `tests/test_structure.py` (imports from `neraium/structure.py`).

## Layout

- `config.py` — assets, paths, columns, `REQUIRED_COLUMNS`, groups, Day 3 structural parameters for `structure.py`
- `main.py` — thin entrypoint → `neraium.cli.dispatch_main` (Day 12)
- `neraium/cli.py` — batch/realtime/health/rebuild + `inspect-sources`, `validate-source`, `ingest-preview` (Day 13)
- `neraium/settings.py` — profiles, `source_type`, mock fixture path, quality thresholds, `reject_invalid_assets`, env overrides, `validate_settings`
- `neraium/connectors/` — `MarketDataConnector`, `CSVConnector`, `MockAPIConnector`, `get_connector` (Day 13)
- `neraium/normalization.py` — canonical OHLCV schema (Day 13)
- `neraium/source_quality.py` — pre-pipeline quality assessment (Day 13)
- `neraium/ingestion.py` — `ingest_market_data`, `build_ingestion_summary` (Day 13)
- `neraium/state_store.py` — pipeline state JSON (Day 14)
- `neraium/snapshots.py` — input/output snapshots, `load_latest_snapshot` (Day 14)
- `neraium/cache.py` — file cache keys and pickle artifacts (Day 14)
- `neraium/incremental.py` — new-data detection, windowing, merge, `build_resume_summary` (Day 14)
- `neraium/manifests.py` — `build_run_manifest`, `save_run_manifest`
- `neraium/runtime.py` — output layout helpers, `safe_run_pipeline`, `fallback_to_last_good_output`, `save_run_status`
- `neraium/realtime.py` — `run_full_pipeline`, `run_realtime_cycle`, polling hook
- `neraium/alerts.py` — alert detection, JSONL log, suppression, operator inbox
- `neraium/health.py` — freshness / health snapshot and JSONL log
- `neraium/data_loader.py` — direct CSV loading (legacy path when `run_full_pipeline` is called without `settings`)
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

---

## Day 11 realtime operator layer (monitoring)

This layer turns the read-only pipeline into a **file-backed monitoring and decision-support** loop for operators. It does **not** place orders, connect to brokers, or automate trading. **Alerts are advisory only.**

### Behavior

- **Batch mode (`python main.py`):** Unchanged Days 1–10 behavior; `output/evidence_log.jsonl` is appended from decision rows.
- **Realtime mode (`python main.py --realtime`):** Runs one full pipeline pass via `run_realtime_cycle`, reads the prior `output/realtime_snapshot.json` if it exists, compares **current vs prior** snapshots, runs `detect_alert_conditions`, applies `should_suppress_alert` against recent `output/alert_log.jsonl` lines, optionally appends a structured alert, computes `compute_system_health`, appends to `output/health_log.jsonl`, writes `output/operator_inbox.csv` (recent rows, **newest first**), and overwrites `output/realtime_snapshot.json`.

### Alert types and severities

- **Types (examples):** `regime_change`, `warning_escalation`, `action_shift`, `confidence_break`, `path_deterioration`
- **Severities:** `info`, `low`, `medium`, `high`
- Triggers include regime label change, material warning escalation, defensive shift in `path_adjusted_market_action`, confidence crossing bands, and high-risk scenario labels.

### Suppression (spam control)

- Suppress if the same **alert_type + alert_severity + market_regime_label** fingerprint matches a recent logged alert.
- Suppress a **lower-severity** alert of the same **alert_type** if a **higher-severity** alert for that type appears in recent history.

### Operator inbox

`build_operator_inbox` builds a **CSV** suitable for review: recent market-state rows plus alert fields when timestamps align with `alert_log.jsonl`. Not a web dashboard.

### System health

`health_status` may be **healthy**, **degraded**, **stale** (e.g. latest timestamp older than 72 hours), or **failed** (pipeline/validation failure). `data_freshness_ok` and `missing_data_rate` summarize timestamp quality on the latest frame.

### Polling

`run_polling_loop` in `neraium/realtime.py` is a simple **for**-loop with `time.sleep(interval_seconds)` and `max_cycles` for tests or scheduled jobs; wire your own scheduler or cron to `python main.py --realtime` as needed.
