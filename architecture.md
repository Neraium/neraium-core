# Neraium Markets Architecture (MVP v1)

## Purpose
Neraium Markets is a **read-only structural intelligence engine** that detects market regime transitions and emits explainable operator-facing posture signals.

## Layers
1. **Ingestion (`markets/data`)**
   - CSV loader for local OHLC/macro proxies.
   - timestamp normalization and alignment.
   - validation for duplicates and missing values.
2. **State Builder (`markets/state` + `markets/features`)**
   - returns, realized volatility, breadth, cross-asset and macro sensitivity features.
   - single state vector per timestamp.
3. **Structural Engine (`markets/structure`)**
   - baseline vs recent windows.
   - correlation geometry shifts, lag shifts, concentration and entropy deltas.
   - aggregate structural drift + instability score.
4. **Regime Engine (`markets/regime`)**
   - deterministic rules for regime/state labels.
   - confidence score from coherence, persistence, quality, contradiction penalties.
   - interpretive gate suppresses low-confidence/noise.
5. **Signal Layer (`markets/signals`)**
   - standardized signal object with scores, posture, rationale, and evidence payload.
6. **Evidence Vault (`markets/evidence`)**
   - append-only JSONL records of every surfaced signal and source snapshot.
7. **API (`markets/app/api.py`)**
   - `/health`, `/run-signals`, `/signals/latest`, `/signals/{asset}`, `/evidence/{signal_id}`

## Tradeoffs
- Deterministic and interpretable over predictive complexity.
- Uses synthetic sample data to stay API-independent.
- Baseline/recent windows are fixed-size in v1 for simplicity.
