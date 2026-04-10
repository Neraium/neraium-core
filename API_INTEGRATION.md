# API Integration for Replacement UI

_Last updated: April 10, 2026._

## Purpose

This guide documents how the `ui` package consumes Neraium API outputs after static router removal.

## Data contract expected by UI

The replacement UI expects records with these keys (missing values are tolerated):

- `timestamp`
- `site_id`
- `asset_id`
- `state`
- `regime_name`
- `structural_drift_score`
- `relational_stability_score`
- `system_health`
- `drift_alert` or `alert`
- `confidence` or `confidence_score`

## Integration path

1. API outputs (for example ingest/result endpoints) produce canonical result records.
2. `ui.core_integration.compact_record` maps records into stable UI payloads.
3. `ui.layouts` composes pilot/operations/demo views.
4. `ui.realtime.create_realtime_feed` may provide websocket-backed status when dependency support exists.

## Notes

- There is no `apps/api/static` UI fallback.
- The API runtime remains independent of the UI package.
- UI launch is optional (`python -m ui.app`) and should be treated as a separate process from the API service.
