# Neraium Service Layer API (Production-Oriented v1)

This document defines the first production-oriented service layer around the validated Neraium engine.

## Canonical Output Contract

All service methods emit or persist a canonical output envelope with one stable schema.

### Required fields

- `schema_version`: contract version string (`2026-03-01`)
- `timestamp`: frame timestamp
- `cycle`: per-run monotonic cycle index
- `attribution`: normalized attribution payload
  - `top_drivers`: list of `{driver, score}`
  - `group_contributions`: optional grouped contributions
- `regime_memory`: regime-memory payload from engine
- `risk_assessment`
  - `risk_level`
  - `trend`
  - `latest_instability`
- `causal_analysis`: engine causal output
- `decision`
  - `state`
  - `action`
  - `reason`
  - `source`
- `confidence`: normalized [0,1] confidence
- `explanation_text`: operator-facing explanation
- `events`: product-facing event flags

### Optional fields

- `session`: `{run_id, customer_id}`
- `aliases`: isolated legacy/experimental aliases (`explanation`, `regime_memory_state`)

## Service Interface

Implemented on `StructuralMonitoringService`:

- `ingest_frame(payload, *, run_id=None, customer_id=None)`
- `ingest_batch(payloads, *, run_id=None, customer_id=None)`
- `get_current_state(*, run_id=None, customer_id=None)`
- `get_recent_history(limit=100, *, run_id=None, customer_id=None)`
- `get_latest_decision(*, run_id=None, customer_id=None)`
- `get_latest_explanation(*, run_id=None, customer_id=None)`

## Persistence / History

SQLite-backed `service_history` table stores, per cycle:

- timestamp/cycle
- risk state
- decision
- confidence
- attribution top drivers
- explanation text
- event flags
- full canonical record JSON

## Event Semantics

Events are derived strictly from existing engine outputs:

- `early_instability_detected`: instability rises while risk not yet high
- `risk_escalated`: risk level increases vs previous cycle
- `decision_available`: decision state/action available
- `inspection_recommended`: high risk or inspect/diagnose action cues
- `deterioration_detected`: rising trend or instability jump

## Example flow

1. `ingest_frame` receives a telemetry frame.
2. Existing engine runs unchanged.
3. Service normalizes output into canonical contract.
4. Service derives product events.
5. Canonical record is persisted in `service_history`.
6. Retrieval methods return latest/current/history decision/explanation views.
