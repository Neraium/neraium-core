# Neraium Service Layer API (Production-Oriented v1)

This document defines the production service layer around the validated Neraium engine using **advisory operational recommendation semantics**.

## Canonical Output Contract

All service methods emit or persist one stable canonical envelope.

### Required fields

- `schema_version`: contract version string (`2026-03-28`)
- `timestamp`: frame timestamp
- `cycle`: per-run monotonic cycle index
- `attribution`
  - `top_drivers`: list of `{driver, score}`
  - `group_contributions`: optional grouped contributions
- `regime_memory`: regime-memory payload from engine
- `risk_assessment`
  - `risk_level`
  - `trend`
  - `latest_instability`
- `causal_analysis`: engine causal output
- `operational_recommendation`
  - `status.available`
  - `status.advisory` (always `true`)
  - `status.reason`
  - `recommended_action`
  - `recommended_target`
  - `priority`
  - `recommendation_confidence`
  - `urgency`
  - `rationale`
  - `supporting_evidence`
  - `operator_note`
- `confidence`: normalized [0,1] confidence view
- `explanation_text`: operator-facing explanation
- `events`: product-facing event flags

### Optional fields

- `session`: `{run_id, customer_id}`
- `aliases`: isolated compatibility aliases (`legacy_decision`, optional legacy `decision`, `explanation`, `regime_memory_state`)

## Operator boundary

Neraium recommendations are advisory outputs intended to support, not replace, qualified operator judgment and site-specific procedures.

## Service Interface

Implemented on `StructuralMonitoringService`:

- `ingest_frame(payload, *, run_id=None, customer_id=None)`
- `ingest_batch(payloads, *, run_id=None, customer_id=None)`
- `get_current_state(*, run_id=None, customer_id=None)`
- `get_recent_history(limit=100, *, run_id=None, customer_id=None)`
- `get_latest_recommendation(*, run_id=None, customer_id=None)`
- `get_latest_decision(*, run_id=None, customer_id=None)` (deprecated compatibility alias)
- `get_latest_explanation(*, run_id=None, customer_id=None)`

## Event Semantics

Events are derived from existing engine outputs:

- `early_instability_detected`: instability rises while risk not yet high
- `risk_escalated`: risk level increases vs previous cycle
- `recommendation_available`: advisory recommendation is available
- `inspection_recommended`: high risk or inspect/diagnose recommendation cues
- `deterioration_detected`: rising trend or instability jump

## HTTP API Layer (FastAPI)

### Endpoints

- `POST /ingest/frame`
- `GET /state`
- `GET /history`
- `GET /recommendation`
- `GET /recommendations/latest`
- `GET /decision` (deprecated compatibility alias)
- `GET /explanation`
- `GET /events/latest`

### Example recommendation block

```json
{
  "operational_recommendation": {
    "status": {
      "available": true,
      "advisory": true,
      "reason": "recommendation_available"
    },
    "recommended_action": "inspect_cooling_loop",
    "recommended_target": "cooling_loop",
    "priority": 1,
    "recommendation_confidence": 0.74,
    "urgency": "medium",
    "rationale": "Recommendation available from converging structural evidence.",
    "supporting_evidence": [{"driver": "temperature", "score": 0.82}],
    "operator_note": "Recommendations are advisory outputs intended to support, not replace, qualified operator judgment and site-specific procedures."
  }
}
```
