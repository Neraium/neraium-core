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

- `session`: `{run_id, customer_id, site_id, asset_id}`
- `aliases`: isolated legacy/experimental aliases (`explanation`, `regime_memory_state`)
- `memory_recall`: structural-memory recall block
  - `status`: recall enabled/scope/count metadata
  - `novelty`: `{is_novel, score, reason}`
  - `nearest_match`: best prior memory match with IDs and scope
  - `top_matches`: top-k evidence matches
  - `pattern_family`: deterministic family label when confidence is sufficient

## Service Interface

Implemented on `StructuralMonitoringService`:

- `ingest_frame(payload, *, run_id=None, customer_id=None)`
- `ingest_batch(payloads, *, run_id=None, customer_id=None)`
- `get_current_state(*, run_id=None, customer_id=None)`
- `get_recent_history(limit=100, *, run_id=None, customer_id=None)`
- `get_latest_decision(*, run_id=None, customer_id=None)`
- `get_latest_explanation(*, run_id=None, customer_id=None)`
- `get_memory_matches(*, run_id=None, customer_id=None, site_id=None, asset_id=None, limit=20)`
- `get_pattern_history(*, customer_id=None, family_label=None, limit=100)`
- `get_novel_events(*, customer_id=None, run_id=None, limit=100)`

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

SQLite-backed `structural_memory` table stores deterministic memory signatures and recall evidence:

- signature fields (top drivers, concentration, regime/risk/decision/hypothesis buckets)
- scoped identifiers (`customer_id`, `run_id`, `site_id`, `asset_id`, `cycle`, `timestamp`)
- novelty marker (`is_novel`, `score`)
- pattern family label (optional deterministic label)
- event flags and trace metadata

Near-duplicate suppression is enabled for repeated same-asset states with very high similarity.

## Event Semantics

Events are derived strictly from existing engine outputs:

- `early_instability_detected`: instability rises while risk not yet high
- `risk_escalated`: risk level increases vs previous cycle
- `decision_available`: decision state/action available
- `inspection_recommended`: high risk or inspect/diagnose action cues
- `deterioration_detected`: rising trend or instability jump
- `known_pattern_recalled`: nearest structural memory match found
- `novel_pattern_detected`: current pattern below conservative similarity/overlap thresholds
- `cross_asset_pattern_match`: nearest recalled match came from a different asset
- `recurring_degradation_pattern`: high-confidence recurring medium/high-risk degradation pattern

## Structural memory signature (deterministic)

The first-pass structural memory signature is compact and explainable:

- top attribution drivers (top 3 names)
- attribution concentration (`max_driver_abs_score / sum_abs_scores`)
- regime key (from regime memory labels/state)
- risk level + trend
- instability bucket (`latest_instability` normalized and clipped)
- decision state + action + derived action target token
- dominant hypothesis (from causal analysis dominant hypothesis/driver)
- confidence bucket

## Similarity and novelty

Similarity score is fully deterministic and interpretable:

- **categorical match score** across regime/risk/decision/hypothesis fields
- **driver overlap score** (Jaccard overlap of top drivers)
- **numeric closeness score** across concentration/instability/confidence buckets
- weighted total: `0.5 * categorical + 0.3 * driver_overlap + 0.2 * numeric`

Novelty is conservative:

- novel when either total similarity `< 0.72` **or** driver overlap `< 0.45`
- if no prior memory exists, mark novel with reason `no_prior_structural_memory`
- novelty reason always returned for traceability

## Example flow

1. `ingest_frame` receives a telemetry frame.
2. Existing engine runs unchanged.
3. Service normalizes output into canonical contract.
4. Service derives product events.
5. Canonical record is persisted in `service_history`.
6. Retrieval methods return latest/current/history decision/explanation views.

---

## HTTP API Layer (FastAPI)

The production API layer exposes the validated service layer without changing engine logic.

### Endpoints

- `POST /ingest/frame`
  - Ingest one telemetry frame and return canonical output.
  - Query params: `run_id`, `customer_id`
  - Body:
    ```json
    {
      "timestamp": "2026-01-01T00:00:00+00:00",
      "site_id": "plant-1",
      "asset_id": "pump-9",
      "sensor_values": {
        "pressure": 50.0,
        "temperature": 80.0
      },
      "customer_id": "customer-a"
    }
    ```

- `GET /state`
  - Return latest canonical state for the run/customer.
  - Query params: `run_id`, `customer_id`

- `GET /history`
  - Return recent canonical history.
  - Query params: `limit` (default `100`), `run_id`, `customer_id`

- `GET /decision`
  - Return latest decision object.
  - Query params: `run_id`, `customer_id`

- `GET /explanation`
  - Return latest explanation text.
  - Query params: `run_id`, `customer_id`

- `GET /events/latest`
  - Return latest product-facing event flags plus cycle/timestamp context.
  - Query params: `run_id`, `customer_id`

### Example responses

- `GET /state`
  ```json
  {
    "state": {
      "schema_version": "2026-03-01",
      "timestamp": "2026-01-01T00:00:00+00:00",
      "cycle": 1,
      "risk_assessment": {
        "risk_level": "LOW",
        "trend": "STABLE",
        "latest_instability": 0.12
      },
      "decision": {
        "state": "STABLE",
        "action": "none",
        "reason": "decision_available",
        "source": {}
      },
      "events": ["decision_available"]
    }
  }
  ```

- `GET /events/latest`
  ```json
  {
    "events": ["decision_available"],
    "cycle": 1,
    "timestamp": "2026-01-01T00:00:00+00:00"
  }
  ```

## Run locally

From repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r apps/api/requirements.txt
uvicorn apps.api.main:app --reload --host 0.0.0.0 --port 8000
```

## One full request flow (exact commands)

```bash
curl -sS -X POST 'http://127.0.0.1:8000/ingest/frame?customer_id=customer-a&run_id=run-prod' \
  -H 'Content-Type: application/json' \
  -d '{"timestamp":"2026-01-01T00:00:00+00:00","site_id":"plant-1","asset_id":"pump-9","sensor_values":{"pressure":50.0,"temperature":80.0}}' | jq

curl -sS 'http://127.0.0.1:8000/state?customer_id=customer-a&run_id=run-prod' | jq
curl -sS 'http://127.0.0.1:8000/history?customer_id=customer-a&run_id=run-prod&limit=5' | jq
curl -sS 'http://127.0.0.1:8000/decision?customer_id=customer-a&run_id=run-prod' | jq
curl -sS 'http://127.0.0.1:8000/explanation?customer_id=customer-a&run_id=run-prod' | jq
curl -sS 'http://127.0.0.1:8000/events/latest?customer_id=customer-a&run_id=run-prod' | jq
```
