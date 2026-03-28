# Operator-facing workflow (lightweight surface)

This workflow adds a minimal operator-facing page on top of the existing validated API/service layer.

## What it shows

The operator page presents the canonical service output in one place:

- current structural state (`timestamp`, `cycle`, `asset_id`, `site_id`, `run_id`)
- `risk_assessment`
- advisory-safe `operational_recommendation`
  - **Recommended next step**
  - **Why this is being recommended**
  - **Recommendation confidence**
  - **Operator note**
- `explanation_text`
- `events`
- `memory_recall`
  - novelty status
  - nearest match
  - top match summary list
- recent timeline (`risk`, recommendation availability/confidence, events, recalled vs novel)
- assistant interaction panel
  - current situation summary
  - explanation mode (`why_recommended`, `what_changed`, `pattern_similarity`)
  - operator handoff note
- Generate Report panel
  - report type selector (`client_report`, `technician_summary`, `inspection_brief`, `handoff_note`)
  - **Generate** action
  - **Copy to clipboard** action

## Run locally

1. Start the API:

```bash
uvicorn apps.api.main:app --reload --port 8000
```

2. Open the operator workflow page:

- `http://127.0.0.1:8000/operator`
- or `http://127.0.0.1:8000/operator/workflow`

3. In the page, set `customer_id` / `run_id`, then:

- click **Seed demo frames** (quick path), or
- ingest your own data first with API calls.

## End-to-end demo sequence

### Option A: browser flow

1. Open `/operator`.
2. Keep defaults (`customer-a`, `run-operator-demo`).
3. Click **Seed demo frames**.
4. Observe:
   - current state + risk assessment
   - advisory recommendation block
   - explanation text
   - events
   - memory recall + top matches
   - timeline rows

### Option B: CLI flow

Use the demo helper script to run ingest + retrieval in one command:

```bash
python tools/run_operator_workflow_demo.py --base-url http://127.0.0.1:8000 --customer-id customer-a --run-id run-operator-demo
```

The script ingests frames, prints current state fields, recommendation + explanation, memory recall, and recent history timeline rows.


## Example report outputs

### Client report (excerpt)

```text
Client Report

Overview
Client-ready advisory report for current Neraium state ...

Risk Assessment
risk_level=MEDIUM, trend=RISING, latest_instability=...
```

### Technician summary (excerpt)

```text
Technician Summary

Current state (concise)
risk_level=MEDIUM, trend=RISING, action=inspect pressure train, cycle=...
```

Intended use: contractor/operator communication and customer-safe status sharing, without changing engine logic or adding autonomous actions.
