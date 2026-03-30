# Ingestion Production Runbook

## 1) Ingestion architecture (operator-facing)

All external ingest entrypoints are normalized into one canonical internal frame contract before the SII core is called.

- `/ingest` and `/ingest/frame` normalize external JSON aliases into canonical frame fields.
- `/ingest/batch` normalizes each row independently and keeps partial-success semantics.
- `/ingest/json` and `/ingest/canonical` normalize to canonical frames via shared normalization utilities.
- `/ingest/csv/preview` infers semantic mapping (timestamp / asset / site / sensor columns) and returns guidance before full ingest.
- `/ingest/csv/upload` streams CSV to a temp file, queues an ingest job, and reports job lifecycle status.

## 2) Canonical payload contract

Canonical records consumed by ingest processing are expected to contain:

- `timestamp` (ISO-8601)
- `asset_id`
- `site_id` (optional; defaults are applied when missing)
- `sensor_values` (numeric map)
- `customer_id` / `run_id` when provided by request scope

Invalid records are blocked before they can enter SII processing.

## 3) Preview vs ingest lifecycle

CSV flow:

1. **Preview** (`/ingest/csv/preview`): parse headers/sample rows, infer mapping, return warnings/issues.
2. **Operator confirmation**: operator can adjust mapping for ambiguous schema.
3. **Upload** (`/ingest/csv/upload`): stream file and start ingest job.
4. **Ingest job polling** (`/ingest/jobs/{job_id}`): monitor queued/processing/completed/partial_success/failed.

## 4) Structured error envelope

All API failures should return a structured envelope:

- `ok` (always `false` for failures)
- `stage` (`preview`, `normalize`, `ingest`, `integration_pull`, or `request`)
- `type`
- `message`
- `actionable_detail`
- `issue_details` / `warning_details`
- `correlation_id` for support/debug correlation

For request-schema failures, `type=validation_error` and `issue_details` includes field-level validation hints.

## 5) Operator-visible failure categories

- `validation_error`: request shape/content problem (fix payload/mapping/file format)
- `csv_preview_empty_or_missing_header`: preview could not locate a valid header row
- `http_error`: explicit API-level rejection (unsupported file type, missing run, etc.)
- `internal_error`: unexpected server-side failure (retry, then escalate with `correlation_id`)

## 6) Runbook: CSV ingest preview fails with HTTP 422

Symptoms:

- Preview call returns `422 Unprocessable Entity`
- UI previously displayed opaque content like `[object Object]`

Actions:

1. Inspect response JSON and capture `correlation_id`.
2. Check `issue_details` for the exact failing field and message.
3. Verify preview request body includes one of:
   - JSON: `csv_sample` (preferred) or legacy `csv_text`
   - multipart form: `file` upload, or `csv_sample`/`csv_text`
4. If mapping is ambiguous, preview returns `requires_confirmation=true`; keep operator in review-blocked state until mapping is confirmed.
5. Retry preview, then proceed to upload only after preview returns headers and mapping guidance.

## 7) Logging and observability notes

Preview and validation paths should log:

- `correlation_id`
- `customer_id`
- ingest path (e.g., JSON vs multipart preview)
- parse issue counts / mapping issue counts

Use `correlation_id` from UI-visible errors to locate matching server logs quickly.
