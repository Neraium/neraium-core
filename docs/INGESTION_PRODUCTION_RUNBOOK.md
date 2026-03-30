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

Job payloads now include explicit lifecycle fields to prevent state ambiguity:

- `lifecycle_phase`: `uploading` → `queued` → `processing` → `terminal`
- `terminal_state`: one of `completed`, `partial_success`, `failed` (terminal only)
- `failure_category`: currently `ingest_failed` for terminal failures caused during ingest processing

## 4) Structured error envelope

All API failures should return a structured envelope:

- `type`
- `message`
- `actionable_detail`
- `detail`
- optional `issue_details`, `warning_details`
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
- Operator cannot progress from preview to upload state because mapping panel is not populated

Actions:

1. Inspect response JSON and capture `correlation_id`.
2. Check `issue_details` for the exact failing field and message.
3. Verify preview request body includes one of:
   - JSON: `csv_sample` (preferred) or legacy `csv_text`
   - multipart form: `file` upload, or `csv_sample`/`csv_text`
4. If mapping is ambiguous, continue via guided mapping UI (not a hard ingest failure).
5. Retry preview, then proceed to upload only after preview returns headers and mapping guidance.

## 7) Runbook: preview succeeds but upload/ingest fails

Symptoms:

- Preview returns headers and suggested mapping, but upload fails or job ends in `failed` / `partial_success`.
- Operator message includes actionable text and a `ref <correlation_id>`.

Actions:

1. **Capture correlation IDs from both phases.**
   - Preview errors use API validation IDs.
   - Upload errors return stream/mapping/file-type IDs and are echoed in `X-Correlation-ID`.
2. **Validate mapping continuity between preview and upload.**
   - Ensure `timestamp` and `asset_id` are still selected after any manual changes.
   - Ensure at least one sensor column is selected.
3. **Check ingest job status payload (`/ingest/jobs/{job_id}`).**
   - Inspect `rows_failed`, `error_samples`, and `message`.
   - Distinguish:
     - `failed`: zero successful rows.
     - `partial_success`: at least one row succeeded; inspect failed rows before retry.
4. **Apply row-level remediation from `error_samples`.**
   - Common causes: non-numeric sensor text, invalid timestamp values, or malformed row column counts.
5. **Retry with narrowed file when needed.**
   - If stream limits or transport instability occurs, split large CSVs and retry smaller chunks.
   - A preview-blocked request (mapping/header issue before upload) is not an ingest job failure; treat it separately.
6. **Escalation checklist for support/on-call.**
   - Provide correlation ID(s), run ID, customer ID, job ID, and a sanitized sample of failed rows.
   - Confirm health endpoint state (`/health`) for persistence/runtime degradation before deeper investigation.

## 8) FD001 replay/validation expectations

- `flatten_validation_result` intentionally includes both raw and smoothed confidence fields:
  - `decision_confidence_raw` and `decision_confidence`
  - `top_hypothesis_confidence_raw` and `top_hypothesis_confidence`
- Replay summaries are expected to keep per-row detail (`unit_id`, `cycle`, `row_index`) separate from per-unit milestone outputs (`first_*_cycle`, `max_cycle_observed`).
- For smoke checks, confirm:
  - replay returns per-row `decision` and `risk_assessment`
  - summary row confidence values are numeric
  - milestone rows remain unit-level only

## 9) Logging and observability notes

Preview and validation paths should log:

- `correlation_id`
- `customer_id`
- ingest path (e.g., JSON vs multipart preview)
- parse issue counts / mapping issue counts

Use `correlation_id` from UI-visible errors to locate matching server logs quickly.
