"""
Canonical ingest job payload fixtures for UI contract tests.

Each fixture is the exact JSON shape returned by GET /ingest/jobs/{job_id}
(i.e. the output of IngestJobsManager.public_ingest_job()) for a specific
scenario. These fixtures lock the contract between the backend and the
ingestion review panel in the frontend UI.

Frontend rendering decisions driven by these payloads (setUploadProgressUI):
  status="completed"       → title "Ingest completed",    progress bar 100%
  status="partial_success" → title "Ingest partial success", bar 100%, errors shown
  status="failed"          → title "Ingest failed",        bar 100%, errors shown
  status="uploading"       → title "Uploading CSV",        bar from bytes
  status="queued"/"processing" → title "Queued for ingest"/"Processing rows", bar 100%
  error_samples non-empty  → #uploadProgressErrors visible, ≤4 items rendered
  error_samples empty/None → #uploadProgressErrors hidden
"""
from __future__ import annotations

from typing import Any

_BASE_TERMINAL: dict[str, Any] = {
    "job_id": "test-job-001",
    "run_id": "run-abc",
    "customer_id": "customer-a",
    "filename": "telemetry.csv",
    "created_at": "2026-01-01T00:00:00+00:00",
    "updated_at": "2026-01-01T00:01:00+00:00",
    "upload_bytes_received": 2048,
    "upload_bytes_total": 2048,
    "message": None,
    "latest_result": None,
    "lifecycle_phase": "terminal",
}


# 1. Full success: all rows processed, no failures.
#    Frontend: title="Ingest completed", errors hidden, rows "50 processed · 50 succeeded · 0 failed"
FULL_SUCCESS: dict[str, Any] = {
    **_BASE_TERMINAL,
    "status": "completed",
    "rows_processed": 50,
    "rows_succeeded": 50,
    "rows_failed": 0,
    "partial_success": False,
    "error_samples": [],
    "ui_state": "completed",
    "terminal_state": "completed",
    "failure_category": None,
    "message": "CSV ingested (50 rows processed).",
}


# 2. Partial success: some rows failed, some succeeded.
#    Frontend: title="Ingest partial success", error_samples section visible.
PARTIAL_SUCCESS: dict[str, Any] = {
    **_BASE_TERMINAL,
    "status": "partial_success",
    "rows_processed": 50,
    "rows_succeeded": 40,
    "rows_failed": 10,
    "partial_success": True,
    "error_samples": [
        {"row": 3, "message": "Row 3: invalid timestamp format"},
        {"row": 7, "message": "Row 7: sensor value out of range"},
        {"row": 12, "message": "Row 12: missing asset_id"},
    ],
    "ui_state": "partial_success",
    "terminal_state": "partial_success",
    "failure_category": None,
    "message": "CSV ingest partial success (40 succeeded, 10 failed).",
}


# 3. All failed: every row failed, zero succeeded.
#    Frontend: title="Ingest failed", failure_category="ingest_failed".
ALL_FAILED: dict[str, Any] = {
    **_BASE_TERMINAL,
    "status": "failed",
    "rows_processed": 10,
    "rows_succeeded": 0,
    "rows_failed": 10,
    "partial_success": False,
    "error_samples": [
        {"row": 1, "message": "Row 1: header mismatch — no recognizable sensor columns"},
        {"row": 2, "message": "Row 2: header mismatch — no recognizable sensor columns"},
        {"row": 3, "message": "Row 3: header mismatch — no recognizable sensor columns"},
    ],
    "ui_state": "failed",
    "terminal_state": "failed",
    "failure_category": "ingest_failed",
    "message": "CSV ingest failed (10 rows failed).",
}


# 4. Empty input: CSV uploaded with only a header row and no data rows.
#    Frontend: title="Ingest completed", rows show 0/0/0, errors hidden.
EMPTY_INPUT: dict[str, Any] = {
    **_BASE_TERMINAL,
    "status": "completed",
    "rows_processed": 0,
    "rows_succeeded": 0,
    "rows_failed": 0,
    "partial_success": False,
    "error_samples": [],
    "ui_state": "completed",
    "terminal_state": "completed",
    "failure_category": None,
    "message": "CSV ingested (0 rows processed).",
    "upload_bytes_received": 64,
    "upload_bytes_total": 64,
}


# 5. Structural failure envelope — NOT a job payload.
#    This is the raw error dict returned by the server on HTTP-level errors
#    (e.g. 422, 500). It has NO job_id and NO status field.
#    The frontend handles this in the XHR error path / catch block.
#    Tests verify: public_ingest_job() applied to this envelope produces
#    a safe output (defaults) rather than crashing.
STRUCTURAL_FAILURE_ENVELOPE: dict[str, Any] = {
    "error": "Fatal ingest error: internal processing failure",
}

# Also test the FastAPI structured error envelope shape:
STRUCTURED_ERROR_ENVELOPE: dict[str, Any] = {
    "ok": False,
    "stage": "ingest",
    "type": "internal_error",
    "message": "Unexpected server error during CSV ingest.",
    "actionable_detail": "Retry. If the problem persists, contact support with the correlation_id.",
    "correlation_id": "corr-abc-123",
    "issue_details": [],
    "warning_details": [],
}


# 6. Truncated errors: error_samples has 5 entries.
#    Backend stores up to DEFAULT_INGEST_JOB_MAX_ERROR_SAMPLES (25).
#    Frontend truncates display to 4 via slice(0, 4) in setUploadProgressUI.
#    Tests verify: API returns full list; truncation is UI-side only.
TRUNCATED_ERRORS: dict[str, Any] = {
    **_BASE_TERMINAL,
    "status": "partial_success",
    "rows_processed": 20,
    "rows_succeeded": 15,
    "rows_failed": 5,
    "partial_success": True,
    "error_samples": [
        {"row": 2, "message": "Row 2: invalid timestamp"},
        {"row": 5, "message": "Row 5: sensor value non-numeric"},
        {"row": 8, "message": "Row 8: missing asset_id"},
        {"row": 11, "message": "Row 11: site_id too long"},
        {"row": 14, "message": "Row 14: duplicate timestamp for asset"},
    ],
    "ui_state": "partial_success",
    "terminal_state": "partial_success",
    "failure_category": None,
    "message": "CSV ingest partial success (15 succeeded, 5 failed).",
}


# 7. Malformed / partial summary payload fallback.
#    Raw job dict with missing, null, or wrong-type fields.
#    public_ingest_job() must normalize without crashing and apply all defaults.
MALFORMED_PARTIAL_PAYLOAD: dict[str, Any] = {
    # status missing → "unknown"
    # rows fields missing → 0
    # error_samples missing → []
    # filename missing → "upload.csv"
    # upload_bytes_total null → None
    "job_id": "test-job-malformed",
    "run_id": None,
    "customer_id": None,
    "upload_bytes_total": None,
}


# All fixture payloads indexed for parametrized tests.
ALL_PAYLOADS: dict[str, dict[str, Any]] = {
    "full_success": FULL_SUCCESS,
    "partial_success": PARTIAL_SUCCESS,
    "all_failed": ALL_FAILED,
    "empty_input": EMPTY_INPUT,
    "truncated_errors": TRUNCATED_ERRORS,
    "malformed_partial_payload": MALFORMED_PARTIAL_PAYLOAD,
}
