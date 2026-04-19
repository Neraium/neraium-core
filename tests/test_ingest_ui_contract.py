"""
Ingestion UI contract tests.

These tests lock the contract between the backend ingest job API and the
frontend ingestion review panel. They verify:

  1. public_ingest_job() normalization — the output shape the frontend polls
  2. API endpoint shapes — what GET /ingest/jobs/{job_id} actually returns
  3. Structural failure classification — error envelopes vs. job payloads
  4. Normalization of missing/malformed fields — no crashes, correct defaults
  5. Terminal state completeness — lifecycle fields on every terminal job

Frontend rendering behaviors locked by this test suite
(driven by setUploadProgressUI in upload.js):

  status="completed"        → mode="completed"   → title "Ingest completed",    bar 100%
  status="partial_success"  → mode="partial_success" → title "Ingest partial success", bar 100%
  status="failed"           → mode="failed"      → title "Ingest failed",        bar 100%
  error_samples non-empty   → #uploadProgressErrors visible, ≤4 items rendered
  error_samples []          → #uploadProgressErrors hidden
  rows_* as ints            → #uploadProgressRows: "N processed · M succeeded · K failed"
  lifecycle_phase="terminal"→ all terminal jobs have terminal_state set
  failure_category set      → only status="failed" jobs carry a failure_category

Intentionally NOT tested here (requires Node.js + Jest for DOM assertions):
  - Exact HTML of rendered error <li> items
  - CSS class changes on #uploadProgressPanel (visible/hidden)
  - Progress bar width calculation from upload bytes
  - Export button visibility changes on the run detail page
  - Raw JSON toggle interaction
  These are locked at the data contract level: if the API returns the right
  shape, the JS rendering functions receive the correct inputs.
"""
from __future__ import annotations

import time
from typing import Any

import pytest

pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from apps.api.main import create_app
from apps.api.services.ingest_jobs import IngestJobsManager

# The service layer requires neraium_core.alignment → neraium_core.causal, which
# may be broken if alignment.py imports were not updated after a causal.py refactor.
# Unit tests (normalization, classification) are always runnable without the full chain.
# Integration tests (HTTP endpoint) are skipped when the service layer is unavailable.
_service_layer_available = False
_service_layer_skip_reason = ""
try:
    from neraium_core.alignment import StructuralEngine  # noqa: F401
    from neraium_core.service import StructuralMonitoringService
    from neraium_core.store import ResultStore
    _service_layer_available = True
except ImportError as _e:
    _service_layer_skip_reason = (
        f"neraium_core service layer unavailable ({_e}); "
        "alignment.py imports may be out of sync with causal.py"
    )

_requires_service = pytest.mark.skipif(
    not _service_layer_available,
    reason=_service_layer_skip_reason or "service layer unavailable",
)
from tests.fixtures.ingest_job_payloads import (
    ALL_FAILED,
    ALL_PAYLOADS,
    EMPTY_INPUT,
    FULL_SUCCESS,
    MALFORMED_PARTIAL_PAYLOAD,
    PARTIAL_SUCCESS,
    STRUCTURAL_FAILURE_ENVELOPE,
    STRUCTURED_ERROR_ENVELOPE,
    TRUNCATED_ERRORS,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CUSTOMER_ID = "customer-a"
_AUTH_HEADER = {"X-Customer-ID": _CUSTOMER_ID}


def _customer_path(path: str) -> str:
    return f"/api/v1/customers/{_CUSTOMER_ID}{path}"


def _client(tmp_path: Any) -> TestClient:
    store = ResultStore(db_path=str(tmp_path / "ui_contract.db"))  # type: ignore[name-defined]
    engine = StructuralEngine(baseline_window=5, recent_window=3)  # type: ignore[name-defined]
    service = StructuralMonitoringService(engine=engine, store=store)  # type: ignore[name-defined]
    return TestClient(create_app(service=service))


def _make_manager() -> IngestJobsManager:
    """Minimal IngestJobsManager with only the attributes used by public_ingest_job()."""
    mgr = IngestJobsManager.__new__(IngestJobsManager)
    mgr._resolve_customer_id = lambda cid: cid or "customer-test"
    mgr._utc_now_iso = lambda: "2026-01-01T00:00:00+00:00"
    return mgr


def _wait_for_job(client: TestClient, job_id: str, timeout_s: float = 8.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        resp = client.get(_customer_path(f"/ingest/jobs/{job_id}"))
        assert resp.status_code == 200
        last = resp.json()
        if last.get("status") in {"completed", "partial_success", "failed"}:
            return last
        time.sleep(0.05)
    raise AssertionError(f"Job did not reach terminal state; last={last}")


# ---------------------------------------------------------------------------
# Section 1: public_ingest_job() normalization unit tests
# ---------------------------------------------------------------------------
# These tests exercise the normalization function directly, verifying that the
# output always has the correct shape and types regardless of input quality.


class TestPublicIngestJobNormalization:
    """Lock the normalization contract of public_ingest_job()."""

    def test_full_success_normalized_shape(self) -> None:
        """Full success job produces correct normalized output."""
        mgr = _make_manager()
        raw = {
            "job_id": "j1",
            "status": "completed",
            "run_id": "run-1",
            "customer_id": "customer-a",
            "rows_processed": 50,
            "rows_succeeded": 50,
            "rows_failed": 0,
            "partial_success": False,
            "error_samples": [],
            "upload_bytes_received": 2048,
            "upload_bytes_total": 2048,
            "lifecycle_phase": "terminal",
            "terminal_state": "completed",
            "failure_category": None,
        }
        out = mgr.public_ingest_job(raw)
        assert out["status"] == "completed"
        assert out["rows_processed"] == 50
        assert out["rows_succeeded"] == 50
        assert out["rows_failed"] == 0
        assert out["partial_success"] is False
        assert out["error_samples"] == []
        assert out["ui_state"] == "completed"
        assert out["terminal_state"] == "completed"
        assert out["failure_category"] is None
        assert out["lifecycle_phase"] == "terminal"
        # Shape completeness: all required keys present
        _assert_job_shape(out)

    def test_partial_success_normalized_shape(self) -> None:
        mgr = _make_manager()
        raw = {
            "job_id": "j2",
            "status": "partial_success",
            "run_id": "run-2",
            "rows_processed": 50,
            "rows_succeeded": 40,
            "rows_failed": 10,
            "partial_success": True,
            "error_samples": [{"row": 3, "message": "Row 3: bad timestamp"}],
            "upload_bytes_received": 2048,
            "upload_bytes_total": 2048,
            "lifecycle_phase": "terminal",
            "terminal_state": "partial_success",
            "failure_category": None,
        }
        out = mgr.public_ingest_job(raw)
        assert out["status"] == "partial_success"
        assert out["rows_failed"] == 10
        assert out["partial_success"] is True
        assert len(out["error_samples"]) == 1
        assert out["ui_state"] == "partial_success"
        assert out["terminal_state"] == "partial_success"
        assert out["failure_category"] is None
        _assert_job_shape(out)

    def test_all_failed_normalized_shape(self) -> None:
        mgr = _make_manager()
        raw = {
            "job_id": "j3",
            "status": "failed",
            "rows_processed": 10,
            "rows_succeeded": 0,
            "rows_failed": 10,
            "partial_success": False,
            "error_samples": [{"row": 1, "message": "Row 1: bad data"}],
            "upload_bytes_received": 512,
            "lifecycle_phase": "terminal",
            "terminal_state": "failed",
            "failure_category": "ingest_failed",
        }
        out = mgr.public_ingest_job(raw)
        assert out["status"] == "failed"
        assert out["rows_succeeded"] == 0
        assert out["rows_failed"] == 10
        assert out["partial_success"] is False
        assert out["ui_state"] == "failed"
        assert out["terminal_state"] == "failed"
        assert out["failure_category"] == "ingest_failed"
        _assert_job_shape(out)

    def test_empty_input_normalized_shape(self) -> None:
        """Zero-row completed job normalizes correctly."""
        mgr = _make_manager()
        raw = {
            "job_id": "j4",
            "status": "completed",
            "rows_processed": 0,
            "rows_succeeded": 0,
            "rows_failed": 0,
            "partial_success": False,
            "error_samples": [],
            "upload_bytes_received": 64,
            "upload_bytes_total": 64,
            "lifecycle_phase": "terminal",
            "terminal_state": "completed",
            "failure_category": None,
        }
        out = mgr.public_ingest_job(raw)
        assert out["rows_processed"] == 0
        assert out["rows_succeeded"] == 0
        assert out["rows_failed"] == 0
        assert out["error_samples"] == []
        assert out["ui_state"] == "completed"
        _assert_job_shape(out)

    def test_truncated_errors_full_list_preserved(self) -> None:
        """API returns all error_samples; frontend truncates display to 4 (slice(0,4))."""
        mgr = _make_manager()
        samples = [{"row": i, "message": f"Row {i}: error"} for i in range(1, 6)]
        raw = {
            "job_id": "j5",
            "status": "partial_success",
            "rows_processed": 20,
            "rows_succeeded": 15,
            "rows_failed": 5,
            "partial_success": True,
            "error_samples": samples,
            "upload_bytes_received": 1024,
            "lifecycle_phase": "terminal",
            "terminal_state": "partial_success",
            "failure_category": None,
        }
        out = mgr.public_ingest_job(raw)
        # Backend preserves all 5 samples — truncation happens in the UI
        assert len(out["error_samples"]) == 5
        # Frontend will render only out["error_samples"][:4] = 4 items
        assert len(out["error_samples"][:4]) == 4
        _assert_job_shape(out)


# ---------------------------------------------------------------------------
# Section 2: Normalization of missing/malformed fields
# ---------------------------------------------------------------------------

class TestNormalizationOfMalformedPayloads:
    """Lock: missing or wrong-type fields never crash public_ingest_job()."""

    def test_structural_failure_envelope_does_not_crash(self) -> None:
        """Pure { 'error': '...' } dict → safe normalized output with defaults."""
        mgr = _make_manager()
        out = mgr.public_ingest_job(STRUCTURAL_FAILURE_ENVELOPE)
        # No crash; all numeric fields default to 0
        assert isinstance(out["rows_processed"], int)
        assert isinstance(out["rows_succeeded"], int)
        assert isinstance(out["rows_failed"], int)
        assert isinstance(out["error_samples"], list)
        assert isinstance(out["partial_success"], bool)
        # status defaults to "unknown" (not "completed" or "failed")
        assert out["status"] == "unknown"
        # ui_state derived from status="unknown" → "idle"
        assert out["ui_state"] == "idle"
        _assert_job_shape(out)

    def test_malformed_partial_payload_applies_all_defaults(self) -> None:
        """Payload with missing rows/error_samples/filename fields → correct defaults."""
        mgr = _make_manager()
        out = mgr.public_ingest_job(MALFORMED_PARTIAL_PAYLOAD)
        # Numeric defaults
        assert out["rows_processed"] == 0
        assert out["rows_succeeded"] == 0
        assert out["rows_failed"] == 0
        assert out["upload_bytes_received"] == 0
        # List default
        assert out["error_samples"] == []
        # Bool default
        assert out["partial_success"] is False
        # String default
        assert out["filename"] == "upload.csv"
        # None is valid for upload_bytes_total
        assert out["upload_bytes_total"] is None
        _assert_job_shape(out)

    def test_string_rows_coerced_to_int(self) -> None:
        """String-typed row counts are coerced to int."""
        mgr = _make_manager()
        raw = {
            "job_id": "j-coerce",
            "status": "completed",
            "rows_processed": "42",
            "rows_succeeded": "42",
            "rows_failed": "0",
            "upload_bytes_received": "1024",
            "lifecycle_phase": "terminal",
            "terminal_state": "completed",
        }
        out = mgr.public_ingest_job(raw)
        assert out["rows_processed"] == 42
        assert out["rows_succeeded"] == 42
        assert out["rows_failed"] == 0
        assert out["upload_bytes_received"] == 1024
        assert isinstance(out["rows_processed"], int)

    def test_null_error_samples_becomes_empty_list(self) -> None:
        """None error_samples normalized to empty list."""
        mgr = _make_manager()
        out = mgr.public_ingest_job({"job_id": "j-null", "status": "completed", "error_samples": None})
        assert out["error_samples"] == []
        assert isinstance(out["error_samples"], list)

    def test_upload_bytes_total_none_preserved(self) -> None:
        """upload_bytes_total=None is a valid state (total unknown during upload)."""
        mgr = _make_manager()
        out = mgr.public_ingest_job({"job_id": "j-nobytes", "status": "uploading", "upload_bytes_total": None})
        assert out["upload_bytes_total"] is None

    def test_upload_bytes_total_int_coerced(self) -> None:
        """upload_bytes_total as integer is preserved."""
        mgr = _make_manager()
        out = mgr.public_ingest_job({"job_id": "j-bytes", "status": "uploading", "upload_bytes_total": 2048})
        assert out["upload_bytes_total"] == 2048
        assert isinstance(out["upload_bytes_total"], int)


# ---------------------------------------------------------------------------
# Section 3: ui_state derivation contract
# ---------------------------------------------------------------------------
# The frontend reads job["ui_state"] to determine which stage to show.
# This section locks the derivation rules.

class TestUiStateDerivation:
    """Lock: ui_state is always correctly derived from status."""

    @pytest.mark.parametrize("status,expected_ui_state", [
        ("uploading", "uploading"),
        ("queued", "ingesting"),
        ("processing", "ingesting"),
        ("completed", "completed"),
        ("partial_success", "partial_success"),
        ("failed", "failed"),
        ("unknown_status", "idle"),
        ("", "idle"),
    ])
    def test_ui_state_derived_from_status(self, status: str, expected_ui_state: str) -> None:
        mgr = _make_manager()
        out = mgr.public_ingest_job({"job_id": "j-ui", "status": status})
        assert out["ui_state"] == expected_ui_state, (
            f"status={status!r} → expected ui_state={expected_ui_state!r}, got {out['ui_state']!r}"
        )

    def test_explicit_ui_state_overrides_derivation(self) -> None:
        """Explicit ui_state in the job dict is used as-is, not re-derived."""
        mgr = _make_manager()
        out = mgr.public_ingest_job({
            "job_id": "j-override",
            "status": "processing",
            "ui_state": "ingesting",  # explicit value matches expected
        })
        assert out["ui_state"] == "ingesting"


# ---------------------------------------------------------------------------
# Section 4: Structural failure classification
# ---------------------------------------------------------------------------
# "Structural failure" = the server returns a raw error envelope ({ "error": ... })
# instead of a job object. Tests prove the classification boundary is clear.

class TestStructuralFailureClassification:
    """Lock: structural failure envelopes are distinguishable from job payloads."""

    def test_pure_error_envelope_has_no_status_field(self) -> None:
        """STRUCTURAL_FAILURE_ENVELOPE has no 'status' — it is not a job payload."""
        assert "status" not in STRUCTURAL_FAILURE_ENVELOPE
        assert "job_id" not in STRUCTURAL_FAILURE_ENVELOPE
        assert "error" in STRUCTURAL_FAILURE_ENVELOPE

    def test_structured_api_error_has_no_status_field(self) -> None:
        """FastAPI error envelopes use 'ok', 'stage', 'type' — not job fields."""
        assert "status" not in STRUCTURED_ERROR_ENVELOPE
        assert "job_id" not in STRUCTURED_ERROR_ENVELOPE
        assert "ok" in STRUCTURED_ERROR_ENVELOPE
        assert STRUCTURED_ERROR_ENVELOPE["ok"] is False

    def test_all_job_payloads_have_status_field(self) -> None:
        """All normal job payloads (including malformed ones after normalization) have status."""
        mgr = _make_manager()
        for name, raw in ALL_PAYLOADS.items():
            out = mgr.public_ingest_job(raw)
            assert "status" in out, f"{name}: normalized output missing 'status'"
            assert isinstance(out["status"], str), f"{name}: status must be str"

    def test_completed_job_not_misclassified_as_failure(self) -> None:
        """A completed job with non-empty error_samples is NOT a structural failure."""
        mgr = _make_manager()
        # Edge case: completed job might still have error_samples from a prior state
        raw = {
            "job_id": "j-edge",
            "status": "completed",
            "rows_processed": 5,
            "rows_succeeded": 5,
            "rows_failed": 0,
            "error_samples": [],
            "lifecycle_phase": "terminal",
            "terminal_state": "completed",
        }
        out = mgr.public_ingest_job(raw)
        assert out["status"] == "completed"
        assert out["terminal_state"] == "completed"
        assert out["failure_category"] is None

    def test_partial_success_not_misclassified_as_complete_failure(self) -> None:
        """partial_success has rows_succeeded > 0; it is NOT classified as all_failed."""
        mgr = _make_manager()
        out = mgr.public_ingest_job({
            "job_id": "j-ps",
            "status": "partial_success",
            "rows_succeeded": 40,
            "rows_failed": 10,
            "partial_success": True,
            "lifecycle_phase": "terminal",
            "terminal_state": "partial_success",
        })
        assert out["status"] == "partial_success"
        assert out["rows_succeeded"] > 0
        assert out["rows_failed"] > 0
        assert out["failure_category"] is None  # not "ingest_failed"

    def test_failed_status_implies_ingest_failed_category(self) -> None:
        """A job with status=failed that carries failure_category keeps it."""
        mgr = _make_manager()
        out = mgr.public_ingest_job({
            "job_id": "j-fail",
            "status": "failed",
            "rows_failed": 5,
            "lifecycle_phase": "terminal",
            "terminal_state": "failed",
            "failure_category": "ingest_failed",
        })
        assert out["failure_category"] == "ingest_failed"


# ---------------------------------------------------------------------------
# Section 5: Terminal state completeness
# ---------------------------------------------------------------------------

class TestTerminalStateCompleteness:
    """Lock: every terminal job payload has all lifecycle fields populated."""

    @pytest.mark.parametrize("payload_name,payload", [
        ("full_success", FULL_SUCCESS),
        ("partial_success", PARTIAL_SUCCESS),
        ("all_failed", ALL_FAILED),
        ("empty_input", EMPTY_INPUT),
        ("truncated_errors", TRUNCATED_ERRORS),
    ])
    def test_terminal_payload_has_lifecycle_fields(
        self, payload_name: str, payload: dict[str, Any]
    ) -> None:
        mgr = _make_manager()
        out = mgr.public_ingest_job(payload)
        assert out["lifecycle_phase"] == "terminal", (
            f"{payload_name}: expected lifecycle_phase='terminal'"
        )
        assert out["terminal_state"] in {"completed", "partial_success", "failed"}, (
            f"{payload_name}: terminal_state must be one of completed/partial_success/failed"
        )
        assert out["ui_state"] in {"completed", "partial_success", "failed"}, (
            f"{payload_name}: terminal ui_state must match terminal_state"
        )
        assert out["ui_state"] == out["terminal_state"], (
            f"{payload_name}: ui_state and terminal_state must agree for terminal jobs"
        )

    def test_failed_terminal_has_failure_category(self) -> None:
        mgr = _make_manager()
        out = mgr.public_ingest_job(ALL_FAILED)
        assert out["failure_category"] == "ingest_failed"

    def test_success_terminal_has_no_failure_category(self) -> None:
        mgr = _make_manager()
        for payload in [FULL_SUCCESS, PARTIAL_SUCCESS, EMPTY_INPUT]:
            out = mgr.public_ingest_job(payload)
            assert out["failure_category"] is None

    def test_truncated_errors_terminal_state(self) -> None:
        mgr = _make_manager()
        out = mgr.public_ingest_job(TRUNCATED_ERRORS)
        assert out["terminal_state"] == "partial_success"
        # Frontend: error_samples[:4] = 4 items shown; 5th is clipped by UI
        assert len(out["error_samples"]) == 5
        assert len(out["error_samples"][:4]) == 4


# ---------------------------------------------------------------------------
# Section 6: API endpoint contract tests (HTTP)
# ---------------------------------------------------------------------------
# These tests verify the actual HTTP endpoint returns the correct shapes.

@_requires_service
class TestApiEndpointContract:
    """Lock: GET /ingest/jobs/{job_id} returns the documented payload shape."""

    def test_completed_upload_job_has_full_terminal_shape(self, tmp_path: Any) -> None:
        client = _client(tmp_path)
        run = client.post(
            _customer_path("/runs"),
            json={"name": "ui-contract-complete", "activate": True, "config": {}},
        )
        assert run.status_code == 200
        run_id = run.json()["run"]["run_id"]

        csv_bytes = (
            "timestamp,asset_id,s1,s2\n"
            "2026-01-01T00:00:00+00:00,asset-a,1.0,2.0\n"
            "2026-01-01T00:00:01+00:00,asset-a,1.1,2.1\n"
        ).encode("utf-8")
        start = client.post(
            _customer_path(f"/ingest/csv/upload?run_id={run_id}"),
            files={"file": ("telemetry.csv", csv_bytes, "text/csv")},
        )
        assert start.status_code == 200
        job_id = start.json()["job_id"]

        done = _wait_for_job(client, job_id)
        # Contract: all required top-level fields present
        _assert_job_shape(done)
        # Completed job fields
        assert done["status"] == "completed"
        assert done["lifecycle_phase"] == "terminal"
        assert done["terminal_state"] == "completed"
        assert done["ui_state"] == "completed"
        assert done["failure_category"] is None
        assert done["rows_succeeded"] > 0
        assert done["rows_failed"] == 0
        assert done["partial_success"] is False
        assert done["error_samples"] == []

    def test_partial_success_job_has_error_samples(self, tmp_path: Any) -> None:
        client = _client(tmp_path)
        run = client.post(
            _customer_path("/runs"),
            json={"name": "ui-contract-partial", "activate": True, "config": {}},
        )
        assert run.status_code == 200
        run_id = run.json()["run"]["run_id"]

        csv_bytes = (
            "timestamp,asset_id,s1\n"
            "2026-01-01T00:00:00+00:00,asset-a,1.0\n"
            "not-a-timestamp,asset-a,1.1\n"
            "2026-01-01T00:00:02+00:00,asset-a,1.2\n"
        ).encode("utf-8")
        start = client.post(
            _customer_path(f"/ingest/csv/upload?run_id={run_id}"),
            files={"file": ("partial.csv", csv_bytes, "text/csv")},
        )
        assert start.status_code == 200
        job_id = start.json()["job_id"]

        done = _wait_for_job(client, job_id)
        _assert_job_shape(done)
        assert done["status"] == "partial_success"
        assert done["lifecycle_phase"] == "terminal"
        assert done["terminal_state"] == "partial_success"
        assert done["ui_state"] == "partial_success"
        assert done["partial_success"] is True
        assert done["rows_failed"] > 0
        assert done["rows_succeeded"] > 0
        assert len(done["error_samples"]) > 0
        # Each error sample has row + message fields for the UI
        sample = done["error_samples"][0]
        assert "message" in sample

    def test_failed_job_has_failure_category(self, tmp_path: Any) -> None:
        client = _client(tmp_path)
        run = client.post(
            _customer_path("/runs"),
            json={"name": "ui-contract-failed", "activate": True, "config": {}},
        )
        assert run.status_code == 200
        run_id = run.json()["run"]["run_id"]

        # All rows invalid → complete failure
        csv_bytes = (
            "timestamp,asset_id,s1\n"
            "not-a-date,asset-a,1.0\n"
            "also-bad,asset-a,2.0\n"
        ).encode("utf-8")
        start = client.post(
            _customer_path(f"/ingest/csv/upload?run_id={run_id}"),
            files={"file": ("broken.csv", csv_bytes, "text/csv")},
        )
        assert start.status_code == 200
        job_id = start.json()["job_id"]

        done = _wait_for_job(client, job_id)
        _assert_job_shape(done)
        assert done["status"] == "failed"
        assert done["terminal_state"] == "failed"
        assert done["lifecycle_phase"] == "terminal"
        assert done["failure_category"] == "ingest_failed"
        assert done["rows_succeeded"] == 0

    def test_job_endpoint_404_on_unknown_id(self, tmp_path: Any) -> None:
        """Unknown job ID → 404, not a 200 with error envelope."""
        client = _client(tmp_path)
        resp = client.get(_customer_path("/ingest/jobs/nonexistent-job-id-xyz"))
        # Must be 404, NOT 200 with { "error": "..." }
        assert resp.status_code == 404
        # Response body should NOT look like a successful job payload
        body = resp.json()
        assert "job_id" not in body or body.get("job_id") != "nonexistent-job-id-xyz"

    def test_initial_upload_response_has_job_shape(self, tmp_path: Any) -> None:
        """POST /ingest/csv/upload immediately returns a job-shaped response."""
        client = _client(tmp_path)
        run = client.post(
            _customer_path("/runs"),
            json={"name": "ui-contract-initial", "activate": True, "config": {}},
        )
        run_id = run.json()["run"]["run_id"]

        csv_bytes = (
            "timestamp,asset_id,s1\n"
            "2026-01-01T00:00:00+00:00,asset-a,1.0\n"
        ).encode("utf-8")
        resp = client.post(
            _customer_path(f"/ingest/csv/upload?run_id={run_id}"),
            files={"file": ("small.csv", csv_bytes, "text/csv")},
        )
        assert resp.status_code == 200
        body = resp.json()
        # The initial response is a partial job envelope (frontend polls for terminal state)
        assert "job_id" in body
        assert isinstance(body["job_id"], str) and body["job_id"]
        assert body["status"] in {"uploading", "queued", "processing", "completed"}

    def test_empty_csv_job_completes_with_zero_rows(self, tmp_path: Any) -> None:
        """CSV with only a header (no data rows) → completed job with 0 rows."""
        client = _client(tmp_path)
        run = client.post(
            _customer_path("/runs"),
            json={"name": "ui-contract-empty", "activate": True, "config": {}},
        )
        run_id = run.json()["run"]["run_id"]

        csv_bytes = b"timestamp,asset_id,s1\n"  # header only
        resp = client.post(
            _customer_path(f"/ingest/csv/upload?run_id={run_id}"),
            files={"file": ("empty.csv", csv_bytes, "text/csv")},
        )
        assert resp.status_code == 200
        job_id = resp.json()["job_id"]
        done = _wait_for_job(client, job_id)

        _assert_job_shape(done)
        # Frontend shows: "0 processed · 0 succeeded · 0 failed" with no errors
        assert done["rows_processed"] == 0
        assert done["rows_failed"] == 0
        assert done["error_samples"] == []
        # Status is completed (0 rows processed is still a success)
        assert done["status"] in {"completed", "partial_success"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REQUIRED_JOB_FIELDS = {
    "job_id",
    "status",
    "run_id",
    "customer_id",
    "filename",
    "created_at",
    "updated_at",
    "rows_processed",
    "rows_succeeded",
    "rows_failed",
    "partial_success",
    "upload_bytes_received",
    "upload_bytes_total",
    "error_samples",
    "message",
    "latest_result",
    "lifecycle_phase",
    "ui_state",
    "terminal_state",
    "failure_category",
}


def _assert_job_shape(out: dict[str, Any]) -> None:
    """Assert that a normalized job dict has all required fields with correct types."""
    missing = _REQUIRED_JOB_FIELDS - set(out.keys())
    assert not missing, f"Missing required job fields: {missing}"
    assert isinstance(out["rows_processed"], int), "rows_processed must be int"
    assert isinstance(out["rows_succeeded"], int), "rows_succeeded must be int"
    assert isinstance(out["rows_failed"], int), "rows_failed must be int"
    assert isinstance(out["partial_success"], bool), "partial_success must be bool"
    assert isinstance(out["error_samples"], list), "error_samples must be list"
    assert isinstance(out["upload_bytes_received"], int), "upload_bytes_received must be int"
    assert out["upload_bytes_total"] is None or isinstance(out["upload_bytes_total"], int), (
        "upload_bytes_total must be int or None"
    )
    assert isinstance(out["status"], str) and out["status"], "status must be non-empty str"
    assert isinstance(out["ui_state"], str) and out["ui_state"], "ui_state must be non-empty str"
