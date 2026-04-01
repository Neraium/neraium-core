from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Body, Depends, File, Form, HTTPException, Query, Request, UploadFile, status
from starlette.responses import JSONResponse

from ..schemas.common import CanonicalOutputResponse, ResultsEnvelope
from ..schemas.ingest import (
    BatchIngestRequest,
    CsvColumnMappingPayload,
    CsvIngestRequest,
    CsvPreviewResponse,
    IngestFrameRequest,
    IngestJobEnvelope,
    IngestRequest,
)
from .dependencies import IngestRouterDependencies

logger = logging.getLogger(__name__)


def build_ingest_router(
    *,
    deps: IngestRouterDependencies,
) -> APIRouter:
    router = APIRouter(tags=["ingest"])

    def _failure(
        *,
        stage: str,
        type_name: str,
        message: str,
        actionable_detail: str,
        correlation_id: str,
        issue_details: list[dict[str, Any]] | None = None,
        warning_details: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        return {
            "ok": False,
            "stage": stage,
            "type": type_name,
            "message": message,
            "actionable_detail": actionable_detail,
            "issue_details": list(issue_details or []),
            "warning_details": list(warning_details or []),
            "correlation_id": correlation_id,
        }

    @router.post("/ingest", response_model=ResultsEnvelope)
    def ingest(payload: IngestRequest, _: None = Depends(deps.require_api_key), run_id: str | None = Query(default=None), customer_id: str | None = Query(default=None)) -> dict[str, Any]:
        try:
            resolved_customer = deps.resolve_customer_id(customer_id or payload.customer_id)
            resolved = deps.resolve_run_id_with_default(deps.service_instance, run_id, customer_id=resolved_customer)
            normalized_payload = deps.normalize_external_payload(payload.model_dump(exclude_none=True), customer_id=resolved_customer)
            result = deps.service_instance.ingest_payload(normalized_payload, run_id=resolved, customer_id=resolved_customer)
            previous_result = None
            recent = deps.service_instance.list_recent_results(limit=2, run_id=resolved, customer_id=resolved_customer)
            if len(recent) >= 2:
                previous_result = recent[1]
            deps.process_alerts_after_ingest(customer_id=resolved_customer, run_id=resolved, latest_result=result, previous_result=previous_result)
            return deps.results_envelope([result], latest=result)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=deps.actionable_validation_detail(str(exc))) from exc

    @router.post("/ingest/frame", response_model=CanonicalOutputResponse)
    def ingest_frame(payload: IngestFrameRequest, _: None = Depends(deps.require_api_key), run_id: str | None = Query(default=None), customer_id: str | None = Query(default=None)) -> dict[str, Any]:
        try:
            resolved_customer = deps.resolve_customer_id(customer_id or payload.customer_id)
            resolved_run = deps.resolve_run_id_with_default(deps.service_instance, run_id, customer_id=resolved_customer)
            normalized_payload = deps.normalize_external_payload(payload.model_dump(exclude_none=True), customer_id=resolved_customer)
            return deps.service_instance.ingest_frame(normalized_payload, run_id=resolved_run, customer_id=resolved_customer)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=deps.actionable_validation_detail(str(exc))) from exc

    @router.post("/ingest/batch", response_model=ResultsEnvelope)
    def ingest_batch(payload: BatchIngestRequest, _: None = Depends(deps.require_api_key), run_id: str | None = Query(default=None), customer_id: str | None = Query(default=None)) -> dict[str, Any]:
        resolved_customer = deps.resolve_customer_id(customer_id or (payload.items[0].customer_id if payload.items else None))
        resolved = deps.resolve_run_id_with_default(deps.service_instance, run_id, customer_id=resolved_customer)
        normalized_rows = []
        errors = []
        for idx, item in enumerate(payload.items, start=1):
            try:
                normalized_rows.append(deps.normalize_external_payload(item.model_dump(exclude_none=True), customer_id=resolved_customer))
            except ValueError as exc:
                errors.append({"row": idx, "error": deps.actionable_validation_detail(str(exc))})
        if not normalized_rows:
            return JSONResponse(status_code=400, content={"status": "error", "message": "ingest failed", "detail": "No valid rows in batch.", "type": "validation_error", "actionable_detail": "Fix invalid rows and retry.", "errors": errors, "count": 0, "results": [], "latest": None})
        results = deps.service_instance.ingest_normalized_frames(normalized_rows, run_id=resolved, customer_id=resolved_customer)
        if results:
            recent = deps.service_instance.list_recent_results(limit=2, run_id=resolved, customer_id=resolved_customer)
            deps.process_alerts_after_ingest(customer_id=resolved_customer, run_id=resolved, latest_result=results[-1], previous_result=recent[1] if len(recent) >= 2 else None)
        env = deps.results_envelope(results, latest=results[-1] if results else None)
        env["status"] = "ok" if not errors else "partial_success"
        env["processed"] = len(results)
        env["run_id"] = resolved
        if errors:
            env["errors"] = errors
        return env

    @router.post("/ingest/json", response_model=ResultsEnvelope)
    def ingest_json(payload: dict[str, Any] | list[dict[str, Any]], _: None = Depends(deps.require_api_key), run_id: str | None = Query(default=None), customer_id: str | None = Query(default=None)) -> dict[str, Any]:
        try:
            payload_customer = payload.get("customer_id") if isinstance(payload, dict) else None
            resolved_customer = deps.resolve_customer_id(customer_id or payload_customer)
            resolved = deps.resolve_run_id_with_default(deps.service_instance, run_id, customer_id=resolved_customer)
            frames, _ = deps.normalize_external_batch_payload(payload, customer_id=resolved_customer)
            results = deps.service_instance.ingest_normalized_frames(frames, run_id=resolved, customer_id=resolved_customer)
            return deps.results_envelope(results, latest=results[-1] if results else None)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=deps.actionable_validation_detail(str(exc))) from exc

    @router.post("/ingest/canonical", response_model=ResultsEnvelope)
    def ingest_canonical(payload: dict[str, Any] | list[dict[str, Any]], _: None = Depends(deps.require_api_key), run_id: str | None = Query(default=None), customer_id: str | None = Query(default=None)) -> dict[str, Any]:
        try:
            payload_customer = payload.get("customer_id") if isinstance(payload, dict) else None
            resolved_customer = deps.resolve_customer_id(customer_id or payload_customer)
            resolved = deps.resolve_run_id_with_default(deps.service_instance, run_id, customer_id=resolved_customer)
            frames = deps.normalize_canonical_records_payload(payload, customer_id=resolved_customer)
            results = deps.service_instance.ingest_normalized_frames(frames, run_id=resolved, customer_id=resolved_customer)
            return deps.results_envelope(results, latest=results[-1] if results else None)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=deps.actionable_validation_detail(str(exc))) from exc

    @router.post("/ingest/csv", response_model=ResultsEnvelope)
    def ingest_csv(payload: CsvIngestRequest, _: None = Depends(deps.require_api_key), run_id: str | None = Query(default=None), customer_id: str | None = Query(default=None)) -> dict[str, Any]:
        try:
            resolved_customer = deps.resolve_customer_id(customer_id or payload.customer_id)
            resolved = deps.resolve_run_id_with_default(deps.service_instance, run_id, customer_id=resolved_customer)
            results = deps.service_instance.ingest_csv(payload.csv_text, run_id=resolved, customer_id=resolved_customer, column_mapping=payload.column_mapping.model_dump() if payload.column_mapping else None)
            return deps.results_envelope(results, latest=results[-1] if results else None)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=deps.actionable_validation_detail(str(exc))) from exc

    @router.post("/ingest/csv/preview", response_model=CsvPreviewResponse)
    async def ingest_csv_preview(
        request: Request,
        payload: Any | None = Body(default=None),
        file: UploadFile | None = File(default=None),
        csv_sample: str | None = Form(default=None),
        csv_text: str | None = Form(default=None),
        json_payload: Any | None = Body(default=None),
        _: None = Depends(deps.require_api_key),
        customer_id: str | None = Query(default=None),
    ) -> dict[str, Any]:
        resolved_customer = deps.resolve_customer_id(customer_id)
        correlation_id = str(getattr(request.state, "correlation_id", "") or f"ing_prev_{uuid4().hex[:12]}")
        ingest_path = "csv_preview"
        body_sample: str | None = None
        logger.info(
            "ingest_csv_preview_received",
            extra={
                "correlation_id": correlation_id,
                "customer_id": customer_id or "default-customer",
                "content_type": request.headers.get("content-type", ""),
            },
        )
        if file is not None:
            preview_bytes = await file.read(524_288)
            body_sample = preview_bytes.decode("utf-8", errors="replace")
            ingest_path = "csv_preview_multipart_file"
        elif csv_sample is not None or csv_text is not None:
            body_sample = str(csv_sample if csv_sample is not None else csv_text)
            ingest_path = "csv_preview_form"
        elif isinstance(payload, dict):
            raw_csv_sample = payload.get("csv_sample")
            raw_csv_text = payload.get("csv_text")
            if raw_csv_sample is not None:
                body_sample = str(raw_csv_sample)
                ingest_path = "csv_preview_json"
            elif raw_csv_text is not None:
                body_sample = str(raw_csv_text)
                ingest_path = "csv_preview_json_legacy_csv_text"
        else:
            try:
                raw = await request.json()
            except Exception:
                raw = None
            if isinstance(raw, dict):
                raw_csv_sample = raw.get("csv_sample")
                raw_csv_text = raw.get("csv_text")
                if raw_csv_sample is not None:
                    body_sample = str(raw_csv_sample)
                elif raw_csv_text is not None:
                    body_sample = str(raw_csv_text)
                    ingest_path = "csv_preview_json_legacy_csv_text"
        logger.info(
            "ingest_csv_preview_received",
            extra={
                "correlation_id": correlation_id,
                "request_id": correlation_id,
                "ingest_path": ingest_path,
                "customer_id": resolved_customer,
                "filename": str(getattr(file, "filename", "") or ""),
                "stage": "preview_received",
            },
        )
        if body_sample is None or not str(body_sample).strip():
            logger.warning(
                "ingest_csv_preview_invalid_request",
                extra={
                    "correlation_id": correlation_id,
                    "ingest_path": ingest_path,
                    "customer_id": resolved_customer,
                },
            )
            raise HTTPException(
                status_code=400,
                detail=_failure(
                    stage="preview",
                    type_name="validation_error",
                    message="CSV preview request is missing CSV content.",
                    actionable_detail="Send csv_sample/csv_text JSON, or upload a CSV file in multipart form.",
                    correlation_id=correlation_id,
                ),
            )
        headers, rows, parse_issues = deps.parse_csv_rows(body_sample)
        logger.info(
            "ingest_csv_preview_parsed",
            extra={
                "correlation_id": correlation_id,
                "ingest_path": ingest_path,
                "customer_id": resolved_customer,
                "header_count": len(headers),
                "sample_row_count": len(rows),
                "parse_issue_count": len(parse_issues),
                "stage": "file_parsed",
            },
        )
        if not headers:
            raise HTTPException(
                status_code=400,
                detail=_failure(
                    stage="preview",
                    type_name="csv_preview_empty_or_missing_header",
                    message="CSV preview could not find a header row.",
                    actionable_detail="Ensure the first row contains column names and retry preview.",
                    issue_details=[deps.issue_to_dict(i) for i in parse_issues],
                    correlation_id=correlation_id,
                ),
            )
        stage = deps.infer_csv_mapping_stage(headers, rows=rows, column_mapping=None)
        mapping = stage.mapping
        mapping_issues = deps.validate_csv_mapping_stage(mapping, headers) if mapping is not None else []
        hard_issues = [*parse_issues, *stage.issues, *mapping_issues]
        warning_issues = list(stage.warnings)
        preview_state = "preview_ready"
        logger.info(
            "ingest_csv_preview_mapping_inferred",
            extra={
                "correlation_id": correlation_id,
                "ingest_path": ingest_path,
                "customer_id": customer_id or "default-customer",
                "requires_confirmation": bool(mapping is None or stage.requires_confirmation),
                "issue_count": len(hard_issues),
                "warning_count": len(warning_issues),
            },
        )
        if mapping is None or stage.requires_confirmation:
            logger.warning(
                "ingest_csv_preview_mapping_blocked",
                extra={
                    "correlation_id": correlation_id,
                    "customer_id": customer_id or "default-customer",
                    "error_type": "ambiguous_mapping",
                    "issue_count": len(hard_issues),
                },
            )
        return CsvPreviewResponse(
            headers=headers,
            suggested_mapping=mapping.to_dict() if mapping else None,
            issues=[i.message for i in hard_issues],
            warnings=[i.message for i in warning_issues],
            issue_details=[deps.issue_to_dict(i) for i in hard_issues],
            warning_details=[deps.issue_to_dict(i) for i in warning_issues],
            requires_confirmation=(mapping is None or stage.requires_confirmation),
            preview_state=preview_state,
        )

    @router.post("/ingest/csv/upload", response_model=IngestJobEnvelope)
    async def ingest_csv_upload(request: Request, file: UploadFile = File(...), _: None = Depends(deps.require_api_key), run_id: str | None = Query(default=None), customer_id: str | None = Query(default=None), mapping: str | None = Form(default=None)) -> dict[str, Any]:
        correlation_id = str(getattr(request.state, "correlation_id", "") or f"ing_up_{uuid4().hex[:12]}")
        filename = str(file.filename or "upload.csv")
        if not filename.lower().endswith(".csv"):
            raise HTTPException(
                status_code=400,
                detail={
                    "type": "csv_upload_invalid_file_type",
                    "message": "Upload must be a .csv file.",
                    "actionable_detail": "Rename or export the telemetry file as .csv and retry.",
                    "correlation_id": correlation_id,
                },
            )
        resolved_customer = deps.resolve_customer_id(customer_id)
        resolved_run = deps.resolve_run_id_with_default(deps.service_instance, run_id, customer_id=resolved_customer)
        column_mapping = None
        if mapping:
            try:
                parsed = json.loads(mapping)
                column_mapping = CsvColumnMappingPayload.model_validate(parsed).model_dump()
            except Exception as exc:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "type": "csv_upload_invalid_mapping",
                        "message": "Invalid column mapping for CSV upload.",
                        "actionable_detail": "Review timestamp/entity/sensor mapping selections and retry.",
                        "detail": str(exc),
                        "correlation_id": correlation_id,
                    },
                ) from exc
        content_length = deps.normalize_content_length(request)
        if content_length is not None and content_length > deps.request_body_limit:
            max_mb = deps.request_body_limit / (1024 * 1024)
            raise HTTPException(status_code=status.HTTP_413_CONTENT_TOO_LARGE, detail=f"Request body too large (max {max_mb:.1f}MB).")
        fd, temp_path = tempfile.mkstemp(prefix="neraium_ingest_", suffix=".csv")
        os.close(fd)
        job_id = f"ingest_{uuid4().hex[:16]}"
        created_at = deps.utc_now_iso()
        initial_job = {
            "job_id": job_id,
            "status": "uploading",
            "run_id": resolved_run,
            "customer_id": resolved_customer,
            "filename": filename,
            "created_at": created_at,
            "updated_at": created_at,
            "rows_processed": 0,
            "rows_succeeded": 0,
            "rows_failed": 0,
            "partial_success": False,
            "upload_bytes_received": 0,
            "upload_bytes_total": content_length,
            "error_samples": [],
            "message": "Upload started.",
            "latest_result": None,
            "lifecycle_phase": "uploading",
            "ui_state": "uploading",
            "terminal_state": None,
            "failure_category": None,
        }
        with deps.ingest_jobs_lock:
            deps.ingest_jobs[job_id] = initial_job
        deps.persist_operational_state(f"ingest_job:{job_id}", initial_job, customer_id=resolved_customer, run_id=resolved_run)
        try:
            bytes_received = await deps.stream_upload_to_tempfile(file, Path(temp_path), job_id)
        except Exception as exc:
            Path(temp_path).unlink(missing_ok=True)
            deps.update_ingest_job(job_id, status="failed", message=f"Upload failed: {exc}")
            logger.exception(
                "ingest_csv_upload_stream_failed",
                extra={"correlation_id": correlation_id, "job_id": job_id, "run_id": resolved_run, "customer_id": resolved_customer},
            )
            raise HTTPException(
                status_code=400,
                detail={
                    "type": "csv_upload_stream_error",
                    "message": "Failed to read upload stream.",
                    "actionable_detail": "Retry upload. If this persists, split the CSV file and retry.",
                    "correlation_id": correlation_id,
                },
            ) from exc
        deps.update_ingest_job(job_id, status="queued", upload_bytes_received=bytes_received, upload_bytes_total=content_length if content_length is not None else bytes_received, message=f"Upload complete ({bytes_received} bytes). Queueing ingest job.")
        logger.info(
            "ingest_csv_upload_queued",
            extra={
                "correlation_id": correlation_id,
                "request_id": correlation_id,
                "job_id": job_id,
                "run_id": resolved_run,
                "customer_id": resolved_customer,
                "filename": filename,
                "ingest_path": "csv_upload",
                "stage": "ingest_started",
            },
        )
        deps.start_ingest_job_worker(job_id=job_id, temp_path=temp_path, run_id=resolved_run, customer_id=resolved_customer, column_mapping=column_mapping)
        with deps.ingest_jobs_lock:
            job = dict(deps.ingest_jobs[job_id])
        return deps.public_ingest_job(job)

    @router.get("/ingest/jobs/{job_id}", response_model=IngestJobEnvelope)
    def get_ingest_job(job_id: str, customer_id: str | None = Query(default=None)) -> dict[str, Any]:
        resolved_customer = deps.resolve_customer_id(customer_id)
        with deps.ingest_jobs_lock:
            job = deps.ingest_jobs.get(job_id)
            if job is None or deps.resolve_customer_id(job.get("customer_id")) != resolved_customer:
                raise HTTPException(status_code=404, detail=f"Unknown ingest job: {job_id}")
            return deps.public_ingest_job(job)

    return router
