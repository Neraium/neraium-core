from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile, status
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)


def build_ingest_router(
    *,
    service_instance: Any,
    require_api_key: Any,
    resolve_customer_id: Any,
    resolve_run_id_with_default: Any,
    actionable_validation_detail: Any,
    normalize_external_payload: Any,
    normalize_external_batch_payload: Any,
    normalize_canonical_records_payload: Any,
    process_alerts_after_ingest: Any,
    results_envelope: Any,
    parse_csv_rows: Any,
    infer_csv_mapping_stage: Any,
    validate_csv_mapping_stage: Any,
    issue_to_dict: Any,
    request_body_limit: int,
    normalize_content_length: Any,
    stream_upload_to_tempfile: Any,
    update_ingest_job: Any,
    public_ingest_job: Any,
    persist_operational_state: Any,
    start_ingest_job_worker: Any,
    ingest_jobs: dict[str, Any],
    ingest_jobs_lock: Any,
    models: Any,
) -> APIRouter:
    router = APIRouter(tags=["ingest"])

    @router.post("/ingest", response_model=models.ResultsEnvelope)
    def ingest(payload: models.IngestRequest, _: None = Depends(require_api_key), run_id: str | None = Query(default=None), customer_id: str | None = Query(default=None)) -> dict[str, Any]:
        try:
            resolved_customer = resolve_customer_id(customer_id or payload.customer_id)
            resolved = resolve_run_id_with_default(service_instance, run_id, customer_id=resolved_customer)
            normalized_payload = normalize_external_payload(payload.model_dump(exclude_none=True), customer_id=resolved_customer)
            result = service_instance.ingest_payload(normalized_payload, run_id=resolved, customer_id=resolved_customer)
            previous_result = None
            recent = service_instance.list_recent_results(limit=2, run_id=resolved, customer_id=resolved_customer)
            if len(recent) >= 2:
                previous_result = recent[1]
            process_alerts_after_ingest(customer_id=resolved_customer, run_id=resolved, latest_result=result, previous_result=previous_result)
            return results_envelope([result], latest=result)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=actionable_validation_detail(str(exc))) from exc

    @router.post("/ingest/frame", response_model=models.CanonicalOutputResponse)
    def ingest_frame(payload: models.IngestFrameRequest, _: None = Depends(require_api_key), run_id: str | None = Query(default=None), customer_id: str | None = Query(default=None)) -> dict[str, Any]:
        try:
            resolved_customer = resolve_customer_id(customer_id or payload.customer_id)
            resolved_run = resolve_run_id_with_default(service_instance, run_id, customer_id=resolved_customer)
            normalized_payload = normalize_external_payload(payload.model_dump(exclude_none=True), customer_id=resolved_customer)
            return service_instance.ingest_frame(normalized_payload, run_id=resolved_run, customer_id=resolved_customer)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=actionable_validation_detail(str(exc))) from exc

    @router.post("/ingest/batch", response_model=models.ResultsEnvelope)
    def ingest_batch(payload: models.BatchIngestRequest, _: None = Depends(require_api_key), run_id: str | None = Query(default=None), customer_id: str | None = Query(default=None)) -> dict[str, Any]:
        batch_size = len(payload.items)
        resolved_customer = resolve_customer_id(customer_id or (payload.items[0].customer_id if payload.items else None))
        resolved = resolve_run_id_with_default(service_instance, run_id, customer_id=resolved_customer)
        normalized_rows = []
        errors = []
        for idx, item in enumerate(payload.items, start=1):
            try:
                normalized_rows.append(normalize_external_payload(item.model_dump(exclude_none=True), customer_id=resolved_customer))
            except ValueError as exc:
                errors.append({"row": idx, "error": actionable_validation_detail(str(exc))})
        if not normalized_rows:
            return JSONResponse(status_code=400, content={"status": "error", "message": "ingest failed", "detail": "No valid rows in batch.", "type": "validation_error", "actionable_detail": "Fix invalid rows and retry.", "errors": errors, "count": 0, "results": [], "latest": None})
        results = service_instance.ingest_normalized_frames(normalized_rows, run_id=resolved, customer_id=resolved_customer)
        if results:
            recent = service_instance.list_recent_results(limit=2, run_id=resolved, customer_id=resolved_customer)
            process_alerts_after_ingest(customer_id=resolved_customer, run_id=resolved, latest_result=results[-1], previous_result=recent[1] if len(recent) >= 2 else None)
        env = results_envelope(results, latest=results[-1] if results else None)
        env["status"] = "ok" if not errors else "partial_success"
        env["processed"] = len(results)
        env["run_id"] = resolved
        if errors:
            env["errors"] = errors
        return env

    @router.post("/ingest/json", response_model=models.ResultsEnvelope)
    def ingest_json(payload: dict[str, Any] | list[dict[str, Any]], _: None = Depends(require_api_key), run_id: str | None = Query(default=None), customer_id: str | None = Query(default=None)) -> dict[str, Any]:
        try:
            payload_customer = payload.get("customer_id") if isinstance(payload, dict) else None
            resolved_customer = resolve_customer_id(customer_id or payload_customer)
            resolved = resolve_run_id_with_default(service_instance, run_id, customer_id=resolved_customer)
            frames, _ = normalize_external_batch_payload(payload, customer_id=resolved_customer)
            results = service_instance.ingest_normalized_frames(frames, run_id=resolved, customer_id=resolved_customer)
            return results_envelope(results, latest=results[-1] if results else None)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=actionable_validation_detail(str(exc))) from exc

    @router.post("/ingest/canonical", response_model=models.ResultsEnvelope)
    def ingest_canonical(payload: dict[str, Any] | list[dict[str, Any]], _: None = Depends(require_api_key), run_id: str | None = Query(default=None), customer_id: str | None = Query(default=None)) -> dict[str, Any]:
        try:
            payload_customer = payload.get("customer_id") if isinstance(payload, dict) else None
            resolved_customer = resolve_customer_id(customer_id or payload_customer)
            resolved = resolve_run_id_with_default(service_instance, run_id, customer_id=resolved_customer)
            frames = normalize_canonical_records_payload(payload, customer_id=resolved_customer)
            results = service_instance.ingest_normalized_frames(frames, run_id=resolved, customer_id=resolved_customer)
            return results_envelope(results, latest=results[-1] if results else None)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=actionable_validation_detail(str(exc))) from exc

    @router.post("/ingest/csv", response_model=models.ResultsEnvelope)
    def ingest_csv(payload: models.CsvIngestRequest, _: None = Depends(require_api_key), run_id: str | None = Query(default=None), customer_id: str | None = Query(default=None)) -> dict[str, Any]:
        try:
            resolved_customer = resolve_customer_id(customer_id or payload.customer_id)
            resolved = resolve_run_id_with_default(service_instance, run_id, customer_id=resolved_customer)
            results = service_instance.ingest_csv(payload.csv_text, run_id=resolved, customer_id=resolved_customer, column_mapping=payload.column_mapping.model_dump() if payload.column_mapping else None)
            return results_envelope(results, latest=results[-1] if results else None)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=actionable_validation_detail(str(exc))) from exc

    @router.post("/ingest/csv/preview", response_model=models.CsvPreviewResponse)
    async def ingest_csv_preview(
        request: Request,
        file: UploadFile | None = File(default=None),
        csv_sample: str | None = Form(default=None),
        csv_text: str | None = Form(default=None),
        _: None = Depends(require_api_key),
        customer_id: str | None = Query(default=None),
    ) -> dict[str, Any]:
        _ = resolve_customer_id(customer_id)
        correlation_id = f"ing_prev_{uuid4().hex[:12]}"
        ingest_path = "csv_preview"
        body_sample: str | None = None
        if file is not None:
            preview_bytes = await file.read(524_288)
            body_sample = preview_bytes.decode("utf-8", errors="replace")
            ingest_path = "csv_preview_multipart_file"
        elif csv_sample is not None or csv_text is not None:
            body_sample = str(csv_sample if csv_sample is not None else csv_text)
            ingest_path = "csv_preview_form"
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
        if body_sample is None or not str(body_sample).strip():
            logger.warning(
                "ingest_csv_preview_invalid_request",
                extra={
                    "correlation_id": correlation_id,
                    "ingest_path": ingest_path,
                    "customer_id": customer_id or "default-customer",
                },
            )
            raise HTTPException(
                status_code=400,
                detail={
                    "type": "validation_error",
                    "message": "CSV preview request is missing CSV content.",
                    "actionable_detail": "Send csv_sample/csv_text JSON, or upload a CSV file in multipart form.",
                    "correlation_id": correlation_id,
                },
            )
        headers, rows, parse_issues = parse_csv_rows(body_sample)
        logger.info(
            "ingest_csv_preview_parsed",
            extra={
                "correlation_id": correlation_id,
                "ingest_path": ingest_path,
                "customer_id": customer_id or "default-customer",
                "header_count": len(headers),
                "sample_row_count": len(rows),
                "parse_issue_count": len(parse_issues),
            },
        )
        if not headers:
            raise HTTPException(
                status_code=400,
                detail={
                    "type": "csv_preview_empty_or_missing_header",
                    "message": "CSV preview could not find a header row.",
                    "actionable_detail": "Ensure the first row contains column names and retry preview.",
                    "issue_details": [issue_to_dict(i) for i in parse_issues],
                    "correlation_id": correlation_id,
                },
            )
        stage = infer_csv_mapping_stage(headers, rows=rows, column_mapping=None)
        mapping = stage.mapping
        mapping_issues = validate_csv_mapping_stage(mapping, headers) if mapping is not None else []
        hard_issues = [*parse_issues, *stage.issues, *mapping_issues]
        warning_issues = list(stage.warnings)
        return models.CsvPreviewResponse(
            headers=headers,
            suggested_mapping=mapping.to_dict() if mapping else None,
            issues=[i.message for i in hard_issues],
            warnings=[i.message for i in warning_issues],
            issue_details=[issue_to_dict(i) for i in hard_issues],
            warning_details=[issue_to_dict(i) for i in warning_issues],
            requires_confirmation=(mapping is None or stage.requires_confirmation),
        )

    @router.post("/ingest/csv/upload", response_model=models.IngestJobEnvelope)
    async def ingest_csv_upload(request: Request, file: UploadFile = File(...), _: None = Depends(require_api_key), run_id: str | None = Query(default=None), customer_id: str | None = Query(default=None), mapping: str | None = Form(default=None)) -> dict[str, Any]:
        filename = str(file.filename or "upload.csv")
        if not filename.lower().endswith(".csv"):
            raise HTTPException(status_code=400, detail="Upload must be a .csv file.")
        resolved_customer = resolve_customer_id(customer_id)
        resolved_run = resolve_run_id_with_default(service_instance, run_id, customer_id=resolved_customer)
        column_mapping = None
        if mapping:
            try:
                parsed = json.loads(mapping)
                column_mapping = models.CsvColumnMappingPayload.model_validate(parsed).model_dump()
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"Invalid column_mapping: {exc}") from exc
        content_length = normalize_content_length(request)
        if content_length is not None and content_length > request_body_limit:
            max_mb = request_body_limit / (1024 * 1024)
            raise HTTPException(status_code=status.HTTP_413_CONTENT_TOO_LARGE, detail=f"Request body too large (max {max_mb:.1f}MB).")
        fd, temp_path = tempfile.mkstemp(prefix="neraium_ingest_", suffix=".csv")
        os.close(fd)
        job_id = f"ingest_{uuid4().hex[:16]}"
        created_at = models.utc_now_iso()
        initial_job = {"job_id": job_id, "status": "uploading", "run_id": resolved_run, "customer_id": resolved_customer, "filename": filename, "created_at": created_at, "updated_at": created_at, "rows_processed": 0, "rows_succeeded": 0, "rows_failed": 0, "partial_success": False, "upload_bytes_received": 0, "upload_bytes_total": content_length, "error_samples": [], "message": "Upload started.", "latest_result": None}
        with ingest_jobs_lock:
            ingest_jobs[job_id] = initial_job
        persist_operational_state(f"ingest_job:{job_id}", initial_job, customer_id=resolved_customer, run_id=resolved_run)
        try:
            bytes_received = await stream_upload_to_tempfile(file, Path(temp_path), job_id)
        except Exception as exc:
            Path(temp_path).unlink(missing_ok=True)
            update_ingest_job(job_id, status="failed", message=f"Upload failed: {exc}")
            raise HTTPException(status_code=400, detail="Failed to read upload stream.") from exc
        update_ingest_job(job_id, status="queued", upload_bytes_received=bytes_received, upload_bytes_total=content_length if content_length is not None else bytes_received, message=f"Upload complete ({bytes_received} bytes). Queueing ingest job.")
        start_ingest_job_worker(job_id=job_id, temp_path=temp_path, run_id=resolved_run, customer_id=resolved_customer, column_mapping=column_mapping)
        with ingest_jobs_lock:
            job = dict(ingest_jobs[job_id])
        return public_ingest_job(job)

    @router.get("/ingest/jobs/{job_id}", response_model=models.IngestJobEnvelope)
    def get_ingest_job(job_id: str, customer_id: str | None = Query(default=None)) -> dict[str, Any]:
        resolved_customer = resolve_customer_id(customer_id)
        with ingest_jobs_lock:
            job = ingest_jobs.get(job_id)
            if job is None or resolve_customer_id(job.get("customer_id")) != resolved_customer:
                raise HTTPException(status_code=404, detail=f"Unknown ingest job: {job_id}")
            return public_ingest_job(job)

    return router
