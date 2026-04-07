#!/usr/bin/env python
from pathlib import Path

p = Path ("apps/api/routers/ingest.py")
s = p.read_text()

old_import = "from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request, status"
new_import = "from fastapi import APIRouter, Body, Depends, File, HTTPException, Query, Request, UploadFile, status"
if old_import in s:
    s = s.replace(old_import, new_import)

old_block = '''    @router.post("/ingest/csv/upload", response_model=IngestJobEnvelope)
    async def ingest_csv_upload(
        request: Request,
        _: None = Depends(deps.require_api_key),
        run_id: str | None = Query(default=None),
        customer_id: str | None = Query(default=None),
        mapping: str | None = Query(default=None),
    ) -> dict[str, Any]:
        correlation_id = str(getattr(request.state, "correlation_id", "") or f"ing_up_{uuid4().hex[:12]}")
        content_type = str(request.headers.get("content-type") or "")
        boundary = ""
        if "boundary=" in content_type:
            boundary = content_type.split("boundary=", 1)[1].strip().strip('"')
        raw_body = await request.body()
        if not boundary:
            return JSONResponse(status_code=400, content={"detail": "Missing uploaded CSV file."})

        file_bytes = b""
        filename = str(request.headers.get("x-filename") or "upload.csv")
        mapping_value = mapping
        marker = f"--{boundary}".encode("utf-8")
        for part in raw_body.split(marker):
            if b"\\r\\n\\r\\n" not in part:
                continue
            head, payload = part.split(b"\\r\\n\\r\\n", 1)
            payload = payload.rstrip(b"\\r\\n")
            if b'name="file"' in head:
                file_bytes = payload
                if b"filename=" in head:
                    try:
                        file_name_raw = head.split(b"filename=", 1)[1].split(b"\\r\\n", 1)[0].strip()
                        filename = file_name_raw.strip(b'"').decode("utf-8", errors="ignore") or filename
                    except Exception:
                        pass
            elif b'name="mapping"' in head and mapping_value is None:
                mapping_value = payload.decode("utf-8", errors="ignore")

        if not file_bytes:
            return JSONResponse(status_code=400, content={"detail": "Missing uploaded CSV file."})
        if not filename.lower().endswith(".csv"):
            return JSONResponse(status_code=400, content={"detail": "Upload must be a .csv file."})
        resolved_customer = deps.resolve_customer_id(customer_id)
        resolved_run = deps.resolve_run_id_with_default(deps.service_instance, run_id, customer_id=resolved_customer)
        column_mapping = None
        if mapping_value:
            try:
                parsed = json.loads(mapping_value)
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
            "upload_filename": filename,
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
            Path(temp_path).write_bytes(file_bytes)
            bytes_received = len(file_bytes)
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
                "upload_filename": filename,
                "ingest_path": "csv_upload",
                "stage": "ingest_started",
            },
        )
        deps.start_ingest_job_worker(job_id=job_id, temp_path=temp_path, run_id=resolved_run, customer_id=resolved_customer, column_mapping=column_mapping)
        with deps.ingest_jobs_lock:
            job = dict(deps.ingest_jobs[job_id])
        return deps.public_ingest_job(job)
'''

new_block = '''    @router.post("/ingest/csv/upload", response_model=IngestJobEnvelope)
    async def ingest_csv_upload(
        request: Request,
        file: UploadFile = File(...),
        _: None = Depends(deps.require_api_key),
        run_id: str | None = Query(default=None),
        customer_id: str | None = Query(default=None),
        mapping: str | None = Query(default=None),
    ) -> dict[str, Any]:
        correlation_id = str(getattr(request.state, "correlation_id", "") or f"ing_up_{uuid4().hex[:12]}")
        filename = file.filename or str(request.headers.get("x-filename") or "upload.csv")

        upload_bytes = await file.read()
        if not upload_bytes:
            return JSONResponse(status_code=400, content={"detail": "Missing uploaded CSV file."})
        if not filename.lower().endswith(".csv"):
            return JSONResponse(status_code=400, content={"detail": "Upload must be a .csv file."})

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
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail=f"Request body too large (max {max_mb:.1f}MB).",
            )

        fd, temp_path = tempfile.mkstemp(prefix="neraium_ingest_", suffix=".csv")
        os.close(fd)
        job_id = f"ingest_{uuid4().hex[:16]}"
        created_at = deps.utc_now_iso()
        initial_job = {
            "job_id": job_id,
            "status": "uploading",
            "run_id": resolved_run,
            "customer_id": resolved_customer,
            "upload_filename": filename,
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
        deps.persist_operational_state(
            f"ingest_job:{job_id}",
            initial_job,
            customer_id=resolved_customer,
            run_id=resolved_run,
        )

        try:
            Path(temp_path).write_bytes(upload_bytes)
            bytes_received = len(upload_bytes)
        except Exception as exc:
            Path(temp_path).unlink(missing_ok=True)
            deps.update_ingest_job(job_id, status="failed", message=f"Upload failed: {exc}")
            logger.exception(
                "ingest_csv_upload_stream_failed",
                extra={
                    "correlation_id": correlation_id,
                    "job_id": job_id,
                    "run_id": resolved_run,
                    "customer_id": resolved_customer,
                },
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
        finally:
            await file.close()

        deps.update_ingest_job(
            job_id,
            status="queued",
            upload_bytes_received=bytes_received,
            upload_bytes_total=content_length if content_length is not None else bytes_received,
            message=f"Upload complete ({bytes_received} bytes). Queueing ingest job.",
        )
        logger.info(
            "ingest_csv_upload_queued",
            extra={
                "correlation_id": correlation_id,
                "request_id": correlation_id,
                "job_id": job_id,
                "run_id": resolved_run,
                "customer_id": resolved_customer,
                "upload_filename": filename,
                "ingest_path": "csv_upload",
                "stage": "ingest_started",
            },
        )
        deps.start_ingest_job_worker(
            job_id=job_id,
            temp_path=temp_path,
            run_id=resolved_run,
            customer_id=resolved_customer,
            column_mapping=column_mapping,
        )
        with deps.ingest_jobs_lock:
            job = dict(deps.ingest_jobs[job_id])
        return deps.public_ingest_job(job)
'''

if old_block not in s:
    raise SystemExit("Could not find expected ingest_csv_upload block to replace.")

s = s.replace(old_block, new_block)
p.write_text(s)
print("Patched apps/api/routers/ingest.py")
PY
pytest -q