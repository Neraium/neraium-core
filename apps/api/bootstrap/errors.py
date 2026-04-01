from __future__ import annotations

import logging
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from starlette.responses import JSONResponse

from ..services.validation_utils import actionable_validation_detail


def infer_stage_from_path(path: str) -> str:
    text = str(path or "").lower()
    if "/ingest/csv/preview" in text:
        return "preview"
    if "/ingest/jobs/" in text or "/ingest/csv/upload" in text:
        return "ingest"
    # Match ``/ingest`` and ``/ingest/...`` (query strings omit a trailing slash after ``ingest``).
    if "/ingest" in text and "/ingest/csv/preview" not in text and "/ingest/jobs/" not in text and "/ingest/csv/upload" not in text:
        return "normalize"
    if "/integrations/" in text:
        return "integration_pull"
    return "request"


def failure_envelope(
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


def register_exception_handlers(app: FastAPI, *, logger: logging.Logger) -> None:
    @app.exception_handler(HTTPException)
    async def _http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        correlation_id = f"api_err_{uuid4().hex[:12]}"
        stage = infer_stage_from_path(request.url.path)
        issue_details: list[dict[str, Any]] = []
        warning_details: list[dict[str, Any]] = []
        if isinstance(exc.detail, dict):
            detail_payload = dict(exc.detail)
            message = str(detail_payload.get("message") or detail_payload.get("detail") or "Request failed.")
            type_name = str(detail_payload.get("type") or "http_error")
            actionable = str(detail_payload.get("actionable_detail") or actionable_validation_detail(message))
            stage = str(detail_payload.get("stage") or stage)
            correlation_id = str(detail_payload.get("correlation_id") or correlation_id)
            issue_details = list(detail_payload.get("issue_details") or [])
            warning_details = list(detail_payload.get("warning_details") or [])
        else:
            message = str(exc.detail)
            type_name = "http_error"
            actionable = actionable_validation_detail(message)
        logger.warning(
            "api_http_exception",
            extra={
                "path": request.url.path,
                "method": request.method,
                "status_code": exc.status_code,
                "stage": stage,
                "type": type_name,
                "correlation_id": correlation_id,
            },
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=failure_envelope(
                stage=stage,
                type_name=type_name,
                message=message,
                actionable_detail=actionable,
                issue_details=issue_details,
                warning_details=warning_details,
                correlation_id=correlation_id,
            ),
        )

    @app.exception_handler(RequestValidationError)
    async def _request_validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        correlation_id = str(getattr(request.state, "correlation_id", "") or f"api_val_{uuid4().hex[:12]}")
        issue_details = []
        for issue in exc.errors():
            location = issue.get("loc", [])
            issue_details.append(
                {
                    "code": issue.get("type", "validation_error"),
                    "message": issue.get("msg", "Invalid request payload."),
                    "field": ".".join(str(part) for part in location if part not in {"body"}),
                    "input": issue.get("input"),
                }
            )
        logger.warning(
            "request_validation_error",
            extra={
                "correlation_id": correlation_id,
                "path": request.url.path,
                "method": request.method,
                "issue_count": len(issue_details),
            },
        )
        return JSONResponse(
            status_code=422,
            content=failure_envelope(
                stage=infer_stage_from_path(request.url.path),
                type_name="validation_error",
                message="Request validation failed.",
                actionable_detail="Review request fields and retry. See issue_details for field-level guidance.",
                issue_details=issue_details,
                correlation_id=correlation_id,
            ),
        )

    @app.exception_handler(Exception)
    async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        correlation_id = f"api_unh_{uuid4().hex[:12]}"
        logger.exception("Unhandled API error")
        msg = "Internal server error."
        return JSONResponse(
            status_code=500,
            content=failure_envelope(
                stage=infer_stage_from_path(request.url.path),
                type_name="internal_error",
                message=msg,
                actionable_detail="Retry request or contact support if issue persists.",
                correlation_id=correlation_id,
            ),
        )
