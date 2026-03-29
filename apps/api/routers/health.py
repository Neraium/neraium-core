from __future__ import annotations

import logging
import os

from fastapi import APIRouter, Response, status

from .._core_imports import get_core_runtime_status

logger = logging.getLogger(__name__)


def build_health_router(
    *,
    service_instance,
    api_key: str | None,
    persistence_available: bool,
    app_version: str,
    resolve_customer_id,
    runtime_state_diagnostics_provider,
    health_response_model,
    client_error_model,
) -> APIRouter:
    router = APIRouter(tags=["health"])

    @router.get("/health", response_model=health_response_model)
    def health() -> health_response_model:
        latest = service_instance.get_latest_result(
            customer_id=resolve_customer_id(os.getenv("NERAIUM_DEFAULT_CUSTOMER_ID"))
        )
        runtime_status = get_core_runtime_status()
        runtime_fallback = bool(runtime_status.get("using_fallback", False))
        overall_ok = persistence_available and not runtime_fallback
        return health_response_model(
            status="ok" if overall_ok else "degraded",
            version=app_version,
            auth_configured=bool(api_key),
            persistence_available=bool(persistence_available),
            latest_result_available=latest is not None,
            core_runtime_mode="degraded" if runtime_fallback else "full",
            core_runtime_fallback=runtime_fallback,
            core_runtime_notes=[str(x) for x in runtime_status.get("notes", [])],
            runtime_state_diagnostics=runtime_state_diagnostics_provider(),
        )

    @router.post("/client-errors", status_code=status.HTTP_204_NO_CONTENT)
    def report_client_error(report: client_error_model) -> Response:
        msg = (report.message or "")[:1500]
        url = (report.url or "")[:1500]
        stack = (report.stack or "")[:4000]
        extra = (report.reason or report.source or "")[:500]
        logger.warning(
            "client_js_error url=%s msg=%s extra=%s stack_snip=%s",
            url,
            msg,
            extra,
            stack[:800].replace("\n", " "),
        )
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    return router
