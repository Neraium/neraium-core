from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from ..integration import load_integration_config, resolve_customer_integration

logger = logging.getLogger(__name__)


def build_integrations_router(*, app: Any, require_api_key: Any, resolve_customer_id: Any, resolve_run_id_with_default: Any, service_instance: Any, pull_manager: Any, log_structured: Any, models: Any) -> APIRouter:
    router = APIRouter(tags=["integrations"])

    @router.post("/integrations/pull/start", response_model=models.PullIntegrationStatusEnvelope)
    def start_pull_integration(payload: models.PullIntegrationStartRequest, _: None = Depends(require_api_key), customer_id: str | None = Query(default=None)) -> dict[str, Any]:
        resolved_customer = resolve_customer_id(customer_id)
        cfg_override = getattr(app.state, "integration_config_override", None)
        if isinstance(cfg_override, dict):
            cfg_doc = cfg_override
        else:
            path_override = getattr(app.state, "integration_config_path_override", None)
            path = str(path_override or "").strip() or os.getenv("NERAIUM_INTEGRATION_CONFIG_PATH")
            cfg_doc = load_integration_config(path)
        resolved_cfg = resolve_customer_integration(customer_id=resolved_customer, config_doc=cfg_doc)
        endpoint_url = pull_manager.validate_endpoint_url(payload.endpoint_url or str(resolved_cfg.get("endpoint_url") or ""))
        auth_type = str(payload.auth_type or resolved_cfg.get("auth_type") or "none")
        username = payload.username if payload.username is not None else resolved_cfg.get("username")
        password = payload.password if payload.password is not None else resolved_cfg.get("password")
        token = payload.token if payload.token is not None else resolved_cfg.get("token")
        polling_interval_seconds = pull_manager.parse_finite_float(payload.polling_interval_seconds if payload.polling_interval_seconds is not None else resolved_cfg.get("polling_interval_seconds") or 30.0, field_name="polling_interval_seconds")
        retry_max_attempts = pull_manager.parse_int(payload.retry_max_attempts if payload.retry_max_attempts is not None else resolved_cfg.get("retry_max_attempts") or 3, field_name="retry_max_attempts")
        retry_backoff_seconds = pull_manager.parse_finite_float(payload.retry_backoff_seconds if payload.retry_backoff_seconds is not None else resolved_cfg.get("retry_backoff_seconds") or 1.0, field_name="retry_backoff_seconds")
        request_timeout_seconds = pull_manager.parse_finite_float(payload.request_timeout_seconds if payload.request_timeout_seconds is not None else resolved_cfg.get("request_timeout_seconds") or 10.0, field_name="request_timeout_seconds")
        if polling_interval_seconds < 0.2:
            raise HTTPException(status_code=400, detail="polling_interval_seconds must be >= 0.2.")
        if retry_max_attempts < 1:
            raise HTTPException(status_code=400, detail="retry_max_attempts must be >= 1.")
        if retry_backoff_seconds < 0.05:
            raise HTTPException(status_code=400, detail="retry_backoff_seconds must be >= 0.05.")
        if request_timeout_seconds < 1.0:
            raise HTTPException(status_code=400, detail="request_timeout_seconds must be >= 1.0.")
        if auth_type == "basic" and (not str(username or "").strip() or password is None):
            raise HTTPException(status_code=400, detail="Basic auth requires username and password.")
        if auth_type == "bearer" and not str(token or "").strip():
            raise HTTPException(status_code=400, detail="Bearer auth requires token.")
        resolved_run = resolve_run_id_with_default(service_instance, payload.run_id, customer_id=resolved_customer)
        state = pull_manager.start_pull_integration(
            customer_id=resolved_customer,
            endpoint_url=endpoint_url,
            run_id=resolved_run,
            auth_type=auth_type,
            username=username,
            password=password,
            token=token,
            polling_interval_seconds=polling_interval_seconds,
            retry_max_attempts=retry_max_attempts,
            retry_backoff_seconds=retry_backoff_seconds,
            request_timeout_seconds=request_timeout_seconds,
        )
        log_structured(logger, event="pull_integration_started", fields={"customer_id": resolved_customer, "run_id": resolved_run, "endpoint_url": endpoint_url, "polling_interval_seconds": polling_interval_seconds, "auth_type": auth_type}, level=logging.INFO)
        return pull_manager.public_pull_state(state, customer_id=resolved_customer)

    @router.post("/integrations/pull/stop", response_model=models.PullIntegrationStatusEnvelope)
    def stop_pull(_: None = Depends(require_api_key), customer_id: str | None = Query(default=None)) -> dict[str, Any]:
        resolved_customer = resolve_customer_id(customer_id)
        return pull_manager.stop_pull_integration(resolved_customer, reason="Pull integration stopped by operator.")

    @router.get("/integrations/pull/status", response_model=models.PullIntegrationStatusEnvelope)
    def pull_status(customer_id: str | None = Query(default=None)) -> dict[str, Any]:
        resolved_customer = resolve_customer_id(customer_id)
        with pull_manager.store.pull_integrations_lock:
            state = pull_manager.store.pull_integrations.get(resolved_customer)
            return pull_manager.public_pull_state(state, customer_id=resolved_customer)

    return router
