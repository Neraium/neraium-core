from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field


class OnboardingVerifyRequest(BaseModel):
    customer_id: str = Field(min_length=1, max_length=200)
    api_key: str = Field(min_length=1, max_length=500)
    environment_label: str | None = Field(default=None, max_length=120)


class OnboardingProfileRequest(BaseModel):
    customer_display_name: str = Field(min_length=1, max_length=200)
    site_name: str = Field(min_length=1, max_length=200)
    default_asset_id: str = Field(min_length=1, max_length=200)
    asset_group: str | None = Field(default=None, max_length=200)
    ingestion_method: Literal["api_push", "csv_upload", "demo_replay"]


def build_onboarding_router(
    *,
    resolve_customer_id: Any,
    service_instance: Any,
    normalize_external_payload: Any,
    is_api_key_valid: Any,
    configured_api_key: str | None,
    ensure_default_run: Any,
    persist_operational_state: Any,
    store_instance: Any | None,
    persisted_state_enabled: bool,
) -> APIRouter:
    router = APIRouter(prefix="/onboarding", tags=["onboarding"])

    def _utc_now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _state_key(customer_id: str) -> str:
        return f"onboarding:{customer_id}"

    def _default_state(customer_id: str) -> dict[str, Any]:
        return {
            "customer_id": customer_id,
            "credentials_verified": False,
            "profile_saved": False,
            "ingestion_method": None,
            "active_run_id": None,
            "test_frame_sent": False,
            "test_frame_accepted": False,
            "profile": {},
            "environment_label": None,
            "verified_at": None,
            "updated_at": _utc_now_iso(),
        }

    def _load_state(customer_id: str) -> dict[str, Any]:
        default = _default_state(customer_id)
        if not persisted_state_enabled or store_instance is None:
            return default
        row = None
        try:
            if hasattr(store_instance, "get_operational_state"):
                row = store_instance.get_operational_state(state_key=_state_key(customer_id))
            else:
                rows = store_instance.list_operational_state(key_prefix=_state_key(customer_id))
                row = rows[0] if rows else None
        except Exception:
            row = None
        payload = row.get("state") if isinstance(row, dict) else None
        if not isinstance(payload, dict):
            return default
        return {
            **default,
            **payload,
            "customer_id": customer_id,
        }

    def _compute_ready(state: dict[str, Any]) -> bool:
        method = str(state.get("ingestion_method") or "").strip().lower()
        csv_or_demo_selected = method in {"csv_upload", "demo_replay"}
        return bool(
            state.get("credentials_verified")
            and state.get("profile_saved")
            and state.get("active_run_id")
            and (state.get("test_frame_sent") or csv_or_demo_selected)
        )

    def _status_payload(state: dict[str, Any]) -> dict[str, Any]:
        return {
            "customer_id": state.get("customer_id"),
            "credentials_verified": bool(state.get("credentials_verified")),
            "profile_saved": bool(state.get("profile_saved")),
            "ingestion_method": state.get("ingestion_method"),
            "active_run_id": state.get("active_run_id"),
            "test_frame_sent": bool(state.get("test_frame_sent")),
            "test_frame_accepted": bool(state.get("test_frame_accepted")),
            "ready": _compute_ready(state),
            "profile": dict(state.get("profile") or {}),
            "environment_label": state.get("environment_label"),
            "verified_at": state.get("verified_at"),
            "updated_at": state.get("updated_at"),
        }

    def _save_state(customer_id: str, state: dict[str, Any]) -> dict[str, Any]:
        outgoing = {
            **state,
            "customer_id": customer_id,
            "updated_at": _utc_now_iso(),
        }
        persist_operational_state(
            _state_key(customer_id),
            outgoing,
            customer_id=customer_id,
            run_id=str(outgoing.get("active_run_id") or "") or None,
        )
        return outgoing

    @router.get("/status")
    def onboarding_status(customer_id: str | None = Query(default=None)) -> dict[str, Any]:
        resolved_customer = resolve_customer_id(customer_id)
        state = _load_state(resolved_customer)
        active = service_instance.get_active_run(customer_id=resolved_customer)
        if active is not None and not state.get("active_run_id"):
            state["active_run_id"] = active.get("run_id")
        return _status_payload(state)

    @router.post("/verify")
    def onboarding_verify(payload: OnboardingVerifyRequest) -> dict[str, Any]:
        resolved_customer = resolve_customer_id(payload.customer_id)
        if not is_api_key_valid(configured_api_key, payload.api_key):
            raise HTTPException(status_code=401, detail="Credential verification failed: API key was rejected.")
        state = _load_state(resolved_customer)
        state.update(
            {
                "credentials_verified": True,
                "customer_id": resolved_customer,
                "environment_label": str(payload.environment_label or "").strip() or None,
                "verified_at": _utc_now_iso(),
            }
        )
        state = _save_state(resolved_customer, state)
        status_payload = _status_payload(state)
        status_payload["verified"] = True
        status_payload["message"] = "Credentials verified."
        return status_payload

    @router.post("/profile")
    def onboarding_profile(payload: OnboardingProfileRequest, customer_id: str | None = Query(default=None)) -> dict[str, Any]:
        resolved_customer = resolve_customer_id(customer_id)
        state = _load_state(resolved_customer)
        state["profile_saved"] = True
        state["ingestion_method"] = payload.ingestion_method
        state["profile"] = {
            "customer_display_name": payload.customer_display_name,
            "site_name": payload.site_name,
            "default_asset_id": payload.default_asset_id,
            "asset_group": payload.asset_group,
        }
        state = _save_state(resolved_customer, state)
        return {
            "ok": True,
            "message": "System identity saved.",
            "status": _status_payload(state),
        }

    @router.post("/bootstrap-run")
    def onboarding_bootstrap_run(customer_id: str | None = Query(default=None)) -> dict[str, Any]:
        resolved_customer = resolve_customer_id(customer_id)
        run = ensure_default_run(service_instance, customer_id=resolved_customer)
        run_id = str(run.get("run_id") or "")
        state = _load_state(resolved_customer)
        state["active_run_id"] = run_id or None
        state = _save_state(resolved_customer, state)
        return {
            "ok": True,
            "run": run,
            "status": _status_payload(state),
        }

    @router.post("/test-frame")
    def onboarding_test_frame(customer_id: str | None = Query(default=None)) -> dict[str, Any]:
        resolved_customer = resolve_customer_id(customer_id)
        state = _load_state(resolved_customer)
        run = ensure_default_run(service_instance, customer_id=resolved_customer)
        run_id = str(run.get("run_id") or "")
        profile = dict(state.get("profile") or {})
        site_name = str(profile.get("site_name") or "site-1")
        asset_id = str(profile.get("default_asset_id") or "asset-1")

        synthetic = normalize_external_payload(
            {
                "customer_id": resolved_customer,
                "timestamp": _utc_now_iso(),
                "site_id": site_name,
                "asset_id": asset_id,
                "sensor_values": {
                    "pressure": 32.4,
                    "flow": 14.8,
                    "temperature": 71.2,
                },
            },
            customer_id=resolved_customer,
        )
        ingest_out = service_instance.ingest_frame(synthetic, run_id=run_id, customer_id=resolved_customer)
        recent = service_instance.list_recent_results(limit=1, run_id=run_id, customer_id=resolved_customer)
        accepted = bool(ingest_out)
        has_results = bool(recent)

        state["active_run_id"] = run_id or None
        state["test_frame_sent"] = accepted
        state["test_frame_accepted"] = accepted
        state = _save_state(resolved_customer, state)

        message = "Test frame accepted and telemetry state is now visible." if accepted and has_results else (
            "Test frame accepted, but no result history is visible yet." if accepted else "Test frame failed. Check run and ingest configuration."
        )
        return {
            "ok": accepted,
            "ingest_accepted": accepted,
            "result_available": has_results,
            "telemetry_connection_status": "connected" if accepted else "failed",
            "active_run_id": run_id or None,
            "message": message,
            "status": _status_payload(state),
        }

    return router
