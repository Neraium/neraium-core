from __future__ import annotations

from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, Query

from ..services.alerts import alert_context


def build_alerts_router(
    *,
    service_instance,
    require_api_key,
    resolve_customer_id,
    request_run_id_or_active,
    utc_now_iso,
    alerts,
    alerts_lock,
    record_alerts_for_customer,
    alerts_envelope_model,
    action_response_model,
    alert_ack_model,
    alert_resolve_model,
) -> APIRouter:
    router = APIRouter(tags=["alerts"])

    @router.get("/alerts", response_model=alerts_envelope_model)
    def list_alerts(
        limit: int = Query(default=50, ge=1, le=500),
        run_id: str | None = Query(default=None),
        customer_id: str | None = Query(default=None),
    ) -> dict[str, Any]:
        resolved_customer = resolve_customer_id(customer_id)
        with alerts_lock:
            items = [dict(a) for a in alerts.get(resolved_customer, [])]
        if run_id:
            items = [a for a in items if str((a.get("context") or {}).get("run_id") or "") == str(run_id)]
        latest = service_instance.get_current_state(run_id=run_id, customer_id=resolved_customer)
        active_alert: dict[str, Any] | None = None
        current_status: dict[str, Any] | None = None
        if isinstance(latest, dict) and isinstance(latest.get("alert_status"), dict):
            current_status = dict(latest["alert_status"])
            current_state = str(current_status.get("state") or current_status.get("alert_state", "")).upper()
            status_item = {
                "id": f"alert_state_{latest.get('cycle', 'latest')}",
                "type": "alert_state",
                "severity": "critical" if current_state == "ESCALATED" else ("high" if latest["alert_status"].get("alert_active") else "info"),
                "message": latest["alert_status"].get("alert_summary"),
                "created_at": latest["alert_status"].get("last_evaluated_at") or latest.get("timestamp"),
                "trigger": latest["alert_status"],
                "context": alert_context(latest),
                "customer_id": resolved_customer,
            }
            items = [status_item, *items]
            if bool(current_status.get("alert_active")):
                active_alert = status_item
        items = items[:limit]
        return {"count": len(items), "alerts": items, "current_status": current_status, "active_alert": active_alert}

    @router.post("/alerts/acknowledge", response_model=action_response_model)
    def acknowledge_alert(
        payload: alert_ack_model,
        _: None = Depends(require_api_key),
    ) -> dict[str, bool]:
        resolved_customer = resolve_customer_id(payload.customer_id)
        resolved_run_id = request_run_id_or_active(service_instance, payload.run_id, customer_id=resolved_customer)
        service_instance.acknowledge_alert(
            run_id=resolved_run_id,
            customer_id=resolved_customer,
            acknowledged_by=payload.acknowledged_by,
        )
        return {"ok": True}

    @router.post("/alerts/resolve", response_model=action_response_model)
    def resolve_alert(
        payload: alert_resolve_model,
        _: None = Depends(require_api_key),
    ) -> dict[str, bool]:
        resolved_customer = resolve_customer_id(payload.customer_id)
        resolved_run_id = request_run_id_or_active(service_instance, payload.run_id, customer_id=resolved_customer)
        service_instance.resolve_alert(
            run_id=resolved_run_id,
            customer_id=resolved_customer,
            resolved_by=payload.resolved_by,
        )
        return {"ok": True}

    @router.post("/alerts/test", response_model=action_response_model)
    def emit_test_alert(
        _: None = Depends(require_api_key),
        customer_id: str | None = Query(default=None),
        run_id: str | None = Query(default=None),
    ) -> dict[str, bool]:
        resolved_customer = resolve_customer_id(customer_id)
        now = utc_now_iso()
        alert = {
            "id": f"alert_{uuid4().hex[:12]}",
            "type": "test_alert",
            "severity": "info",
            "message": "Test alert generated manually.",
            "created_at": now,
            "trigger": {"manual": True},
            "context": {
                "result_id": None,
                "run_id": run_id,
                "timestamp": now,
                "state": "TEST",
                "risk_level": "UNKNOWN",
                "structural_drift_score": 0.0,
                "composite_instability": 0.0,
            },
        }
        record_alerts_for_customer(resolved_customer, [alert])
        return {"ok": True}

    return router
