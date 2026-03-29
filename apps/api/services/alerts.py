from __future__ import annotations

import logging
import os
from typing import Any
from uuid import uuid4

import numpy as np

from .._core_imports import log_structured


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(f):
        return float(default)
    return float(f)


def alert_thresholds() -> tuple[float, float]:
    try:
        instability = float(os.getenv("NERAIUM_ALERT_INSTABILITY_THRESHOLD", "1.5"))
    except (TypeError, ValueError):
        instability = 1.5
    try:
        drift_rapid = float(os.getenv("NERAIUM_ALERT_RAPID_DRIFT_DELTA", "0.2"))
    except (TypeError, ValueError):
        drift_rapid = 0.2
    return max(0.0, instability), max(0.0, drift_rapid)


def alert_context(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "result_id": result.get("result_id"),
        "run_id": result.get("run_id"),
        "timestamp": result.get("timestamp") or result.get("persisted_at"),
        "state": result.get("state") or result.get("interpreted_state"),
        "risk_level": result.get("risk_level"),
        "structural_drift_score": _safe_float(result.get("structural_drift_score"), 0.0),
        "composite_instability": _safe_float(result.get("latest_instability"), 0.0),
        "alert_status": result.get("alert_status") if isinstance(result.get("alert_status"), dict) else {},
    }


def evaluate_alerts(
    *,
    current: dict[str, Any] | None,
    previous: dict[str, Any] | None,
    instability_threshold: float,
    rapid_drift_delta: float,
    now_iso: str,
) -> list[dict[str, Any]]:
    del instability_threshold, rapid_drift_delta
    if not isinstance(current, dict):
        return []

    current_status = current.get("alert_status") if isinstance(current.get("alert_status"), dict) else {}
    previous_status = previous.get("alert_status") if isinstance(previous, dict) and isinstance(previous.get("alert_status"), dict) else {}

    current_state = str(current_status.get("state") or current_status.get("alert_state", "CLEAR")).upper()
    previous_state = str(previous_status.get("state") or previous_status.get("alert_state", "CLEAR")).upper()

    alerts: list[dict[str, Any]] = []
    should_emit_activation = current_state in {"ACTIVE_UNACKNOWLEDGED", "ESCALATED"} and previous_state not in {"ACTIVE_UNACKNOWLEDGED", "ACTIVE_ACKNOWLEDGED", "ESCALATED"}
    should_emit_renotify = bool(current_status.get("renotify_due")) and current_state in {"ACTIVE_UNACKNOWLEDGED", "ESCALATED"}

    if should_emit_activation or should_emit_renotify:
        severity = "critical" if current_state == "ESCALATED" else "high"
        if should_emit_activation:
            alert_type = "persistent_alert_activated"
            message = f"Persistent alert activated after {int(current_status.get('hit_window_threshold', 3))} consecutive hits."
        else:
            alert_type = "persistent_alert_renotify"
            message = "Persistent alert remains active and unacknowledged."

        alerts.append(
            {
                "id": f"alert_{uuid4().hex[:12]}",
                "type": alert_type,
                "severity": severity,
                "message": message,
                "created_at": now_iso,
                "trigger": {
                    "state": current_state,
                    "reason": current_status.get("reason") or current_status.get("alert_reason"),
                    "consecutive_hit_count": int(current_status.get("consecutive_hit_count", 0)),
                    "hit_window_threshold": int(current_status.get("hit_window_threshold", 3)),
                    "unacknowledged_duration": int(current_status.get("unacknowledged_duration", 0)),
                },
                "context": alert_context(current),
            }
        )

    return alerts


def dispatch_alert_stubs(
    *,
    logger: logging.Logger,
    alert: dict[str, Any],
    webhook_url: str | None,
    email_to: str | None,
) -> None:
    if webhook_url:
        log_structured(
            logger,
            event="alert_webhook_stub",
            fields={
                "webhook_url": webhook_url,
                "alert_id": alert.get("id"),
                "alert_type": alert.get("type"),
                "severity": alert.get("severity"),
            },
            level=logging.INFO,
        )
    if email_to:
        log_structured(
            logger,
            event="alert_email_stub",
            fields={
                "email_to": email_to,
                "alert_id": alert.get("id"),
                "alert_type": alert.get("type"),
                "severity": alert.get("severity"),
            },
            level=logging.INFO,
        )
