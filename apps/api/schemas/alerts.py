from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class AlertsEnvelope(BaseModel):
    count: int
    alerts: list[dict[str, Any]]
    current_status: dict[str, Any] | None = None
    active_alert: dict[str, Any] | None = None


class AlertAcknowledgeRequest(BaseModel):
    run_id: str | None = None
    customer_id: str | None = None
    acknowledged_by: str | None = None


class AlertResolveRequest(BaseModel):
    run_id: str | None = None
    customer_id: str | None = None
    resolved_by: str | None = None
