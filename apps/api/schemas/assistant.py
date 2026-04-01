from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class AssistantRequest(BaseModel):
    run_id: str | None = None
    customer_id: str | None = None
    mode: Literal["summary", "why_recommended", "what_changed", "pattern_similarity", "handoff"] | None = None
    history_limit: int = Field(default=20, ge=2, le=100)


class AssistantResponse(BaseModel):
    mode: str
    text: str
    grounding: dict[str, Any]
    context: dict[str, Any]


class ReportRequest(BaseModel):
    run_id: str | None = None
    customer_id: str | None = None
    mode: Literal["client_report", "technician_summary", "inspection_brief", "handoff_note"]
    history_limit: int = Field(default=20, ge=2, le=100)


class ReportResponse(BaseModel):
    mode: str
    report_text: str
    sections: dict[str, str]
