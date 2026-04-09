from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatContext(BaseModel):
    customer_id: str | None = None
    run_id: str | None = None
    site_id: str | None = None
    asset_id: str | None = None


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    context: ChatContext | None = None
    recent_metrics_snapshot: dict[str, Any] | None = None


class ChatResponse(BaseModel):
    observed: list[str] = Field(default_factory=list)
    inferred: list[str] = Field(default_factory=list)
    suggested_next_step: str
    response_text: str
    uncertainty: float | None = None
    grounding: dict[str, Any] = Field(default_factory=dict)
