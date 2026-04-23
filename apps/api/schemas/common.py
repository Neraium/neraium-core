from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ExportEnvelope(BaseModel):
    run_id: str | None = None
    format: Literal["json", "csv"]
    count: int
    content_type: str
    filename: str
    content: str


class HealthResponse(BaseModel):
    status: str
    version: str
    auth_configured: bool
    persistence_available: bool
    latest_result_available: bool
    core_runtime_mode: str = "full"
    core_runtime_fallback: bool = False
    core_runtime_notes: list[str] = Field(default_factory=list)
    analysis_runtime_available: bool = True
    runtime_state_diagnostics: dict[str, Any] = Field(default_factory=dict)


class ClientErrorReport(BaseModel):
    """Browser-reported script errors (product UI telemetry)."""

    message: str = ""
    stack: str | None = None
    url: str | None = None
    source: str | None = None
    lineno: int | None = None
    colno: int | None = None
    reason: str | None = None


class ResultsEnvelope(BaseModel):
    status: str | None = None
    run_id: str | None = None
    processed: int | None = None
    latest: dict[str, Any] | None = None
    alert_status: dict[str, Any] | None = None
    memory_recall: dict[str, Any] | None = None
    count: int
    results: list[dict[str, Any]]


class ActionResponse(BaseModel):
    ok: bool


class ResultEnvelope(BaseModel):
    result: dict[str, Any]


class GeometryEnvelope(BaseModel):
    run_id: str | None = None
    result_id: int | None = None
    timestamp: str | None = None
    available: bool
    reason: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    nodes: list[dict[str, Any]] = Field(default_factory=list)
    edges: list[dict[str, Any]] = Field(default_factory=list)
    views: dict[str, Any] = Field(default_factory=dict)
    summary: dict[str, Any] = Field(default_factory=dict)
    projection: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)
    graph_analytics: dict[str, Any] | None = None
    system_state: dict[str, Any] | None = None


class CanonicalOutputResponse(BaseModel):
    schema_version: str
    timestamp: str
    cycle: int
    attribution: dict[str, Any]
    regime_memory: dict[str, Any]
    risk_assessment: dict[str, Any]
    causal_analysis: dict[str, Any]
    operational_recommendation: dict[str, Any]
    confidence: float
    explanation_text: str
    events: list[str]
    session: dict[str, Any] | None = None
    aliases: dict[str, Any] | None = None
    history_id: int | None = None
    persisted_at: str | None = None
    customer_id: str | None = None
    run_id: str | None = None
    alert_status: dict[str, Any] | None = None
    memory_recall: dict[str, Any] | None = None


class CurrentStateEnvelope(BaseModel):
    state: CanonicalOutputResponse | None = None


class DecisionContractV2Envelope(BaseModel):
    state: dict[str, Any] | None = None


class OperatorActionEnvelope(BaseModel):
    operator_action: dict[str, Any] | None = None


class DecisionContractV2HistoryEnvelope(BaseModel):
    count: int
    history: list[dict[str, Any]]


class DecisionContractV2LatestEnvelope(BaseModel):
    count: int
    latest: dict[str, Any] | None = None


class HistoryEnvelope(BaseModel):
    count: int
    history: list[CanonicalOutputResponse]


class RecommendationEnvelope(BaseModel):
    operational_recommendation: dict[str, Any] | None = None


class DecisionEnvelope(BaseModel):
    """Deprecated compatibility envelope. Prefer RecommendationEnvelope."""

    decision: dict[str, Any] | None = None


class ExplanationEnvelope(BaseModel):
    explanation_text: str | None = None


class EventsEnvelope(BaseModel):
    events: list[str] = Field(default_factory=list)
    cycle: int | None = None
    timestamp: str | None = None
