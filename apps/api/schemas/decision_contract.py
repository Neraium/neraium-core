from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ConfidenceAssessment(BaseModel):
    overall_confidence: float = Field(ge=0.0, le=1.0)
    confidence_band: Literal["low", "medium", "high"]
    low_confidence: bool
    confidence_reason: str
    unknowns: list[str] = Field(default_factory=list)


class EvidenceItem(BaseModel):
    code: str
    detail: str
    source: str


class DataQualityWarning(BaseModel):
    code: str
    severity: Literal["info", "warning", "critical"]
    message: str


class StructuralStateV2(BaseModel):
    cycle: int | None = None
    timestamp: str | None = None
    risk_level: str
    trend: str
    instability_index: float
    events: list[str] = Field(default_factory=list)


class PolicyMetadataV2(BaseModel):
    canonical_schema_version: str
    decision_policy_version: str
    alert_policy: dict[str, int] | None = None


class EvidenceExplanationV2(BaseModel):
    summary: str
    top_drivers: list[dict[str, Any]] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    unknowns: list[str] = Field(default_factory=list)


class OperatorActionV2(BaseModel):
    recommendation_available: bool
    recommended_action: str | None = None
    recommended_target: str | None = None
    urgency: str | None = None
    priority: str | None = None
    rationale: str
    operator_message: str
    operator_note: str
    requires_human_review: bool = True


class DataQualityV2(BaseModel):
    has_warnings: bool
    warnings: list[DataQualityWarning] = Field(default_factory=list)


class DecisionContractV2(BaseModel):
    contract_version: Literal["decision-contract.v2"]
    structural_state: StructuralStateV2
    confidence_quality: ConfidenceAssessment
    evidence_explanation: EvidenceExplanationV2
    operator_action: OperatorActionV2
    policy: PolicyMetadataV2
    data_quality: DataQualityV2
    legacy_aliases: dict[str, Any] = Field(default_factory=dict)


class DecisionContractStateEnvelope(BaseModel):
    state: DecisionContractV2 | None = None


class DecisionContractHistoryEnvelope(BaseModel):
    count: int
    history: list[DecisionContractV2]


class DecisionContractRecommendationEnvelope(BaseModel):
    operator_action: OperatorActionV2 | None = None


class DecisionContractResultsEnvelope(BaseModel):
    count: int
    latest: DecisionContractV2 | None = None
    results: list[DecisionContractV2]
