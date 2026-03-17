from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class TelemetryFrame:
    timestamp: str
    site_id: str
    asset_id: str
    sensor_values: dict[str, float | None]
    sensor_quality: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class StructuralResult:
    id: str | None
    event_type: str
    timestamp: str
    site_id: str
    asset_id: str
    state: str
    structural_drift_score: float
    relational_stability_score: float
    system_health: int
    drift_alert: bool
    lead_time_hours: float | None
    lead_time_confidence: float
    drift_velocity: float
    structural_driver: str
    predicted_impact: str
    explanation: str
    mahalanobis_score: float
    covariance_drift_score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ValidationIssue:
    row: int
    field: str
    message: str


@dataclass(slots=True)
class IngestSummary:
    accepted: int
    rejected: int
    issues: list[ValidationIssue] = field(default_factory=list)


@dataclass(slots=True)
class IngestResponse:
    summary: IngestSummary
    results: list[StructuralResult] = field(default_factory=list)
