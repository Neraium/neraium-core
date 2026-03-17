from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from neraium_core.pipeline import normalize_rest_payload, parse_csv_text
from run_engine import StructuralEngine


class StructuralResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: int | None = None
    event_type: str
    timestamp: str
    site_id: str
    asset_id: str
    state: str
    structural_drift_score: float
    relational_stability_score: float
    system_health: int
    drift_alert: bool
    lead_time_hours: float | None = None
    lead_time_confidence: float
    drift_velocity: float
    structural_driver: str
    predicted_impact: str
    explanation: str
    mahalanobis_score: float
    covariance_drift_score: float


class StructuralIntelligenceService:
    """Application-facing API over the structural engine."""

    def __init__(self) -> None:
        self._engine = StructuralEngine(baseline_window=24, recent_window=8)
        self._latest_result: StructuralResult | None = None

    def ingest_payload(self, payload: dict[str, Any]) -> StructuralResult:
        frame = normalize_rest_payload(payload)
        result = self._engine.process_frame(frame)
        self._latest_result = StructuralResult.model_validate(result)
        return self._latest_result

    def ingest_batch(self, payloads: list[dict[str, Any]]) -> StructuralResult:
        if not payloads:
            raise ValueError("payloads must not be empty")

        latest: StructuralResult | None = None
        for payload in payloads:
            latest = self.ingest_payload(payload)

        if latest is None:
            raise ValueError("No payloads processed")
        return latest

    def ingest_csv(self, csv_text: str) -> StructuralResult:
        frames = parse_csv_text(csv_text)
        if not frames:
            raise ValueError("CSV produced no frames")

        latest: StructuralResult | None = None
        for frame in frames:
            result = self._engine.process_frame(frame)
            latest = StructuralResult.model_validate(result)

        if latest is None:
            raise ValueError("CSV produced no frames")

        self._latest_result = latest
        return latest

    def get_latest_result(self) -> StructuralResult:
        if self._latest_result is None:
            raise ValueError("No result available")
        return self._latest_result

    def reset(self) -> None:
        self._engine.frames.clear()
        self._engine.prev_drift = None
        self._engine.latest_result = None
        self._engine.sensor_order = []
        self._latest_result = None
