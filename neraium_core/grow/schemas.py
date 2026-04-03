from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


ScopeType = Literal["plant", "zone", "room", "subsystem", "facility", "portfolio"]
PrimaryState = Literal["STABLE", "DRIFTING", "AT_RISK", "DEGRADING"]
Urgency = Literal["LOW", "MEDIUM", "HIGH", "IMMEDIATE"]
TrendDirection = Literal["IMPROVING", "STABLE", "WORSENING"]
BiologicalRisk = Literal[
    "LOW_BIOLOGICAL_RISK",
    "ELEVATED_BIOLOGICAL_RISK",
    "YIELD_RISK_EMERGING",
    "QUALITY_RISK_EMERGING",
]
PlantMode = Literal["measured_plant_state", "inferred_plant_risk_view", "not_applicable"]


class SetupSummary(BaseModel):
    service_ok: bool = True
    deployment_mode: str = "grow"
    telemetry_source_count: int = 0
    signal_count_discovered: int = 0
    rooms_discovered: int = 0
    zones_discovered: int = 0
    plants_discovered: int = 0
    assets_discovered: int = 0
    subsystems_discovered: int = 0
    facilities_discovered: int = 0
    portfolio_sites_discovered: int = 0
    baseline_learned: bool = False
    model_ready: bool = False
    readiness_percent: int = 0
    estimated_time_to_readiness_minutes: int = 60
    last_ingest_timestamp: str | None = None


class GrowTelemetryRecord(BaseModel):
    timestamp: datetime
    facility_id: str
    site_id: str | None = None
    building_id: str | None = None
    room_id: str | None = None
    zone_id: str | None = None
    plant_id: str | None = None
    asset_id: str | None = None
    asset_type: str | None = None
    subsystem_id: str | None = None
    controller_id: str | None = None
    crop_type: str | None = None
    strain_or_crop_type: str | None = None
    growth_stage: str | None = None
    batch_id: str | None = None
    telemetry_source: str | None = None
    signals: dict[str, float | int | bool | str | None] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _identity_guardrails(self) -> "GrowTelemetryRecord":
        if not any([self.plant_id, self.zone_id, self.room_id, self.asset_id, self.subsystem_id]):
            raise ValueError("one of plant_id, zone_id, room_id, asset_id, or subsystem_id is required")
        return self


class EvidenceItem(BaseModel):
    summary: str
    severity: str = "info"
    subsystem: str | None = None
    signals: list[str] = Field(default_factory=list)


class ComparisonSummary(BaseModel):
    baseline_delta: float
    peer_median_delta: float
    best_performer_delta: float
    ranking_position: int
    peer_count: int
    relative_drift: float
    relative_inefficiency: float
    likely_causal_differences: list[str] = Field(default_factory=list)


class GrowStateResponse(BaseModel):
    timestamp: str
    scope_type: ScopeType
    scope_id: str
    parent_scope_id: str | None = None
    facility_id: str
    site_id: str | None = None
    room_id: str | None = None
    zone_id: str | None = None
    plant_id: str | None = None
    asset_id: str | None = None
    subsystem: str | None = None
    state: PrimaryState
    confidence: float
    structural_drift_score: float
    composite_instability: float
    time_in_state_minutes: int
    trend_direction: TrendDirection
    dominant_relationship_shift: str
    cross_system_driver: str
    biological_risk: BiologicalRisk
    inefficiency_summary: str
    energy_inefficiency: float
    climate_inefficiency: float
    process_inefficiency: float
    biological_inefficiency: float
    recommended_action: str
    urgency: Urgency
    estimated_time_to_impact_hours: float | None
    threshold_breach_detected: bool
    threshold_breach_message: str
    operator_message: str
    evidence: list[EvidenceItem]
    comparison_summary: ComparisonSummary
    setup_summary: SetupSummary
    plant_view_mode: PlantMode = "not_applicable"
    measured_plant_signals_present: bool = False
    trend_score: float = 0.0


class DecisionRecord(BaseModel):
    decision_id: str
    timestamp: str
    scope_type: ScopeType
    scope_id: str
    state: PrimaryState
    confidence: float
    time_in_state: str
    trend_direction: TrendDirection
    dominant_relationship_shift: str
    cross_system_driver: str
    biological_risk: BiologicalRisk
    inefficiency_summary: str
    recommended_action: str
    urgency: Urgency
    estimated_time_to_impact: str
    evidence_summary: list[str]
    threshold_breach_detected: bool
    threshold_breach_message: str


class IngestResponse(BaseModel):
    accepted: int
    deployment_mode: str = "grow"
    read_only: bool = True
    setup_summary: SetupSummary
    latest_decisions: list[DecisionRecord]


class CompareResponse(BaseModel):
    anchor: GrowStateResponse
    peers: list[GrowStateResponse]
    summary: dict[str, Any]


class FleetSummary(BaseModel):
    ranked_plants_by_local_risk: list[dict[str, Any]]
    ranked_zones_by_risk: list[dict[str, Any]]
    ranked_rooms_by_risk: list[dict[str, Any]]
    ranked_subsystems_by_drift_contribution: list[dict[str, Any]]
    ranked_facilities_by_instability: list[dict[str, Any]]
    highest_inefficiency_areas: list[dict[str, Any]]
    fastest_deteriorating_entities: list[dict[str, Any]]
    entities_requiring_action_soonest: list[dict[str, Any]]
    top_cross_system_drivers: list[dict[str, Any]]
