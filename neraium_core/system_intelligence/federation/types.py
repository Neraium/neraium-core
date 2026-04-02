from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PacketMetadata:
    source_instance_id: str
    schema_version: str = "federation.v1"
    model_version: str = "structural-intelligence.v1"
    generated_at_step: int = 0
    recency_score: float = 1.0


@dataclass
class TrajectoryFamilySummary:
    family_id: str
    path_family: str
    support: int
    escalation_frequency: float
    phase_shift_frequency: float
    novelty_rate: float
    representative_segment: list[list[float]] = field(default_factory=list)
    reliability: float = 0.5


@dataclass
class LawEvidenceSummary:
    law_key: str
    condition_family: str
    mechanism: str
    outcome_family: str
    support: int
    effect_estimate: float
    uncertainty: float
    consistency: float
    source_diversity: int


@dataclass
class InterventionEvidenceSummary:
    context_bucket: str
    trajectory_family: str
    intervention_type: str
    intervention_target: str
    support: int
    effectiveness: float
    worsening_rate: float
    reliability: float


@dataclass
class CalibrationSummary:
    bucket: str
    support_band: str
    novelty_band: str
    support: int
    confidence_mean: float
    realized_success_rate: float


@dataclass
class SharedKnowledgePacket:
    metadata: PacketMetadata
    trajectory_families: list[TrajectoryFamilySummary] = field(default_factory=list)
    law_evidence: list[LawEvidenceSummary] = field(default_factory=list)
    intervention_evidence: list[InterventionEvidenceSummary] = field(default_factory=list)
    calibration: list[CalibrationSummary] = field(default_factory=list)
    mechanism_family_associations: list[dict[str, Any]] = field(default_factory=list)
