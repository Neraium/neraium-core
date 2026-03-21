from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict


InterpretedState = Literal[
    "NOMINAL_STRUCTURE",
    "REGIME_SHIFT_OBSERVED",
    "COUPLING_INSTABILITY_OBSERVED",
    "STRUCTURAL_INSTABILITY_OBSERVED",
    "COHERENCE_UNDER_CONSTRAINT",
]
DecisionState = Literal["STABLE", "WATCH", "ALERT"]
ConfidenceLevel = Literal["low", "medium", "high"]

ALLOWED_INTERPRETED_STATES: set[str] = {
    "NOMINAL_STRUCTURE",
    "REGIME_SHIFT_OBSERVED",
    "COUPLING_INSTABILITY_OBSERVED",
    "STRUCTURAL_INSTABILITY_OBSERVED",
    "COHERENCE_UNDER_CONSTRAINT",
}
ALLOWED_STATES: set[str] = {"STABLE", "WATCH", "ALERT"}
ALLOWED_CONFIDENCE: set[str] = {"low", "medium", "high"}

# Compatibility aliases used by other modules/tests.
INTERPRETED_STATES = ALLOWED_INTERPRETED_STATES
STATES = ALLOWED_STATES
CONFIDENCE_LEVELS = ALLOWED_CONFIDENCE


class StructuralTelemetryFrame(TypedDict):
    timestamp: str
    site_id: str
    asset_id: str
    sensor_values: dict[str, float | int | str | None]


@dataclass(frozen=True)
class SIIFrame:
    timestamp: str
    site_id: str
    asset_id: str
    sensor_values: dict[str, float | None]
    metadata: dict[str, Any] = field(default_factory=dict)


# Backward-compatible name used by some internal modules.
TelemetryFrame = SIIFrame


@dataclass(frozen=True)
class StructuralIndicators:
    structural_drift_score: float
    relational_instability_score: float
    regime_distance: float
    coherence_loss_score: float
    graph_deformation_score: float
    coupling_instability_score: float


@dataclass(frozen=True)
class DecisionResult:
    interpreted_state: InterpretedState
    state: DecisionState
    confidence: ConfidenceLevel
    confidence_score: float
    reason: str


@dataclass(frozen=True)
class GraphState:
    adjacency: Any
    feature_names: list[str]
    density: float = 0.0
    avg_degree: float = 0.0
    l1_deformation: float = 0.0


@dataclass(frozen=True)
class GraphSnapshot:
    adjacency: Any
    node_count: int
    edge_count: int
    density: float
    mean_abs_weight: float
    spectral_radius: float
    laplacian_trace: float
    degree_centrality: dict[str, float]


class SIIResult(TypedDict):
    timestamp: str
    site_id: str
    asset_id: str
    state: DecisionState
    interpreted_state: InterpretedState
    confidence: ConfidenceLevel
    structural_drift_score: float
    relational_instability_score: float
    regime_distance: float
    coherence_score: float
    graph_deformation_score: float
    dominant_drivers: list[str]
    confidence_reasoning: list[str]
    explanation: str
    read_only: bool
    system_health: int
    data_quality_summary: dict[str, Any]
    experimental_analytics: dict[str, Any]
