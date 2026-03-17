from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel


class TelemetryPoint(BaseModel):
    timestamp: datetime
    signals: dict[str, float | None]


class TelemetryIngestRequest(BaseModel):
    points: list[TelemetryPoint]


class WindowConfig(BaseModel):
    size: int = 50
    step: int = 1


class WindowSnapshot(BaseModel):
    start: datetime
    end: datetime
    rows: int
    columns: list[str]


class GraphMetrics(BaseModel):
    mean_connectivity: float
    density: float
    connected_components: int
    clustering_coefficient: float


class SpectralMetrics(BaseModel):
    spectral_radius: float
    spectral_gap: float
    dominant_eigenvector: list[float]
    ranked_signal_loadings: list[tuple[str, float]]


class DirectionalMetrics(BaseModel):
    causal_energy: float
    causal_asymmetry: float
    causal_divergence: float


class EarlyWarningMetrics(BaseModel):
    variance_avg: float
    lag1_autocorr_avg: float


class DriftMetrics(BaseModel):
    frobenius_drift: float
    mean_absolute_drift: float
    regime_relative_drift: float


class ForecastMetrics(BaseModel):
    slope: float
    recent_slope: float
    acceleration: float
    time_to_instability: float | None


class SubsystemMetrics(BaseModel):
    subsystem_count: int
    max_subsystem_instability: float


class ObservableSnapshot(BaseModel):
    timestamp: datetime
    entropy: float
    centrality: dict[str, float]
    graph: GraphMetrics
    spectral: SpectralMetrics
    directional: DirectionalMetrics
    early_warning: EarlyWarningMetrics
    subsystems: SubsystemMetrics
    score: float


class RegimeState(BaseModel):
    regime: str
    distance: float


class HealthResponse(BaseModel):
    status: str = "ok"


class PersistedState(BaseModel):
    key: str
    value: dict[str, Any]
