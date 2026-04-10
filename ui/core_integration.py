from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .config import UIConfig
from .utils import clamp, l2_norm, parse_iso8601, safe_float


@dataclass(frozen=True)
class TrajectoryPoint:
    x: float
    y: float
    t: str


@dataclass(frozen=True)
class StructuralEdge:
    source: str
    target: str
    strength: float
    trend: str


@dataclass(frozen=True)
class TimelineEvent:
    t: str
    regime: str
    drift_delta: float
    reaction_window_minutes: float


@dataclass(frozen=True)
class SystemState:
    position: TrajectoryPoint
    velocity: tuple[float, float]
    trajectory_history: list[TrajectoryPoint]
    projected_cone: list[TrajectoryPoint]
    stability_regions: dict[str, tuple[float, float]]
    structural_relationships: list[StructuralEdge]
    timeline: list[TimelineEvent]
    drift_intensity: float
    regime_state: str
    system_health: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "position": asdict(self.position),
            "velocity": {"dx": self.velocity[0], "dy": self.velocity[1]},
            "trajectory_history": [asdict(p) for p in self.trajectory_history],
            "projected_cone": [asdict(p) for p in self.projected_cone],
            "stability_regions": self.stability_regions,
            "structural_relationships": [asdict(e) for e in self.structural_relationships],
            "timeline": [asdict(e) for e in self.timeline],
            "drift_intensity": self.drift_intensity,
            "regime_state": self.regime_state,
            "system_health": self.system_health,
            "confidence": self.confidence,
        }


def _point_from_row(row: dict[str, Any], fallback_t: str) -> TrajectoryPoint:
    drift = clamp(safe_float(row.get("structural_drift_score"), 0.0), 0.0, 1.0)
    stability_loss = 1.0 - clamp(safe_float(row.get("relational_stability_score"), 1.0), 0.0, 1.0)
    return TrajectoryPoint(x=round(drift, 6), y=round(stability_loss, 6), t=str(row.get("timestamp") or fallback_t))


def _velocity(points: list[TrajectoryPoint]) -> tuple[float, float]:
    if len(points) < 2:
        return (0.0, 0.0)
    return (round(points[-1].x - points[-2].x, 6), round(points[-1].y - points[-2].y, 6))


def _projected_cone(anchor: TrajectoryPoint, velocity: tuple[float, float], steps: int) -> list[TrajectoryPoint]:
    vx, vy = velocity
    spread = max(0.02, l2_norm((vx, vy)) * 0.3)
    projection: list[TrajectoryPoint] = []
    for i in range(1, steps + 1):
        step_weight = i / max(steps, 1)
        projection.append(
            TrajectoryPoint(
                x=clamp(anchor.x + (vx * i) + spread * step_weight, 0.0, 1.0),
                y=clamp(anchor.y + (vy * i) - spread * step_weight, 0.0, 1.0),
                t=f"+{i}",
            )
        )
    return projection


def build_system_state(records: list[dict[str, Any]] | None, *, config: UIConfig) -> SystemState:
    rows = records or [{}]
    fallback_t = parse_iso8601(None).isoformat()
    history = [_point_from_row(row, fallback_t) for row in rows][-config.trajectory_tail :]
    position = history[-1]
    velocity = _velocity(history)

    latest = rows[-1]
    regime = str(latest.get("regime_name") or latest.get("state") or "baseline")
    health = str(latest.get("system_health") or "nominal")
    confidence = clamp(safe_float(latest.get("confidence_score", latest.get("confidence", 0.0))), 0.0, 1.0)
    drift = position.x

    relationships = [
        StructuralEdge("baseline_cluster", "current_cluster", round(max(0.0, 1.0 - drift), 4), "weakening" if velocity[0] > 0 else "stable"),
        StructuralEdge("correlation_core", "causal_frontier", round(max(0.0, 1.0 - position.y), 4), "weakening" if velocity[1] > 0 else "strengthening"),
    ]

    timeline: list[TimelineEvent] = []
    for idx, point in enumerate(history):
        prev = history[idx - 1] if idx > 0 else point
        timeline.append(
            TimelineEvent(
                t=point.t,
                regime=regime,
                drift_delta=round(point.x - prev.x, 6),
                reaction_window_minutes=round(max(1.0, config.reaction_window_default_minutes * (1.0 - point.x)), 2),
            )
        )

    return SystemState(
        position=position,
        velocity=velocity,
        trajectory_history=history,
        projected_cone=_projected_cone(position, velocity, config.projection_horizon_steps),
        stability_regions={
            "stable_basin": (0.0, 0.35),
            "transition_band": (0.35, 0.65),
            "divergence_zone": (0.65, 1.0),
        },
        structural_relationships=relationships,
        timeline=timeline,
        drift_intensity=drift,
        regime_state=regime,
        system_health=health,
        confidence=confidence,
    )
