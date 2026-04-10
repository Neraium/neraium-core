from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DeteriorationCriteria:
    instability_min: float
    drift_min: float


@dataclass(frozen=True)
class PersistenceThresholds:
    min_consecutive_cycles: int
    min_window_fraction: float


@dataclass(frozen=True)
class CorroborationThresholds:
    min_confirming_signals: int
    min_confidence: float


@dataclass(frozen=True)
class UncertaintyOverrideBehavior:
    refuse_on_low_confidence: bool
    min_confidence: float
    refusal_message: str


@dataclass(frozen=True)
class Doctrine:
    doctrine_version: str
    allowed_assertions: tuple[str, ...]
    forbidden_outputs: tuple[str, ...]
    deterioration_criteria: DeteriorationCriteria
    persistence_thresholds: PersistenceThresholds
    corroboration_thresholds: CorroborationThresholds
    refusal_conditions: tuple[str, ...]
    uncertainty_override: UncertaintyOverrideBehavior
    metadata: dict[str, Any] = field(default_factory=dict)
