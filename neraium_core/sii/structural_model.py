from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .types import StructuralCoherenceState


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


@dataclass(frozen=True)
class LawfulStructuralState:
    """
    Canonical structural model for the engine.

    This object is the center of the runtime: downstream scores, alerts,
    and operator-facing recommendations are derived from these quantities.
    """

    operator_drift: float
    residual_structure: float
    regime_transition_pressure: float
    coherence_margin: float
    structural_drift: float

    def as_dict(self) -> dict[str, float]:
        return {
            "operator_drift": float(self.operator_drift),
            "residual_structure": float(self.residual_structure),
            "regime_transition_pressure": float(self.regime_transition_pressure),
            "coherence_margin": float(self.coherence_margin),
            "structural_drift": float(self.structural_drift),
        }


def build_lawful_structural_state(state: StructuralCoherenceState) -> LawfulStructuralState:
    indicators = state.indicators
    operator_drift = _clip01(
        0.55 * float(indicators.structural_drift_score)
        + 0.30 * float(indicators.relational_instability_score)
        + 0.15 * float(indicators.graph_deformation_score)
    )
    residual_structure = _clip01(
        0.35 * float(indicators.mean_shift_score)
        + 0.35 * float(indicators.covariance_shift_score)
        + 0.20 * float(indicators.subspace_rotation_score)
        + 0.10 * float(indicators.path_length_shift_score)
    )
    regime_transition = _clip01(
        0.85 * float(indicators.regime_distance)
        + 0.15 * float(state.regime.get("uncertainty", 0.0))
    )
    coherence_margin = float(state.coherence_score) - max(operator_drift, regime_transition, residual_structure)
    structural_drift = _clip01(
        0.40 * operator_drift
        + 0.30 * regime_transition
        + 0.20 * residual_structure
        + 0.10 * float(indicators.coherence_loss_score)
    )
    return LawfulStructuralState(
        operator_drift=float(operator_drift),
        residual_structure=float(residual_structure),
        regime_transition_pressure=float(regime_transition),
        coherence_margin=float(coherence_margin),
        structural_drift=float(structural_drift),
    )


def derived_signal_view(lawful: LawfulStructuralState) -> dict[str, Any]:
    """
    Build a clearly secondary interface layer from the central structural state.
    """
    tension = _clip01(max(0.0, -float(lawful.coherence_margin)))
    composite_instability = _clip01(
        0.45 * float(lawful.structural_drift)
        + 0.35 * float(lawful.regime_transition_pressure)
        + 0.20 * tension
    )
    return {
        "composite_instability": float(composite_instability),
        "coherence_tension": float(tension),
    }
