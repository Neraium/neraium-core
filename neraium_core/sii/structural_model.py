from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any

from .types import StructuralCoherenceState


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _squash_positive(value: float, scale: float = 1.0) -> float:
    v = max(0.0, float(value))
    s = max(1e-6, float(scale))
    return float(v / (v + s))


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
    raw = state.raw_components

    # Operator drift is now grounded in raw operator-level deformation terms when available.
    raw_structural = float(raw.get("structural_drift", indicators.structural_drift_score))
    raw_relational = float(raw.get("relational_instability", indicators.relational_instability_score))
    raw_graph = float(raw.get("graph_deformation", indicators.graph_deformation_score))
    operator_energy = sqrt(raw_structural * raw_structural + raw_relational * raw_relational + raw_graph * raw_graph)
    operator_drift = _clip01(_squash_positive(operator_energy, scale=sqrt(3.0)))

    # Residual structure is derived from actual residual statistics (not score recombination).
    raw_mean_shift = float(raw.get("mean_shift", indicators.mean_shift_score))
    raw_cov_shift = float(raw.get("covariance_shift", indicators.covariance_shift_score))
    raw_subspace_shift = float(raw.get("subspace_rotation", indicators.subspace_rotation_score))
    raw_path_shift = float(raw.get("path_length_shift", indicators.path_length_shift_score))
    residual_energy = sqrt(
        raw_mean_shift * raw_mean_shift
        + raw_cov_shift * raw_cov_shift
        + raw_subspace_shift * raw_subspace_shift
        + raw_path_shift * raw_path_shift
    )
    residual_structure = _clip01(_squash_positive(residual_energy, scale=2.0))

    regime_distance = float(state.regime.get("distance", raw.get("regime_distance", indicators.regime_distance)))
    regime_uncertainty = float(state.regime.get("uncertainty", 0.0))
    regime_transition = _clip01(_squash_positive(regime_distance, scale=0.7) + 0.20 * regime_uncertainty)

    coherence_margin = float(state.coherence_score) - max(operator_drift, regime_transition, residual_structure)
    coherence_tension = _clip01(-coherence_margin)

    # Structural drift is the norm of the lawful structural axes.
    drift_energy = sqrt(
        operator_drift * operator_drift
        + residual_structure * residual_structure
        + regime_transition * regime_transition
        + coherence_tension * coherence_tension
    )
    structural_drift = _clip01(_squash_positive(drift_energy, scale=2.0))

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
