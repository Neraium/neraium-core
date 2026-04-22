from __future__ import annotations

from neraium_core.sii.structural_model import build_lawful_structural_state, derived_signal_view
from neraium_core.sii.types import StructuralCoherenceState, StructuralIndicators


def test_lawful_structural_state_tracks_operator_and_regime_pressure() -> None:
    state = StructuralCoherenceState(
        indicators=StructuralIndicators(
            structural_drift_score=0.80,
            relational_instability_score=0.70,
            regime_distance=0.65,
            coherence_loss_score=0.40,
            graph_deformation_score=0.55,
            coupling_instability_score=0.60,
            mean_shift_score=0.45,
            covariance_shift_score=0.35,
            subspace_rotation_score=0.30,
            path_length_shift_score=0.20,
        ),
        coherence_score=0.50,
        composite_instability=0.0,
        regime={"uncertainty": 0.25},
        graph_metrics={},
        raw_components={},
        component_extensions={},
    )
    lawful = build_lawful_structural_state(state)

    assert lawful.operator_drift > lawful.residual_structure
    assert lawful.regime_transition_pressure > 0.5
    assert lawful.coherence_margin < 0.0


def test_derived_signal_view_is_downstream_of_lawful_state() -> None:
    state = StructuralCoherenceState(
        indicators=StructuralIndicators(
            structural_drift_score=0.10,
            relational_instability_score=0.10,
            regime_distance=0.10,
            coherence_loss_score=0.05,
            graph_deformation_score=0.10,
            coupling_instability_score=0.10,
            mean_shift_score=0.05,
            covariance_shift_score=0.05,
            subspace_rotation_score=0.05,
            path_length_shift_score=0.05,
        ),
        coherence_score=0.95,
        composite_instability=0.0,
        regime={"uncertainty": 0.02},
        graph_metrics={},
        raw_components={},
        component_extensions={},
    )
    derived = derived_signal_view(build_lawful_structural_state(state))
    assert 0.0 <= derived["composite_instability"] <= 1.0
    assert derived["coherence_tension"] == 0.0
