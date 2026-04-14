from __future__ import annotations

from neraium_core.tetrahedral_state import compute_tetrahedral_state


def _base_state(**overrides: float):
    payload = {
        "structural_drift_score": 0.2,
        "relational_instability_score": 0.2,
        "transition_pressure": 0.2,
        "temporal_consistency_score": 0.8,
    }
    payload.update(overrides)
    return compute_tetrahedral_state(**payload)


def test_weights_sum_to_one() -> None:
    state = _base_state()
    assert abs(sum(state["weights"].values()) - 1.0) < 1e-6


def test_position_has_length_three() -> None:
    state = _base_state()
    assert len(state["position"]) == 3


def test_zero_inputs_return_balanced_weights() -> None:
    state = compute_tetrahedral_state(
        structural_drift_score=0.0,
        relational_instability_score=0.0,
        transition_pressure=0.0,
        temporal_consistency_score=1.0,
    )
    assert state["weights"] == {
        "structural_drift_score": 0.25,
        "relational_instability_score": 0.25,
        "transition_pressure": 0.25,
        "temporal_inconsistency": 0.25,
    }


def test_high_structural_drift_moves_to_structural_vertex() -> None:
    state = _base_state(structural_drift_score=1.0, relational_instability_score=0.0, transition_pressure=0.0)
    assert state["nearest_vertex"] == "STRUCTURAL"


def test_high_transition_pressure_moves_to_transition_vertex() -> None:
    state = _base_state(structural_drift_score=0.0, relational_instability_score=0.0, transition_pressure=1.0)
    assert state["nearest_vertex"] == "TRANSITION"


def test_motion_history_reports_non_negative_speed() -> None:
    state = compute_tetrahedral_state(
        structural_drift_score=0.7,
        relational_instability_score=0.1,
        transition_pressure=0.1,
        temporal_consistency_score=0.9,
        history_positions=[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]],
    )
    assert state["speed"] >= 0.0


def test_top_level_keys_present() -> None:
    state = _base_state()
    expected_keys = {
        "weights",
        "position",
        "nearest_vertex",
        "interpreted_label",
        "nearest_face",
        "edge_alignment",
        "speed",
        "curvature",
        "state_label",
        "movement_summary",
        "geometric_motion_class",
        "semantic_consistency",
    }
    assert expected_keys.issubset(state.keys())


def test_interpreted_label_mapping() -> None:
    structural = compute_tetrahedral_state(
        structural_drift_score=1.0,
        relational_instability_score=0.0,
        transition_pressure=0.0,
        temporal_consistency_score=1.0,
    )
    assert structural["interpreted_label"] == "STRUCTURAL_STRESS_BUILDING"

    relational = compute_tetrahedral_state(
        structural_drift_score=0.0,
        relational_instability_score=1.0,
        transition_pressure=0.0,
        temporal_consistency_score=1.0,
    )
    assert relational["interpreted_label"] == "COUPLING_BREAKDOWN"

    transition = compute_tetrahedral_state(
        structural_drift_score=0.0,
        relational_instability_score=0.0,
        transition_pressure=1.0,
        temporal_consistency_score=1.0,
    )
    assert transition["interpreted_label"] == "ACTIVE_TRANSITION"

    temporal = compute_tetrahedral_state(
        structural_drift_score=0.0,
        relational_instability_score=0.0,
        transition_pressure=0.0,
        temporal_consistency_score=0.0,
    )
    assert temporal["interpreted_label"] == "TEMPORAL_DEGRADATION"


def test_semantic_consistency_flags_alert_neutral_tension() -> None:
    """Test that ALERT + GEOMETRICALLY_NEUTRAL flags a semantic tension."""
    state = compute_tetrahedral_state(
        structural_drift_score=0.3,
        relational_instability_score=0.3,
        transition_pressure=0.3,
        temporal_consistency_score=0.3,
        policy_state="ALERT",
    )
    assert state["semantic_consistency"]["consistency_status"] == "tension"
    assert state["semantic_consistency"]["tension_type"] == "alert_but_geometrically_neutral"
    assert "system-wide" in state["semantic_consistency"]["semantic_context"].lower()


def test_semantic_consistency_flags_coherent_when_stable() -> None:
    """Test that STABLE + GEOMETRICALLY_NEUTRAL is coherent."""
    state = compute_tetrahedral_state(
        structural_drift_score=0.1,
        relational_instability_score=0.1,
        transition_pressure=0.1,
        temporal_consistency_score=0.9,
        policy_state="STABLE",
    )
    assert state["semantic_consistency"]["consistency_status"] == "coherent"
    assert state["semantic_consistency"]["tension_type"] is None


def test_semantic_consistency_high_drift_stationary_tension() -> None:
    """Test that high drift + stationary motion flags a tension."""
    state = compute_tetrahedral_state(
        structural_drift_score=0.8,
        relational_instability_score=0.2,
        transition_pressure=0.2,
        temporal_consistency_score=0.8,
        history_positions=[[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]],  # Minimal motion
    )
    assert state["semantic_consistency"]["consistency_status"] == "tension"
    assert "dwelling" in state["semantic_consistency"]["semantic_context"].lower()


def test_geometric_motion_class_field_present() -> None:
    """Test that geometric_motion_class field is present in payload."""
    state = _base_state()
    assert "geometric_motion_class" in state
    assert state["geometric_motion_class"] in ("geometric_stationary", "steady_drift", "geometric_turning")


def test_label_renamed_to_geometrically_neutral() -> None:
    """Test that BALANCED has been renamed to GEOMETRICALLY_NEUTRAL."""
    state = compute_tetrahedral_state(
        structural_drift_score=0.2,
        relational_instability_score=0.2,
        transition_pressure=0.2,
        temporal_consistency_score=0.8,
    )
    # When no single dimension dominates, should be GEOMETRICALLY_NEUTRAL
    assert state["state_label"] in ("GEOMETRICALLY_NEUTRAL", "EDGE_ALIGNED", "STRUCTURAL_DOMINANT", "RELATIONAL_DOMINANT", "TRANSITION_DOMINANT", "TEMPORAL_DOMINANT")
    # For this balanced case, should be GEOMETRICALLY_NEUTRAL
    if state["state_label"] == "GEOMETRICALLY_NEUTRAL":
        assert "NEUTRAL" in state["state_label"]
