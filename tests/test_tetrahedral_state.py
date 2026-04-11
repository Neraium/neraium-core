from __future__ import annotations

from neraium_core.alignment import StructuralEngine
from neraium_core.tetrahedral_state import (
    compute_motion_features,
    compute_tetrahedral_state,
    compute_tetrahedral_weights,
    weights_to_position,
)


def test_compute_tetrahedral_weights_normalized() -> None:
    weights = compute_tetrahedral_weights(
        structural_drift_score=0.8,
        relational_instability_score=0.4,
        transition_pressure=0.2,
        temporal_consistency_score=0.5,
    )
    assert set(weights.keys()) == {
        "structural_drift_score",
        "relational_instability_score",
        "transition_pressure",
        "temporal_inconsistency",
    }
    assert abs(sum(weights.values()) - 1.0) < 1e-9


def test_compute_motion_features_curvature_and_speed() -> None:
    motion = compute_motion_features([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [0.3, 0.3, 0.0]])
    assert motion["speed"] > 0.0
    assert motion["curvature"] > 0.0
    assert motion["movement_summary"] == "turning"


def test_compute_tetrahedral_state_payload_shape() -> None:
    payload = compute_tetrahedral_state(
        structural_drift_score=0.9,
        relational_instability_score=0.1,
        transition_pressure=0.2,
        temporal_consistency_score=0.3,
        history_positions=[[0.0, 0.0, 0.0]],
        regime_drift=0.2,
        reversibility=0.4,
        geometry_curvature=0.5,
    )
    assert len(payload["position"]) == 3
    assert payload["nearest_vertex"] in {"STRUCTURAL", "RELATIONAL", "TRANSITION", "TEMPORAL"}
    assert "weights" in payload
    assert "movement_summary" in payload


def test_alignment_attaches_tetrahedral_state() -> None:
    engine = StructuralEngine(baseline_window=4, recent_window=3)
    result = {}
    for i in range(12):
        result = engine.process_frame(
            {
                "timestamp": float(i),
                "site_id": "S",
                "asset_id": "A",
                "sensor_values": {
                    "s1": 1.0 + 0.1 * i,
                    "s2": 2.0 - 0.05 * i,
                    "s3": 0.5 + 0.03 * i,
                },
            }
        )

    tetra = result.get("tetrahedral_state")
    assert isinstance(tetra, dict)
    assert "weights" in tetra
    assert "state_label" in tetra
    analytics = result.get("experimental_analytics", {})
    assert analytics.get("tetrahedral_state") == tetra
    pos = weights_to_position(tetra["weights"])
    assert len(pos) == 3
