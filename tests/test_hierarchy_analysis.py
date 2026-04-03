from __future__ import annotations

from neraium_core.experimental_analytics.hierarchy_analysis import analyze_hierarchy_cascade


def test_hierarchy_localized_failure_marks_local_origin_and_subsystem() -> None:
    out = analyze_hierarchy_cascade(
        sensor_names=["s1", "s2", "s3", "s4"],
        subsystem={"clusters": [[0, 1], [2, 3]], "max_instability": 1.2},
        graph={"connectivity": 1.0, "density": 0.6},
        causal_propagation={
            "spread_scores": [0.9, 0.7, 0.05, 0.02],
            "top_pairs": [{"src_idx": 0, "dst_idx": 1, "edge_weight": 0.8}],
        },
        counterfactual_guidance={
            "top_structural_break_contributors": [
                {"sensor_a": "s1", "sensor_b": "s2", "contributor_score": 0.95},
                {"sensor_a": "s1", "sensor_b": "s2", "contributor_score": 0.82},
            ]
        },
        transition={"transition_pressure": 0.25, "shock_triggered": 0.0},
    )

    assert out["available"] is True
    assert out["origin_scope"] == "LOCAL"
    assert out["origin_subsystem"] == "cluster_0"


def test_hierarchy_spreading_scenario_marks_high_propagation_risk() -> None:
    out = analyze_hierarchy_cascade(
        sensor_names=["a", "b", "c", "d"],
        subsystem={"clusters": [[0, 1], [2, 3]], "max_instability": 1.5},
        graph={"connectivity": 0.4, "density": 0.35},
        causal_propagation={
            "spread_scores": [1.0, 0.8, 0.7, 0.6],
            "top_pairs": [
                {"src_idx": 0, "dst_idx": 2, "edge_weight": 0.9},
                {"src_idx": 1, "dst_idx": 3, "edge_weight": 0.85},
            ],
        },
        counterfactual_guidance={
            "top_structural_break_contributors": [
                {"sensor_a": "a", "sensor_b": "b", "contributor_score": 0.7},
                {"sensor_a": "c", "sensor_b": "d", "contributor_score": 0.65},
            ]
        },
        transition={"transition_pressure": 1.32, "shock_triggered": 1.0, "shock_boost": 0.9},
    )

    assert out["propagation_risk"] == "HIGH"
    assert out["cascade_direction"]


def test_hierarchy_broad_instability_marks_system_wide() -> None:
    out = analyze_hierarchy_cascade(
        sensor_names=["s1", "s2", "s3", "s4", "s5", "s6"],
        subsystem={"clusters": [[0, 1], [2, 3], [4, 5]], "max_instability": 1.6},
        graph={"connectivity": 0.3, "density": 0.3},
        causal_propagation={
            "spread_scores": [1.0, 0.85, 0.9, 0.8, 0.88, 0.86],
            "top_pairs": [
                {"src_idx": 0, "dst_idx": 2, "edge_weight": 0.7},
                {"src_idx": 2, "dst_idx": 4, "edge_weight": 0.72},
                {"src_idx": 4, "dst_idx": 1, "edge_weight": 0.75},
            ],
        },
        counterfactual_guidance={
            "top_structural_break_contributors": [
                {"sensor_a": "s1", "sensor_b": "s2", "contributor_score": 0.6},
                {"sensor_a": "s3", "sensor_b": "s4", "contributor_score": 0.62},
                {"sensor_a": "s5", "sensor_b": "s6", "contributor_score": 0.64},
            ]
        },
        transition={"transition_pressure": 1.1, "shock_triggered": 0.6},
    )

    assert out["origin_scope"] == "SYSTEM_WIDE"
    assert out["origin_subsystem"] is None


def test_hierarchy_missing_relational_analytics_returns_fallback() -> None:
    out = analyze_hierarchy_cascade(
        sensor_names=["only_sensor"],
        subsystem=None,
        graph=None,
        causal_propagation=None,
        counterfactual_guidance=None,
        transition=None,
    )

    assert out["available"] is False
    assert out["reason"] == "relational_analytics_unavailable"
    assert out["origin_subsystem"] is None
