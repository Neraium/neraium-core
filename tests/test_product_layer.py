"""Smoke tests for product layer builders (deterministic, no engine dependency)."""

from neraium_core.product import build_fleet_summary, build_product_layer


def _minimal_result() -> dict:
    return {
        "state": "WATCH",
        "structural_drift_score": 2.1,
        "transition_pressure": 0.55,
        "transition_state": "EMERGING_TRANSITION",
        "confidence_score": 0.72,
        "interpreted_state": "EMERGING_TRANSITION",
        "mahalanobis_score": 1.2,
        "covariance_drift_score": 0.8,
        "drift_velocity": 0.05,
        "readiness": {
            "ready": True,
            "structural_ready": True,
            "transition_classification_ready": True,
            "reason": "ready",
            "unavailable_components": {},
        },
        "transition_outputs_actionable": True,
        "engine_transition_detectable": True,
        "experimental_analytics": {
            "trajectory_analysis": {"dominant_path": "DIVERGING", "path_confidence": 0.7, "available": True},
            "horizon_analysis": {"risk_horizon": "NEAR_TERM", "horizon_score": 0.62, "available": True},
            "constraint_analysis": {"lock_in_score": 0.45, "available": True},
            "hierarchy_analysis": {"origin_scope": "LOCAL", "available": True},
        },
        "early_warning": {
            "early_warning_state": "pre_instability",
            "pre_instability_score": 0.5,
            "stability_erosion_score": 0.4,
            "coherence_breakdown_score": 0.3,
        },
        "geometry": {"available": True, "curvature": 0.2, "directional_consistency": 0.7},
        "state_space_statistics": {"available": True, "local_volume": 0.3, "state_contraction_score": 0.2},
        "state_graph": {"available": True, "branching_factor": 0.25, "path_commitment_score": 0.4},
    }


def test_build_product_layer_keys_and_nested():
    r = _minimal_result()
    pl = build_product_layer(r)
    assert "decision_layer" in pl
    assert "attribution" in pl
    assert "trust_layer" in pl
    assert "operational_recommendation" in pl
    dl = pl["decision_layer"]
    assert dl["current_state"] == "WATCH"
    assert dl["trajectory"] == "DIVERGING"
    assert "evidence_summary" in dl and isinstance(dl["evidence_summary"], list)
    assert "top_signal_drivers" in pl["attribution"]
    assert pl["trust_layer"]["safe_to_act"] is True
    assert pl["operational_recommendation"]["status"] == "observe"


def test_fleet_summary_ranking():
    import pandas as pd

    df = pd.DataFrame(
        [
            {"unit": 1, "state": "STABLE", "transition_pressure": 0.2, "structural_drift_score": 1.0, "early_warning_pre_instability_score": 0.1},
            {"unit": 2, "state": "ALERT", "transition_pressure": 0.8, "structural_drift_score": 3.0, "early_warning_pre_instability_score": 0.7},
        ]
    )
    fs = build_fleet_summary(df)
    assert fs["asset_count"] == 2
    assert fs["top_asset"]["unit"] == 2
    assert fs["ranked_assets"][0]["fleet_rank"] == 1
