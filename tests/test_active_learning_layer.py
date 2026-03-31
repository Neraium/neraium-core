from __future__ import annotations

from neraium_core.system_intelligence.active_learning.uncertainty import decompose_uncertainty
from neraium_core.system_intelligence.experiment_design.discriminative import identify_discriminative_experiments
from neraium_core.system_intelligence.experiment_design.exploration import score_exploration_vs_exploitation
from neraium_core.system_intelligence.experiment_design.planner import build_experiment_plan
from neraium_core.system_intelligence.hypothesis.tracker import track_competing_hypotheses
from neraium_core.system_intelligence.observability.recommendation import recommend_measurements
from neraium_core.system_intelligence.platform import StructuralSystemIntelligencePlatform


def test_uncertainty_decomposition_identifies_dominant_component() -> None:
    hypotheses = [
        {"hypothesis": "A", "likelihood": 0.42},
        {"hypothesis": "B", "likelihood": 0.4},
    ]
    out = decompose_uncertainty(
        transition={"uncertainty": 0.8},
        trajectory={"support_count": 1, "novelty_score": 0.9},
        hypotheses=hypotheses,
        reliability={"intervention_recommendation": {"recommendation_calibrated_confidence": 0.2}},
        cross_system={"transfer_adaptation": {"transfer_confidence": 0.3}},
    )
    assert out["dominant_uncertainty_type"] in out["uncertainty_vector"]
    assert out["uncertainty_magnitude"] > 0.5


def test_hypothesis_tracking_returns_competing_hypotheses() -> None:
    out = track_competing_hypotheses(
        trajectory={"current_trajectory_path_family": "oscillatory", "support_count": 2, "novelty_score": 0.6},
        mechanism={"mechanism_candidates": [{"mechanism": "pressure->temperature"}]},
        laws={"law_candidates": [{"law": "localized thermal lock-in"}], "falsification_summary": {"contradiction_rate": 0.2}},
        transition={"transition_path": "oscillatory"},
    )
    assert len(out["active_hypotheses"]) >= 3
    assert abs(sum(h["likelihood"] for h in out["active_hypotheses"]) - 1.0) < 0.02


def test_discriminative_experiments_rank_information_gain() -> None:
    hypotheses = [
        {"hypothesis": "Drift-dominant degradation", "likelihood": 0.45},
        {"hypothesis": "Oscillatory instability", "likelihood": 0.43},
        {"hypothesis": "Localized failure", "likelihood": 0.12},
    ]
    out = identify_discriminative_experiments(
        hypotheses=hypotheses,
        uncertainty={"uncertainty_magnitude": 0.7},
        intervention_ranking=[{"name": "remove_top_driver_contribution"}],
    )
    assert out["candidate_experiments"]
    assert out["expected_information_gain"] >= out["candidate_experiments"][-1]["expected_information_gain"]


def test_exploration_vs_exploitation_tradeoff_behaves() -> None:
    out = score_exploration_vs_exploitation(
        intervention={
            "recommendation": {
                "ranked_interventions": [
                    {
                        "name": "remove_top_driver_contribution",
                        "confidence": 0.5,
                        "evidence_sources": {"historical_evidence": {"intervention_effectiveness": 0.4, "uncertainty": 0.6}},
                    },
                    {
                        "name": "restore_relationship_cluster_to_baseline",
                        "confidence": 0.8,
                        "evidence_sources": {"historical_evidence": {"intervention_effectiveness": 0.5, "uncertainty": 0.2}},
                    },
                ]
            }
        },
        candidate_experiments=[{"name": "increase_sampling_rate_sensor_Y", "experiment_type": "measurement", "expected_information_gain": 0.8, "discriminative_power": 0.7}],
    )
    assert out["exploit_recommendations"]
    assert out["explore_recommendations"]
    assert out["exploration_vs_exploitation_tradeoff"]["recommended_mode"] in {"explore", "exploit"}


def test_measurement_recommendations_target_ambiguity() -> None:
    hypotheses = [
        {"hypothesis": "A", "likelihood": 0.48},
        {"hypothesis": "B", "likelihood": 0.45},
    ]
    out = recommend_measurements(
        observation={"subsystem_impact": {"subsystem_A": 0.8}, "sensor_readings": [1.0]},
        trajectory={"novelty_score": 0.7},
        hypotheses=hypotheses,
        uncertainty={"uncertainty_vector": {"structural_ambiguity": 0.8}},
    )
    assert out["observability_gap_score"] > 0.0
    assert out["expected_uncertainty_reduction"] > 0.0


def test_experiment_plan_is_interpretable_and_hypothesis_conditioned() -> None:
    plan = build_experiment_plan(
        hypotheses=[{"hypothesis": "drift-based degradation"}, {"hypothesis": "oscillatory instability"}],
        candidate_experiments=[
            {
                "name": "temporarily_damp_subsystem_X",
                "safety_profile": "low_risk_reversible",
                "expected_information_gain": 0.42,
                "expected_outcomes": {"drift-based degradation": 0.31, "oscillatory instability": 0.79},
            }
        ],
        measurements=[{"measurement": "increase_sampling_rate_sensor_Y"}],
        uncertainty={"uncertainty_magnitude": 0.66},
    )
    exp = plan["experiment_plan"]
    assert "disambiguate" in exp["objective"]
    assert exp["risk_level"] == "low"
    assert exp["advisory"] is True


def test_platform_emits_active_learning_block_and_confusion_flags() -> None:
    platform = StructuralSystemIntelligencePlatform(operating_mode="experimental")
    latest = {}
    for i in range(18):
        latest = platform.update(
            {
                "asset_id": "asset-x",
                "domain": "water",
                "system_type": "pump",
                "top_relationships": [
                    {"source": "pressure", "target": "flow"},
                    {"source": "vibration", "target": "temperature"},
                ],
                "subsystem_impact": {"subsystem_A": 0.9, "subsystem_B": 0.4},
                "sensor_readings": [0.1 + 0.01 * i],
                "composite_instability": 0.35 + 0.02 * i,
            }
        )
    active = latest["active_learning"]
    assert active["advisory_only"] is True
    assert active["candidate_experiments"]
    assert active["experiment_plan"]["expected_information_gain"] >= 0.0
    assert isinstance(active["confusion_flags"], list)
    assert "learning_efficiency_score" in active["learning_efficiency"]
