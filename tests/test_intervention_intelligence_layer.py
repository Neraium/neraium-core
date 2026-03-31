from __future__ import annotations

from neraium_core.system_intelligence.intervention_intelligence.engine import InterventionIntelligenceEngine
from neraium_core.system_intelligence.platform import StructuralSystemIntelligencePlatform


def _state(step: int, asset: str = "asset-1", intervention: dict | None = None) -> dict:
    obs = {
        "asset_id": asset,
        "structural_drift_score": 0.25 + 0.02 * step,
        "relational_instability_score": 0.22 + 0.02 * step,
        "regime_distance": 0.2 + 0.02 * step,
        "graph_deformation_score": 0.2 + 0.01 * step,
        "coherence_score": 0.8 - 0.02 * step,
        "coupling_instability_score": 0.3 + 0.01 * step,
        "risk_pressure": 0.35 + 0.02 * step,
        "path_length_shift": 0.1 + 0.01 * step,
        "subspace_rotation": 0.08 + 0.01 * step,
        "mean_shift": 0.09 + 0.01 * step,
        "covariance_shift": 0.08 + 0.01 * step,
        "top_relationships": [{"source": "A", "target": "B"}, {"source": "B", "target": "C"}],
        "subsystem_impact": {"sub1": 0.6, "sub2": 0.4},
        "composite_instability": 0.55 + 0.03 * step,
        "contribution_scores": {"coupling": 0.8, "entropy": 0.3},
    }
    if intervention is not None:
        obs["applied_intervention"] = intervention
    return obs


def _simulate_record(engine: InterventionIntelligenceEngine, helpful: bool = True) -> None:
    transition_pre = {
        "regime": "critical",
        "transition_path": "escalating",
        "escalation_probability": 0.85,
        "reversibility_score": 0.2,
        "distance_to_critical_region": 0.1,
    }
    transition_post = {
        "regime": "transitional",
        "transition_path": "drift",
        "escalation_probability": 0.35 if helpful else 0.9,
        "reversibility_score": 0.55 if helpful else 0.18,
        "distance_to_critical_region": 0.45 if helpful else 0.07,
    }
    trajectory = {"current_trajectory_path_family": "escalating", "novelty_score": 0.1}
    mechanism = {"mechanism_candidates": [{"mechanism": "cluster_decoupling:a->b"}]}
    laws = {"law_candidates": [{"law": "cluster_decoupling"}]}
    counterfactuals = {
        "scenario_rankings": [
            {"name": "remove_top_driver_contribution", "risk_delta": 0.3},
            {"name": "restore_relationship_cluster_to_baseline", "risk_delta": 0.2},
        ]
    }

    engine.update(
        asset_id="asset-x",
        observation={"latent_embedding": [0.1, 0.2, 0.3], "applied_intervention": {"type": "remove_or_suppress_top_driver", "target": "top_driver"}},
        transition=transition_pre,
        trajectory=trajectory,
        mechanism=mechanism,
        laws=laws,
        counterfactuals=counterfactuals,
    )
    engine.update(
        asset_id="asset-x",
        observation={"latent_embedding": [0.15, 0.16, 0.1]},
        transition=transition_post,
        trajectory=trajectory,
        mechanism=mechanism,
        laws=laws,
        counterfactuals=counterfactuals,
    )


def test_intervention_memory_records_created_and_updated() -> None:
    engine = InterventionIntelligenceEngine()
    _simulate_record(engine, helpful=True)
    summary = engine.memory.support_summary()
    assert summary["support_count"] == 1
    assert summary["helpful_interventions"]


def test_effectiveness_improves_with_repeated_helpful_support() -> None:
    engine = InterventionIntelligenceEngine()
    _simulate_record(engine, helpful=True)
    out1 = engine.update(
        asset_id="asset-y",
        observation={"latent_embedding": [0.2, 0.2, 0.2]},
        transition={"regime": "critical", "transition_path": "escalating", "escalation_probability": 0.8, "reversibility_score": 0.2, "distance_to_critical_region": 0.1},
        trajectory={"current_trajectory_path_family": "escalating", "novelty_score": 0.1},
        mechanism={"mechanism_candidates": [{"mechanism": "cluster_decoupling:a->b"}]},
        laws={"law_candidates": [{"law": "cluster_decoupling"}]},
        counterfactuals={"scenario_rankings": [{"name": "remove_top_driver_contribution", "risk_delta": 0.2}]},
    )
    score1 = out1["historical_evidence"]["effectiveness_by_intervention"]["remove_top_driver_contribution"]["intervention_effectiveness"]

    for _ in range(3):
        _simulate_record(engine, helpful=True)

    out2 = engine.update(
        asset_id="asset-z",
        observation={"latent_embedding": [0.2, 0.2, 0.2]},
        transition={"regime": "critical", "transition_path": "escalating", "escalation_probability": 0.82, "reversibility_score": 0.2, "distance_to_critical_region": 0.08},
        trajectory={"current_trajectory_path_family": "escalating", "novelty_score": 0.1},
        mechanism={"mechanism_candidates": [{"mechanism": "cluster_decoupling:a->b"}]},
        laws={"law_candidates": [{"law": "cluster_decoupling"}]},
        counterfactuals={"scenario_rankings": [{"name": "remove_top_driver_contribution", "risk_delta": 0.2}]},
    )
    score2 = out2["historical_evidence"]["effectiveness_by_intervention"]["remove_top_driver_contribution"]["intervention_effectiveness"]
    assert score2 >= score1


def test_weak_support_and_novel_context_are_penalized() -> None:
    engine = InterventionIntelligenceEngine()
    out = engine.update(
        asset_id="novel-1",
        observation={"latent_embedding": [0.4, 0.9, 0.7]},
        transition={"regime": "critical", "transition_path": "escalating", "escalation_probability": 0.88, "reversibility_score": 0.1, "distance_to_critical_region": 0.03},
        trajectory={"current_trajectory_path_family": "unknown_new_family", "novelty_score": 0.98},
        mechanism={"mechanism_candidates": [{"mechanism": "new_mechanism"}]},
        laws={"law_candidates": [{"law": "new_law"}]},
        counterfactuals={"scenario_rankings": [{"name": "restore_relationship_cluster_to_baseline", "risk_delta": 0.3}]},
    )
    eff = out["historical_evidence"]["effectiveness_by_intervention"]["restore_relationship_cluster_to_baseline"]
    assert eff["support"] == 0
    assert eff["novelty_penalty"] >= 0.6
    assert eff["uncertainty"] >= 0.5


def test_harmful_interventions_rank_lower_than_helpful() -> None:
    engine = InterventionIntelligenceEngine()
    for _ in range(3):
        _simulate_record(engine, helpful=True)
    for _ in range(3):
        _simulate_record(engine, helpful=False)

    out = engine.update(
        asset_id="rank-1",
        observation={"latent_embedding": [0.1, 0.3, 0.5]},
        transition={"regime": "critical", "transition_path": "escalating", "escalation_probability": 0.86, "reversibility_score": 0.2, "distance_to_critical_region": 0.1},
        trajectory={"current_trajectory_path_family": "escalating", "novelty_score": 0.2},
        mechanism={"mechanism_candidates": [{"mechanism": "cluster_decoupling:a->b"}]},
        laws={"law_candidates": [{"law": "cluster_decoupling"}]},
        counterfactuals={
            "scenario_rankings": [
                {"name": "remove_top_driver_contribution", "risk_delta": 0.25},
                {"name": "restore_relationship_cluster_to_baseline", "risk_delta": 0.2},
            ]
        },
    )
    ranked = out["recommendation"]["ranked_interventions"]
    assert ranked[0]["name"] == "remove_top_driver_contribution"


def test_platform_integration_preserves_operator_flow_and_warmup_safe() -> None:
    platform = StructuralSystemIntelligencePlatform()
    out = platform.update(_state(0))
    assert "compatibility" in out
    assert "intervention_intelligence" in out
    assert out["intervention_intelligence"]["recommendation"]["decision_trace"]

    out2 = platform.update(
        _state(
            1,
            intervention={"type": "remove_or_suppress_top_driver", "target": "top_driver", "confidence": 0.8},
        )
    )
    assert out2["intervention_intelligence"]["evidence_update"]["status"] in {"no_new_record", "recorded"}


def test_zero_memory_fallback_is_conservative() -> None:
    engine = InterventionIntelligenceEngine()
    out = engine.update(
        asset_id="cold-start",
        observation={"latent_embedding": [0.12, 0.22, 0.32]},
        transition={"regime": "transitional", "transition_path": "drift", "escalation_probability": 0.51, "reversibility_score": 0.22, "distance_to_critical_region": 0.12},
        trajectory={"current_trajectory_path_family": "drift", "novelty_score": 0.4},
        mechanism={"mechanism_candidates": []},
        laws={"law_candidates": []},
        counterfactuals={"scenario_rankings": [{"name": "remove_top_driver_contribution", "risk_delta": 0.12}]},
    )
    rec = out["recommendation"]
    assert out["historical_evidence"]["support_summary"]["support_count"] == 0
    assert rec["confidence"] < 0.5
    assert rec["advisory"] is True


def test_novelty_penalty_reduces_recommendation_confidence() -> None:
    engine = InterventionIntelligenceEngine()
    for _ in range(4):
        _simulate_record(engine, helpful=True)

    low_novel = engine.update(
        asset_id="low-novel",
        observation={"latent_embedding": [0.2, 0.2, 0.2]},
        transition={"regime": "critical", "transition_path": "escalating", "escalation_probability": 0.8, "reversibility_score": 0.2, "distance_to_critical_region": 0.1},
        trajectory={"current_trajectory_path_family": "escalating", "novelty_score": 0.1},
        mechanism={"mechanism_candidates": [{"mechanism": "cluster_decoupling:a->b"}]},
        laws={"law_candidates": [{"law": "cluster_decoupling"}]},
        counterfactuals={"scenario_rankings": [{"name": "remove_top_driver_contribution", "risk_delta": 0.2}]},
    )
    high_novel = engine.update(
        asset_id="high-novel",
        observation={"latent_embedding": [0.9, 0.9, 0.9]},
        transition={"regime": "critical", "transition_path": "escalating", "escalation_probability": 0.8, "reversibility_score": 0.2, "distance_to_critical_region": 0.1},
        trajectory={"current_trajectory_path_family": "unknown_new_family", "novelty_score": 0.95},
        mechanism={"mechanism_candidates": [{"mechanism": "cluster_decoupling:a->b"}]},
        laws={"law_candidates": [{"law": "cluster_decoupling"}]},
        counterfactuals={"scenario_rankings": [{"name": "remove_top_driver_contribution", "risk_delta": 0.2}]},
    )
    assert high_novel["recommendation"]["confidence"] <= low_novel["recommendation"]["confidence"]


def test_conflicting_evidence_keeps_confidence_moderate() -> None:
    engine = InterventionIntelligenceEngine()
    for _ in range(3):
        _simulate_record(engine, helpful=True)
        _simulate_record(engine, helpful=False)

    out = engine.update(
        asset_id="conflict-1",
        observation={"latent_embedding": [0.2, 0.3, 0.4]},
        transition={"regime": "critical", "transition_path": "escalating", "escalation_probability": 0.83, "reversibility_score": 0.2, "distance_to_critical_region": 0.08},
        trajectory={"current_trajectory_path_family": "escalating", "novelty_score": 0.2},
        mechanism={"mechanism_candidates": [{"mechanism": "cluster_decoupling:a->b"}]},
        laws={"law_candidates": [{"law": "cluster_decoupling"}]},
        counterfactuals={"scenario_rankings": [{"name": "remove_top_driver_contribution", "risk_delta": 0.22}]},
    )
    assert out["recommendation"]["confidence"] < 0.75


def test_compatibility_adapter_stays_conservative_when_confidence_is_weak() -> None:
    platform = StructuralSystemIntelligencePlatform()
    out = platform.update(_state(0))
    rec = str(out["compatibility"]["operational_recommendation"]).lower()
    assert "evidence limited" in rec or "monitor" in rec


def test_intervention_memory_ablation_meaningfully_changes_ranking_quality() -> None:
    engine = InterventionIntelligenceEngine()
    for _ in range(5):
        _simulate_record(engine, helpful=True)

    seeded = engine.update(
        asset_id="seeded",
        observation={"latent_embedding": [0.2, 0.2, 0.2]},
        transition={"regime": "critical", "transition_path": "escalating", "escalation_probability": 0.82, "reversibility_score": 0.22, "distance_to_critical_region": 0.1},
        trajectory={"current_trajectory_path_family": "escalating", "novelty_score": 0.1},
        mechanism={"mechanism_candidates": [{"mechanism": "cluster_decoupling:a->b"}]},
        laws={"law_candidates": [{"law": "cluster_decoupling"}]},
        counterfactuals={"scenario_rankings": [{"name": "remove_top_driver_contribution", "risk_delta": 0.2}]},
    )

    cold = InterventionIntelligenceEngine().update(
        asset_id="cold",
        observation={"latent_embedding": [0.2, 0.2, 0.2]},
        transition={"regime": "critical", "transition_path": "escalating", "escalation_probability": 0.82, "reversibility_score": 0.22, "distance_to_critical_region": 0.1},
        trajectory={"current_trajectory_path_family": "escalating", "novelty_score": 0.1},
        mechanism={"mechanism_candidates": [{"mechanism": "cluster_decoupling:a->b"}]},
        laws={"law_candidates": [{"law": "cluster_decoupling"}]},
        counterfactuals={"scenario_rankings": [{"name": "remove_top_driver_contribution", "risk_delta": 0.2}]},
    )
    seeded_conf = float(seeded["recommendation"]["confidence"])
    cold_conf = float(cold["recommendation"]["confidence"])
    assert seeded_conf - cold_conf >= 0.12


def test_structural_uncertainty_mode_activates_for_high_novelty_weak_support() -> None:
    engine = InterventionIntelligenceEngine()
    out = engine.update(
        asset_id="uncertain-1",
        observation={"latent_embedding": [0.85, 0.83, 0.82], "drift_warning": True},
        transition={"regime": "critical", "transition_path": "escalating", "escalation_probability": 0.9, "reversibility_score": 0.1, "distance_to_critical_region": 0.04},
        trajectory={"current_trajectory_path_family": "unknown_new_family", "novelty_score": 0.97, "family_similarity": 0.2, "support": 0},
        mechanism={"mechanism_candidates": []},
        laws={"law_candidates": []},
        counterfactuals={"scenario_rankings": [{"name": "remove_top_driver_contribution", "risk_delta": 0.4}]},
    )
    mode = out["structural_uncertainty_mode"]
    assert mode["active"] is True
    assert mode["recommended_posture"] == "human_review_required"
    assert out["recommendation"]["best_intervention"]["name"] == "monitor"
    assert out["recommendation"]["fallback_triggered"] is True


def test_production_hard_override_forces_monitor_and_human_review(monkeypatch) -> None:
    platform = StructuralSystemIntelligencePlatform(operating_mode="production")

    def _mock_intervention_update(**_: object) -> dict[str, object]:
        return {
            "recommendation": {
                "best_intervention": {"name": "remove_top_driver_contribution", "confidence": 0.91, "rank_score": 0.91},
                "ranked_interventions": [
                    {"name": "remove_top_driver_contribution", "confidence": 0.91},
                    {"name": "restore_relationship_cluster_to_baseline", "confidence": 0.62},
                ],
                "fallback_triggered": False,
                "fallback_reasons": [],
                "recommended_posture": "standard_advisory",
            },
            "historical_evidence": {"support_summary": {"support_count": 1}},
            "structural_uncertainty_mode": {
                "active": True,
                "reason": "high_novelty,weak_support",
                "novelty": 0.97,
                "support": 1,
                "reliability": 0.3,
                "recommended_posture": "human_review_required",
            },
        }

    monkeypatch.setattr(platform.production.intervention_intelligence, "update", _mock_intervention_update)
    out = platform.update(
        {
            **_state(0),
            "asset_id": "override-1",
            "drift_warning": True,
        }
    )

    intervention = out["production_intelligence"]["intervention_intelligence"]
    recommendation = intervention["recommendation"]
    compatibility = out["compatibility"]
    trace = out["production_intelligence"]["core_decision_trace"]

    assert recommendation["best_intervention"]["name"] == "monitor"
    assert recommendation["recommended_posture"] == "human_review_required"
    assert recommendation["override_applied"] is True
    assert compatibility["recommended_intervention"] == "monitor"
    assert compatibility["forced_posture"] == "human_review_required"
    assert compatibility["intervention_overridden"] is True
    assert trace["override_applied"] is True


def test_correlation_trap_penalty_is_exposed_in_ranking_trace() -> None:
    engine = InterventionIntelligenceEngine()
    _simulate_record(engine, helpful=True)
    _simulate_record(engine, helpful=True)
    out = engine.update(
        asset_id="corr-1",
        observation={"latent_embedding": [0.1, 0.2, 0.3]},
        transition={"regime": "critical", "transition_path": "escalating", "escalation_probability": 0.82, "reversibility_score": 0.2, "distance_to_critical_region": 0.1},
        trajectory={"current_trajectory_path_family": "escalating", "novelty_score": 0.2, "family_similarity": 0.8, "support": 4},
        mechanism={"mechanism_candidates": [{"mechanism": "cluster_decoupling:a->b"}]},
        laws={"law_candidates": [{"law": "cluster_decoupling"}]},
        counterfactuals={"scenario_rankings": [{"name": "remove_top_driver_contribution", "risk_delta": 0.2}]},
    )
    ranked = out["recommendation"]["ranked_interventions"][0]
    assert "correlation_trap_penalty" in ranked["ranking_factors"]
    assert "counterfactual_specificity_guard" in out["recommendation"]["decision_trace"]["confidence_contributions"]


def test_override_divergence_clusters_are_recorded() -> None:
    engine = InterventionIntelligenceEngine()
    engine.memory.ingest_feedback_event(intervention_type="remove_or_suppress_top_driver", action="overridden", outcome_label="helpful")
    engine.memory.ingest_feedback_event(intervention_type="remove_or_suppress_top_driver", action="overridden", outcome_label="neutral")
    engine.memory.ingest_feedback_event(intervention_type="remove_or_suppress_top_driver", action="accepted", outcome_label="harmful")
    summary = engine.memory.support_summary()
    clusters = summary["override_divergence_clusters"]
    assert "remove_or_suppress_top_driver" in clusters
    assert clusters["remove_or_suppress_top_driver"]["pattern"] in {
        "system_wrong_operator_better",
        "operator_wrong_system_better",
        "unresolved_tradeoff",
    }
