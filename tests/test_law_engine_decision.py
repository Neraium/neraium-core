from __future__ import annotations

from neraium_core.system_intelligence.law_engine import StructuralLawDecisionEngine
from neraium_core.system_intelligence.platform import StructuralSystemIntelligencePlatform


def _law(
    law_id: str,
    support: int,
    effect: float,
    robustness: float,
    novelty_ready_conf: float,
    family: str = "escalating",
    cf_consistency: float = 0.65,
) -> dict:
    return {
        "law_id": law_id,
        "mechanism": "triad_weakening:A|B|C",
        "support": support,
        "estimated_effect": effect,
        "robustness": robustness,
        "confidence": novelty_ready_conf,
        "consistency_across_assets_runs": 0.8,
        "trajectory_family_support": 0.82,
        "counterfactual_consistency": cf_consistency,
        "outcome": "Escalation tendency increases in subsequent transition phase.",
        "applicability": {"trajectory_family": family, "archetypes": ["degeneration"]},
    }


def test_strong_supported_law_influences_recommendations_when_decision_grade() -> None:
    engine = StructuralLawDecisionEngine()
    out = {}
    for support in range(4, 13):
        out = engine.evaluate(
            law_candidates={"status": "ready", "law_candidates": [_law("law::strong", support, 0.42, 0.9, 0.88)]},
            trajectory_info={"current_trajectory_path_family": "escalating", "novelty_score": 0.08},
            mechanism_info={"mechanism_candidates": [{"mechanism": "triad_weakening:A|B|C", "family_conditioned_predictive_value": 0.88}]},
            risk_assessment={"current_risk_level": "medium", "projected_score": 0.62},
            operator_guidance={"recommended_actions": ["Inspect subsystem A"]},
            counterfactuals={"best_intervention": {"name": "restore_relationship_cluster_to_baseline"}, "scenario_rankings": [{"risk_delta": 0.55}]},
        )

    assert out["decision_eligible_laws"]
    assert out["matched_law_ids"] == ["law::strong"]
    assert out["law_influence_weight"] > 0.0
    assert out["law_informed_recommendations"][0].startswith("Law-layer match law::strong")


def test_weak_or_novel_law_stays_suppressed_and_bounded() -> None:
    engine = StructuralLawDecisionEngine()
    out = engine.evaluate(
        law_candidates={"status": "ready", "law_candidates": [_law("law::weak", 2, 0.07, 0.22, 0.2)]},
        trajectory_info={"current_trajectory_path_family": "escalating", "novelty_score": 0.95},
        mechanism_info={"mechanism_candidates": []},
        risk_assessment={"current_risk_level": "high", "projected_score": 0.88},
        operator_guidance={"recommended_actions": ["Inspect subsystem A"]},
        counterfactuals={"best_intervention": {"name": "remove_top_driver_contribution"}},
    )

    assert not out["decision_eligible_laws"]
    assert out["suppressed_laws"]
    assert abs(out["law_risk_delta"]) <= 0.12
    assert out["law_layer_message"].startswith("Law-layer evidence is weak/novel")


def test_conflicting_laws_reduce_influence_weight() -> None:
    engine = StructuralLawDecisionEngine()
    escalating = _law("law::escalate", 10, 0.33, 0.88, 0.9, cf_consistency=0.6)
    recovery = _law("law::recover", 10, -0.31, 0.86, 0.89, cf_consistency=0.58)
    recovery["outcome"] = "Recovery tendency increases under matched family."

    out = {}
    for _ in range(4):
        out = engine.evaluate(
            law_candidates={"status": "ready", "law_candidates": [escalating, recovery]},
            trajectory_info={"current_trajectory_path_family": "escalating", "novelty_score": 0.05},
            mechanism_info={"mechanism_candidates": []},
            risk_assessment={"current_risk_level": "medium", "projected_score": 0.55},
            operator_guidance={"recommended_actions": ["Inspect subsystem A"]},
            counterfactuals={"best_intervention": {"name": "suppress_subsystem_instability"}, "scenario_rankings": [{"risk_delta": 0.6}]},
        )

    assert len(out["matched_law_ids"]) == 2
    assert out["law_influence_weight"] <= 0.35


def test_platform_exposes_traceable_law_decision_outputs() -> None:
    platform = StructuralSystemIntelligencePlatform(operating_mode="research_assistive")
    out = {}
    for i in range(30):
        out = platform.update(
            {
                "asset_id": "asset-1" if i % 2 == 0 else "asset-2",
                "structural_drift_score": 0.08 + 0.03 * i,
                "relational_instability_score": 0.05 + 0.026 * i,
                "regime_distance": 0.03 + 0.02 * i,
                "graph_deformation_score": 0.05 + 0.02 * i,
                "coherence_score": max(0.1, 0.95 - 0.02 * i),
                "coupling_instability_score": 0.04 + 0.02 * i,
                "risk_pressure": 0.2 + 0.02 * i,
                "path_length_shift": 0.01 + 0.01 * i,
                "subspace_rotation": 0.01 + 0.01 * i,
                "mean_shift": 0.02 + 0.01 * i,
                "covariance_shift": 0.02 + 0.01 * i,
                "composite_instability": 0.04 + 0.02 * i,
                "contribution_scores": {"structural": 0.7, "graph": 0.6},
                "top_relationships": [{"source": "A", "target": "B"}, {"source": "B", "target": "C"}],
                "subsystem_impact": {"sub_a": 0.8, "sub_b": 0.7},
            }
        )

    law = out["law_engine_decision"]
    assert "law_decision_trace" in law
    assert "suppressed_candidates" in law
    assert "matched_law_ids" in law
    assert "law_adjusted_risk" in law
    assert "structural_law_governance" in law
