from __future__ import annotations

from neraium_core.system_intelligence.law_governance import StructuralLawGovernance


def _candidate(
    *,
    law_id: str = "law::x",
    support: int = 10,
    confidence: float = 0.86,
    robustness: float = 0.9,
    consistency: float = 0.82,
    counterfactual_consistency: float = 0.6,
) -> dict:
    return {
        "law_id": law_id,
        "condition": "family escalation with weakening triad",
        "support": support,
        "robustness": robustness,
        "confidence": confidence,
        "consistency_across_assets_runs": consistency,
        "counterfactual_consistency": counterfactual_consistency,
        "applicability": {"archetypes": ["degeneration"]},
        "mechanism": "triad_weakening:A|B|C",
    }


def test_contradictory_evidence_blocks_promotion() -> None:
    governance = StructuralLawGovernance()
    out = governance.evaluate(
        law_candidates=[_candidate()],
        trajectory_info={"novelty_score": 0.1},
        falsification={
            "falsification_summary": {
                "law_evaluations": [
                    {"law_id": "law::x", "counterexample_summary": {"contradiction_count": 12}},
                ]
            }
        },
    )
    law = out["laws"][0]
    assert law["current_stage"] == "rejected_or_falsified"
    assert out["blocked_promotions"]


def test_intervention_evidence_is_required_for_decision_grade() -> None:
    governance = StructuralLawGovernance()
    out = governance.evaluate(
        law_candidates=[_candidate(counterfactual_consistency=0.1)],
        trajectory_info={"novelty_score": 0.05},
        reliability={
            "law_applicability": [
                {
                    "law_id": "law::x",
                    "calibrated_applicability_confidence": 0.9,
                    "reliability_trace": {"reliability_band": "high", "support_band": "strong", "novelty_penalty": 0.1, "transfer_penalty": 0.05},
                }
            ]
        },
    )
    law = out["laws"][0]
    assert law["current_stage"] != "decision_grade_heuristic"
    assert "intervention_signal_too_weak" in law["promotion_blockers"]


def test_demotion_occurs_when_evidence_weakens() -> None:
    governance = StructuralLawGovernance()
    base = _candidate(counterfactual_consistency=0.8)

    first = governance.evaluate(
        law_candidates=[base],
        trajectory_info={"novelty_score": 0.05},
        reliability={
            "law_applicability": [
                {
                    "law_id": "law::x",
                    "calibrated_applicability_confidence": 0.9,
                    "reliability_trace": {"reliability_band": "high", "support_band": "strong", "novelty_penalty": 0.08, "transfer_penalty": 0.05},
                }
            ]
        },
        intervention_info={"scenario_rankings": [{"risk_delta": 0.6}]},
    )
    assert first["laws"][0]["current_stage"] in {"intervention_sensitive_pattern", "decision_grade_heuristic"}

    second = governance.evaluate(
        law_candidates=[_candidate(counterfactual_consistency=0.1, robustness=0.2)],
        trajectory_info={"novelty_score": 0.05},
        falsification={
            "falsification_summary": {
                "law_evaluations": [
                    {"law_id": "law::x", "counterexample_summary": {"contradiction_count": 8}},
                ]
            }
        },
    )
    assert second["demotions"]
    assert second["laws"][0]["current_stage"] in {"repeated_pattern", "context_conditioned_pattern", "rejected_or_falsified"}


def test_decision_grade_requires_real_world_and_intervention_sensitive_evidence() -> None:
    governance = StructuralLawGovernance()
    governance.ingest_real_world_validation(
        law_id="law::x",
        helpful_count=8,
        harmful_count=1,
        neutral_count=1,
        intervention_sensitive_helpful_count=1,
        intervention_sensitive_harmful_count=0,
    )
    out = governance.evaluate(
        law_candidates=[_candidate(counterfactual_consistency=0.9, support=14, confidence=0.9, consistency=0.88)],
        trajectory_info={"novelty_score": 0.05},
        reliability={
            "law_applicability": [
                {
                    "law_id": "law::x",
                    "calibrated_applicability_confidence": 0.9,
                    "reliability_trace": {"reliability_band": "high", "support_band": "strong", "novelty_penalty": 0.08, "transfer_penalty": 0.05},
                }
            ]
        },
        intervention_info={"scenario_rankings": [{"risk_delta": 0.6}]},
    )
    law = out["laws"][0]
    assert law["current_stage"] != "decision_grade_heuristic"
    assert "intervention_sensitive_evidence_insufficient" in law["promotion_blockers"]
