from __future__ import annotations

from validation.pipeline import RealWorldValidationPipeline
from validation.release_policy import evaluate_corpus_release


def test_core_validation_marks_novelty_blocker_cases() -> None:
    artifact = RealWorldValidationPipeline._build_core_validation_artifact(
        step_logs=[
            {
                "timestep": 0,
                "timestamp": 0.0,
                "asset_id": "a-1",
                "domain": "energy",
                "scenario_family": "novel",
                "scenario_id": "s-1",
                "system_type": "pump",
                "recommended_intervention": "remove_top_driver_contribution",
                "actual_intervention": "remove_top_driver_contribution",
                "confidence": 0.8,
                "advisory_mode": "human_review_required",
                "fallback_triggered": True,
                "fallback_reasons": ["high_novelty", "weak_support"],
                "intervention_memory_contribution": {},
                "novelty": 0.92,
                "support_count": 1,
                "drift_warning": True,
            }
        ],
        outcomes=[{"timestep": 0, "outcome_label": "harmful", "attribution_confidence": 0.6}],
        metrics={"decision_accuracy": 0.0, "harm_rate": 1.0, "calibration_error": 0.8, "calibration_quality": 0.2, "false_confidence_events": 1},
        drift={"status": "warning", "warnings": ["high_drift"]},
    )
    row = artifact["timeline"][0]
    assert row["novelty_blocker_case"] is True
    assert artifact["cohort_breakdowns"]["novelty_weak_support_drift_subset"]["count"] == 1


def test_release_policy_surfaces_novelty_fallback_failure_tag() -> None:
    core_validation_report = {
        "summary": {
            "decision_accuracy": 0.5,
            "harm_rate": 0.3,
            "calibration_error": 0.4,
            "false_confidence_rate": 0.15,
            "drift_warning_rate": 0.2,
        },
        "corpus_summary": {"domain_count": 2, "asset_count": 3},
        "cohort_breakdowns": {
            "novelty_weak_support_drift_subset": {
                "count": 10,
                "harm_rate": 0.5,
                "false_confidence_rate": 0.2,
            }
        },
    }
    out = evaluate_corpus_release(core_validation_report, corpus_type="transfer_cross_domain")
    assert "novelty_fallback_failure" in out["failure_mode_tags"]
