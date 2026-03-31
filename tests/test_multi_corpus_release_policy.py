from __future__ import annotations

from validation.release_policy import aggregate_multi_corpus_release, classify_failure_modes, evaluate_corpus_release


def _core_report(acc: float = 0.82, harm: float = 0.08, calibration_error: float = 0.21) -> dict:
    return {
        "summary": {
            "decision_accuracy": acc,
            "harm_rate": harm,
            "calibration_error": calibration_error,
            "calibration_quality": round(max(0.0, 1.0 - calibration_error), 6),
            "false_confidence_rate": 0.05,
            "drift_warning_rate": 0.1,
        },
        "corpus_summary": {"total_records": 80, "domain_count": 2, "asset_count": 6},
    }


def test_multi_corpus_aggregation_blocks_required_class_failure() -> None:
    results = [
        {"corpus_id": "b", "corpus_type": "baseline_clean", "release_passed": True, "gate_breakdown": [], "failure_mode_tags": []},
        {"corpus_id": "n", "corpus_type": "noisy_realistic", "release_passed": False, "gate_breakdown": [{"gate": "minimum_decision_accuracy", "passed": False}], "failure_mode_tags": ["trajectory_misclassification"]},
        {"corpus_id": "t", "corpus_type": "transfer_cross_domain", "release_passed": True, "gate_breakdown": [], "failure_mode_tags": []},
    ]
    agg = aggregate_multi_corpus_release(results)
    assert agg["release_passed"] is False
    assert "noisy_realistic" in agg["blocking_corpus_classes"]


def test_adversarial_limited_degradation_tolerated() -> None:
    results = [
        {"corpus_id": "b", "corpus_type": "baseline_clean", "release_passed": True, "gate_breakdown": [], "failure_mode_tags": []},
        {"corpus_id": "n", "corpus_type": "noisy_realistic", "release_passed": True, "gate_breakdown": [], "failure_mode_tags": []},
        {"corpus_id": "t", "corpus_type": "transfer_cross_domain", "release_passed": True, "gate_breakdown": [], "failure_mode_tags": []},
        {"corpus_id": "a1", "corpus_type": "adversarial", "release_passed": False, "gate_breakdown": [{"gate": "minimum_decision_accuracy", "passed": False}], "failure_mode_tags": ["trajectory_misclassification"]},
        {"corpus_id": "a2", "corpus_type": "adversarial", "release_passed": True, "gate_breakdown": [], "failure_mode_tags": []},
        {"corpus_id": "a3", "corpus_type": "adversarial", "release_passed": True, "gate_breakdown": [], "failure_mode_tags": []},
    ]
    agg = aggregate_multi_corpus_release(results)
    assert agg["release_passed"] is True


def test_transfer_failure_tagging() -> None:
    gate_breakdown = [{"gate": "minimum_decision_accuracy", "passed": False}]
    tags = classify_failure_modes("transfer_cross_domain", gate_breakdown)
    assert "transfer_failure" in tags


def test_corpus_specific_thresholds_enforced() -> None:
    baseline_gate = evaluate_corpus_release(_core_report(acc=0.70), corpus_type="baseline_clean")
    adversarial_gate = evaluate_corpus_release(_core_report(acc=0.70), corpus_type="adversarial")
    assert baseline_gate["release_passed"] is False
    assert adversarial_gate["release_passed"] is True
