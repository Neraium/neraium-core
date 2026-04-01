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


def test_tiny_deprecated_corpora_excluded_from_default_release_decision() -> None:
    results = [
        {"corpus_id": "baseline_clean_ops", "corpus_type": "baseline_clean", "release_passed": True, "gate_breakdown": [], "failure_mode_tags": [], "corpus_summary": {"total_records": 120}},
        {"corpus_id": "noisy_realistic_ops", "corpus_type": "noisy_realistic", "release_passed": True, "gate_breakdown": [], "failure_mode_tags": [], "corpus_summary": {"total_records": 120}},
        {"corpus_id": "transfer_cross_domain_large_scale", "corpus_type": "transfer_cross_domain", "release_passed": True, "gate_breakdown": [], "failure_mode_tags": [], "corpus_summary": {"total_records": 120}},
        {"corpus_id": "adv_misleading_stability", "corpus_type": "adversarial", "release_passed": True, "gate_breakdown": [], "failure_mode_tags": [], "corpus_summary": {"total_records": 120}},
        {"corpus_id": "corpus_v1", "corpus_type": "baseline_clean", "release_passed": False, "gate_breakdown": [{"gate": "minimum_decision_accuracy", "passed": False}], "failure_mode_tags": ["trajectory_misclassification"], "corpus_summary": {"total_records": 4}},
    ]

    default_agg = aggregate_multi_corpus_release(results, min_credible_records=50)
    assert default_agg["release_passed"] is True
    assert "corpus_v1" in default_agg["excluded_non_credible_corpora"]

    diagnostics_agg = aggregate_multi_corpus_release(results, min_credible_records=50, include_ineligible=True)
    assert diagnostics_agg["release_passed"] is False
    assert "baseline_clean" in diagnostics_agg["blocking_corpus_classes"]


def test_missing_required_classes_reported_separately_from_blocking_failures() -> None:
    results = [
        {"corpus_id": "baseline_clean_ops", "corpus_type": "baseline_clean", "release_passed": False, "gate_breakdown": [{"gate": "maximum_calibration_error", "passed": False}], "failure_mode_tags": ["calibration_failure"]},
    ]

    agg = aggregate_multi_corpus_release(results)
    assert agg["release_passed"] is False
    assert agg["blocking_corpus_classes"] == ["baseline_clean"]
    assert agg["missing_required_corpus_classes"] == ["noisy_realistic", "transfer_cross_domain"]
    assert agg["failing_corpora"] == ["baseline_clean_ops"]
