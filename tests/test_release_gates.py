from __future__ import annotations

from validation.release_gates import ReleaseGateThresholds, evaluate_release_gates


def _core_report(
    *,
    decision_accuracy: float = 0.82,
    harm_rate: float = 0.08,
    calibration_error: float = 0.21,
    false_confidence_rate: float = 0.04,
    drift_warning_rate: float = 0.10,
    total_records: int = 120,
    domain_count: int = 2,
    asset_count: int = 8,
) -> dict:
    return {
        "summary": {
            "decision_accuracy": decision_accuracy,
            "harm_rate": harm_rate,
            "calibration_error": calibration_error,
            "calibration_quality": round(max(0.0, 1.0 - calibration_error), 6),
            "false_confidence_rate": false_confidence_rate,
            "drift_warning_rate": drift_warning_rate,
            "intervention_memory_contribution": 0.05,
        },
        "corpus_summary": {
            "total_records": total_records,
            "domain_count": domain_count,
            "asset_count": asset_count,
        },
    }


def test_release_gates_pass_when_all_thresholds_met() -> None:
    report = evaluate_release_gates(_core_report())
    assert report["release_passed"] is True
    assert report["release_recommendation"] == "ship"


def test_release_gates_fail_on_low_accuracy() -> None:
    report = evaluate_release_gates(_core_report(decision_accuracy=0.50))
    assert report["release_passed"] is False
    assert any(g["gate"] == "minimum_decision_accuracy" and not g["passed"] for g in report["gate_breakdown"])


def test_release_gates_fail_on_high_harm() -> None:
    report = evaluate_release_gates(_core_report(harm_rate=0.35))
    assert report["release_passed"] is False
    assert any(g["gate"] == "maximum_harm_rate" and not g["passed"] for g in report["gate_breakdown"])


def test_release_gates_fail_on_calibration_drop() -> None:
    report = evaluate_release_gates(_core_report(calibration_error=0.50))
    assert report["release_passed"] is False
    assert any(g["gate"] == "maximum_calibration_error" and not g["passed"] for g in report["gate_breakdown"])


def test_release_gates_fail_on_drift_warning_rate() -> None:
    report = evaluate_release_gates(_core_report(drift_warning_rate=0.45))
    assert report["release_passed"] is False
    assert any(g["gate"] == "maximum_drift_warning_rate" and not g["passed"] for g in report["gate_breakdown"])


def test_release_gates_fail_on_small_corpus() -> None:
    report = evaluate_release_gates(_core_report(total_records=8, asset_count=1))
    assert report["release_passed"] is False
    failed = {g["gate"] for g in report["gate_breakdown"] if not g["passed"]}
    assert "minimum_data_volume" in failed
    assert "minimum_asset_coverage" in failed


def test_release_gates_regression_blocker() -> None:
    current = _core_report(decision_accuracy=0.70, harm_rate=0.2, calibration_error=0.3, drift_warning_rate=0.2)
    previous = _core_report(decision_accuracy=0.80, harm_rate=0.1, calibration_error=0.2, drift_warning_rate=0.1)
    thresholds = ReleaseGateThresholds(regression_accuracy_drop_limit=0.05)
    report = evaluate_release_gates(current, previous_core_report=previous, thresholds=thresholds)
    assert report["release_passed"] is False
    assert report["regression_analysis"]["release_regression_blockers"]


def test_release_gates_fail_on_severe_representativeness_skew() -> None:
    report = evaluate_release_gates(
        {
            **_core_report(),
            "corpus_summary": {
                "total_records": 120,
                "domain_count": 2,
                "asset_count": 8,
                "representativeness": {"warnings": ["dominant_asset_skew"]},
            },
        }
    )
    assert report["release_passed"] is False
    assert any(g["gate"] == "representativeness_safety" and not g["passed"] for g in report["gate_breakdown"])
