from __future__ import annotations

from validation.metrics.backtest import compute_backtest_metrics


def test_calibration_confidence_improves_error_without_changing_accuracy_or_harm() -> None:
    decision_logs = [
        {
            "timestep": 0,
            "recommended_intervention": "monitor",
            "actual_intervention": None,
            "fallback_triggered": True,
            "confidence": 0.15,
            "calibration_confidence": 0.68,
        },
        {
            "timestep": 1,
            "recommended_intervention": "remove_top_driver_contribution",
            "actual_intervention": "remove_top_driver_contribution",
            "fallback_triggered": False,
            "confidence": 0.88,
            "calibration_confidence": 0.95,
        },
    ]
    outcomes = [
        {"timestep": 0, "outcome_label": "neutral"},
        {"timestep": 1, "outcome_label": "helpful"},
    ]

    baseline_logs = [{**row, "calibration_confidence": row["confidence"]} for row in decision_logs]
    baseline = compute_backtest_metrics(baseline_logs, outcomes)
    refined = compute_backtest_metrics(decision_logs, outcomes)

    assert refined["decision_accuracy"] == baseline["decision_accuracy"]
    assert refined["harm_rate"] == baseline["harm_rate"]
    assert refined["calibration_error"] < baseline["calibration_error"]


def test_monitoring_frequency_is_treated_as_non_intervention_for_alignment() -> None:
    decision_logs = [
        {"timestep": 0, "recommended_intervention": "no_action_recommended", "actual_intervention": None, "confidence": 0.1},
        {"timestep": 1, "recommended_intervention": "no_action_recommended", "actual_intervention": "increase_monitoring_frequency", "confidence": 0.1},
        {"timestep": 2, "recommended_intervention": "no_action_recommended", "actual_intervention": "remove_top_driver_contribution", "confidence": 0.1},
        {"timestep": 3, "recommended_intervention": "no_action_recommended", "actual_intervention": None, "confidence": 0.1},
    ]
    outcomes = [
        {"timestep": 0, "outcome_label": "neutral"},
        {"timestep": 1, "outcome_label": "harmful"},
        {"timestep": 2, "outcome_label": "helpful"},
        {"timestep": 3, "outcome_label": "neutral"},
    ]
    metrics = compute_backtest_metrics(decision_logs, outcomes)
    assert metrics["decision_accuracy"] == 0.75
