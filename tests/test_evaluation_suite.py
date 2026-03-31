from __future__ import annotations

from evaluation.ablation import run_ablation_suite
from evaluation.metrics import EvaluationRecord, compute_metrics
from evaluation.runner import run_evaluation
from evaluation.scenarios import build_benchmark_scenarios


def test_scenarios_run_without_error(tmp_path) -> None:
    scenarios = build_benchmark_scenarios(seed=7, include_noise_variants=False)
    out = run_evaluation(scenarios=scenarios, output_dir=tmp_path)
    assert out["records"]
    assert "neraium_full_stack" in out["aggregate_metrics"]


def test_metrics_compute_correctly() -> None:
    rows = [
        EvaluationRecord(
            model_id="m1",
            scenario_id="stable_system",
            scenario_type="stable system",
            domain="d",
            variant="base",
            predicted_risk=0.1,
            predicted_unstable=False,
            first_alert_step=None,
            recommended_intervention="none",
            confidence=0.2,
            calibration_probability=0.1,
            intervention_effect="neutral",
            critical_step=None,
            actual_unstable=False,
        ),
        EvaluationRecord(
            model_id="m1",
            scenario_id="case2",
            scenario_type="slow drift",
            domain="d",
            variant="base",
            predicted_risk=0.9,
            predicted_unstable=True,
            first_alert_step=3,
            recommended_intervention="remove_top_driver_contribution",
            confidence=0.8,
            calibration_probability=0.9,
            intervention_effect="helpful",
            critical_step=6,
            actual_unstable=True,
        ),
    ]
    m = compute_metrics(rows)["m1"]
    assert m["decision_quality"]["success_rate"] == 0.5
    assert m["false_positive_rate"] == 0.0
    assert m["early_warning_utility"]["lead_time_before_critical"] == 3.0


def test_ablation_toggles_work_and_consistent_outputs(tmp_path) -> None:
    scenarios = build_benchmark_scenarios(seed=5, include_noise_variants=False)
    run_evaluation(scenarios=scenarios, output_dir=tmp_path)
    ablation = run_ablation_suite(scenarios=scenarios, output_dir=tmp_path)
    assert ablation["performance_delta_per_component"]
    assert ablation["component_importance_report"]["most_important_component"] is not None
    assert (tmp_path / "aggregate_metrics.json").exists()
    assert (tmp_path / "ablation_report.json").exists()
