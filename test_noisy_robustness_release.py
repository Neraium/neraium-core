from __future__ import annotations

from argparse import Namespace

from tools.run_validation import execute_validation
from validation.release_policy import CORPUS_TYPE_THRESHOLDS


def _run_corpus(corpus_id: str) -> dict:
    return execute_validation(
        Namespace(
            input=None,
            format=None,
            corpus_id=corpus_id,
            corpus_root="validation/corpus",
            output=f"/tmp/{corpus_id}_test_report.json",
            core_output=f"/tmp/{corpus_id}_test_core.json",
            release_gate_output=f"/tmp/{corpus_id}_test_release.json",
            prior_core_report=None,
            history_root="/tmp/validation_history_tests",
            write_history=False,
            mark_baseline=False,
        )
    )


def test_noisy_realistic_ops_calibration_improves_without_clean_regression() -> None:
    baseline = _run_corpus("baseline_clean_ops")
    noisy = _run_corpus("noisy_realistic_ops")

    baseline_summary = baseline["core_validation_report"]["summary"]
    noisy_summary = noisy["core_validation_report"]["summary"]
    baseline_release = baseline["release_gate_report"]
    noisy_release = noisy["release_gate_report"]

    baseline_thresholds = CORPUS_TYPE_THRESHOLDS["baseline_clean"]
    noisy_thresholds = CORPUS_TYPE_THRESHOLDS["noisy_realistic"]

    # Baseline must remain release-safe.
    assert baseline_release["release_passed"] is True
    assert baseline_summary["decision_accuracy"] >= baseline_thresholds.min_decision_accuracy
    assert baseline_summary["calibration_error"] <= baseline_thresholds.max_calibration_error

    # Noisy robustness: calibration should improve from previously-verified failing range.
    assert noisy_summary["calibration_error"] <= 0.55
    assert noisy_summary["decision_accuracy"] >= noisy_thresholds.min_decision_accuracy

    # Release reporting consistency checks.
    assert noisy_release["release_decision"]["release_recommendation"] == noisy_release["release_recommendation"]
    assert baseline_release["release_decision"]["release_recommendation"] == baseline_release["release_recommendation"]
