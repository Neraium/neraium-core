"""Tests for multi-corpus release aggregation (reporting consistency)."""

from __future__ import annotations

from validation.release_policy import aggregate_multi_corpus_release


def _passing_row(*, corpus_id: str, corpus_type: str) -> dict:
    return {
        "corpus_id": corpus_id,
        "corpus_type": corpus_type,
        "eligible_for_release_decision": True,
        "release_passed": True,
        "gate_breakdown": [{"gate": "minimum_decision_accuracy", "passed": True}],
        "failure_mode_tags": [],
    }


def test_no_false_release_fail_when_all_required_classes_pass() -> None:
    """If every required corpus class is represented and passes, aggregate must ship."""
    results = [
        _passing_row(corpus_id="baseline_clean_ops", corpus_type="baseline_clean"),
        _passing_row(corpus_id="noisy_realistic_ops", corpus_type="noisy_realistic"),
        _passing_row(corpus_id="transfer_small", corpus_type="transfer_cross_domain"),
    ]
    agg = aggregate_multi_corpus_release(results, min_credible_records=50)
    assert agg["release_passed"] is True
    assert agg["release_recommendation"] == "ship"
    assert agg["blocking_corpus_classes"] == []
    assert agg["missing_required_corpus_classes"] == []
    assert agg["failing_corpora"] == []


def test_missing_required_classes_are_blocking_and_reported() -> None:
    """Incomplete runs must fail with explicit missing class blockers, not empty lists."""
    results = [_passing_row(corpus_id="baseline_clean_ops", corpus_type="baseline_clean")]
    agg = aggregate_multi_corpus_release(results, min_credible_records=50)
    assert agg["release_passed"] is False
    assert agg["release_recommendation"] == "no-ship"
    assert set(agg["missing_required_corpus_classes"]) == {"noisy_realistic", "transfer_cross_domain"}
    assert set(agg["blocking_corpus_classes"]) == {"noisy_realistic", "transfer_cross_domain"}
    assert agg["failing_corpora"] == []


def test_status_matches_aggregate_release_passed() -> None:
    """Final status line must match aggregate release_passed (no split-brain)."""
    results = [_passing_row(corpus_id="baseline_clean_ops", corpus_type="baseline_clean")]
    agg = aggregate_multi_corpus_release(results, min_credible_records=50)
    status = "RELEASE_PASS" if agg.get("release_passed") else "RELEASE_FAIL"
    assert status == "RELEASE_FAIL"
    assert agg["release_recommendation"] == "no-ship"
