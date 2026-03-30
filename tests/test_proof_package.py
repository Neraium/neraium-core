from __future__ import annotations

import json
from pathlib import Path

from neraium_core.proof_package import run_proof_package


def _scenario(summary: dict, name: str) -> dict:
    return next(s for s in summary["scenarios"] if s["scenario"] == name)


def test_proof_runner_completes_and_writes_outputs(tmp_path: Path) -> None:
    summary = run_proof_package(tmp_path)
    assert summary["scenario_count"] == 5
    artifacts = summary["artifacts"]
    assert Path(artifacts["summary_json"]).exists()
    assert Path(artifacts["summary_md"]).exists()
    assert Path(artifacts["founder_one_glance_md"]).exists()


def test_stable_scenario_does_not_false_escalate(tmp_path: Path) -> None:
    summary = run_proof_package(tmp_path)
    stable = _scenario(summary, "stable_baseline")
    assert stable["threshold_first_trigger_cycle"] is None
    assert stable["neraium_warning_cycle"] is None


def test_gradual_drift_warning_not_later_than_threshold(tmp_path: Path) -> None:
    summary = run_proof_package(tmp_path)
    drift = _scenario(summary, "gradual_drift")
    assert drift["neraium_warning_cycle"] is not None
    assert drift["threshold_first_trigger_cycle"] is not None
    assert drift["lead_cycles_neraium_vs_threshold"] >= 0


def test_spike_recovers_without_permanent_escalation(tmp_path: Path) -> None:
    summary = run_proof_package(tmp_path)
    spike = _scenario(summary, "abrupt_spike")
    assert spike["neraium_warning_cycle"] is not None
    assert spike["sustained_warning_in_tail"] is False


def test_messy_data_emits_explicit_quality_states(tmp_path: Path) -> None:
    summary = run_proof_package(tmp_path)
    messy = _scenario(summary, "messy_data")
    statuses = set(messy["quality_statuses_seen"])
    assert statuses
    assert "DATA_QUALITY_LIMITED" in statuses or "TEMPORAL_IRREGULARITY" in statuses


def test_proof_package_deterministic_summary(tmp_path: Path) -> None:
    first = run_proof_package(tmp_path / "run1")
    second = run_proof_package(tmp_path / "run2")

    def normalized(payload: dict) -> dict:
        copied = json.loads(json.dumps(payload))
        copied.pop("generated_at", None)
        copied.pop("artifacts", None)
        for s in copied["scenarios"]:
            s.pop("artifacts", None)
        return copied

    assert normalized(first) == normalized(second)
