from __future__ import annotations

import json
from pathlib import Path

from neraium_core.system_intelligence.law_governance import StructuralLawGovernance
from neraium_core.system_intelligence.platform import StructuralSystemIntelligencePlatform
from validation import RealWorldValidationPipeline
from validation.replay.adapters import load_dataset


def _dataset() -> list[dict]:
    return [
        {
            "timestamp": 1,
            "asset_id": "A",
            "observation": {"x": 0.1, "sensor": 1.0},
            "actual_intervention": "remove_top_driver_contribution",
            "outcome_label": "helpful",
        },
        {
            "timestamp": 2,
            "asset_id": "A",
            "observation": {"x": 0.2, "sensor": 1.1},
            "actual_intervention": "restore_relationship_cluster_to_baseline",
            "outcome_label": "harmful",
        },
        {
            "timestamp": 3,
            "asset_id": "A",
            "observation": {"x": 0.15, "sensor": 1.05},
            "actual_intervention": None,
            "outcome_label": None,
        },
    ]


def test_replay_and_outcome_pipeline_runs() -> None:
    platform = StructuralSystemIntelligencePlatform(operating_mode="production")
    pipeline = RealWorldValidationPipeline(decision_fn=platform.update)
    report = pipeline.run(_dataset())

    assert len(report["replay"]["step_logs"]) == 3
    assert len(report["outcomes"]) == 3
    assert "real_world_validation" in report
    assert "drift_signals" in report["real_world_validation"]
    assert "core_validation_report" in report


def test_feedback_updates_intervention_memory_and_reliability() -> None:
    platform = StructuralSystemIntelligencePlatform(operating_mode="production")
    pipeline = RealWorldValidationPipeline(decision_fn=platform.update)
    report = pipeline.run(_dataset())

    memory = platform.production.intervention_intelligence.memory
    for rec in report["feedback"]:
        memory.ingest_feedback_event(
            intervention_type=str(rec.get("recommended_intervention") or "other"),
            action=str(rec.get("action")),
            outcome_label=str(rec.get("outcome_label") or "neutral"),
        )
    summary = memory.support_summary()
    assert "feedback_evidence" in summary

    finalized = platform.production.reliability.ingest_feedback_records(
        asset_id="A",
        step=10,
        feedback_records=[
            {"outcome_label": "harmful", "confidence": 0.92},
            {"outcome_label": "helpful", "confidence": 0.8},
        ],
    )
    assert finalized == 2


def test_real_world_law_validation_updates_score() -> None:
    governance = StructuralLawGovernance()
    evidence = governance.ingest_real_world_validation(
        law_id="law::rw",
        helpful_count=6,
        harmful_count=2,
        neutral_count=1,
    )
    assert evidence["support_count"] >= 7
    assert evidence["validation_score"] > 0.0


def test_json_adapter_supports_validation_inputs(tmp_path: Path) -> None:
    payload = {"events": _dataset()}
    path = tmp_path / "events.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    rows = load_dataset(path, "json")
    assert len(rows) == 3
    assert rows[0]["asset_id"] == "A"


def test_outcome_attribution_surfaces_confidence_and_source() -> None:
    platform = StructuralSystemIntelligencePlatform(operating_mode="production")
    pipeline = RealWorldValidationPipeline(decision_fn=platform.update)
    report = pipeline.run(_dataset())
    first = report["outcomes"][0]
    assert "attribution_confidence" in first
    assert "attribution_source" in first


def test_novelty_fallback_forces_conservative_monitoring() -> None:
    def decision_fn(_obs: dict) -> dict:
        return {
            "intervention_intelligence": {
                "recommendation": {
                    "best_intervention": {"name": "aggressive_action", "confidence": 0.91},
                    "intervention_memory_contribution": {"status": "available", "best_confidence_delta": 0.03, "choice_changed_without_memory": False},
                }
            },
            "reliability_intelligence": {
                "intervention_recommendation": {
                    "recommendation_calibrated_confidence": 0.42,
                    "reliability_trace": {"warnings": ["weak_support"]},
                }
            },
            "structural_law_intelligence": {"structural_law_governance": {"laws": []}},
        }

    pipeline = RealWorldValidationPipeline(decision_fn=decision_fn)
    rows = [
        {
            "timestamp": 1,
            "asset_id": "A",
            "novelty": 0.9,
            "support_count": 1,
            "drift_warning": True,
            "observation": {"x": 1.0},
            "actual_intervention": "aggressive_action",
            "outcome_label": "harmful",
        }
    ]
    report = pipeline.run(rows)
    step = report["replay"]["step_logs"][0]
    assert step["recommended_intervention"] == "monitor"
    assert step["fallback_triggered"] is True
    assert step["advisory_mode"] == "conservative_monitoring"


def test_corpus_summary_surfaces_representativeness_warnings() -> None:
    def decision_fn(_obs: dict) -> dict:
        return {"intervention_intelligence": {"recommendation": {"best_intervention": {"name": "monitor", "confidence": 0.3}}}}

    pipeline = RealWorldValidationPipeline(decision_fn=decision_fn)
    rows = [
        {"timestamp": i, "asset_id": "dominant", "domain": "single", "observation": {"x": i}, "actual_intervention": None, "outcome_label": "neutral"}
        for i in range(12)
    ]
    report = pipeline.run(rows)
    warnings = set(report["core_validation_report"]["corpus_summary"]["representativeness"]["warnings"])
    assert "dominant_asset_skew" in warnings
    assert "weak_domain_diversity" in warnings


def test_intervention_memory_contribution_not_mirrored_to_accuracy() -> None:
    def decision_fn(_obs: dict) -> dict:
        return {
            "intervention_intelligence": {
                "recommendation": {
                    "best_intervention": {"name": "monitor", "confidence": 0.5},
                    "intervention_memory_contribution": {
                        "status": "available",
                        "best_confidence_delta": 0.02,
                        "choice_changed_without_memory": True,
                    },
                }
            }
        }

    pipeline = RealWorldValidationPipeline(decision_fn=decision_fn)
    rows = [
        {"timestamp": i, "asset_id": f"A{i%2}", "domain": "d1", "observation": {"x": i}, "actual_intervention": None, "outcome_label": "neutral"}
        for i in range(8)
    ]
    report = pipeline.run(rows)
    contribution = report["core_validation_report"]["summary"]["intervention_memory_contribution"]
    assert contribution["status"] == "available"
    assert contribution["mean_best_confidence_delta"] != report["core_validation_report"]["summary"]["decision_accuracy"]
