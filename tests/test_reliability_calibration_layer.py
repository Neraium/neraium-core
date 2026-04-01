from __future__ import annotations

from neraium_core.system_intelligence.reliability.layer import StructuralReliabilityLayer


def _base_inputs(*, novelty: float = 0.2, rec_support: int = 1, rec_conf: float = 0.7, law_conf: float = 0.7, law_support: int = 3):
    transition = {"regime": "critical", "transition_path": "escalating", "escalation_probability": 0.7, "uncertainty": 0.2}
    forecast = {
        "likely_next_path_family": "escalating",
        "trajectory_conditioned_escalation_probability": 0.7,
        "matched_historical_progressions": [{"path_family": "escalating", "frequency": 2}],
        "uncertainty": 0.25,
        "novelty_aware_confidence_penalty": novelty,
    }
    laws = {
        "law_candidates": [
            {
                "law_id": "law::1",
                "mechanism": "A->B",
                "support": law_support,
                "confidence": law_conf,
                "consistency_across_assets_runs": 0.75,
                "applicability": {"trajectory_family": "escalating"},
            }
        ]
    }
    intervention = {
        "context": {"trajectory_family": "escalating", "regime": "critical", "novelty_score": novelty},
        "historical_evidence": {"support_summary": {"support_count": rec_support}},
        "uncertainty_summary": {"confidence": rec_conf},
        "recommendation": {"best_intervention": {"name": "remove_top_driver_contribution", "confidence": rec_conf}, "ranked_interventions": []},
    }
    cross = {
        "evidence_source_weights": {"local_evidence_weight": 0.8, "global_evidence_weight": 0.2, "mismatch_penalty": 0.1},
        "shared_trajectory_knowledge": {"global_support": 12},
        "shared_law_intelligence": [{"law_id": "law::1", "status": "globally_consistent", "generalization_confidence": 0.72, "cross_system_support": 10}],
        "shared_intervention_evidence": [],
        "stale_evidence_penalty": 0.0,
    }
    return transition, forecast, laws, intervention, cross


def test_weak_support_and_novelty_reduce_calibrated_confidence() -> None:
    layer = StructuralReliabilityLayer()
    transition, forecast, laws, intervention, cross = _base_inputs(novelty=0.8, rec_support=1, rec_conf=0.85)
    out = layer.calibrate_all(
        asset_id="a",
        step=1,
        transition=transition,
        trajectory_forecast=forecast,
        laws=laws,
        intervention=intervention,
        cross=cross,
    )
    rec = out["intervention_recommendation"]
    assert rec["recommendation_calibrated_confidence"] < rec["recommendation_raw_confidence"]
    assert rec["novelty_penalty"] > 0.15


def test_context_bucket_fallback_and_warmup_safe_behavior() -> None:
    layer = StructuralReliabilityLayer()
    transition, forecast, laws, intervention, cross = _base_inputs(rec_support=0)
    out = layer.calibrate_all(
        asset_id="a",
        step=1,
        transition=transition,
        trajectory_forecast=forecast,
        laws=laws,
        intervention=intervention,
        cross=cross,
    )
    trace = out["trajectory_forecast"]["reliability_trace"]
    assert trace["fallback_used"] is True
    assert trace["bucket_support"] == 0
    assert trace["calibrated_confidence"] <= trace["raw_confidence"]


def test_intervention_reliability_improves_with_repeated_success() -> None:
    layer = StructuralReliabilityLayer()
    transition, forecast, laws, intervention, cross = _base_inputs(rec_support=5, rec_conf=0.6)

    for step in range(1, 14):
        layer.finalize_due(asset_id="asset", step=step, transition=transition)
        out = layer.calibrate_all(
            asset_id="asset",
            step=step,
            transition=transition,
            trajectory_forecast=forecast,
            laws=laws,
            intervention=intervention,
            cross=cross,
        )
    rec = out["intervention_recommendation"]
    assert rec["recommendation_calibrated_confidence"] >= 0.4
    assert rec["reliability_trace"]["bucket_support"] >= 6


def test_law_confidence_reduced_when_cross_system_conflicts() -> None:
    layer = StructuralReliabilityLayer()
    transition, forecast, laws, intervention, cross = _base_inputs(law_conf=0.78, law_support=8)
    cross["shared_law_intelligence"][0]["status"] = "globally_conflicting"
    out = layer.calibrate_all(
        asset_id="asset",
        step=1,
        transition=transition,
        trajectory_forecast=forecast,
        laws=laws,
        intervention=intervention,
        cross=cross,
    )
    law = out["law_candidates"][0]
    assert law["calibrated_applicability_confidence"] < law["local_law_confidence"]


def test_transfer_penalty_increases_with_mismatch_and_local_first() -> None:
    layer = StructuralReliabilityLayer()
    transition, forecast, laws, intervention, cross = _base_inputs(rec_conf=0.7)
    cross["evidence_source_weights"] = {"local_evidence_weight": 0.85, "global_evidence_weight": 0.65, "mismatch_penalty": 0.8}

    out = layer.calibrate_all(
        asset_id="asset",
        step=1,
        transition=transition,
        trajectory_forecast=forecast,
        laws=laws,
        intervention=intervention,
        cross=cross,
    )
    trace = out["intervention_recommendation"]["reliability_trace"]
    assert trace["transfer_penalty"] > 0.15
    assert trace["local_evidence_weight"] > 0.8


def test_horizon_aware_outputs_and_support_fields_present() -> None:
    layer = StructuralReliabilityLayer()
    transition, forecast, laws, intervention, cross = _base_inputs()
    out = layer.calibrate_all(
        asset_id="asset",
        step=1,
        transition=transition,
        trajectory_forecast=forecast,
        laws=laws,
        intervention=intervention,
        cross=cross,
    )
    traj = out["trajectory_forecast"]
    assert "calibrated_short_horizon_confidence" in traj
    assert "calibrated_medium_horizon_confidence" in traj
    assert "horizon_support" in traj


def test_calibrated_confidence_tracks_empirical_success_better_than_raw() -> None:
    layer = StructuralReliabilityLayer()
    transition, forecast, laws, intervention, cross = _base_inputs(rec_support=5, rec_conf=0.55)

    for step in range(1, 20):
        layer.finalize_due(asset_id="asset", step=step, transition=transition)
        layer.calibrate_all(
            asset_id="asset",
            step=step,
            transition=transition,
            trajectory_forecast=forecast,
            laws=laws,
            intervention=intervention,
            cross=cross,
        )

    intervention["recommendation"]["best_intervention"]["confidence"] = 0.85
    out = layer.calibrate_all(
        asset_id="asset",
        step=21,
        transition=transition,
        trajectory_forecast=forecast,
        laws=laws,
        intervention=intervention,
        cross=cross,
    )
    rec = out["intervention_recommendation"]
    assert abs(rec["recommendation_calibrated_confidence"] - transition["escalation_probability"]) < abs(
        rec["recommendation_raw_confidence"] - transition["escalation_probability"]
    )


def test_repeated_high_confidence_failures_are_penalized() -> None:
    layer = StructuralReliabilityLayer()
    transition, forecast, laws, intervention, cross = _base_inputs(rec_support=6, rec_conf=0.9)
    for step in range(1, 7):
        layer.ingest_feedback_records(
            asset_id="asset",
            step=step,
            feedback_records=[
                {
                    "outcome_label": "harmful",
                    "confidence": 0.92,
                    "trajectory_family": "escalating",
                    "transition_path": "escalating",
                    "regime": "critical",
                    "novelty": 0.1,
                    "support_count": 6,
                }
            ],
        )

    out = layer.calibrate_all(
        asset_id="asset",
        step=20,
        transition=transition,
        trajectory_forecast=forecast,
        laws=laws,
        intervention=intervention,
        cross=cross,
    )
    trace = out["intervention_recommendation"]["reliability_trace"]
    assert out["intervention_recommendation"]["recommendation_calibrated_confidence"] < 0.7
    assert any("high-confidence misses" in msg.lower() for msg in trace["warnings"])
