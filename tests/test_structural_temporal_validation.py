from __future__ import annotations

from neraium_core.sii.reporting import (
    compute_structural_signal_ranking,
    compute_structural_temporal_validation,
    evaluate_structural_risk_decision_policy,
)
from neraium_core.sii.calibration import derive_calibration_from_validation, load_config, save_config


def test_structural_temporal_validation_detects_leading_spikes_before_transitions() -> None:
    results = [
        {
            "state": "STABLE",
            "interpreted_state": "NOMINAL_STRUCTURE",
            "structural_state": {
                "operator_deformation_energy": 0.10,
                "residual_energy": 0.10,
                "regime_distance": 0.10,
                "coherence_margin_raw": 0.80,
                "structural_drift_magnitude": 0.18,
            },
        },
        {
            "state": "STABLE",
            "interpreted_state": "NOMINAL_STRUCTURE",
            "structural_state": {
                "operator_deformation_energy": 0.11,
                "residual_energy": 0.10,
                "regime_distance": 0.12,
                "coherence_margin_raw": 0.77,
                "structural_drift_magnitude": 0.20,
            },
        },
        {
            "state": "WATCH",
            "interpreted_state": "REGIME_SHIFT_OBSERVED",
            "structural_state": {
                "operator_deformation_energy": 1.60,
                "residual_energy": 0.11,
                "regime_distance": 0.14,
                "coherence_margin_raw": -0.20,
                "structural_drift_magnitude": 1.70,
            },
        },
        {
            "state": "ALERT",
            "interpreted_state": "STRUCTURAL_INSTABILITY_OBSERVED",
            "structural_state": {
                "operator_deformation_energy": 1.65,
                "residual_energy": 0.12,
                "regime_distance": 0.16,
                "coherence_margin_raw": -0.32,
                "structural_drift_magnitude": 1.80,
            },
        },
    ]

    report = compute_structural_temporal_validation(results, lead_window=1)

    assert report["transition_count"] == 2
    op = report["signals"]["operator_deformation_energy"]
    assert op["spike_steps"] == [2]
    assert op["leading_transition_hits"] == 1
    assert op["lead_recall"] == 0.5


def test_structural_temporal_validation_uses_top_level_fields_when_present() -> None:
    results = [
        {
            "state": "STABLE",
            "interpreted_state": "NOMINAL_STRUCTURE",
            "operator_deformation_energy": 0.2,
            "residual_energy": 0.1,
            "regime_distance": 0.15,
            "coherence_margin_raw": 0.7,
            "structural_drift_magnitude": 0.3,
        },
        {
            "state": "ALERT",
            "interpreted_state": "STRUCTURAL_INSTABILITY_OBSERVED",
            "operator_deformation_energy": 0.9,
            "residual_energy": 0.8,
            "regime_distance": 0.85,
            "coherence_margin_raw": -0.5,
            "structural_drift_magnitude": 1.1,
        },
    ]

    report = compute_structural_temporal_validation(results)

    assert report["sample_count"] == 2
    assert report["transition_count"] == 1
    assert report["signals"]["residual_energy"]["peak"] == 0.8


def test_structural_signal_ranking_orders_by_predictive_metrics() -> None:
    temporal_validation = {
        "transition_count": 3,
        "signals": {
            "operator_deformation_energy": {
                "lead_recall": 0.66,
                "lead_precision": 0.5,
                "leading_transition_hits": 2,
                "concurrent_transition_hits": 0,
                "lagging_transition_hits": 1,
                "spike_steps": [2, 4, 8, 11],
                "mean_abs_change": 0.5,
                "spike_threshold": 0.6,
            },
            "residual_energy": {
                "lead_recall": 0.33,
                "lead_precision": 1.0,
                "leading_transition_hits": 1,
                "concurrent_transition_hits": 2,
                "lagging_transition_hits": 0,
                "spike_steps": [4],
                "mean_abs_change": 0.2,
                "spike_threshold": 0.4,
            },
            "regime_distance": {
                "lead_recall": 0.0,
                "lead_precision": 0.0,
                "leading_transition_hits": 0,
                "concurrent_transition_hits": 1,
                "lagging_transition_hits": 2,
                "spike_steps": [5, 6, 7],
                "mean_abs_change": 0.1,
                "spike_threshold": 0.25,
            },
        },
    }

    ranking = compute_structural_signal_ranking(temporal_validation)

    assert ranking["transition_count"] == 3
    assert [item["signal"] for item in ranking["ranking"]] == [
        "operator_deformation_energy",
        "residual_energy",
        "regime_distance",
    ]
    assert ranking["ranking"][0]["temporal_alignment"]["dominant_role"] == "leading"
    assert ranking["ranking"][1]["temporal_alignment"]["dominant_role"] == "concurrent"
    assert ranking["ranking"][2]["temporal_alignment"]["dominant_role"] == "lagging"
    assert ranking["ranking"][0]["predictive_metrics"]["transition_coverage"] == 1.0


def test_structural_signal_ranking_handles_empty_inputs() -> None:
    ranking = compute_structural_signal_ranking({"transition_count": 0, "signals": {}})
    assert ranking == {"transition_count": 0, "ranking": []}


def test_structural_risk_decision_policy_uses_ranked_leading_signals() -> None:
    ranking = {
        "transition_count": 3,
        "ranking": [
            {
                "signal": "operator_deformation_energy",
                "rank": 1,
                "predictive_metrics": {"lead_recall": 0.66, "lead_precision": 0.5, "transition_coverage": 1.0},
                "behavior_metrics": {"mean_abs_change": 0.4, "spike_threshold": 0.5},
                "temporal_alignment": {"dominant_role": "leading"},
            },
            {
                "signal": "residual_energy",
                "rank": 2,
                "predictive_metrics": {"lead_recall": 0.1, "lead_precision": 0.3, "transition_coverage": 0.2},
                "behavior_metrics": {"mean_abs_change": 0.1, "spike_threshold": 0.2},
                "temporal_alignment": {"dominant_role": "lagging"},
            },
        ],
    }

    policy = evaluate_structural_risk_decision_policy(
        signal_ranking=ranking,
        raw_structural_magnitudes={"operator_deformation_energy": 1.1, "residual_energy": 1.2},
    )

    assert policy["risk_flag"] is True
    assert policy["risk_level"] == "elevated"
    assert policy["decision_posture"] == "elevated_structural_risk_watch"
    assert [s["signal"] for s in policy["contributing_signals"]] == ["operator_deformation_energy"]
    lagging_eval = next(item for item in policy["signal_evaluation"] if item["signal"] == "residual_energy")
    assert lagging_eval["eligible_for_early_risk"] is False


def test_structural_risk_decision_policy_requires_eligible_crossings() -> None:
    ranking = {
        "transition_count": 2,
        "ranking": [
            {
                "signal": "operator_deformation_energy",
                "rank": 1,
                "predictive_metrics": {"lead_recall": 0.7, "lead_precision": 0.7, "transition_coverage": 1.0},
                "behavior_metrics": {"mean_abs_change": 0.5, "spike_threshold": 0.6},
                "temporal_alignment": {"dominant_role": "leading"},
            }
        ],
    }

    policy = evaluate_structural_risk_decision_policy(
        signal_ranking=ranking,
        raw_structural_magnitudes={"operator_deformation_energy": 0.7},
    )

    assert policy["risk_flag"] is False
    assert policy["risk_level"] == "normal"
    assert policy["contributing_signals"] == []


def test_run_structural_pipeline_returns_single_inspectable_output() -> None:
    from neraium_core.sii.orchestration import run_structural_pipeline

    payloads = [
        {
            "timestamp": "2026-01-01T00:00:00Z",
            "site_id": "site-a",
            "asset_id": "asset-a",
            "sensor_values": {"temperature": 10.0, "pressure": 100.0},
        },
        {
            "timestamp": "2026-01-01T00:01:00Z",
            "site_id": "site-a",
            "asset_id": "asset-a",
            "sensor_values": {"temperature": 10.2, "pressure": 99.7},
        },
        {
            "timestamp": "2026-01-01T00:02:00Z",
            "site_id": "site-a",
            "asset_id": "asset-a",
            "sensor_values": {"temperature": 10.4, "pressure": 99.5},
        },
    ]

    output = run_structural_pipeline(payloads)

    assert set(output.keys()) >= {
        "structural_state",
        "validation_results",
        "signal_ranking",
        "decision_output",
    }
    assert isinstance(output["structural_state"], dict)
    assert isinstance(output["validation_results"], dict)
    assert isinstance(output["signal_ranking"], dict)
    assert isinstance(output["decision_output"], dict)
    assert output["validation_results"]["sample_count"] == len(payloads)


def test_structural_policy_applies_config_threshold_overrides() -> None:
    ranking = {
        "transition_count": 1,
        "ranking": [
            {
                "signal": "operator_deformation_energy",
                "rank": 1,
                "predictive_metrics": {"lead_recall": 0.8, "lead_precision": 0.8, "transition_coverage": 1.0},
                "behavior_metrics": {"mean_abs_change": 0.4, "spike_threshold": 0.5},
                "temporal_alignment": {"dominant_role": "leading"},
            }
        ],
    }
    policy = evaluate_structural_risk_decision_policy(
        signal_ranking=ranking,
        raw_structural_magnitudes={"operator_deformation_energy": 0.95},
        config={
            "min_lead_recall": 0.2,
            "min_lead_precision": 0.2,
            "top_k_rank_window": 2,
            "decision_thresholds": {"default": 1.2, "operator_deformation_energy": 1.2},
        },
    )
    assert policy["risk_flag"] is False
    assert policy["signal_evaluation"][0]["threshold_comparison"]["decision_multiplier"] == 1.2


def test_calibration_round_trip_and_derivation(tmp_path) -> None:
    validation = {
        "sample_count": 20,
        "transition_count": 4,
        "signals": {
            "operator_deformation_energy": {
                "lead_recall": 0.5,
                "lead_precision": 0.5,
                "mean_abs_change": 0.2,
                "spike_threshold": 0.4,
                "spike_steps": [2, 7, 9, 11],
            }
        },
    }
    derived = derive_calibration_from_validation(validation)
    target = tmp_path / "calibration.json"
    save_config(derived, target)
    loaded = load_config(target)

    assert loaded["lead_window"] >= 1
    assert loaded["top_k_rank_window"] >= 1
    assert "default" in loaded["decision_thresholds"]
