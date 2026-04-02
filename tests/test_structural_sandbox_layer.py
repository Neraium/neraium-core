from __future__ import annotations

from neraium_core.system_intelligence.sandbox import StructuralSandboxEngine


def _base_payload(novelty: float = 0.2) -> dict[str, object]:
    return {
        "latent_state": {
            "embedding": [0.25, 0.12, 0.06],
            "trajectory": [
                [0.2, 0.1, 0.06],
                [0.21, 0.1, 0.06],
                [0.22, 0.11, 0.06],
                [0.23, 0.11, 0.06],
                [0.24, 0.12, 0.06],
                [0.25, 0.12, 0.06],
            ],
        },
        "transition": {
            "regime": "stable",
            "transition_path": "drifting",
            "escalation_probability": 0.18,
            "reversibility_score": 0.62,
            "distance_to_critical_region": 0.9,
            "uncertainty": 0.18,
        },
        "trajectory": {
            "current_trajectory_path_family": "drift",
            "family_similarity": 0.78,
            "novelty_score": novelty,
        },
        "laws": {
            "law_candidates": [
                {
                    "law_id": "law::drift",
                    "estimated_effect": 0.22,
                    "confidence": 0.71,
                }
            ]
        },
        "intervention": {
            "recommendation": {
                "best_intervention": {
                    "name": "damp_subsystem_instability",
                    "predicted_escalation_reduction": 0.24,
                    "confidence": 0.66,
                }
            }
        },
        "reliability": {
            "transition_dynamics": {
                "transition_confidence": 0.73,
            }
        },
    }


def test_baseline_continuation_stable_regime_stays_bounded() -> None:
    engine = StructuralSandboxEngine()
    p = _base_payload()
    out = engine.evaluate(**p)["structural_sandbox"]
    risks = out["current_state_simulation"]["risk_evolution"]
    assert max(risks) < 0.55
    assert min(risks) >= 0.0


def test_intervention_branches_differ_from_no_intervention() -> None:
    engine = StructuralSandboxEngine()
    out = engine.evaluate(**_base_payload())["structural_sandbox"]
    by_name = {b["branch"]: b for b in out["scenario_branches"]}
    assert by_name["best_ranked_intervention"]["escalation_probability"] < by_name["no_intervention"]["escalation_probability"]


def test_competing_world_models_are_distinguishable() -> None:
    engine = StructuralSandboxEngine()
    out = engine.evaluate(**_base_payload())["structural_sandbox"]
    scores = out["world_model_fit_scores"]
    assert len(scores) >= 3
    assert len(set(scores.values())) > 1


def test_replay_surfaces_prediction_errors() -> None:
    engine = StructuralSandboxEngine()
    p = _base_payload()
    p["latent_state"]["trajectory"] = [
        [0.2, 0.1, 0.06],
        [0.3, 0.12, 0.08],
        [0.45, 0.2, 0.1],
        [0.65, 0.32, 0.16],
        [0.86, 0.44, 0.22],
    ]
    out = engine.evaluate(**p)["structural_sandbox"]
    assert len(out["prediction_error_trace"]) > 0
    assert any(e > 0.05 for e in out["prediction_error_trace"])


def test_divergence_detection_identifies_mismatch_type() -> None:
    engine = StructuralSandboxEngine()
    out = engine.evaluate(**_base_payload(novelty=0.92))["structural_sandbox"]
    divergence_type = out["divergence_analysis"]["divergence_type"]
    assert divergence_type in {
        "novelty_unseen_behavior",
        "wrong_law",
        "wrong_archetype_family",
        "intervention_mismatch",
        "insufficient_observability",
    }


def test_low_support_degrades_simulation_confidence() -> None:
    engine = StructuralSandboxEngine()
    p = _base_payload(novelty=0.95)
    p["trajectory"]["family_similarity"] = 0.18
    p["laws"]["law_candidates"][0]["confidence"] = 0.1
    out = engine.evaluate(**p)["structural_sandbox"]
    rel = out["simulation_reliability"]
    assert rel["simulation_confidence"] < 0.45
    assert rel["degraded"] is True


def test_what_if_outputs_bounded_and_interpretable() -> None:
    engine = StructuralSandboxEngine()
    out = engine.evaluate(**_base_payload())["structural_sandbox"]
    assert out["what_if_analysis"]
    for scenario in out["what_if_analysis"]:
        assert -1.0 <= scenario["delta_vs_baseline"] <= 1.0
        assert 0.0 <= scenario["projected_escalation_probability"] <= 1.0
