from __future__ import annotations

import numpy as np

from neraium_core.system_intelligence.forecast.trajectory_conditioned import TrajectoryConditionedForecaster
from neraium_core.system_intelligence.law_extraction.extractor import StructuralLawExtractor
from neraium_core.system_intelligence.mechanisms.discovery import MechanismDiscoveryLayer
from neraium_core.system_intelligence.platform import StructuralSystemIntelligencePlatform
from neraium_core.system_intelligence.trajectory_archetypes.archetypes import TrajectorySegmentEncoder, dtw_distance
from neraium_core.system_intelligence.trajectory_memory.memory import CrossSystemTrajectoryMemory


def _obs(asset: str, drift: float, rel: float, regime_d: float, coherence: float, relationships: list[dict] | None = None) -> dict:
    return {
        "asset_id": asset,
        "structural_drift_score": drift,
        "relational_instability_score": rel,
        "regime_distance": regime_d,
        "graph_deformation_score": drift * 0.8,
        "coherence_score": coherence,
        "coupling_instability_score": rel * 0.7,
        "risk_pressure": max(drift, rel),
        "path_length_shift": regime_d * 0.5,
        "subspace_rotation": rel * 0.3,
        "mean_shift": drift * 0.4,
        "covariance_shift": rel * 0.4,
        "top_relationships": relationships or [{"source": "A", "target": "B"}, {"source": "B", "target": "C"}],
        "subsystem_impact": {"sub_a": drift, "sub_b": rel},
    }


def _run(memory: CrossSystemTrajectoryMemory, asset: str, seq: np.ndarray, escalating_cutoff: float = 0.9) -> dict:
    out = {}
    for i, emb in enumerate(seq):
        out = memory.update(
            asset_id=asset,
            embedding=emb.tolist(),
            escalating=float(np.linalg.norm(emb)) >= escalating_cutoff,
            in_critical_region=float(np.linalg.norm(emb)) >= escalating_cutoff + 0.25,
            phase_shift=i > 2,
            mechanism_names=["m:a->b"],
        )
    return out


def test_alignment_prefers_shape_match_over_endpoint_only() -> None:
    encoder = TrajectorySegmentEncoder(window_size=10)
    base = np.asarray([[i * 0.08, i * 0.05, i * 0.09] for i in range(10)], dtype=float)
    same_shape = base + np.asarray([0.03, -0.02, 0.02], dtype=float)
    same_endpoint_diff_shape = np.asarray(
        [[0.0, 0.0, 0.0], [0.5, 0.8, 0.2], [0.2, 0.9, 0.3], [0.8, 0.4, 0.9], [0.3, 0.1, 0.7], [0.4, 0.2, 0.6], [0.6, 0.1, 0.8], [0.55, 0.35, 0.75], [0.62, 0.4, 0.82], base[-1]],
        dtype=float,
    )

    d_shape = dtw_distance(base, same_shape)
    d_diff = dtw_distance(base, same_endpoint_diff_shape)
    endpoint_d_shape = float(np.linalg.norm(base[-1] - same_shape[-1]))
    endpoint_d_diff = float(np.linalg.norm(base[-1] - same_endpoint_diff_shape[-1]))

    assert endpoint_d_diff < endpoint_d_shape
    assert d_shape < d_diff
    assert encoder._infer_path_family(base) in {"escalating", "drift"}


def test_trajectory_families_separate_stable_drift_escalating_reversible() -> None:
    memory = CrossSystemTrajectoryMemory(window_size=10, min_similarity_for_match=0.7)
    t = np.linspace(0.0, 1.0, 20)
    stable = np.stack([0.2 + 0.02 * np.sin(4 * t), 0.21 + 0.01 * np.cos(5 * t), 0.19 + 0.01 * np.sin(3 * t)], axis=1)
    drift = np.stack([0.2 + 0.18 * t, 0.15 + 0.09 * t, 0.2 + 0.11 * t], axis=1)
    escalating = np.stack([0.12 + 1.15 * t**2, 0.09 + 0.85 * t**2, 0.1 + 0.95 * t**2], axis=1)
    reversible = np.stack([0.9 - 0.75 * t + 0.05 * np.sin(4 * t), 0.8 - 0.6 * t, 0.85 - 0.65 * t + 0.02 * np.cos(3 * t)], axis=1)

    out_s = _run(memory, "stable-1", stable, escalating_cutoff=1.5)
    out_d = _run(memory, "drift-1", drift, escalating_cutoff=1.5)
    out_e = _run(memory, "escalate-1", escalating, escalating_cutoff=0.8)
    out_r = _run(memory, "recover-1", reversible, escalating_cutoff=1.5)

    assert out_s["status"] == "ready"
    assert out_e["status"] == "ready"
    assert out_s["current_trajectory_path_family"] != out_e["current_trajectory_path_family"]
    assert out_r["current_trajectory_path_family"] in {"reversible", "drift"}
    assert out_d["support"] >= 1


def test_encoder_identifies_reversible_pattern() -> None:
    encoder = TrajectorySegmentEncoder(window_size=10)
    t = np.linspace(0.0, 1.0, 10)
    reversible = np.stack([1.1 - 0.9 * t, 0.95 - 0.7 * t + 0.03 * np.sin(4 * t), 1.0 - 0.8 * t], axis=1)
    assert encoder._infer_path_family(reversible) == "reversible"


def test_family_conditioned_forecast_beats_point_only_in_controlled_case() -> None:
    forecaster = TrajectoryConditionedForecaster()
    traj = {
        "status": "ready",
        "current_trajectory_path_family": "drift",
        "novelty_score": 0.05,
        "nearest_trajectory_families": [
            {"family_similarity": 0.94, "escalation_frequency": 0.91, "support": 24},
            {"family_similarity": 0.76, "escalation_frequency": 0.79, "support": 16},
        ],
        "matched_historical_progressions": [{"path_family": "escalating", "frequency": 0.67, "support": 12}],
    }
    point_only = {"escalation_probability": 0.34, "transition_path": "drift"}
    out = forecaster.forecast(trajectory_intelligence=traj, transition_dynamics=point_only)

    assert out["trajectory_conditioned_escalation_probability"] > point_only["escalation_probability"]
    assert out["likely_next_path_family"] == "escalating"
    assert out["uncertainty"] < 0.6

    novel_traj = dict(traj)
    novel_traj["novelty_score"] = 0.95
    novel = forecaster.forecast(trajectory_intelligence=novel_traj, transition_dynamics=point_only)
    assert novel["trajectory_conditioned_escalation_probability"] < out["trajectory_conditioned_escalation_probability"]
    assert novel["novelty_aware_confidence_penalty"] > out["novelty_aware_confidence_penalty"]


def test_mechanism_signals_gain_family_specificity() -> None:
    layer = MechanismDiscoveryLayer()
    rel = [{"source": "A", "target": "B"}, {"source": "B", "target": "C"}, {"source": "A", "target": "C"}]

    out = {}
    for _ in range(6):
        out = layer.update(
            top_relationships=rel,
            subsystem_impact={"s1": 0.85, "s2": 0.72},
            escalating=True,
            trajectory_family="escalating",
            recovering=False,
        )
    for _ in range(5):
        layer.update(
            top_relationships=rel,
            subsystem_impact={"s1": 0.85, "s2": 0.72},
            escalating=False,
            trajectory_family="reversible",
            recovering=True,
        )

    candidate = out["mechanism_candidates"][0]
    assert "family_conditioned_predictive_value" in candidate
    assert "family_conditioned_recovery_association" in candidate
    assert out["evidence_by_family"]


def test_platform_outputs_trajectory_family_and_forecast_sections() -> None:
    platform = StructuralSystemIntelligencePlatform()
    out = {}
    for i in range(24):
        out = platform.update(
            _obs(
                asset="asset-1",
                drift=0.07 + 0.035 * i,
                rel=0.05 + 0.03 * i,
                regime_d=0.03 + 0.02 * i,
                coherence=max(0.05, 0.93 - 0.03 * i),
            )
        )

    traj = out["trajectory_archetypes"]
    forecast = out["trajectory_forecast"]
    assert traj["status"] in {"ready", "warming"}
    if traj["status"] == "ready":
        assert "current_trajectory_family" in traj
        assert "nearest_trajectory_families" in traj
        assert "family_similarity" in traj
        assert "novelty_score" in traj
        assert "representative_segment" in traj
        assert "escalation_frequency_by_family" in traj
    assert "likely_next_path_family" in forecast
    assert "trajectory_conditioned_escalation_probability" in forecast
    assert "estimated_steps_to_critical_region" in forecast


def test_structural_laws_require_support_and_effect_thresholds() -> None:
    extractor = StructuralLawExtractor(min_support=3, robust_support=8)
    trajectory = {"status": "ready", "current_trajectory_archetype": "trajectory_family_1", "current_trajectory_path_family": "escalating"}
    mechanism = {"mechanism_candidates": [{"mechanism": "cluster_decoupling:x->y"}]}

    for _ in range(2):
        out = extractor.update(trajectory_info=trajectory, mechanism_info=mechanism, transition={"transition_path": "escalating", "escalation_probability": 0.8, "regime": "transitional"})
    out = extractor.update(trajectory_info=trajectory, mechanism_info=mechanism, transition={"transition_path": "escalating", "escalation_probability": 0.8, "regime": "critical"})

    assert out["law_candidates"][0]["classification"] in {"weak_pattern", "candidate_structural_law"}
    assert out["law_candidates"][0]["support"] >= 3
