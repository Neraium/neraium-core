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


def test_trajectory_shape_similarity_outperforms_endpoint_matching() -> None:
    encoder = TrajectorySegmentEncoder(window_size=8)
    base = np.asarray([[i * 0.1, i * 0.08, i * 0.12] for i in range(8)], dtype=float)
    same_shape = base + np.asarray([0.02, -0.01, 0.0], dtype=float)
    different_shape_same_end = np.vstack([
        np.asarray([[0.0, 0.0, 0.0], [0.7, 0.2, 0.6], [0.3, 0.8, 0.2], [0.9, 0.9, 0.9]], dtype=float),
        np.linspace([0.9, 0.9, 0.9], base[-1], 4),
    ])

    d_shape = dtw_distance(base, same_shape)
    d_different = dtw_distance(base, different_shape_same_end)

    point_distance_similar = float(np.linalg.norm(base[-1] - same_shape[-1]))
    point_distance_different = float(np.linalg.norm(base[-1] - different_shape_same_end[-1]))

    assert point_distance_different < point_distance_similar
    assert d_shape < d_different
    assert encoder._infer_path_family(base) == "escalating"


def test_trajectory_memory_separates_path_families() -> None:
    memory = CrossSystemTrajectoryMemory(window_size=8, min_similarity_for_match=0.5)

    for i in range(12):
        escalating = [0.15 * i, 0.11 * i, 0.12 * i]
        stable = [0.2 + 0.01 * np.sin(i), 0.19 + 0.01 * np.cos(i), 0.18]
        recovery = [1.2 - 0.08 * i, 1.0 - 0.07 * i, 0.95 - 0.06 * i]

        out_e = memory.update(asset_id="asset-e", embedding=escalating, escalating=True, in_critical_region=i > 8, phase_shift=True, mechanism_names=[])
        out_s = memory.update(asset_id="asset-s", embedding=stable, escalating=False, in_critical_region=False, phase_shift=False, mechanism_names=[])
        out_r = memory.update(asset_id="asset-r", embedding=recovery, escalating=False, in_critical_region=False, phase_shift=True, mechanism_names=[])

    assert out_e["status"] == "ready"
    assert out_s["status"] == "ready"
    assert out_r["status"] == "ready"
    assert out_e["current_trajectory_path_family"] == "escalating"
    assert out_s["current_trajectory_path_family"] == "stable"
    assert out_r["current_trajectory_path_family"] in {"reversible", "drifting"}


def test_trajectory_conditioned_forecast_is_evidence_bounded() -> None:
    forecaster = TrajectoryConditionedForecaster()
    traj = {
        "status": "ready",
        "current_trajectory_path_family": "drifting",
        "nearest_trajectory_archetypes": [
            {"similarity": 0.9, "historical_escalation_frequency": 0.85, "support": 20},
            {"similarity": 0.7, "historical_escalation_frequency": 0.8, "support": 12},
        ],
        "matched_historical_progressions": [{"path_family": "escalating", "frequency": 0.62, "support": 11}],
    }
    base = {"escalation_probability": 0.35, "transition_path": "drifting"}
    out = forecaster.forecast(trajectory_intelligence=traj, transition_dynamics=base)

    assert out["status"] == "ready"
    assert out["trajectory_conditioned_escalation_probability"] > base["escalation_probability"]
    assert out["likely_next_path_family"] == "escalating"
    assert out["estimated_steps_to_critical_region"]["range"][0] <= out["estimated_steps_to_critical_region"]["median"]


def test_structural_laws_require_support_and_effect_thresholds() -> None:
    extractor = StructuralLawExtractor(min_support=3, robust_support=8)
    trajectory = {"status": "ready", "current_trajectory_archetype": "trajectory_archetype_1", "current_trajectory_path_family": "escalating"}
    mechanism = {"mechanism_candidates": [{"mechanism": "cluster_decoupling:x->y"}]}

    labels = []
    for _ in range(2):
        out = extractor.update(trajectory_info=trajectory, mechanism_info=mechanism, transition={"transition_path": "escalating", "escalation_probability": 0.8, "regime": "transitional"})
        labels.append(out["law_candidates"][0]["classification"])
    out = extractor.update(trajectory_info=trajectory, mechanism_info=mechanism, transition={"transition_path": "escalating", "escalation_probability": 0.8, "regime": "critical"})

    assert labels[-1] == "unsupported_hypothesis"
    assert out["law_candidates"][0]["classification"] in {"weak_pattern", "candidate_structural_law"}
    assert out["law_candidates"][0]["support"] >= 3


def test_mechanism_associations_are_family_conditioned() -> None:
    layer = MechanismDiscoveryLayer()
    rel = [{"source": "A", "target": "B"}, {"source": "B", "target": "C"}, {"source": "A", "target": "C"}]

    escalating_out = {}
    for _ in range(5):
        escalating_out = layer.update(
            top_relationships=rel,
            subsystem_impact={"s1": 0.8, "s2": 0.6},
            escalating=True,
            trajectory_family="escalating",
            recovering=False,
        )
    for _ in range(4):
        layer.update(
            top_relationships=rel,
            subsystem_impact={"s1": 0.8, "s2": 0.6},
            escalating=False,
            trajectory_family="reversible",
            recovering=True,
        )

    assoc = escalating_out["mechanism_trajectory_associations"]
    assert assoc
    assert any(item["association"] == "escalation_correlated" for item in assoc)


def test_platform_keeps_existing_outputs_and_adds_trajectory_sections() -> None:
    platform = StructuralSystemIntelligencePlatform()
    out = {}
    for i in range(20):
        out = platform.update(
            _obs(
                asset="asset-1",
                drift=0.08 + 0.03 * i,
                rel=0.06 + 0.025 * i,
                regime_d=0.04 + 0.02 * i,
                coherence=max(0.05, 0.92 - 0.03 * i),
            )
        )

    assert "archetype_intelligence" in out
    assert "mechanism_discovery" in out
    assert "compatibility" in out
    assert "trajectory_archetypes" in out
    assert "trajectory_forecast" in out
    assert "structural_law_candidates" in out
    assert out["trajectory_archetypes"]["status"] in {"ready", "warming"}
