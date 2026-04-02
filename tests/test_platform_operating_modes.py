from __future__ import annotations

from neraium_core.system_intelligence.platform import StructuralSystemIntelligencePlatform


def _obs(step: int = 0) -> dict:
    return {
        "asset_id": "asset-1",
        "domain": "water",
        "system_type": "pump",
        "structural_drift_score": 0.2 + 0.01 * step,
        "relational_instability_score": 0.18 + 0.01 * step,
        "regime_distance": 0.16 + 0.01 * step,
        "graph_deformation_score": 0.17 + 0.01 * step,
        "coherence_score": 0.8 - 0.01 * step,
        "coupling_instability_score": 0.2 + 0.01 * step,
        "risk_pressure": 0.3 + 0.02 * step,
        "path_length_shift": 0.05 + 0.005 * step,
        "subspace_rotation": 0.05 + 0.005 * step,
        "mean_shift": 0.05 + 0.005 * step,
        "covariance_shift": 0.05 + 0.005 * step,
        "top_relationships": [{"source": "A", "target": "B"}],
        "subsystem_impact": {"sub_a": 0.6},
        "composite_instability": 0.4 + 0.02 * step,
    }


def test_default_mode_is_production_safe() -> None:
    platform = StructuralSystemIntelligencePlatform()
    out = platform.update(_obs())

    assert out["operating_mode"] == "production"
    assert out["advisory_intelligence"] == {}
    assert out["experimental_intelligence"] == {}
    assert "law_engine_decision" not in out


def test_full_mode_exposes_advisory_and_experimental_sections() -> None:
    platform = StructuralSystemIntelligencePlatform(operating_mode="full")
    latest = {}
    for i in range(14):
        latest = platform.update(_obs(i))

    assert latest["advisory_intelligence"]
    assert latest["experimental_intelligence"]
    assert latest["capability_boundaries"]["law_engine_decision"]["can_drive_operator_recommendations"] is False
    assert latest["capability_boundaries"]["active_learning"]["status"] == "experimental"
