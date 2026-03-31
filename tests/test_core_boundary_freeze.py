from __future__ import annotations

from neraium_core.system_intelligence.platform import StructuralSystemIntelligencePlatform


def _obs() -> dict:
    return {
        "asset_id": "asset-boundary",
        "structural_drift_score": 0.42,
        "relational_instability_score": 0.41,
        "regime_distance": 0.4,
        "graph_deformation_score": 0.35,
        "coherence_score": 0.62,
        "risk_pressure": 0.58,
    }


def test_production_compatibility_is_isolated_from_advisory_experimental_modes() -> None:
    platform = StructuralSystemIntelligencePlatform(operating_mode="production")
    prod = platform.update(_obs(), operating_mode="production")
    full = platform.update(_obs(), operating_mode="full")

    assert prod["compatibility"] == full["compatibility"]


def test_core_decision_trace_is_complete_and_bounded() -> None:
    platform = StructuralSystemIntelligencePlatform(operating_mode="production")
    out = platform.update(_obs())
    trace = out["production_intelligence"]["core_decision_trace"]
    assert "state_context" in trace
    assert "top_intervention_factors" in trace
    assert "calibrated_confidence" in trace
    assert "intervention_history_evidence" in trace
    assert trace["law_influence"]["status"] == "excluded_from_core_trace"
