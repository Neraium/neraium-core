from __future__ import annotations

from neraium_core.alignment import StructuralEngine


def _frame(i: int) -> dict[str, object]:
    return {
        "timestamp": float(i),
        "site_id": "site-a",
        "asset_id": "asset-a",
        "sensor_values": {
            "s1": 1.0 + 0.05 * i,
            "s2": 0.8 + 0.03 * i,
            "s3": 1.2 - 0.02 * i,
        },
    }


def test_fast_mode_still_returns_tetrahedral_state(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("NERAIUM_FAST_MODE", "1")
    engine = StructuralEngine(baseline_window=5, recent_window=3, regime_store_path=str(tmp_path / "regime.json"))

    result = {}
    for i in range(10):
        result = engine.process_frame(_frame(i))

    tetra = result.get("tetrahedral_state")
    assert isinstance(tetra, dict)
    assert isinstance(tetra.get("position"), list)
    assert "weights" in tetra


def test_fast_mode_preserves_key_output_fields(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("NERAIUM_FAST_MODE", "1")
    engine = StructuralEngine(baseline_window=5, recent_window=3, regime_store_path=str(tmp_path / "regime.json"))

    result = {}
    for i in range(10):
        result = engine.process_frame(_frame(i))

    required_keys = {
        "structural_drift_score",
        "relational_instability_score",
        "transition_pressure",
        "temporal_consistency_score",
        "tetrahedral_state",
        "policy_state",
        "interpreted_state",
        "experimental_analytics",
    }
    assert required_keys.issubset(result.keys())
