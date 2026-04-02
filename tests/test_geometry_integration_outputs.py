from __future__ import annotations

from neraium_core.alignment import StructuralEngine


def _frame(t: int, vals: dict[str, float]) -> dict[str, object]:
    return {
        "timestamp": str(t),
        "site_id": "site-x",
        "asset_id": "asset-x",
        "sensor_values": vals,
    }


def test_structural_engine_emits_statistical_geometry_outputs() -> None:
    engine = StructuralEngine(baseline_window=24, recent_window=8, representation_mode="combined")
    out = None
    for t in range(75):
        s1 = 0.02 * t
        s2 = 0.018 * t + 0.12 * max(0, t - 45)
        s3 = -0.012 * t + 0.08 * max(0, t - 45)
        out = engine.process_frame(_frame(t, {"a": s1, "b": s2, "c": s3}))
    assert out is not None
    assert "geometry" in out
    assert "state_space_statistics" in out
    assert "state_graph" in out
    assert "geometry_explanations" in out
    assert out["geometry"].get("curvature", 0.0) >= 0.0
    assert out["state_graph"].get("node_count", 0) >= 1
    exp = out.get("experimental_analytics") or {}
    assert "geometry" in exp
    assert "state_space_statistics" in exp
    assert "state_graph" in exp
