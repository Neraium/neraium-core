from __future__ import annotations

from neraium_core.alignment import StructuralEngine


def _frame(timestamp: int, sensor_values: dict[str, float | None]) -> dict[str, object]:
    return {
        "timestamp": f"2026-01-01T00:00:{timestamp:02d}Z",
        "site_id": "s1",
        "asset_id": "a1",
        "sensor_values": sensor_values,
    }


def _run_engine(engine: StructuralEngine, frames: list[dict[str, object]]) -> dict:
    result = {}
    for frame in frames:
        result = engine.process_frame(frame)
    return result


def test_empty_sensor_input_has_safe_output() -> None:
    engine = StructuralEngine(baseline_window=5, recent_window=3)
    frames = [_frame(i, {}) for i in range(8)]

    result = _run_engine(engine, frames)

    assert result["n_signals"] == 0
    assert result["structural_drift_score"] >= 0.0
    assert result["relational_stability_score"] == 0.0
    assert result["experimental_analytics"]["composite_instability"] == 0.0


def test_single_sensor_input_skips_relational_metrics() -> None:
    engine = StructuralEngine(baseline_window=5, recent_window=3)
    frames = [_frame(i, {"temp": float(i)}) for i in range(8)]

    result = _run_engine(engine, frames)

    assert result["n_signals"] == 1
    assert result["relational_stability_score"] == 0.0
    assert result["experimental_analytics"]["directional"]["causal_divergence"] == 0.0


def test_partially_missing_data_treated_as_structurally_invalid() -> None:
    engine = StructuralEngine(baseline_window=5, recent_window=3)
    frames = [_frame(i, {"temp": float(i), "pressure": None}) for i in range(8)]

    result = _run_engine(engine, frames)

    assert result["n_signals"] == 1
    assert result["experimental_analytics"]["subsystems"]["subsystem_count"] == 0.0
    assert result["experimental_analytics"]["composite_instability"] == 0.0


def test_valid_multi_sensor_input_still_computes_relational_metrics() -> None:
    engine = StructuralEngine(baseline_window=5, recent_window=3)
    frames = [
        _frame(i, {"temp": float(i), "pressure": float(i) * 1.5 + 2.0})
        for i in range(8)
    ]

    result = _run_engine(engine, frames)

    assert result["n_signals"] == 2
    assert "experimental_analytics" in result
    assert result["experimental_analytics"]["composite_instability"] >= 0.0
