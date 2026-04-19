from __future__ import annotations

import numpy as np

from neraium_core.alignment import StructuralEngine
from neraium_core.context_invariant_representation import (
    TemporalRepresentationConfig,
    build_temporal_representation,
    _rolling_median,
)


def _make_series(offset: float, degrade_start: int | None = None) -> np.ndarray:
    rows: list[list[float]] = []
    for t in range(80):
        base = np.array(
            [
                offset + 0.015 * t,
                2.0 * offset + 0.009 * t,
                -0.5 * offset + 0.012 * t,
            ],
            dtype=float,
        )
        if degrade_start is not None and t >= degrade_start:
            drift = float(t - degrade_start)
            base = base + np.array([0.07 * drift, -0.04 * drift, 0.05 * drift], dtype=float)
        rows.append(base.tolist())
    return np.asarray(rows, dtype=float)


def _window_distance(a: np.ndarray, b: np.ndarray, end: int, width: int = 8) -> float:
    sa = a[max(0, end - width) : end]
    sb = b[max(0, end - width) : end]
    return float(np.mean(np.linalg.norm(sa - sb, axis=1)))


def test_dynamic_representation_reduces_early_context_separation() -> None:
    healthy_low = _make_series(offset=0.0, degrade_start=None)
    healthy_high = _make_series(offset=4.0, degrade_start=None)

    raw_cfg = TemporalRepresentationConfig(mode="raw", reference_strategy="initial_window")
    combined_cfg = TemporalRepresentationConfig(mode="combined", reference_strategy="robust")

    raw_low = build_temporal_representation(healthy_low, raw_cfg).transformed
    raw_high = build_temporal_representation(healthy_high, raw_cfg).transformed
    comb_low = build_temporal_representation(healthy_low, combined_cfg).transformed
    comb_high = build_temporal_representation(healthy_high, combined_cfg).transformed

    early_raw_sep = _window_distance(raw_low, raw_high, end=20)
    early_combined_sep = _window_distance(comb_low, comb_high, end=20)

    assert early_combined_sep < early_raw_sep


def test_dynamic_representation_amplifies_late_evolution_divergence() -> None:
    nominal = _make_series(offset=1.0, degrade_start=None)
    degrading = _make_series(offset=1.0, degrade_start=45)

    cfg = TemporalRepresentationConfig(mode="combined", reference_strategy="robust")
    nominal_repr = build_temporal_representation(nominal, cfg).transformed
    degrading_repr = build_temporal_representation(degrading, cfg).transformed

    early_sep = _window_distance(nominal_repr, degrading_repr, end=25)
    late_sep = _window_distance(nominal_repr, degrading_repr, end=79)

    assert late_sep > early_sep * 1.8


def test_structural_engine_emits_context_diagnostics() -> None:
    engine = StructuralEngine(
        baseline_window=24,
        recent_window=8,
        representation_mode="combined",
        reference_strategy="robust",
        context_diagnostics_enabled=True,
    )
    out = None
    for t in range(70):
        s1 = 0.02 * t
        s2 = 0.01 * t + (0.15 * max(0, t - 45))
        s3 = -0.015 * t + (0.08 * max(0, t - 45))
        out = engine.process_frame(
            {
                "timestamp": f"{t}",
                "site_id": "site-a",
                "asset_id": "asset-a",
                "sensor_values": {"s1": s1, "s2": s2, "s3": s3},
            }
        )

    assert out is not None
    assert "context_dominance_score" in out
    assert "dynamic_signal_strength" in out
    assert "early_separation_flag" in out
    exp = out.get("experimental_analytics") or {}
    assert "context_diagnostics" in exp
    assert exp.get("representation", {}).get("mode") == "combined"


def test_rolling_median_no_index_error_on_window_transition() -> None:
    """Regression test for IndexError in rolling_median at window completion.

    Previously, the condition `if t < w - 1:` caused an off-by-one error where
    at t = w-1 (first complete window), the code would incorrectly use sliding
    window logic and try to access a[t-w] = a[-1] (last element) instead of
    rebuilding the complete window. This test ensures the fix is in place.

    See: https://github.com/neraium/neraium-core/issues/XXXX
    """
    # Single column case that triggered the original error
    arr = np.array([
        [1.0],
        [2.0],
        [3.0],
        [4.0],
        [5.0],
    ], dtype=float)

    # Should not raise IndexError
    result = _rolling_median(arr, window=3)

    # Verify results are correct
    assert result.shape == arr.shape
    # First value: median of [1.0] = 1.0
    assert np.isclose(result[0, 0], 1.0)
    # Second value: median of [1.0, 2.0] = 1.5
    assert np.isclose(result[1, 0], 1.5)
    # Third value: median of [1.0, 2.0, 3.0] = 2.0
    assert np.isclose(result[2, 0], 2.0)
    # Fourth value: median of [2.0, 3.0, 4.0] = 3.0
    assert np.isclose(result[3, 0], 3.0)
    # Fifth value: median of [3.0, 4.0, 5.0] = 4.0
    assert np.isclose(result[4, 0], 4.0)


def test_rolling_median_multi_column() -> None:
    """Test rolling_median with multiple columns after window transition fix."""
    arr = np.array([
        [1.0, 10.0],
        [2.0, 20.0],
        [3.0, 30.0],
        [4.0, 40.0],
        [5.0, 50.0],
    ], dtype=float)

    result = _rolling_median(arr, window=3)

    assert result.shape == arr.shape
    # Check column 0
    assert np.isclose(result[2, 0], 2.0)
    assert np.isclose(result[3, 0], 3.0)
    # Check column 1
    assert np.isclose(result[2, 1], 20.0)
    assert np.isclose(result[3, 1], 30.0)


def test_rolling_median_larger_window() -> None:
    """Test rolling_median with larger windows to ensure sustained correctness."""
    arr = np.arange(20, dtype=float).reshape(-1, 1)

    result = _rolling_median(arr, window=5)

    assert result.shape == arr.shape
    # At index 4 (first complete window): median of [0, 1, 2, 3, 4] = 2
    assert np.isclose(result[4, 0], 2.0)
    # At index 10: median of [6, 7, 8, 9, 10] = 8
    assert np.isclose(result[10, 0], 8.0)
