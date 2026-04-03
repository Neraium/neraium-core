from __future__ import annotations

import numpy as np

from neraium_core.alignment import StructuralEngine
from neraium_core.context_invariant_representation import (
    TemporalRepresentationConfig,
    build_temporal_representation,
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
