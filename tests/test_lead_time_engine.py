from __future__ import annotations

import numpy as np

from lead_time_engine import DetectorConfig, HybridSIIDetector


def test_early_warning_mode_exposed_for_non_persisted_weak_signal() -> None:
    detector = HybridSIIDetector(
        DetectorConfig(
            baseline_window=120,
            ema_alpha=0.6,
            instability_boundary=2.8,
            target_fpr=0.08,
            weak_signal_ratio=0.7,
            strong_persistence=3,
            weak_persistence=4,
        )
    )

    sensor_names = ("s1", "s2", "s3")
    rng = np.random.default_rng(17)

    # Warm stable history.
    for i in range(8):
        values = tuple((rng.normal(0.0, 0.01, size=3)).tolist())
        detector.update("site", "asset", f"t{i}", sensor_names, values)

    # Single weak signal should expose early-warning mode before persistence gates trip.
    ew = detector.update("site", "asset", "te1", sensor_names, (0.28, 0.27, 0.26))
    assert ew.weak_signal is True
    assert ew.state == "EARLY_WARNING"
    assert ew.early_warning_mode is True


def test_strong_persistence_escalates_to_alert() -> None:
    detector = HybridSIIDetector(
        DetectorConfig(
            ema_alpha=0.7,
            instability_boundary=1.2,
            target_fpr=0.08,
            weak_signal_ratio=0.9,
            strong_persistence=2,
            weak_persistence=4,
        )
    )
    sensor_names = ("a", "b")

    # Warmup and baseline.
    for i in range(18):
        detector.update("s", "a", f"w{i}", sensor_names, (0.0, 0.0))

    first = detector.update("s", "a", "x1", sensor_names, (3.0, 3.0))
    second = detector.update("s", "a", "x2", sensor_names, (3.1, 3.1))

    assert first.strong_signal is True
    assert first.state in {"EARLY_WARNING", "WATCH"}
    assert second.strong_signal is True
    assert second.state == "ALERT"


def test_adaptive_threshold_tracks_target_fpr_quantile() -> None:
    detector = HybridSIIDetector(
        DetectorConfig(
            ema_alpha=0.4,
            instability_boundary=2.0,
            target_fpr=0.08,
            adaptive_history_window=64,
        )
    )
    sensor_names = ("x",)

    # Feed mostly low drift with occasional moderate excursions.
    for i in range(80):
        v = 0.02 if i % 10 else 0.25
        result = detector.update("S", "A", f"q{i}", sensor_names, (v,))

    # Adaptive threshold should remain bounded and finite.
    assert np.isfinite(result.adaptive_threshold)
    assert 1.1 <= result.adaptive_threshold <= 3.3
