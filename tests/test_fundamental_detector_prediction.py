import numpy as np

from fundamental_detector import FundamentalChangeDetector


def test_predict_future_requires_history():
    detector = FundamentalChangeDetector(n_sensors=4, window_size=32, min_samples=5)
    out = detector.predict_future(hours_ahead=168)
    assert "error" in out
    assert out["predictions"] == {}


def test_predict_future_emits_7_day_horizon():
    rng = np.random.default_rng(7)
    detector = FundamentalChangeDetector(n_sensors=4, window_size=40, min_samples=5)
    base = np.array([1.0, 2.0, 3.0, 4.0])
    for i in range(60):
        noise = rng.normal(0.0, 0.05, size=4)
        drift = np.array([0.01 * i, 0.012 * i, -0.008 * i, 0.009 * i])
        detector.update(base + drift + noise)
    forecast = detector.predict_future(hours_ahead=168)
    assert "predictions" in forecast
    for key in [6, 24, 48, 72, 96, 120, 144, 168]:
        assert key in forecast["predictions"]
        assert "status" in forecast["predictions"][key]
        assert "confidence" in forecast["predictions"][key]
