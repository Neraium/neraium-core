from neraium_core.decision_layer import evaluate_signal


def _row(cycle: int, instability: float, drift: float, phase: str, risk: str = "LOW") -> dict:
    return {
        "cycle": cycle,
        "composite_instability": instability,
        "structural_drift_score": drift,
        "phase": phase,
        "risk_level": risk,
    }


def test_evaluate_signal_stable_system_no_signal():
    timeseries = [
        _row(1, 0.2, 0.10, "stable"),
        _row(2, 0.25, 0.11, "stable"),
        _row(3, 0.22, 0.12, "stable"),
        _row(4, 0.20, 0.10, "stable"),
        _row(5, 0.24, 0.11, "stable"),
    ]
    result = evaluate_signal(timeseries, {"peak_instability": 0.25})

    assert result["signal_emitted"] is False
    assert result["signal_strength"] == "low"


def test_evaluate_signal_gradual_instability_emits_signal():
    timeseries = [
        _row(1, 0.50, 0.20, "stable", "LOW"),
        _row(2, 0.65, 0.30, "drift", "MEDIUM"),
        _row(3, 0.74, 0.40, "drift", "MEDIUM"),
        _row(4, 0.79, 0.52, "unstable", "HIGH"),
        _row(5, 0.84, 0.61, "unstable", "HIGH"),
        _row(6, 0.88, 0.73, "unstable", "HIGH"),
    ]
    result = evaluate_signal(timeseries, {"peak_instability": 0.88})

    assert result["signal_emitted"] is True
    assert result["signal_strength"] in {"medium", "high"}
    assert result["confidence"] in {"medium", "high"}


def test_evaluate_signal_noisy_data_suppresses_signal():
    timeseries = [
        _row(1, 0.35, 0.20, "stable", "LOW"),
        _row(2, 0.90, 0.45, "unstable", "HIGH"),
        _row(3, 0.50, 0.40, "stable", "LOW"),
        _row(4, 0.86, 0.50, "unstable", "HIGH"),
        _row(5, 0.55, 0.44, "stable", "LOW"),
        _row(6, 0.87, 0.54, "unstable", "HIGH"),
    ]
    result = evaluate_signal(timeseries, {"peak_instability": 0.90})

    assert result["signal_emitted"] is False
    assert any("suppressed" in reason for reason in result["reason"])
