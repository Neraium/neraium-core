import numpy as np

from fd00x.qit_detector import create_qit_detector


def _baseline(n=200, d=6):
    rng = np.random.default_rng(42)
    return rng.normal(0.0, 1.0, size=(n, d))


def test_qit_detector_api_and_output_shape():
    baseline = _baseline()
    sensors = [f"s{i}" for i in range(baseline.shape[1])]
    detector = create_qit_detector(sensors=sensors, baseline_data=baseline)

    result = detector.detect(baseline[0])

    assert set(result.keys()) == {"scores", "fused", "can_alert", "diagnostics"}
    assert set(result["scores"].keys()) == {
        "quantum",
        "information",
        "free_energy",
        "topological",
        "algorithmic",
    }
    assert 0.0 <= result["fused"]["total"] <= 1.0
    assert result["fused"]["state"] in {"GREEN", "AMBER", "YELLOW", "RED"}


def test_qit_detector_escalates_under_sustained_shift():
    baseline = _baseline(n=240)
    sensors = [f"s{i}" for i in range(baseline.shape[1])]
    detector = create_qit_detector(sensors=sensors, baseline_data=baseline)

    # burn-in with healthy behavior
    for row in baseline[:40]:
        detector.detect(row)

    shifted = baseline[40:] + 4.0
    states = [detector.detect(row)["fused"]["state"] for row in shifted]

    assert "AMBER" in states or "YELLOW" in states or "RED" in states
