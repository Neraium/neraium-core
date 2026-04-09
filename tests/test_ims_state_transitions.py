import tempfile

from neraium_core.alignment import StructuralEngine


def _engine_for_state_machine(**kwargs) -> StructuralEngine:
    d = tempfile.mkdtemp(prefix="ims_state_machine_")
    return StructuralEngine(
        baseline_window=8,
        recent_window=4,
        regime_store_path=f"{d}/regimes.json",
        **kwargs,
    )


def test_alert_latches_and_releases_after_deep_recovery():
    engine = _engine_for_state_machine(
        drift_smoothing_window=1,
        watch_persistence=5,
        alert_persistence=3,
        fast_trigger_multiplier=1.25,
    )
    engine._drift_watch_alert_thresholds = (0.8, 1.3)

    states = []
    for _ in range(5):
        state, _ = engine._update_drift_state_machine(0.95)
        states.append(state)
    assert states[-1] == "WATCH"

    for _ in range(8):
        state, _ = engine._update_drift_state_machine(1.5)
    assert state == "ALERT"
    assert engine._alert_latched

    # Mild recovery does not clear a latched alert.
    for _ in range(4):
        state, _ = engine._update_drift_state_machine(0.9)
    assert state == "ALERT"

    # Deep recovery below release ratio clears latch.
    state, _ = engine._update_drift_state_machine(0.4)
    assert state in {"WATCH", "STABLE"}
    assert not engine._alert_latched

    for _ in range(20):
        state, _ = engine._update_drift_state_machine(0.4)
    assert state == "STABLE"


def test_smoothing_and_persistence_prevent_single_spike_flicker():
    engine = _engine_for_state_machine(drift_smoothing_window=5, watch_persistence=4, alert_persistence=3)
    engine._drift_watch_alert_thresholds = (0.9, 1.4)

    states = []
    signal = [0.3] * 8 + [2.4] + [0.3] * 8
    for value in signal:
        state, _ = engine._update_drift_state_machine(value)
        states.append(state)

    assert "WATCH" not in states
    assert "ALERT" not in states


def test_fast_trigger_override_latches_immediately():
    engine = _engine_for_state_machine(
        drift_smoothing_window=1,
        watch_persistence=5,
        alert_persistence=3,
        fast_trigger_multiplier=1.25,
    )
    engine._drift_watch_alert_thresholds = (0.8, 1.2)
    state, _ = engine._update_drift_state_machine(1.55)
    assert state == "ALERT"
    assert engine._alert_latched


def test_process_frame_exposes_explicit_policy_outputs():
    engine = _engine_for_state_machine()
    out = engine.process_frame(
        {
            "timestamp": 0.0,
            "site_id": "ims",
            "asset_id": "bearing",
            "sensor_values": {"s1": 0.1, "s2": 0.2, "s3": 0.3},
        }
    )
    assert out["state"] == out["policy_state"]
    assert "policy_watch" in out
    assert "policy_alert" in out
    assert "drift_smooth" in out
    assert "watch_threshold" in out
    assert "alert_threshold" in out
