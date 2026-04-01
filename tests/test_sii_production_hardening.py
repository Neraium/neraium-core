from __future__ import annotations

from datetime import datetime, timedelta, timezone

from neraium_core.sii.config import SIIConfig
from neraium_core.sii.engine import SIIEngine
from neraium_core.sii.types import SIIFrame


def _engine(tmp_path, name: str) -> SIIEngine:
    return SIIEngine(
        config=SIIConfig(
            baseline_window=8,
            recent_window=4,
            max_history=200,
            min_samples_for_alerts=8,
            allow_context_provider=False,
            regime_store_path=str(tmp_path / f"{name}.json"),
        )
    )


def _frame(ts: datetime, sensors: dict[str, float | None]) -> SIIFrame:
    return SIIFrame(
        timestamp=ts.isoformat(),
        site_id="site-test",
        system_id="system-test",
        asset_id="asset-test",
        node_id="node-test",
        sensor_values=dict(sensors),
    )


def test_stable_input_remains_stable(tmp_path) -> None:
    engine = _engine(tmp_path, "stable")
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    outputs = []
    for i in range(24):
        outputs.append(
            engine.process_frame(
                _frame(
                    start + timedelta(minutes=i),
                    {
                        "pressure": 50.0 + (0.03 * (i % 3)),
                        "flow": 12.0 - (0.02 * (i % 3)),
                        "vibration": 1.0 + (0.01 * (i % 2)),
                    },
                )
            )
        )
    assert all(o["state"] == "STABLE" for o in outputs[-10:])


def test_gradual_drift_increases_instability_signal(tmp_path) -> None:
    engine = _engine(tmp_path, "drift")
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    outputs = []
    for i in range(30):
        outputs.append(
            engine.process_frame(
                _frame(
                    start + timedelta(minutes=i),
                    {
                        "pressure": 50.0 + 0.35 * i,
                        "flow": 12.5 - 0.05 * i - 0.01 * (i**1.2),
                        "vibration": 1.1 + 0.05 * i + (0.02 if i % 3 == 0 else -0.01),
                    },
                )
            )
        )
    assert any(o["interpreted_state"] != "NOMINAL_STRUCTURE" for o in outputs[12:])


def test_abrupt_spike_triggers_elevated_state(tmp_path) -> None:
    engine = _engine(tmp_path, "spike")
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    outputs = []
    for i in range(26):
        sensors = {"pressure": 50.0, "flow": 12.0, "vibration": 1.1}
        if i >= 18:
            sensors = {"pressure": 70.0, "flow": 8.0, "vibration": 5.7}
        outputs.append(engine.process_frame(_frame(start + timedelta(minutes=i), sensors)))
    assert any(o["state"] in {"WATCH", "ALERT"} for o in outputs[18:])


def test_missing_sensors_force_quality_warning_mode(tmp_path) -> None:
    engine = _engine(tmp_path, "missing")
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for i in range(12):
        engine.process_frame(
            _frame(
                start + timedelta(minutes=i),
                {"pressure": 50.0 + i * 0.1, "flow": 12.0 - i * 0.05, "vibration": 1.0 + i * 0.02},
            )
        )
    out = None
    for j in range(13, 18):
        out = engine.process_frame(
            _frame(
                start + timedelta(minutes=j),
                {"pressure": None, "flow": None, "vibration": 1.0},
            )
        )
    assert out is not None
    assert out["state"] == "STABLE"
    assert (out["experimental_analytics"] or {}).get("processing", {}).get("code") == "data_quality_gate_failed"


def test_reordered_timestamp_is_dropped_explicitly(tmp_path) -> None:
    engine = _engine(tmp_path, "reordered")
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    first = engine.process_frame(_frame(start, {"a": 1.0, "b": 2.0}))
    assert first["state"] == "STABLE"
    dropped = engine.process_frame(_frame(start - timedelta(minutes=1), {"a": 1.1, "b": 2.1}))
    assert dropped["experimental_analytics"]["processing"]["code"] == "out_of_order_timestamp_dropped"
    assert dropped["experimental_analytics"]["processing"]["dropped"] is True


def test_inconsistent_feature_dimensions_expand_schema_and_remain_deterministic(tmp_path) -> None:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    seq = []
    for i in range(22):
        sensors = {"pressure": 50.0 + i * 0.2, "flow": 11.8 - i * 0.06, "vibration": 1.1 + i * 0.05}
        if i >= 10:
            sensors["temperature"] = 78.0 + i * 0.3
        seq.append((start + timedelta(minutes=i), sensors))

    outputs = []
    for run in (1, 2):
        engine = _engine(tmp_path, f"schema_{run}")
        run_outputs = [engine.process_frame(_frame(ts, sensors)) for ts, sensors in seq]
        outputs.append(
            [
                (
                    o["state"],
                    o["interpreted_state"],
                    round(float((o["experimental_analytics"] or {}).get("composite_instability", 0.0)), 6),
                )
                for o in run_outputs
            ]
        )
    assert outputs[0] == outputs[1]
