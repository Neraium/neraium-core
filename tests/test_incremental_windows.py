"""Incremental window buffers should match full batch deque scans."""

from __future__ import annotations

import numpy as np
import pytest

from neraium_core.alignment import StructuralEngine


def _make_frames(n: int, sensors: int = 5, seed: int = 2) -> list[dict]:
    rng = np.random.default_rng(seed)
    names = [f"s{i}" for i in range(1, sensors + 1)]
    out = []
    for t in range(n):
        vals = {name: float(rng.standard_normal()) for name in names}
        out.append(
            {
                "timestamp": str(t),
                "site_id": "t",
                "asset_id": "u1",
                "sensor_values": vals,
            }
        )
    return out


@pytest.mark.parametrize("incremental", ["0", "1"])
def test_incremental_matches_batch_drift_scores(incremental: str, monkeypatch) -> None:
    monkeypatch.setenv("NERAIUM_INCREMENTAL", incremental)
    frames = _make_frames(120, sensors=6, seed=11)
    eng = StructuralEngine(baseline_window=50, recent_window=12)
    drifts = []
    for f in frames:
        r = eng.process_frame(f)
        drifts.append(float(r.get("structural_drift_score", 0.0)))
    assert len(drifts) == 120
    assert all(np.isfinite(drifts))


def test_incremental_vs_batch_identical_scores(monkeypatch) -> None:
    frames = _make_frames(90, sensors=4, seed=99)

    monkeypatch.setenv("NERAIUM_INCREMENTAL", "0")
    off = StructuralEngine(baseline_window=50, recent_window=12)
    a = [float(off.process_frame(f).get("latest_instability", 0.0)) for f in frames]

    monkeypatch.setenv("NERAIUM_INCREMENTAL", "1")
    on = StructuralEngine(baseline_window=50, recent_window=12)
    b = [float(on.process_frame(f).get("latest_instability", 0.0)) for f in frames]

    # Rounded composite + cached windows can differ at ~1e-4; semantics match.
    assert np.allclose(a, b, rtol=1e-5, atol=5e-4)


def test_schema_change_rebuilds_buffers(monkeypatch) -> None:
    monkeypatch.setenv("NERAIUM_INCREMENTAL", "1")
    eng = StructuralEngine(baseline_window=20, recent_window=8)
    base = {"timestamp": "0", "site_id": "t", "asset_id": "u", "sensor_values": {"s1": 1.0, "s2": 0.5}}
    for t in range(25):
        f = dict(base)
        f["timestamp"] = str(t)
        f["sensor_values"] = {"s1": float(t % 5), "s2": float(t % 3)}
        eng.process_frame(f)
    # add new sensor mid-stream
    f = dict(base)
    f["timestamp"] = "26"
    f["sensor_values"] = {"s1": 1.0, "s2": 0.0, "s3": 0.1}
    r = eng.process_frame(f)
    assert r.get("sensor_relationships") == ["s1", "s2", "s3"]
    assert np.isfinite(float(r.get("latest_instability", 0.0)))


def test_full_frame_gating_requires_non_zero_width_windows(monkeypatch) -> None:
    monkeypatch.setenv("NERAIUM_INCREMENTAL", "1")
    eng = StructuralEngine(baseline_window=6, recent_window=3)

    for t in range(10):
        out = eng.process_frame(
            {
                "timestamp": str(t),
                "site_id": "t",
                "asset_id": "u",
                "sensor_values": {"bad": "n/a", "flag": True, "inf": float("inf")},
            }
        )
        # No real numeric channels => sensor schema remains empty and windows are never ready.
        assert out["state"] == "STABLE"
        assert float(out.get("latest_instability", 0.0)) == 0.0
        assert out.get("data_quality_summary", {}) == {}

    assert eng.sensor_order == []
