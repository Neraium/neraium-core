from __future__ import annotations

import math
import tempfile

import numpy as np

from neraium_core.alignment import StructuralEngine
from neraium_core.service import StructuralMonitoringService
from neraium_core.store import ResultStore


def _frame(t: int, sensors: dict[str, float]) -> dict:
    return {
        "timestamp": str(t),
        "site_id": "transition-site",
        "asset_id": "transition-asset",
        "sensor_values": sensors,
    }


def test_transition_pressure_higher_for_active_structural_break_than_steady_bad() -> None:
    with tempfile.TemporaryDirectory() as d:
        engine = StructuralEngine(baseline_window=40, recent_window=12, regime_store_path=f"{d}/r.json")
        pressure_transition: list[float] = []
        pressure_steady: list[float] = []
        pressure_transition_pre: list[float] = []

        # Stable warmup.
        for t in range(90):
            base = math.sin(0.04 * t)
            out = engine.process_frame(_frame(t, {"s1": base, "s2": 0.95 * base, "s3": 1.02 * base}))
            if t > 70:
                pressure_steady.append(float(out.get("transition_pressure", 0.0)))
            if t >= 78:
                pressure_transition_pre.append(float(out.get("transition_pressure", 0.0)))

        # Active structural departure: dynamic phase + coupling drift.
        onset_transition: list[float] = []
        for t in range(90, 150):
            phase = 0.05 * (t - 90)
            base = math.sin(0.09 * t + phase) + 0.15 * math.sin(0.2 * t)
            sensors = {
                "s1": base + 0.02 * (t - 90),
                "s2": 0.75 * base + 0.15 * math.sin(0.13 * t + phase),
                "s3": 0.4 * base + 0.6 * math.sin(0.21 * t),
            }
            out = engine.process_frame(_frame(t, sensors))
            pressure_transition.append(float(out.get("transition_pressure", 0.0)))
            if t < 102:
                onset_transition.append(float(out.get("transition_pressure", 0.0)))

        # Continuously bad but steady: hold a degraded, fixed post-break regime.
        for t in range(150, 220):
            base = 1.8 + 0.04 * math.sin(0.04 * t)
            sensors = {
                "s1": base,
                "s2": 0.18 * base + 0.35,
                "s3": -0.12 * base + 0.28,
            }
            out = engine.process_frame(_frame(t, sensors))
            if t >= 180:
                pressure_steady.append(float(out.get("transition_pressure", 0.0)))

        assert max(onset_transition) > np.mean(pressure_transition_pre) + 0.12
        assert max(pressure_transition) > np.mean(pressure_steady[-24:]) + 0.15


def test_transition_pressure_decays_after_stabilization() -> None:
    with tempfile.TemporaryDirectory() as d:
        engine = StructuralEngine(baseline_window=40, recent_window=12, regime_store_path=f"{d}/r.json")

        for t in range(90):
            base = math.sin(0.04 * t)
            engine.process_frame(_frame(t, {"s1": base, "s2": 0.95 * base, "s3": 1.02 * base}))

        active_pressures: list[float] = []
        for t in range(90, 145):
            phase = 0.07 * (t - 90)
            base = math.sin(0.1 * t + phase) + 0.2 * math.sin(0.16 * t)
            out = engine.process_frame(
                _frame(
                    t,
                    {
                        "s1": base + 0.015 * (t - 90),
                        "s2": 0.5 * base + 0.45 * math.sin(0.13 * t + phase),
                        "s3": 0.25 * base - 0.48 * math.sin(0.17 * t),
                    },
                )
            )
            active_pressures.append(float(out.get("transition_pressure", 0.0)))

        stabilized_pressures: list[float] = []
        for t in range(145, 215):
            base = 1.65 + 0.03 * math.sin(0.04 * t)
            out = engine.process_frame(_frame(t, {"s1": base, "s2": 0.22 * base + 0.34, "s3": -0.14 * base + 0.24}))
            if t >= 182:
                stabilized_pressures.append(float(out.get("transition_pressure", 0.0)))

        assert active_pressures
        assert stabilized_pressures
        assert np.mean(stabilized_pressures) < 0.65 * max(active_pressures)


def test_shock_term_spikes_on_sharp_structural_break() -> None:
    with tempfile.TemporaryDirectory() as d:
        engine = StructuralEngine(baseline_window=40, recent_window=12, regime_store_path=f"{d}/r.json")

        pre_pressures: list[float] = []
        onset_pressures: list[float] = []
        shock_boosts: list[float] = []

        for t in range(170):
            if t < 95:
                base = math.sin(0.04 * t)
                sensors = {"s1": base, "s2": 0.95 * base, "s3": 1.02 * base}
            elif t == 95:
                # Deliberate sharp structural discontinuity.
                sensors = {"s1": 2.2, "s2": -1.8, "s3": 1.4}
            else:
                base = 1.7 + 0.03 * math.sin(0.05 * t)
                sensors = {"s1": base, "s2": -0.32 * base + 0.42, "s3": 0.18 * base - 0.37}

            out = engine.process_frame(_frame(t, sensors))
            pressure = float(out.get("transition_pressure", 0.0))
            shock_boosts.append(float(out.get("experimental_analytics", {}).get("transition", {}).get("shock_boost", 0.0)))
            if 82 <= t < 95:
                pre_pressures.append(pressure)
            if 95 <= t < 101:
                onset_pressures.append(pressure)

        assert max(shock_boosts) > 0.35
        assert max(onset_pressures) > np.mean(pre_pressures) + 0.25


def test_shock_term_not_sustained_in_noisy_oscillation() -> None:
    with tempfile.TemporaryDirectory() as d:
        engine = StructuralEngine(baseline_window=40, recent_window=12, regime_store_path=f"{d}/r.json")

        shock_flags: list[int] = []
        for t in range(220):
            base = math.sin(0.11 * t) + 0.45 * math.sin(0.23 * t)
            noise = 0.05 * math.sin(0.91 * t) + 0.03 * math.cos(1.09 * t)
            out = engine.process_frame(
                _frame(
                    t,
                    {
                        "s1": base + noise,
                        "s2": 0.92 * base - 0.36 * noise,
                        "s3": 1.04 * base + 0.24 * noise,
                    },
                )
            )
            shock = float(out.get("experimental_analytics", {}).get("transition", {}).get("shock_triggered", 0.0))
            shock_flags.append(1 if shock > 0.5 else 0)

        longest_streak = 0
        curr = 0
        for flag in shock_flags:
            if flag == 1:
                curr += 1
                longest_streak = max(longest_streak, curr)
            else:
                curr = 0
        assert longest_streak <= 2


def test_rolling_baseline_pauses_during_transition_state() -> None:
    with tempfile.TemporaryDirectory() as d:
        engine = StructuralEngine(baseline_window=30, recent_window=10, regime_store_path=f"{d}/r.json")

        for t in range(85):
            base = math.sin(0.05 * t)
            out = engine.process_frame(_frame(t, {"a": base, "b": 0.93 * base, "c": 1.01 * base}))

        assert engine._rolling_baseline_corr is not None
        baseline_before = np.array(engine._rolling_baseline_corr, copy=True)
        saw_transition = False
        transition_baseline_deltas: list[float] = []
        prev_baseline = np.array(engine._rolling_baseline_corr, copy=True)

        for t in range(85, 150):
            phase = 0.12 * (t - 85)
            base = math.sin(0.08 * t + phase)
            sensors = {
                "a": base + 0.01 * (t - 85),
                "b": 0.45 * base + 0.55 * math.sin(0.18 * t + phase),
                "c": 0.2 * base - 0.5 * math.sin(0.12 * t),
            }
            out = engine.process_frame(_frame(t, sensors))
            if out.get("transition_state") in {"EMERGING_TRANSITION", "SUSTAINED_TRANSITION"}:
                saw_transition = True
                if engine._rolling_baseline_corr is not None:
                    curr_baseline = np.array(engine._rolling_baseline_corr, copy=True)
                    transition_baseline_deltas.append(float(np.linalg.norm(curr_baseline - prev_baseline)))
                    prev_baseline = curr_baseline

        assert saw_transition
        assert transition_baseline_deltas
        significant = [d for d in transition_baseline_deltas if d > 1e-8]
        assert len(significant) <= 1
        assert np.all(np.isfinite(engine._rolling_baseline_corr))
        assert np.linalg.norm(np.array(engine._rolling_baseline_corr) - baseline_before) < 1e-5


def test_service_interpretation_escalates_on_transition_pressure(tmp_path) -> None:
    service = StructuralMonitoringService(
        engine=StructuralEngine(baseline_window=5, recent_window=3, regime_store_path=str(tmp_path / "r.json")),
        store=ResultStore(db_path=str(tmp_path / "test.db")),
    )
    interpreted = service._interpret(  # pylint: disable=protected-access
        {
            "state": "STABLE",
            "structural_drift_score": 0.1,
            "transition_pressure": 1.25,
            "transition_state": "SUSTAINED_TRANSITION",
        }
    )

    assert interpreted["risk_level"] == "HIGH"
    assert interpreted["action_state"] == "ALERT"
    assert "transition pressure" in interpreted["operator_message"].lower()


def test_transition_aware_toggle_restores_legacy_interpretation(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("NERAIUM_TRANSITION_AWARE", "0")
    service = StructuralMonitoringService(
        engine=StructuralEngine(baseline_window=5, recent_window=3, regime_store_path=str(tmp_path / "r.json")),
        store=ResultStore(db_path=str(tmp_path / "test.db")),
    )
    interpreted = service._interpret(  # pylint: disable=protected-access
        {
            "state": "STABLE",
            "structural_drift_score": 0.1,
            "transition_pressure": 1.25,
            "transition_state": "SUSTAINED_TRANSITION",
        }
    )

    assert interpreted["risk_level"] == "LOW"
    assert interpreted["action_state"] == "STABLE"
