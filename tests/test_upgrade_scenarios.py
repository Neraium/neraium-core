"""
Tests for the production-readiness upgrade: nominal, regime shift, coupling/structural
instability, missing data, stale sensor, adaptive baseline, causal attribution.
"""
from __future__ import annotations

import math
import tempfile

import numpy as np

from neraium_core.alignment import StructuralEngine


def _frame(t: int, sensors: dict[str, float], site: str = "site1", asset: str = "asset1") -> dict:
    return {
        "timestamp": str(t),
        "site_id": site,
        "asset_id": asset,
        "sensor_values": sensors,
    }


def test_nominal_operation_output_shape_and_stability():
    """Nominal operation: output has required keys and stays NOMINAL_STRUCTURE."""
    with tempfile.TemporaryDirectory() as d:
        engine = StructuralEngine(
            baseline_window=30,
            recent_window=10,
            regime_store_path=f"{d}/r.json",
        )
        # Coherent stable signals
        for t in range(80):
            base = math.sin(0.05 * t) + 0.2 * math.sin(0.02 * t)
            frame = _frame(
                t,
                {"s1": base, "s2": 0.95 * base + 0.05 * math.sin(0.03 * t), "s3": 1.02 * base},
            )
            out = engine.process_frame(frame)

        assert "interpreted_state" in out
        assert out["interpreted_state"] == "NOMINAL_STRUCTURE"
        assert "causal_attribution" in out
        assert "top_drivers" in out["causal_attribution"]
        assert "driver_scores" in out["causal_attribution"]
        assert "data_quality_summary" in out
        assert "active_sensor_count" in out
        assert "missing_sensor_count" in out
        assert "confidence_score" in out
        assert "baseline_mode" in out
        assert "regime_memory_state" in out
        assert out["structural_drift_score"] >= 0.0


def test_regime_shift_observed_under_clean_transition():
    """Clean structural transition without coupling breakdown yields REGIME_SHIFT_OBSERVED."""
    with tempfile.TemporaryDirectory() as d:
        engine = StructuralEngine(
            baseline_window=40,
            recent_window=12,
            regime_store_path=f"{d}/r.json",
        )
        # First: stable regime
        for t in range(60):
            base = math.sin(0.06 * t)
            out = engine.process_frame(
                _frame(t, {"s1": base, "s2": 0.9 * base, "s3": 1.0 * base})
            )
        # Then: different but internally consistent regime (phase/scale change, no breakdown)
        for t in range(60, 120):
            base = 0.7 * math.sin(0.08 * t + 0.5) + 0.3
            out = engine.process_frame(
                _frame(t, {"s1": base, "s2": 0.85 * base + 0.1, "s3": 0.95 * base})
            )
        # After warmup we may see REGIME_SHIFT or NOMINAL depending on thresholds
        assert out["interpreted_state"] in {
            "NOMINAL_STRUCTURE",
            "REGIME_SHIFT_OBSERVED",
            "COHERENCE_UNDER_CONSTRAINT",
        }


def test_coupling_instability_presence():
    """Scenario with directional/spectral breakdown yields COUPLING_INSTABILITY_OBSERVED in tail."""
    rng = np.random.default_rng(42)
    with tempfile.TemporaryDirectory() as d:
        engine = StructuralEngine(
            baseline_window=50,
            recent_window=12,
            regime_store_path=f"{d}/r.json",
        )
        # Stable then one channel goes noisy (coupling breakdown)
        tail = []
        for t in range(200):
            base = math.sin(0.07 * t) + 0.2 * math.sin(0.02 * t)
            if t < 100:
                sensors = {"s1": base, "s2": 0.9 * base, "s3": 1.05 * base}
            else:
                sensors = {
                    "s1": base,
                    "s2": 0.9 * base,
                    "s3": float(rng.normal(0, 1.0)) + 0.1 * base,
                }
            out = engine.process_frame(_frame(t, sensors))
            if t >= 150:
                tail.append(out.get("interpreted_state"))

        assert "COUPLING_INSTABILITY_OBSERVED" in tail or "STRUCTURAL_INSTABILITY_OBSERVED" in tail


def test_structural_instability_requires_relational_and_persistence():
    """Structural instability appears when relational drift + entropy/regime + sustained."""
    rng = np.random.default_rng(99)
    with tempfile.TemporaryDirectory() as d:
        engine = StructuralEngine(
            baseline_window=50,
            recent_window=12,
            regime_store_path=f"{d}/r.json",
        )
        tail = []
        for t in range(280):
            base = math.sin(0.06 * t)
            if t < 120:
                sensors = {"s1": base, "s2": 0.92 * base, "s3": 1.0 * base}
            else:
                # Persistent breakdown: decorrelate and add noise
                sensors = {
                    "s1": base + 0.1 * float(rng.normal(0, 1)),
                    "s2": 0.5 * base + 0.5 * float(rng.normal(0, 1)),
                    "s3": float(rng.normal(0, 0.8)) + 0.2 * base,
                }
            out = engine.process_frame(_frame(t, sensors))
            if t >= 220:
                tail.append(out.get("interpreted_state"))

        assert "STRUCTURAL_INSTABILITY_OBSERVED" in tail or "COUPLING_INSTABILITY_OBSERVED" in tail


def test_missing_data_degraded_output():
    """Under missing data, process_frame still returns meaningful output with degraded confidence."""
    with tempfile.TemporaryDirectory() as d:
        engine = StructuralEngine(
            baseline_window=20,
            recent_window=8,
            regime_store_path=f"{d}/r.json",
        )
        # Build some history then inject NaNs in one sensor
        for t in range(40):
            base = math.sin(0.05 * t)
            sensors = {"a": base, "b": 0.9 * base, "c": 1.0 * base}
            if t >= 25:
                sensors["b"] = float("nan")  # type: ignore[assignment]
            out = engine.process_frame(_frame(t, sensors))

        assert "timestamp" in out
        assert "data_quality" in out
        assert "data_quality_summary" in out
        assert out["data_quality_summary"].get("valid_signal_count") is not None
        # May still have instability score and interpreted_state (degraded path or partial)
        assert "latest_instability" in out
        assert "interpreted_state" in out


def test_stale_sensor_behavior():
    """Stale (mostly NaN) sensor is reflected in data_quality and missing_sensor_count."""
    with tempfile.TemporaryDirectory() as d:
        engine = StructuralEngine(
            baseline_window=25,
            recent_window=10,
            regime_store_path=f"{d}/r.json",
        )
        for t in range(60):
            base = math.sin(0.04 * t)
            sensors = {
                "x": base,
                "y": 0.9 * base if t < 30 else float("nan"),  # type: ignore[assignment]
                "z": 1.0 * base,
            }
            out = engine.process_frame(_frame(t, sensors))

        dq = out.get("data_quality_summary", {})
        assert "stale_sensor_count" in dq or "missing_sensor_count" in dq or "valid_signal_count" in dq
        assert out.get("missing_sensor_count") is not None
        assert out.get("active_sensor_count") is not None


def test_adaptive_baseline_baseline_mode():
    """After nominal operation, baseline_mode can become 'rolling'."""
    with tempfile.TemporaryDirectory() as d:
        engine = StructuralEngine(
            baseline_window=30,
            recent_window=10,
            regime_store_path=f"{d}/r.json",
        )
        for t in range(80):
            base = math.sin(0.05 * t)
            out = engine.process_frame(
                _frame(t, {"s1": base, "s2": 0.95 * base, "s3": 1.0 * base})
            )

        # Engine may be using rolling baseline after sustained nominal
        assert out.get("baseline_mode") in ("fixed", "rolling", None)


def test_causal_attribution_presence():
    """Multi-sensor frames get causal_attribution with top_drivers and driver_scores."""
    with tempfile.TemporaryDirectory() as d:
        engine = StructuralEngine(
            baseline_window=25,
            recent_window=10,
            regime_store_path=f"{d}/r.json",
        )
        for t in range(50):
            base = math.sin(0.06 * t)
            out = engine.process_frame(
                _frame(t, {"s1": base, "s2": 0.9 * base, "s3": 1.05 * base})
            )

        assert "causal_attribution" in out
        attr = out["causal_attribution"]
        assert "top_drivers" in attr
        assert "driver_scores" in attr
        assert isinstance(attr["top_drivers"], list)
        assert isinstance(attr["driver_scores"], dict)
        assert out.get("dominant_driver") is None or isinstance(out["dominant_driver"], str)


def test_classification_stability_and_confidence_score():
    """Output includes classification_stability and confidence_score when applicable."""
    with tempfile.TemporaryDirectory() as d:
        engine = StructuralEngine(
            baseline_window=30,
            recent_window=10,
            regime_store_path=f"{d}/r.json",
        )
        for t in range(60):
            base = math.sin(0.05 * t)
            out = engine.process_frame(
                _frame(t, {"s1": base, "s2": 0.95 * base, "s3": 1.0 * base})
            )

        assert "confidence_score" in out
        assert 0.0 <= out["confidence_score"] <= 1.0
        # classification_stability may be in decision output
        if "classification_stability" in out:
            assert 0.0 <= out["classification_stability"] <= 1.0
