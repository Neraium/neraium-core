"""Neraium engine readiness and transition classification gating (alignment + detection)."""

from __future__ import annotations

import math
import tempfile

import numpy as np

from neraium_core.alignment import StructuralEngine
from neraium_core.detection.composite_signal import composite_structural_pressure
from neraium_core.detection.readiness import compute_engine_readiness
from neraium_core.detection.transition_config import TransitionDetectionConfig
from neraium_core.detection.transition_detector import detect_transition, detect_transition_from_signals


def _frame(t: int, sensors: dict[str, float]) -> dict:
    return {
        "timestamp": str(t),
        "site_id": "s",
        "asset_id": "a",
        "sensor_values": sensors,
    }


def test_compute_engine_readiness_not_ready_before_windows() -> None:
    st = compute_engine_readiness(
        frame_count=3,
        baseline_window=10,
        recent_window=5,
        transition_pressure_history_len=0,
        warmup_margin_frames=2,
        min_transition_history=4,
    )
    assert st.structural_ready is False
    assert st.transition_classification_ready is False


def test_compute_engine_readiness_classification_ready() -> None:
    st = compute_engine_readiness(
        frame_count=20,
        baseline_window=10,
        recent_window=5,
        transition_pressure_history_len=8,
        warmup_margin_frames=2,
        min_transition_history=6,
    )
    assert st.structural_ready is True
    assert st.transition_classification_ready is True


def test_alignment_readiness_fields_and_warmup_state() -> None:
    with tempfile.TemporaryDirectory() as d:
        engine = StructuralEngine(
            baseline_window=10,
            recent_window=5,
            regime_store_path=f"{d}/r.json",
        )
        engine.transition_stabilization_margin_frames = 4
        engine.transition_classification_min_history = 4
        last = None
        for t in range(20):
            base = math.sin(0.1 * t)
            last = engine.process_frame(_frame(t, {"s1": base, "s2": 0.9 * base, "s3": 1.02 * base}))
        assert last is not None
        assert "readiness" in last
        assert "engine_ready" in last
        assert "engine_stabilization_progress" in last
        assert "engine_warmup_progress" in last
        assert "engine_min_history_required" in last
        assert "transition_outputs_actionable" in last
        assert "engine_transition_detectable" in last
        assert "neraium" in last
        assert last["readiness"]["structural_ready"] is True
        # Early frames in run should have seen WARMUP; late frame should be classifiable or NONE
        assert last["transition_state"] in {"NONE", "WARMUP", "EMERGING_TRANSITION", "SUSTAINED_TRANSITION", "METASTABLE"}


def test_detect_transition_readiness_gate() -> None:
    import pandas as pd

    df = pd.DataFrame(
        {
            "cycle": [1.0, 2.0, 3.0, 4.0],
            "transition_pressure": [0.1, 0.9, 0.95, 0.96],
        }
    )
    cfg = TransitionDetectionConfig(index_column="cycle", warmup_index_floor=1.0, threshold=0.5, sustained_points=2)
    rd = {"transition_classification_ready": False, "reason": "test"}
    r = detect_transition(df, cfg, readiness=rd)
    assert r["detected"] is False
    assert r["detection_reason"] == "readiness_not_met"


def test_detect_transition_from_signals() -> None:
    n = 20
    sig = np.linspace(0.1, 0.95, n)
    r = detect_transition_from_signals(
        {"transition_pressure": sig},
        config=TransitionDetectionConfig(index_column="cycle", warmup_index_floor=0.0, threshold=0.5),
    )
    assert r["detected"] is True


def test_composite_structural_pressure() -> None:
    score, meta = composite_structural_pressure(
        {"transition_pressure": 0.2, "structural_drift_score": 0.8, "latest_instability": 0.1},
        weights={"transition_pressure": 1.0, "structural_drift_score": 1.0, "latest_instability": 1.0},
    )
    assert 0.0 <= score <= 1.0
    assert "agreement" in meta


def test_signal_artifact_window_skips_early_crossing() -> None:
    import pandas as pd

    p = np.concatenate([np.linspace(0.9, 0.95, 8), np.linspace(0.2, 0.9, 40)])
    df = pd.DataFrame({"cycle": np.arange(1, len(p) + 1, dtype=float), "transition_pressure": p})
    cfg = TransitionDetectionConfig(
        warmup_index_floor=1.0,
        threshold=0.5,
        sustained_points=2,
        signal_artifact_window=6,
        smoothing_window=1,
        fallback_mode="none",
    )
    r = detect_transition(df, cfg)
    assert r["detected"] is True
    # First index considered for crossing is ``artifact_start`` (0-based row in post-warmup slice).
    assert float(r["transition_position"]) >= 7.0
