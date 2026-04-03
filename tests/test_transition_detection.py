"""Tests for neraium_core.detection (platform transition detection)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neraium_core.detection.transition_config import TransitionDetectionConfig
from neraium_core.detection.transition_detector import (
    detect_transition,
    detect_transition_from_signal,
    summarize_transition_detection,
)
from neraium_core.detection.transition_evaluation import summarize_entity_transitions


def _cycles_df(n: int, pressure: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"cycle": np.arange(1, n + 1, dtype=float), "transition_pressure": pressure})


def test_warmup_excludes_early_cycles() -> None:
    p = np.concatenate([np.linspace(0.9, 0.95, 40), np.linspace(0.2, 0.3, 60)])
    df = _cycles_df(len(p), p)
    cfg = TransitionDetectionConfig(warmup_index_floor=40.0, threshold=0.5, sustained_points=2, smoothing_window=3)
    r = detect_transition(df, cfg)
    assert r["detected"] is True
    assert float(r["transition_position"]) >= 40.0


def test_constant_signal_no_detection() -> None:
    df = _cycles_df(50, np.ones(50) * 0.3)
    r = detect_transition(df, TransitionDetectionConfig(warmup_index_floor=1.0))
    assert r["detected"] is False
    assert r["detection_reason"] in ("flat_or_invalid_signal", "no_detection")


def test_missing_values_interpolated() -> None:
    p = np.linspace(0.1, 0.9, 30).astype(float)
    p[10:15] = np.nan
    df = _cycles_df(len(p), p)
    r = detect_transition(df, TransitionDetectionConfig(warmup_index_floor=1.0, threshold=0.5, sustained_points=2))
    assert r["detected"] is True


def test_sustained_threshold_crossing() -> None:
    p = np.concatenate([np.linspace(0.1, 0.2, 20), np.linspace(0.2, 0.85, 15)])
    df = _cycles_df(len(p), p)
    cfg = TransitionDetectionConfig(
        warmup_index_floor=1.0,
        threshold=0.5,
        sustained_points=3,
        smoothing_window=1,
        fallback_mode="none",
    )
    r = detect_transition(df, cfg)
    assert r["detected"] is True
    assert r["detection_reason"] == "threshold_crossing"


def test_fallback_max_slope_when_no_crossing() -> None:
    p = np.linspace(0.1, 0.15, 50)
    df = _cycles_df(len(p), p)
    cfg = TransitionDetectionConfig(warmup_index_floor=1.0, threshold=0.99, sustained_points=5, fallback_mode="max_slope")
    r = detect_transition(df, cfg)
    assert r["detected"] is True
    assert r["detection_reason"] == "max_slope_increase"


def test_fallback_none() -> None:
    p = np.linspace(0.1, 0.15, 50)
    df = _cycles_df(len(p), p)
    cfg = TransitionDetectionConfig(warmup_index_floor=1.0, threshold=0.99, sustained_points=5, fallback_mode="none")
    r = detect_transition(df, cfg)
    assert r["detected"] is False


def test_warmup_fraction() -> None:
    p = np.concatenate([np.ones(30) * 0.9, np.linspace(0.2, 0.9, 40)])
    df = _cycles_df(len(p), p)
    cfg = TransitionDetectionConfig(
        warmup_fraction=0.35,
        threshold=0.5,
        sustained_points=2,
        min_history=2,
    )
    r = detect_transition(df, cfg)
    assert r["detected"] is True


def test_composite_signal_mode() -> None:
    n = 60
    cyc = np.arange(1, n + 1, dtype=float)
    a = np.concatenate([np.linspace(0.1, 0.2, 35), np.linspace(0.85, 0.95, 25)])
    b = np.concatenate([np.linspace(0.2, 0.25, 35), np.linspace(0.8, 0.9, 25)])
    df = pd.DataFrame(
        {
            "cycle": cyc,
            "transition_pressure": a,
            "structural_drift_score": b,
        }
    )
    cfg = TransitionDetectionConfig(
        signal_mode="composite",
        composite_weights={"transition_pressure": 0.5, "structural_drift_score": 0.5},
        warmup_index_floor=5.0,
        threshold=0.55,
        sustained_points=2,
    )
    r = detect_transition(df, cfg)
    assert r["detected"] is True
    assert "composite" in (r.get("signal_used") or "")


def test_grouped_multi_entity() -> None:
    rows = []
    for u in (1, 2):
        n = 40
        p = np.concatenate([np.linspace(0.1, 0.2, 25), np.linspace(0.7, 0.9, 15)])
        for i in range(n):
            rows.append({"unit": u, "cycle": float(i + 1), "transition_pressure": p[i]})
    df = pd.DataFrame(rows)
    cfg = TransitionDetectionConfig(warmup_index_floor=5.0, entity_group_column="unit")
    summary, agg = summarize_entity_transitions(df, "unit", config=cfg)
    assert len(summary) == 2
    assert agg["n_entities"] == 2
    assert agg["n_detected"] >= 1


def test_summarize_transition_detection_api() -> None:
    df = pd.DataFrame(
        {
            "asset": ["a", "a", "b", "b"],
            "cycle": [1.0, 2.0, 1.0, 2.0],
            "transition_pressure": [0.1, 0.9, 0.2, 0.95],
        }
    )
    cfg = TransitionDetectionConfig(index_column="cycle", warmup_index_floor=1.0, entity_group_column="asset")
    s, agg = summarize_transition_detection(df, cfg)
    assert len(s) == 2
    assert agg["n_entities"] == 2


def test_detect_transition_from_signal() -> None:
    sig = np.concatenate([np.linspace(0.1, 0.2, 15), np.linspace(0.8, 0.95, 15)])
    r = detect_transition_from_signal(sig, config=TransitionDetectionConfig(warmup_index_floor=0.0, threshold=0.5))
    assert r["detected"] is True


def test_alternate_primary_signal_priority() -> None:
    n = 50
    df = pd.DataFrame(
        {
            "t": np.arange(n, dtype=float),
            "foo": np.ones(n) * 0.1,
            "latest_instability": np.concatenate([np.ones(30) * 0.1, np.linspace(0.85, 0.95, 20)]),
        }
    )
    cfg = TransitionDetectionConfig(
        index_column="t",
        signal_priority=("transition_pressure", "latest_instability"),
        warmup_index_floor=0.0,
        threshold=0.5,
        sustained_points=2,
    )
    r = detect_transition(df, cfg)
    assert r["detected"] is True
    assert r["signal_used"] == "latest_instability"


@pytest.mark.parametrize("warmup_index_floor", [0.0, 1.0])
def test_insufficient_history(warmup_index_floor: float) -> None:
    df = pd.DataFrame({"cycle": [1.0, 2.0], "transition_pressure": [0.1, 0.2]})
    cfg = TransitionDetectionConfig(min_history=10, warmup_index_floor=warmup_index_floor)
    r = detect_transition(df, cfg)
    assert r["detected"] is False
    assert r["detection_reason"] == "insufficient_post_trim"
