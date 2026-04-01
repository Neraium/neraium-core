"""Unit tests for structural attribution helpers."""

import numpy as np

from neraium_core.structural_attribution import (
    compute_structural_attribution,
    infer_primary_structural_shift,
    rank_signal_pairs,
)


def test_rank_signal_pairs_sorted():
    rng = np.random.default_rng(0)
    names = ["a", "b", "c"]
    xb = rng.normal(size=(30, 3))
    xr = xb + 0.5 * rng.normal(size=(30, 3))
    pairs = rank_signal_pairs(sensor_order=names, baseline_matrix=xb, recent_matrix=xr)
    assert len(pairs) >= 1
    assert pairs[0]["relationship_shift_score"] >= pairs[-1]["relationship_shift_score"]


def test_compute_unavailable_without_baseline():
    out = compute_structural_attribution(
        sensor_order=["s1"],
        baseline=None,
        current_vector=np.array([1.0]),
        baseline_matrix=None,
        recent_matrix=None,
        drift_score=0.0,
        covariance_drift=0.0,
        transition_pressure=0.0,
        transition_state="NONE",
        experimental_analytics={},
        readiness={},
        frame_count=2,
        min_history_required=40,
        geometry={},
        state_space={},
        state_graph={},
    )
    assert out["available"] is False
    assert out["attribution_confidence"] == 0.0
    assert "reason" in out


def test_infer_primary_stable():
    p = infer_primary_structural_shift(
        drift_score=0.1,
        cov_drift=0.05,
        transition_pressure=0.1,
        transition_state="NONE",
        dominant_path="STABILIZING",
        top_driver_share=0.2,
        pair_shift_max=0.05,
    )
    assert p == "stable baseline telemetry"


def test_engine_has_attribution():
    from run_engine import StructuralEngine

    eng = StructuralEngine(baseline_window=10, recent_window=5, max_frames=100)
    for i in range(15):
        r = eng.process_frame(
            {
                "timestamp": str(i),
                "site_id": "s",
                "asset_id": "a",
                "sensor_values": {f"s{k}": float(k) + 0.01 * i for k in range(1, 8)},
            }
        )
    assert "attribution" in r
    assert "available" in r["attribution"]
    assert "top_signal_drivers" in r["attribution"]
