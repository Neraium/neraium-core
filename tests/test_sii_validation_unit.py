from __future__ import annotations

import math
import os
import tempfile

import numpy as np

from neraium_core.alignment import StructuralEngine
from neraium_core.causal_graph import causal_propagation_spread


def _frame(t: int, sensors: dict[str, float | None]) -> dict:
    # StructuralEngine only looks at these keys.
    # Use stable site/asset ids so regime memory behaves deterministically.
    cleaned: dict[str, float | None] = dict(sensors)
    return {
        "timestamp": str(t),
        "site_id": "site-A",
        "asset_id": "asset-A",
        "sensor_values": cleaned,
    }


def test_node_index_bounds_for_causal_propagation_spread() -> None:
    rng = np.random.default_rng(123)
    n = 5
    C = rng.normal(0.0, 1.0, size=(n, n))
    # Ensure some directed edges are present.
    C = np.abs(C) * (rng.random((n, n)) > 0.5)
    np.fill_diagonal(C, 0.0)

    out = causal_propagation_spread(C, threshold=0.2, max_steps=2, top_k=4)
    top_sources = out["top_sources"]
    assert isinstance(top_sources, list)
    assert len(top_sources) <= 4
    for idx in top_sources:
        assert 0 <= int(idx) < n


def test_source_attribution_toy_3node_known_propagation() -> None:
    """
    Directed edges:
      0 -> 1, 0 -> 2  (source 0 can reach both nodes within 1 hop)
      2 -> 1           (source 2 can reach node 1 within 1 hop)
      1 -> *           (no outgoing edges)
    With max_steps=2 and top_k=2, expected top sources are [0, 2].
    """

    # causal_propagation_spread treats C[i,j] as edge i -> j.
    C = np.array(
        [
            [0.0, 0.2, 0.2],  # 0 -> {1,2}
            [0.0, 0.0, 0.0],  # 1 -> {}
            [0.0, 0.2, 0.0],  # 2 -> {1}
        ],
        dtype=float,
    )
    out = causal_propagation_spread(C, threshold=0.1, max_steps=2, top_k=2)
    assert out["top_sources"] == [0, 2]


def test_no_alerts_on_stable_nominal_sequence() -> None:
    with tempfile.TemporaryDirectory() as d:
        engine = StructuralEngine(
            baseline_window=40,
            recent_window=10,
            window_stride=1,
            regime_store_path=os.path.join(d, "regimes.json"),
        )

        # Perfectly proportional relationships (should keep correlation geometry unchanged).
        for t in range(140):
            base = math.sin(0.05 * t)
            sensors = {
                "s1": base,
                "s2": 0.985 * base,
                "s3": 1.02 * base,
            }
            out = engine.process_frame(_frame(t, sensors))
            assert out["state"] == "STABLE"


def test_alert_only_after_warmup() -> None:
    with tempfile.TemporaryDirectory() as d:
        engine = StructuralEngine(
            baseline_window=40,
            recent_window=10,
            window_stride=1,
            regime_store_path=os.path.join(d, "regimes.json"),
        )

        warmup_end = max(engine.baseline_window, engine.recent_window) - 1
        drift_start = 80
        rng = np.random.default_rng(999)

        first_alert: int | None = None
        for t in range(160):
            base = math.sin(0.05 * t)

            if t < drift_start:
                sensors = {
                    "s1": base,
                    "s2": 0.985 * base + 0.0005 * math.sin(0.11 * t),
                    "s3": 1.02 * base + 0.0005 * math.cos(0.07 * t),
                }
            else:
                # Break correlation geometry persistently: s3 becomes noise-dominated.
                sensors = {
                    "s1": base,
                    "s2": 0.985 * base + 0.0005 * math.sin(0.11 * t),
                    "s3": float(rng.normal(0.0, 1.2)),
                }

            out = engine.process_frame(_frame(t, sensors))
            if out["state"] in {"WATCH", "ALERT"} and first_alert is None:
                first_alert = t

            if t < warmup_end:
                assert out["state"] == "STABLE"

        assert first_alert is not None, "expected WATCH/ALERT after introducing drift"
        assert first_alert >= warmup_end


def test_robustness_with_noise_and_missing_data() -> None:
    with tempfile.TemporaryDirectory() as d:
        engine = StructuralEngine(
            baseline_window=40,
            recent_window=10,
            window_stride=1,
            regime_store_path=os.path.join(d, "regimes.json"),
        )

        rng = np.random.default_rng(2026)
        warmup_end = max(engine.baseline_window, engine.recent_window) - 1
        drift_start = 85
        missing_prob = 0.2

        pre_drift_alerts = 0
        post_drift_alert = False

        for t in range(175):
            base = math.sin(0.05 * t)

            missing_s2 = rng.random() < missing_prob
            missing_s3 = rng.random() < missing_prob

            if t < drift_start:
                s1 = base
                s2 = 0.985 * base + 0.02 * math.sin(0.11 * t) + float(rng.normal(0.0, 0.03))
                s3 = 1.02 * base + 0.02 * math.cos(0.07 * t) + float(rng.normal(0.0, 0.03))
            else:
                s1 = base
                s2 = 0.985 * base + 0.02 * math.sin(0.11 * t) + float(rng.normal(0.0, 0.03))
                s3 = float(rng.normal(0.0, 1.3))  # drift: uncorrelated noise

            sensors: dict[str, float | None] = {
                "s1": s1,
                "s2": None if missing_s2 else s2,
                "s3": None if missing_s3 else s3,
            }
            out = engine.process_frame(_frame(t, sensors))

            if t >= warmup_end and t < drift_start and out["state"] in {"WATCH", "ALERT"}:
                pre_drift_alerts += 1

            if t >= drift_start and out["state"] in {"WATCH", "ALERT"}:
                post_drift_alert = True

            # If causal propagation produced a top source list, indices must remain in-range.
            analytics = out.get("experimental_analytics") or {}
            causal_prop = analytics.get("causal_propagation") or {}
            top_sources = causal_prop.get("top_sources")
            if isinstance(top_sources, list) and engine.sensor_order:
                n = len(engine.sensor_order)
                for idx in top_sources:
                    assert 0 <= int(idx) < n

        assert pre_drift_alerts <= 2, f"too many pre-drift alerts: {pre_drift_alerts}"
        assert post_drift_alert, "expected WATCH/ALERT after drift under noise/missing data"

