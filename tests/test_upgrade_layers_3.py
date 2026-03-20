from __future__ import annotations

import math
import tempfile

import numpy as np

from neraium_core.alignment import StructuralEngine
from neraium_core.causal_graph import causal_propagation_spread


def test_causal_propagation_spread_reachability_chain():
    """
    Causal proxy reachability:
    0 -> 1 -> 2 within 2 steps means source 0 can reach {1,2}.
    """
    C = np.array(
        [
            [0.0, 0.2, 0.0],
            [0.0, 0.0, 0.2],
            [0.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    out = causal_propagation_spread(C, threshold=0.1, max_steps=2, top_k=2)

    top_sources = out["top_sources"]
    assert top_sources and top_sources[0] == 0

    spread_scores = out["spread_scores"]
    assert len(spread_scores) == 3
    # n=3 => denom n-1=2, source0 reach_count=2 => 1.0; source1 reach_count=1 => 0.5
    assert math.isclose(float(spread_scores[0]), 1.0, rel_tol=1e-6)
    assert math.isclose(float(spread_scores[1]), 0.5, rel_tol=1e-6)


def test_engine_outputs_causal_propagation_and_scenario_projections():
    """
    Upgrade layers should add new observables into experimental_analytics:
    - causal_propagation
    - forecasting.scenario_projections
    """
    with tempfile.TemporaryDirectory() as d:
        engine = StructuralEngine(
            baseline_window=8,
            recent_window=4,
            regime_store_path=f"{d}/r.json",
        )

        last = None
        for t in range(80):
            base = math.sin(0.05 * t)
            frame = {
                "timestamp": str(t),
                "site_id": "site1",
                "asset_id": "asset1",
                "sensor_values": {"s1": base, "s2": 0.9 * base + 0.01 * math.sin(0.1 * t), "s3": 1.02 * base},
            }
            last = engine.process_frame(frame)

        assert last is not None
        analytics = last.get("experimental_analytics", {})
        assert "causal_propagation" in analytics

        forecasting = analytics.get("forecasting", {})
        assert "scenario_projections" in forecasting

