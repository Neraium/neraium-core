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


def test_engine_outputs_uncertainty_and_root_cause_chains(monkeypatch):
    """
    The upgrades should add:
    - uncertainty block (how sure + what limited evidence)
    - causal_root_cause_chains + root_cause_narrative (human-readable chain)
    """
    monkeypatch.setenv("NERAIUM_AUTONOMOUS_RESPONSE", "0")
    monkeypatch.setenv("NERAIUM_CAUSAL_ROOT_CAUSE_CHAINS", "1")
    monkeypatch.setenv("NERAIUM_CAUSAL_INTELLIGENCE", "1")
    monkeypatch.setenv("NERAIUM_TEMPORAL_SCENARIOS", "1")

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
                "sensor_values": {
                    "s1": base,
                    "s2": 0.9 * base + 0.01 * math.sin(0.1 * t),
                    "s3": 1.02 * base,
                },
            }
            last = engine.process_frame(frame)

        assert last is not None
        assert "uncertainty" in last
        u = last["uncertainty"]
        assert "confidence_score" in u
        assert "evidence_confidence" in u
        assert "gate_passed" in u
        assert "data_quality_summary" in u

        assert "causal_root_cause_chains" in last
        assert isinstance(last["causal_root_cause_chains"], list)
        assert "root_cause_narrative" in last
        assert isinstance(last["root_cause_narrative"], str)


def test_engine_outputs_ranked_response_recommendations(monkeypatch):
    """
    When NERAIUM_AUTONOMOUS_RESPONSE=1, decision layer should emit ranked
    recommendations with risk/cost/time tiers.
    """
    monkeypatch.setenv("NERAIUM_AUTONOMOUS_RESPONSE", "1")
    monkeypatch.setenv("NERAIUM_CAUSAL_ROOT_CAUSE_CHAINS", "1")
    monkeypatch.setenv("NERAIUM_CAUSAL_INTELLIGENCE", "1")
    monkeypatch.setenv("NERAIUM_TEMPORAL_SCENARIOS", "1")

    with tempfile.TemporaryDirectory() as d:
        engine = StructuralEngine(
            baseline_window=8,
            recent_window=4,
            regime_store_path=f"{d}/r.json",
        )

        last = None
        for t in range(80):
            base = math.sin(0.07 * t)
            frame = {
                "timestamp": str(t),
                "site_id": "site1",
                "asset_id": "asset1",
                "sensor_values": {
                    "s1": base,
                    "s2": 0.9 * base + 0.02 * math.sin(0.2 * t),
                    "s3": 1.05 * base,
                },
            }
            last = engine.process_frame(frame)

        assert last is not None
        assert last.get("autonomous_response_enabled") is True
        recs = last.get("response_recommendations")
        assert isinstance(recs, list) and recs
        r0 = recs[0]
        for k in ("rank", "risk", "cost_tier", "time_impact_tier", "action_type", "rationale"):
            assert k in r0

