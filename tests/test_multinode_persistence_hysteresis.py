import math
import tempfile

import numpy as np

from neraium_core.alignment import StructuralEngine


def _frame(node: str, t: int, sensors: dict[str, float]) -> dict:
    return {
        "timestamp": str(t),
        "site_id": f"node-{node}",
        "asset_id": f"asset-{node}",
        "sensor_values": sensors,
    }


def _bounded_coherent_signals(t: int, correction_gain: float = 0.12) -> dict[str, float]:
    """
    Coherent, bounded, continuously corrected behavior.

    - Base oscillation shared across sensors (coherent structure)
    - Small bounded correction term that changes slowly (persistent correction activity)
    - Relationships remain bounded; no runaway degradation.
    """
    base = math.sin(0.08 * t) + 0.25 * math.sin(0.02 * t + 0.6)
    corr = correction_gain * math.sin(0.015 * t + 1.1)
    # Keep a stable linear relationship plus bounded correction activity.
    return {
        "s1": base + 0.02 * math.sin(0.11 * t),
        "s2": 0.9 * base + 0.1 * math.sin(0.07 * t + 0.2) + corr,
        "s3": 1.05 * base - 0.06 * math.sin(0.05 * t + 0.9) - corr * 0.6,
    }


def _true_instability_signals(t: int, rng: np.random.Generator) -> dict[str, float]:
    """
    True structural breakdown: after t>160, one sensor loses coordination (noise-dominated),
    breaking correlation geometry persistently.
    """
    base = math.sin(0.08 * t) + 0.25 * math.sin(0.02 * t + 0.6)
    if t <= 160:
        return {
            "s1": base + 0.02 * math.sin(0.11 * t),
            "s2": 0.95 * base + 0.04 * math.sin(0.07 * t + 0.2),
            "s3": 1.05 * base - 0.06 * math.sin(0.05 * t + 0.9),
        }
    # Persistent breakdown: s3 becomes largely uncorrelated noise + small residual.
    return {
        "s1": base + 0.02 * math.sin(0.11 * t),
        "s2": 0.95 * base + 0.04 * math.sin(0.07 * t + 0.2),
        "s3": float(rng.normal(0.0, 1.2)) + 0.1 * base,
    }


def test_bounded_multinode_correction_does_not_collapse_to_instability():
    with tempfile.TemporaryDirectory() as d:
        engines = {
            n: StructuralEngine(
                baseline_window=50,
                recent_window=12,
                window_stride=1,
                regime_store_path=f"{d}/regimes_{n}.json",
            )
            for n in ["A", "B", "C", "D"]
        }

        interpreted_tail = {n: [] for n in engines}
        state_tail = {n: [] for n in engines}

        for t in range(240):
            for node, engine in engines.items():
                sensors = _bounded_coherent_signals(t, correction_gain=0.12 + 0.02 * (ord(node) - ord("A")))
                out = engine.process_frame(_frame(node, t, sensors))
                if t > 120:  # after warmup + some persistence
                    interpreted_tail[node].append(out.get("interpreted_state"))
                    state_tail[node].append(out.get("state"))

        for node in engines:
            # Should not default into persistent structural instability for coherent bounded correction.
            interpreted = interpreted_tail[node]
            assert interpreted, "expected interpreted_state after warmup"
            instability_rate = sum(1 for s in interpreted if s == "STRUCTURAL_INSTABILITY_OBSERVED") / len(interpreted)
            assert instability_rate < 0.35

            states = state_tail[node]
            alert_rate = sum(1 for s in states if s == "ALERT") / len(states)
            assert alert_rate < 0.15


def test_true_breakdown_escalates_with_persistence():
    rng = np.random.default_rng(12345)
    with tempfile.TemporaryDirectory() as d:
        engine = StructuralEngine(
            baseline_window=50,
            recent_window=12,
            window_stride=1,
            regime_store_path=f"{d}/regimes_instability.json",
        )

        tail = []
        for t in range(260):
            sensors = _true_instability_signals(t, rng=rng)
            out = engine.process_frame(_frame("A", t, sensors))
            if t > 200:
                tail.append(out.get("interpreted_state"))

        # In a persistent breakdown segment, instability should appear.
        assert tail
        assert "STRUCTURAL_INSTABILITY_OBSERVED" in tail or "COUPLING_INSTABILITY_OBSERVED" in tail

