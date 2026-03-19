#!/usr/bin/env python3
"""
Upgraded 4-node A/B persistence test using StructuralEngine.

Nodes:
  A: nominal baseline
  B: sustained load asymmetry after t=60
  C: coupling degradation / noisy interaction breakdown after t=60
  D: missing-data dropout and rejoin from t=60 to t=80

Temporal conditions:
  coherent_time:   regular timesteps 0, 1, 2, ...
  disturbed_time: irregular/jittered timestamps (optionally anchored to GAL-2 time API)

GAL-2 time API (optional): set GAL2_API_KEY and optionally GAL2_TIME_URL in the
environment. When set, disturbed_time timestamps are anchored to GAL-2 time.
Never commit API keys; use env vars or a .env file.
"""
from __future__ import annotations

import json
import math
import os
import tempfile
import time
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from neraium_core.alignment import StructuralEngine


# -----------------------------------------------------------------------------
# GAL-2 time API (credentials from environment only)
# -----------------------------------------------------------------------------
def _get_gal2_time() -> float | None:
    """
    Return current time from GAL-2 API. Uses GAL2_API_KEY and GAL2_TIME_URL
    from the environment. Returns None if key is unset or request fails.
    """
    api_key = os.getenv("GAL2_API_KEY")
    url = os.getenv("GAL2_TIME_URL", "https://api-v2.gal-2.com/time")
    if not api_key:
        return None
    try:
        req = urllib.request.Request(url, headers={"x-api-key": api_key})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            # API may return {"time": <sec>} or {"gal2_time": <sec>}
            t = data.get("time") or data.get("gal2_time")
            return float(t) if t is not None else None
    except Exception:
        return None


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
NODES = ["A", "B", "C", "D"]
CONDITIONS = ["coherent_time", "disturbed_time"]
T_START = 0
T_END = 140
T_LOAD_ASYMMETRY = 60
T_DROPOUT_START = 60
T_DROPOUT_END = 80
BASELINE_WINDOW = 50
RECENT_WINDOW = 12
SEED = 42
OUTPUT_JSON = "upgraded_multinode_test_results.json"
RNG = np.random.default_rng(SEED)


# -----------------------------------------------------------------------------
# Signal generators (sensor_values dict per node at time t)
# -----------------------------------------------------------------------------
def sensors_node_a(t: int) -> dict[str, float]:
    """Nominal baseline: coherent, stable correlation structure."""
    base = math.sin(0.08 * t) + 0.25 * math.sin(0.02 * t + 0.6)
    return {
        "s1": base + 0.02 * math.sin(0.11 * t),
        "s2": 0.9 * base + 0.1 * math.sin(0.07 * t + 0.2),
        "s3": 1.05 * base - 0.06 * math.sin(0.05 * t + 0.9),
    }


def sensors_node_b(t: int) -> dict[str, float]:
    """Sustained load asymmetry after t=60: one channel scaled and offset."""
    base = math.sin(0.08 * t) + 0.25 * math.sin(0.02 * t + 0.6)
    s1 = base + 0.02 * math.sin(0.11 * t)
    s2 = 0.9 * base + 0.1 * math.sin(0.07 * t + 0.2)
    s3 = 1.05 * base - 0.06 * math.sin(0.05 * t + 0.9)
    if t >= T_LOAD_ASYMMETRY:
        # Persistent load asymmetry: s2 drifts in scale/offset
        s2 = 1.4 * s2 + 0.5
    return {"s1": s1, "s2": s2, "s3": s3}


def sensors_node_c(t: int) -> dict[str, float]:
    """Coupling degradation after t=60: one sensor becomes noisy (interaction breakdown)."""
    base = math.sin(0.08 * t) + 0.25 * math.sin(0.02 * t + 0.6)
    if t < T_LOAD_ASYMMETRY:
        return {
            "s1": base + 0.02 * math.sin(0.11 * t),
            "s2": 0.95 * base + 0.04 * math.sin(0.07 * t + 0.2),
            "s3": 1.05 * base - 0.06 * math.sin(0.05 * t + 0.9),
        }
    return {
        "s1": base + 0.02 * math.sin(0.11 * t),
        "s2": 0.95 * base + 0.04 * math.sin(0.07 * t + 0.2),
        "s3": float(RNG.normal(0.0, 1.2)) + 0.1 * base,
    }


def sensors_node_d(t: int) -> dict[str, float]:
    """Missing-data dropout t=60..80, then rejoin."""
    out = sensors_node_a(t)
    if T_DROPOUT_START <= t < T_DROPOUT_END:
        out["s2"] = float("nan")  # type: ignore[assignment]
    return out


NODE_SENSOR_FN = {
    "A": sensors_node_a,
    "B": sensors_node_b,
    "C": sensors_node_c,
    "D": sensors_node_d,
}


# -----------------------------------------------------------------------------
# Temporal conditions: (condition_name, n_steps) -> list of timestamps (numeric or str)
# -----------------------------------------------------------------------------
def coherent_timestamps(n_steps: int) -> list[str]:
    """Regular timesteps 0, 1, ..., n_steps-1."""
    return [str(t) for t in range(n_steps)]


def disturbed_timestamps(n_steps: int) -> tuple[list[str], bool]:
    """
    Irregular/jittered timestamps. When GAL2_API_KEY is set, anchor to GAL-2 time.
    Returns (timestamps, gal2_used).
    """
    gal2_base = _get_gal2_time()
    if gal2_base is not None:
        # Anchor to GAL-2 time: base + step index + jitter
        jitter = RNG.uniform(-0.3, 0.3, size=n_steps)
        gaps = RNG.choice(n_steps, size=min(n_steps // 10, 15), replace=False)
        for i in gaps:
            jitter[i] += RNG.uniform(0.5, 1.5)
        ts = np.array([gal2_base + i + jitter[i] for i in range(n_steps)])
        ts = np.maximum(ts, 0)
        ts = np.sort(ts)
        return [str(round(t, 4)) for t in ts], True

    base = np.linspace(0, n_steps - 1, n_steps)
    jitter = RNG.uniform(-0.3, 0.3, size=n_steps)
    gaps = RNG.choice(n_steps, size=min(n_steps // 10, 15), replace=False)
    for i in gaps:
        jitter[i] += RNG.uniform(0.5, 1.5)
    ts = base + jitter
    ts = np.maximum(ts, 0)
    ts = np.sort(ts)
    return [str(round(t, 4)) for t in ts], False


def timestamps_for_condition(condition: str, n_steps: int) -> tuple[list[str], bool]:
    """Return (timestamps, gal2_used). gal2_used is True only when disturbed_time used GAL-2 API."""
    if condition == "coherent_time":
        return coherent_timestamps(n_steps), False
    if condition == "disturbed_time":
        return disturbed_timestamps(n_steps)
    raise ValueError(f"Unknown condition: {condition}")


# -----------------------------------------------------------------------------
# Frame building and run
# -----------------------------------------------------------------------------
def build_frame(node: str, step_index: int, t_value: int, ts: str, sensors: dict[str, float]) -> dict:
    return {
        "timestamp": ts,
        "site_id": f"node-{node}",
        "asset_id": f"asset-{node}",
        "sensor_values": sensors,
    }


def run_node_condition(
    node: str,
    condition: str,
    tmp_dir: str,
) -> tuple[list[dict[str, Any]], bool]:
    """Run one (node, condition) through StructuralEngine. Returns (results, gal2_used)."""
    n_steps = T_END - T_START
    timestamps, gal2_used = timestamps_for_condition(condition, n_steps)
    sensor_fn = NODE_SENSOR_FN[node]
    regime_path = os.path.join(tmp_dir, f"regime_{node}_{condition}.json")

    engine = StructuralEngine(
        baseline_window=BASELINE_WINDOW,
        recent_window=RECENT_WINDOW,
        regime_store_path=regime_path,
    )
    results = []
    for i in range(n_steps):
        t = T_START + i
        ts = timestamps[i] if i < len(timestamps) else str(t)
        sensors = sensor_fn(t)
        frame = build_frame(node, i, t, ts, sensors)
        out = engine.process_frame(frame)
        results.append(out)
    return results, gal2_used


def summarize_run(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate one run's process_frame outputs into summary metrics."""
    if not results:
        return {}

    states = [r.get("state") for r in results if r.get("state") is not None]
    interpreted = [r.get("interpreted_state") for r in results if r.get("interpreted_state") is not None]
    scores = []
    confidences = []
    class_stability = []
    missing = []
    active = []
    dominant_drivers = []
    dq_gate_passed = []
    dq_valid = []

    for r in results:
        s = r.get("latest_instability")
        if s is not None:
            scores.append(float(s))
        c = r.get("confidence_score")
        if c is not None:
            confidences.append(float(c))
        cs = r.get("classification_stability")
        if cs is not None:
            class_stability.append(float(cs))
        mc = r.get("missing_sensor_count")
        if mc is not None:
            missing.append(int(mc))
        ac = r.get("active_sensor_count")
        if ac is not None:
            active.append(int(ac))
        dd = r.get("dominant_driver")
        if dd is not None:
            dominant_drivers.append(dd)
        dq = r.get("data_quality_summary") or {}
        if "gate_passed" in dq:
            dq_gate_passed.append(bool(dq["gate_passed"]))
        if "valid_signal_count" in dq:
            dq_valid.append(int(dq["valid_signal_count"]))

    # Volatility: std of score over run (or 0 if single/few points)
    volatility = float(np.std(scores)) if len(scores) > 1 else 0.0

    # Causal attribution summary: from last frame with content
    causal_summary: dict[str, Any] = {}
    for r in reversed(results):
        attr = r.get("causal_attribution")
        if attr and isinstance(attr, dict) and attr.get("top_drivers"):
            causal_summary = {
                "top_drivers": attr.get("top_drivers", []),
                "driver_count": len(attr.get("driver_scores") or {}),
            }
            break

    # Data quality summary (aggregate)
    dq_summary: dict[str, Any] = {}
    if dq_gate_passed:
        dq_summary["gate_passed_rate"] = sum(dq_gate_passed) / len(dq_gate_passed)
    if dq_valid:
        dq_summary["mean_valid_signal_count"] = float(np.mean(dq_valid))
        dq_summary["min_valid_signal_count"] = int(min(dq_valid))
        dq_summary["max_valid_signal_count"] = int(max(dq_valid))

    return {
        "state_counts": dict(Counter(states)),
        "interpreted_state_counts": dict(Counter(interpreted)),
        "mean_score": float(np.mean(scores)) if scores else None,
        "max_score": float(max(scores)) if scores else None,
        "min_score": float(min(scores)) if scores else None,
        "final_state": states[-1] if states else None,
        "final_interpreted_state": interpreted[-1] if interpreted else None,
        "mean_confidence": float(np.mean(confidences)) if confidences else None,
        "confidence_std": float(np.std(confidences)) if len(confidences) > 1 else None,
        "classification_stability_mean": float(np.mean(class_stability)) if class_stability else None,
        "classification_stability_final": class_stability[-1] if class_stability else None,
        "volatility": volatility,
        "missing_sensor_count_mean": float(np.mean(missing)) if missing else None,
        "missing_sensor_count_max": int(max(missing)) if missing else None,
        "active_sensor_count_mean": float(np.mean(active)) if active else None,
        "active_sensor_count_min": int(min(active)) if active else None,
        "dominant_driver_final": dominant_drivers[-1] if dominant_drivers else None,
        "dominant_driver_sample": list(dict.fromkeys(dominant_drivers[-10:])) if dominant_drivers else [],
        "causal_attribution_summary": causal_summary,
        "data_quality_summary": dq_summary,
        "n_frames": len(results),
    }


def make_serializable(obj: Any) -> Any:
    """Recurse to replace non-JSON-serializable values."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_serializable(x) for x in obj]
    return str(obj)


def print_console_summary(payload: dict[str, Any]) -> None:
    """Print concise comparison: coherent_time vs disturbed_time per node."""
    print("\n" + "=" * 60)
    print("Upgraded multinode test — coherent_time vs disturbed_time")
    print("=" * 60)
    gal2 = payload.get("gal2_api_configured", False)
    gal2_used = payload.get("gal2_used_for_disturbed_time", False)
    print(f"GAL-2 API configured: {gal2}  |  disturbed_time used GAL-2: {gal2_used}")
    if gal2:
        print(f"GAL-2 URL: {payload.get('gal2_time_url', 'N/A')}")

    for node in NODES:
        print(f"\n--- Node {node} ---")
        coh = payload.get("runs", {}).get(node, {}).get("coherent_time", {})
        dist = payload.get("runs", {}).get(node, {}).get("disturbed_time", {})

        def row(label: str, c_val: Any, d_val: Any) -> None:
            print(f"  {label}: coherent={c_val}  disturbed={d_val}")

        row("final_interpreted_state", coh.get("final_interpreted_state"), dist.get("final_interpreted_state"))
        row("final_state", coh.get("final_state"), dist.get("final_state"))
        row("mean_score", round(coh.get("mean_score"), 4) if coh.get("mean_score") is not None else "-",
            round(dist.get("mean_score"), 4) if dist.get("mean_score") is not None else "-")
        row("max_score", round(coh.get("max_score"), 4) if coh.get("max_score") is not None else "-",
            round(dist.get("max_score"), 4) if dist.get("max_score") is not None else "-")
        row("mean_confidence", round(coh.get("mean_confidence"), 4) if coh.get("mean_confidence") is not None else "-",
            round(dist.get("mean_confidence"), 4) if dist.get("mean_confidence") is not None else "-")
        row("volatility", round(coh.get("volatility"), 4) if coh.get("volatility") is not None else "-",
            round(dist.get("volatility"), 4) if dist.get("volatility") is not None else "-")
        row("active_sensor_count_mean", coh.get("active_sensor_count_mean"), dist.get("active_sensor_count_mean"))
        row("missing_sensor_count_mean", coh.get("missing_sensor_count_mean"), dist.get("missing_sensor_count_mean"))
        row("dominant_driver_final", coh.get("dominant_driver_final"), dist.get("dominant_driver_final"))

    print("\n" + "=" * 60)
    print(f"Results written to: {payload.get('output_path', OUTPUT_JSON)}")
    print("=" * 60 + "\n")


def main() -> int:
    output_path = Path(OUTPUT_JSON)
    gal2_configured = bool(os.getenv("GAL2_API_KEY"))
    gal2_url = os.getenv("GAL2_TIME_URL", "https://api-v2.gal-2.com/time")
    gal2_used_any = False

    with tempfile.TemporaryDirectory(prefix="neraium_upgraded_multinode_") as tmp_dir:
        runs: dict[str, dict[str, dict[str, Any]]] = {node: {} for node in NODES}

        for node in NODES:
            for condition in CONDITIONS:
                results, gal2_used = run_node_condition(node, condition, tmp_dir)
                if condition == "disturbed_time" and gal2_used:
                    gal2_used_any = True
                summary = summarize_run(results)
                runs[node][condition] = summary

        payload = {
            "description": "4-node A/B persistence test with coherent_time vs disturbed_time",
            "nodes": NODES,
            "conditions": CONDITIONS,
            "t_start": T_START,
            "t_end": T_END,
            "baseline_window": BASELINE_WINDOW,
            "recent_window": RECENT_WINDOW,
            "seed": SEED,
            "gal2_api_configured": gal2_configured,
            "gal2_time_url": gal2_url if gal2_configured else None,
            "gal2_used_for_disturbed_time": gal2_used_any,
            "runs": runs,
            "output_path": str(output_path),
        }

    out_obj = make_serializable(payload)
    output_path.write_text(json.dumps(out_obj, indent=2), encoding="utf-8")
    print_console_summary(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
