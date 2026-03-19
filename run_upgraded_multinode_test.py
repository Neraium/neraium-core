#!/usr/bin/env python3
"""
Credible 4-node Systemic Infrastructure Intelligence (SII) structural perturbation benchmark.

Purpose:
  - Validate that StructuralEngine emits operator-safe deviations primarily for the target node (C).
  - Produce measurable alert precision/recall and baseline stability during controlled perturbations.
  - Preserve GAL-2 time integration and the 4-node coherent vs disturbed timing design.

Ground truth used by metrics:
  - Target/anomaly node: C
  - Anomaly window: perturbation phase only (t in [PERT_START, PERT_END])
  - Nodes A/B/D are negatives (any alerts are false positives), with D expected to show weak spillover.

Output artifacts:
  - JSON results file (reproducible summary + metrics)
  - CSV timeseries and CSV node/metrics summaries
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
try:
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover
    # Colab typically includes pandas, but local environments may not.
    # Install at runtime so the benchmark stays turnkey.
    import subprocess
    import sys

    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pandas"])
    import pandas as pd

from neraium_core.alignment import StructuralEngine


# -----------------------------------------------------------------------------
# GAL-2 time API (credentials from environment only)
# -----------------------------------------------------------------------------
def _get_gal2_time() -> float | None:
    """
    Return current time from GAL-2 API.

    Uses:
      - GAL2_API_KEY (required)
      - GAL2_TIME_URL (optional; default is https://api-v2.gal-2.com/time)

    Returns None if key is unset or request fails.
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
# Allowed enums (must remain exactly these values)
# -----------------------------------------------------------------------------
INTERPRETED_STATE_ALLOWED = {
    "NOMINAL_STRUCTURE",
    "REGIME_SHIFT_OBSERVED",
    "COUPLING_INSTABILITY_OBSERVED",
    "STRUCTURAL_INSTABILITY_OBSERVED",
    "COHERENCE_UNDER_CONSTRAINT",
}

STATE_ALLOWED = {"STABLE", "WATCH", "ALERT"}

CONFIDENCE_ALLOWED = {"low", "medium", "high"}


# -----------------------------------------------------------------------------
# Benchmark configuration
# -----------------------------------------------------------------------------
NODES = ["A", "B", "C", "D"]
CONDITIONS = ["coherent_time", "disturbed_time"]

T_START = 0
T_END = 240  # 0..239
N_STEPS = T_END - T_START

# 3-phase benchmark structure
BASELINE_END = 79
PERT_START = 80
PERT_END = 159
RECOVERY_START = 160

# StructuralEngine windows
BASELINE_WINDOW = 50
RECENT_WINDOW = 12

# Reproducibility
SEED = 42
RNG = np.random.default_rng(SEED)

# Output artifacts (keep JSON output name stable for Colab workflows)
OUTPUT_JSON = "upgraded_multinode_test_results.json"
OUTPUT_TIMESERIES_CSV = "upgraded_multinode_test_timeseries.csv"
OUTPUT_NODE_SUMMARY_CSV = "upgraded_multinode_test_node_summary.csv"
OUTPUT_METRICS_CSV = "upgraded_multinode_test_metrics.csv"


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def parse_timestamp_float(ts: str) -> float:
    try:
        return float(ts)
    except (TypeError, ValueError):
        return 0.0


def phase_for_step(step_i: int) -> str:
    if step_i <= BASELINE_END:
        return "baseline"
    if PERT_START <= step_i <= PERT_END:
        return "perturbation"
    if step_i >= RECOVERY_START:
        return "recovery"
    return "unknown"


def coherent_timestamps(n_steps: int) -> Tuple[List[str], bool]:
    """Regular timesteps 0, 1, 2, ... (gal2_used False)."""
    return [str(float(t)) for t in range(n_steps)], False


def disturbed_timestamps(n_steps: int) -> Tuple[List[str], bool]:
    """
    Irregular/jittered timestamps.

    When GAL2_API_KEY is set, anchor to GAL-2 time so the perturbed timeline
    changes in absolute time but remains reproducible for this run.
    """

    gal2_base = _get_gal2_time()
    if gal2_base is not None:
        # Anchor to GAL-2 time: base + step index + jitter
        jitter = RNG.uniform(-0.55, 0.55, size=n_steps)
        # Add larger gaps at sparse indices.
        gaps = RNG.choice(n_steps, size=min(max(10, n_steps // 12), 25), replace=False)
        for i in gaps:
            jitter[i] += RNG.uniform(0.75, 1.75)

        ts = np.array([gal2_base + i + jitter[i] for i in range(n_steps)], dtype=float)
        ts = np.maximum(ts, 0.0)
        ts = np.sort(ts)
        return [str(round(float(t), 4)) for t in ts], True

    # No GAL-2: use deterministic irregular sampling around integer index.
    base = np.linspace(0, n_steps - 1, n_steps, dtype=float)
    jitter = RNG.uniform(-0.55, 0.55, size=n_steps)
    gaps = RNG.choice(n_steps, size=min(max(10, n_steps // 12), 25), replace=False)
    for i in gaps:
        jitter[i] += RNG.uniform(0.75, 1.75)

    ts = base + jitter
    ts = np.maximum(ts, 0.0)
    ts = np.sort(ts)
    return [str(round(float(t), 4)) for t in ts], False


def timestamps_for_condition(condition: str, n_steps: int) -> Tuple[List[str], bool]:
    if condition == "coherent_time":
        return coherent_timestamps(n_steps)
    if condition == "disturbed_time":
        return disturbed_timestamps(n_steps)
    raise ValueError(f"Unknown condition: {condition}")


# -----------------------------------------------------------------------------
# Synthetic telemetry model (calibrated for node specificity)
# -----------------------------------------------------------------------------
#
# Design principle:
#   - Nominal nodes are generated from a shared latent factor Z(t) using exact
#     linear sensor mixing. This makes correlation geometry stable (low drift).
#   - Target node C breaks that coupling only during perturbation to ensure
#     high recall without degrading precision.
#   - Node D has weak spillover during perturbation (small decoupling + mild
#     missingness), and returns to nominal during recovery.
#

# Latent and anomaly parameters (tuned to align with StructuralEngine thresholds)
LATENT_AMPLITUDE = 0.20
LATENT_W1 = 2 * math.pi / 64.0
LATENT_W2 = 2 * math.pi / 23.0
LATENT_PHI2 = 0.6

# Nominal linear mixing coefficients
A1, A2, A3 = 1.00, 0.90, 1.05
NOMINAL_OFFSET2 = 0.00
NOMINAL_OFFSET3 = 0.00

# Node B perturbation: correlation-preserving asymmetry (scaling + offset on one sensor)
B_S2_SCALE = 1.07
B_S2_OFFSET = 0.03

# Node C perturbation: strong coupling breakdown
#   - s1 and s2 remain proportional to latent
#   - s3 becomes a near-independent time-varying signal with small latent residual
C_S3_RESIDUAL = 0.00
C_NOISE_AMPLITUDE = 1.20
C_S2_NOISE_AMPLITUDE = 0.22

# Node D spillover: weak decoupling + mild missingness
D_S3_NOISE_AMPLITUDE = 0.05

# Mild missingness probability during perturbation for D
D_MISSING_PROB = 0.015

# Keep missingness deterministic per-step via seed+step index.
def missing_on_step_d(step_i: int) -> bool:
    # Deterministic pseudo-random event without keeping RNG state changes across conditions.
    # Uses a simple hash with SEED and step index.
    x = (SEED * 2654435761 + step_i * 97531) % 10_000
    p = x / 10_000.0
    return p < D_MISSING_PROB


def latent_z(t: float) -> float:
    return math.sin(LATENT_W1 * t) + 0.5 * math.sin(LATENT_W2 * t + LATENT_PHI2)


def sensors_node_nominal(t: float) -> Dict[str, float]:
    z = LATENT_AMPLITUDE * latent_z(t)
    s1 = A1 * z
    s2 = A2 * z + NOMINAL_OFFSET2
    s3 = A3 * z + NOMINAL_OFFSET3
    return {"s1": float(s1), "s2": float(s2), "s3": float(s3)}


def sensors_node_a(step_i: int, t_real: float, rng: np.random.Generator) -> Dict[str, float]:
    # Always nominal
    return sensors_node_nominal(t_real)


def sensors_node_b(step_i: int, t_real: float, rng: np.random.Generator) -> Dict[str, float]:
    # Correlation-preserving asymmetry only during perturbation.
    base = sensors_node_nominal(t_real)
    if PERT_START <= step_i <= PERT_END:
        z = LATENT_AMPLITUDE * latent_z(t_real)
        s1 = A1 * z
        s2 = B_S2_SCALE * (A2 * z) + B_S2_OFFSET
        s3 = A3 * z
        return {"s1": float(s1), "s2": float(s2), "s3": float(s3)}
    return base


def sensors_node_c(step_i: int, t_real: float, rng: np.random.Generator) -> Dict[str, float]:
    # Strong coupling breakdown on s3 (and a small perturbation on s2) during perturbation.
    # Use truly random, seeded noise so correlation geometry breaks consistently
    # across windowing and does not "accidentally" align.
    if PERT_START <= step_i <= PERT_END:
        z = LATENT_AMPLITUDE * latent_z(t_real)
        s1 = A1 * z
        # Add random perturbation components. Correlation after z-score
        # normalization depends on waveform shape; randomness is the most
        # reliable way to force near-zero coupling consistently.
        noise_s2 = float(rng.normal(0.0, 1.0))
        noise_s3 = float(rng.normal(0.0, 1.0))
        s2 = A2 * z + C_S2_NOISE_AMPLITUDE * noise_s2
        residual = C_S3_RESIDUAL * (A3 * z)
        s3 = residual + C_NOISE_AMPLITUDE * noise_s3
        return {"s1": float(s1), "s2": float(s2), "s3": float(s3)}
    return sensors_node_nominal(t_real)


def sensors_node_d(step_i: int, t_real: float, rng: np.random.Generator) -> Dict[str, float]:
    # Weak spillover during perturbation: mild decoupling + occasional NaNs.
    if PERT_START <= step_i <= PERT_END:
        z = LATENT_AMPLITUDE * latent_z(t_real)
        s1 = A1 * z
        # Occasionally drop s2 (mild data quality degradation).
        if missing_on_step_d(step_i):
            s2 = float("nan")
        else:
            s2 = A2 * z
        # s3 stays mostly coupled, with small noise.
        s3 = A3 * z + D_S3_NOISE_AMPLITUDE * float(rng.normal(0.0, 1.0))
        return {"s1": float(s1), "s2": float(s2), "s3": float(s3)}
    return sensors_node_nominal(t_real)


NODE_SENSOR_FN: Dict[str, Callable[[int, float, np.random.Generator], Dict[str, float]]] = {
    "A": sensors_node_a,
    "B": sensors_node_b,
    "C": sensors_node_c,
    "D": sensors_node_d,
}


# -----------------------------------------------------------------------------
# Frame building and validation
# -----------------------------------------------------------------------------
def build_frame(node: str, step_i: int, ts: str, sensors: Dict[str, float]) -> Dict[str, Any]:
    return {
        "timestamp": ts,
        "site_id": f"node-{node}",
        "asset_id": f"asset-{node}",
        "sensor_values": sensors,
    }


def validate_engine_output(out: Dict[str, Any], *, context: str) -> None:
    # During StructuralEngine warmup the payload may omit `interpreted_state` and
    # `confidence`. For benchmark reliability we default them to the most
    # conservative nominal assumptions.
    state = str(out.get("state", "STABLE"))
    interpreted = str(out.get("interpreted_state", "NOMINAL_STRUCTURE"))
    conf = str(out.get("confidence", "low"))

    if state not in STATE_ALLOWED:
        raise AssertionError(f"Invalid `state` enum: {state!r} ({context})")
    if interpreted not in INTERPRETED_STATE_ALLOWED:
        raise AssertionError(f"Invalid `interpreted_state` enum: {interpreted!r} ({context})")
    if conf not in CONFIDENCE_ALLOWED:
        raise AssertionError(f"Invalid `confidence` enum: {conf!r} ({context})")


def alert_indicator_from_out(out: Dict[str, Any]) -> bool:
    return str(out.get("state")) in {"WATCH", "ALERT"}


def interpreted_deviation_indicator(out: Dict[str, Any]) -> bool:
    return str(out.get("interpreted_state")) != "NOMINAL_STRUCTURE"


# -----------------------------------------------------------------------------
# Run a node/condition through StructuralEngine
# -----------------------------------------------------------------------------
def run_node_condition(
    node: str,
    condition: str,
    tmp_dir: str,
    timestamps: List[str],
) -> List[Dict[str, Any]]:
    """Run one (node, condition) through StructuralEngine. Returns per-frame outputs."""

    cond_offset = 0 if condition == "coherent_time" else 10_000
    node_offset = ord(node) - ord("A")
    noise_seed = SEED + cond_offset + 1_000 * node_offset
    noise_rng = np.random.default_rng(noise_seed)

    regime_path = os.path.join(tmp_dir, f"regime_{node}_{condition}.json")
    engine = StructuralEngine(
        baseline_window=BASELINE_WINDOW,
        recent_window=RECENT_WINDOW,
        regime_store_path=regime_path,
    )

    out_frames: List[Dict[str, Any]] = []
    for step_i in range(N_STEPS):
        ts = timestamps[step_i]
        t_real = parse_timestamp_float(ts)

        sensors = NODE_SENSOR_FN[node](step_i, t_real, noise_rng)
        frame = build_frame(node, step_i, ts, sensors)
        out = engine.process_frame(frame)

        validate_engine_output(out, context=f"node={node} condition={condition} step={step_i}")

        # Apply the same warmup defaults used by `validate_engine_output` so
        # metrics can be computed uniformly across all steps.
        out_state = str(out.get("state", "STABLE"))
        out_interpreted_state = str(out.get("interpreted_state", "NOMINAL_STRUCTURE"))
        out_confidence = str(out.get("confidence", "low"))

        out_frames.append(
            {
                "node": node,
                "condition": condition,
                "step": step_i,
                "phase": phase_for_step(step_i),
                "timestamp": ts,
                "state": out_state,
                "interpreted_state": out_interpreted_state,
                "confidence": out_confidence,
                "signal_emitted": bool(out.get("signal_emitted")),
                "latest_instability": float(out.get("latest_instability", 0.0)),
                "structural_drift_score": float(out.get("structural_drift_score", 0.0)),
                "relational_stability_score": float(out.get("relational_stability_score", 0.0)),
                "drift_alert": bool(out.get("drift_alert")),
                "data_quality_gate_passed": bool(out.get("data_quality_summary", {}).get("gate_passed", True)),
                "missing_sensor_count": out.get("missing_sensor_count"),
                "active_sensor_count": out.get("active_sensor_count"),
                "dominant_driver": out.get("dominant_driver"),
                "deviation_hit": out_interpreted_state != "NOMINAL_STRUCTURE",
                "alert_hit": out_state in {"WATCH", "ALERT"},
            }
        )

    return out_frames


# -----------------------------------------------------------------------------
# Summaries (node/condition)
# -----------------------------------------------------------------------------
@dataclass
class NodeConditionSummary:
    node: str
    condition: str
    state_counts: Dict[str, int]
    interpreted_state_counts: Dict[str, int]
    deviation_hit_count: int
    deviation_hit_rate: float
    alert_hit_count: int
    alert_hit_rate: float
    peak_instability: float
    mean_instability: float
    baseline_stable_rate: float
    recovery_nominal_rate: float
    data_quality_gate_pass_rate: float


def summarize_node_condition(rows: List[Dict[str, Any]]) -> NodeConditionSummary:
    if not rows:
        raise ValueError("Cannot summarize empty run")

    node = str(rows[0]["node"])
    condition = str(rows[0]["condition"])
    state_counts = dict(Counter(str(r["state"]) for r in rows))
    interpreted_state_counts = dict(Counter(str(r["interpreted_state"]) for r in rows))

    deviation_hits = int(sum(1 for r in rows if r["deviation_hit"]))
    deviation_rate = deviation_hits / float(len(rows))

    alert_hits = int(sum(1 for r in rows if r["alert_hit"]))
    alert_rate = alert_hits / float(len(rows))

    inst = [float(r["latest_instability"]) for r in rows]
    peak_inst = float(max(inst)) if inst else 0.0
    mean_inst = float(float(np.mean(inst))) if inst else 0.0

    baseline_and_recovery = [r for r in rows if r["phase"] in {"baseline", "recovery"}]
    nominal = sum(
        1
        for r in baseline_and_recovery
        if r["state"] == "STABLE" and r["interpreted_state"] == "NOMINAL_STRUCTURE"
    )
    baseline_stable_rate = float(nominal) / float(len(baseline_and_recovery)) if baseline_and_recovery else 0.0

    recovery_rows = [r for r in rows if r["phase"] == "recovery"]
    recovery_nominal = sum(
        1
        for r in recovery_rows
        if r["state"] == "STABLE" and r["interpreted_state"] == "NOMINAL_STRUCTURE"
    )
    recovery_nominal_rate = float(recovery_nominal) / float(len(recovery_rows)) if recovery_rows else 0.0

    gate_pass = sum(1 for r in rows if bool(r["data_quality_gate_passed"]))
    gate_pass_rate = float(gate_pass) / float(len(rows))

    return NodeConditionSummary(
        node=node,
        condition=condition,
        state_counts=state_counts,
        interpreted_state_counts=interpreted_state_counts,
        deviation_hit_count=deviation_hits,
        deviation_hit_rate=deviation_rate,
        alert_hit_count=alert_hits,
        alert_hit_rate=alert_rate,
        peak_instability=peak_inst,
        mean_instability=mean_inst,
        baseline_stable_rate=baseline_stable_rate,
        recovery_nominal_rate=recovery_nominal_rate,
        data_quality_gate_pass_rate=gate_pass_rate,
    )


# -----------------------------------------------------------------------------
# Metrics (precision/recall/stability) across nodes
# -----------------------------------------------------------------------------
def compute_precision_recall(
    rows: List[Dict[str, Any]],
    *,
    positive_predicate: Callable[[Dict[str, Any]], bool],
    anomaly_label: Callable[[Dict[str, Any]], bool],
) -> Tuple[float, float, int, int, int]:
    """
    Compute:
      - precision = TP / (TP + FP)
      - recall = TP / (TP + FN)
    over rows where predictions and labels are defined.
    """

    tp = fp = fn = 0
    for r in rows:
        y_true = anomaly_label(r)
        y_pred = positive_predicate(r)
        if y_true and y_pred:
            tp += 1
        elif (not y_true) and y_pred:
            fp += 1
        elif y_true and (not y_pred):
            fn += 1

    precision = float(tp) / float(tp + fp) if tp + fp > 0 else 0.0
    recall = float(tp) / float(tp + fn) if tp + fn > 0 else 0.0
    return precision, recall, tp, fp, fn


def compute_benchmark_metrics(all_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Metrics computed per condition:
      - alert_precision/recall for target node C during perturbation.
      - baseline_stability_rate for A/B negative nodes during baseline+recovery.
      - recovery_success_rate for C during recovery.
      - false_positive_rate for non-target nodes during perturbation.
      - C vs others peak-instability separation during perturbation.
    """

    metrics: Dict[str, Any] = {}
    by_condition = {c: [r for r in all_rows if r["condition"] == c] for c in CONDITIONS}

    for condition, rows_c in by_condition.items():
        # Ground truth label: C is anomalous only during perturbation.
        def is_anomaly(r: Dict[str, Any]) -> bool:
            return r["node"] == "C" and r["phase"] == "perturbation"

        def is_alert(r: Dict[str, Any]) -> bool:
            return bool(r["alert_hit"])

        precision, recall, tp, fp, fn = compute_precision_recall(
            rows_c,
            positive_predicate=is_alert,
            anomaly_label=is_anomaly,
        )

        # Baseline stability: A/B negative nodes should remain nominal.
        ab_rows = [r for r in rows_c if r["node"] in {"A", "B"} and r["phase"] in {"baseline", "recovery"}]
        ab_nominal = sum(
            1
            for r in ab_rows
            if r["state"] == "STABLE" and r["interpreted_state"] == "NOMINAL_STRUCTURE"
        )
        baseline_stability_rate = float(ab_nominal) / float(len(ab_rows)) if ab_rows else 0.0

        # False positives during perturbation: any alert on non-C nodes.
        pert_rows = [r for r in rows_c if r["phase"] == "perturbation"]
        non_target_pert = [r for r in pert_rows if r["node"] != "C"]
        fp_count = sum(1 for r in non_target_pert if bool(r["alert_hit"]))
        false_positive_rate = float(fp_count) / float(len(non_target_pert)) if non_target_pert else 0.0

        # C recovery success: should return to nominal during recovery.
        c_recovery_rows = [r for r in rows_c if r["node"] == "C" and r["phase"] == "recovery"]
        c_recovery_nominal = sum(
            1 for r in c_recovery_rows if r["state"] == "STABLE" and r["interpreted_state"] == "NOMINAL_STRUCTURE"
        )
        recovery_success_rate = (
            float(c_recovery_nominal) / float(len(c_recovery_rows)) if c_recovery_rows else 0.0
        )

        # Peak separation: C should stand out vs non-target nodes in perturbation.
        c_pert = [r for r in rows_c if r["node"] == "C" and r["phase"] == "perturbation"]
        others_pert = [r for r in rows_c if r["node"] != "C" and r["phase"] == "perturbation"]
        c_peak = float(max((r["latest_instability"] for r in c_pert), default=0.0))
        others_peak = float(max((r["latest_instability"] for r in others_pert), default=0.0))
        peak_separation = c_peak - others_peak

        # C perturbation alert-rate (recall uses per-frame, but we also expose rate for stakeholders).
        c_pert_alert_rate = float(sum(1 for r in c_pert if bool(r["alert_hit"]))) / float(len(c_pert)) if c_pert else 0.0

        metrics[condition] = {
            "alert_precision": precision,
            "alert_recall": recall,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "baseline_stability_rate": baseline_stability_rate,
            "false_positive_rate": false_positive_rate,
            "recovery_success_rate": recovery_success_rate,
            "peak_separation": peak_separation,
            "c_perturbation_alert_rate": c_pert_alert_rate,
            "c_peak_instability": c_peak,
            "others_peak_instability": others_peak,
        }

    # Keep condition-level comparison useful for quick reads.
    metrics["comparison"] = {
        "disturbed_minus_coherent_peak_separation": metrics["disturbed_time"]["peak_separation"]
        - metrics["coherent_time"]["peak_separation"],
    }
    return metrics


def to_summary_rows(summaries: List[NodeConditionSummary]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for s in summaries:
        row = {
            "node": s.node,
            "condition": s.condition,
            "deviation_hit_count": s.deviation_hit_count,
            "deviation_hit_rate": round(s.deviation_hit_rate, 6),
            "alert_hit_count": s.alert_hit_count,
            "alert_hit_rate": round(s.alert_hit_rate, 6),
            "peak_instability": round(s.peak_instability, 6),
            "mean_instability": round(s.mean_instability, 6),
            "baseline_stable_rate": round(s.baseline_stable_rate, 6),
            "recovery_nominal_rate": round(s.recovery_nominal_rate, 6),
            "data_quality_gate_pass_rate": round(s.data_quality_gate_pass_rate, 6),
            "state_counts": json.dumps(s.state_counts, sort_keys=True),
            "interpreted_state_counts": json.dumps(s.interpreted_state_counts, sort_keys=True),
        }
        rows.append(row)
    return rows


def print_node_state_distributions(summaries: List[NodeConditionSummary]) -> None:
    print("\n" + "=" * 60)
    print("Node state distribution (state=STABLE/WATCH/ALERT)")
    print("=" * 60)
    for node in NODES:
        print(f"\n--- Node {node} ---")
        for cond in CONDITIONS:
            s = next(x for x in summaries if x.node == node and x.condition == cond)
            c = s.state_counts
            print(f"  {cond}: ALERT={c.get('ALERT', 0)} WATCH={c.get('WATCH', 0)} STABLE={c.get('STABLE', 0)}")


def print_executive_summary(metrics_by_condition: Dict[str, Any], summaries: List[NodeConditionSummary]) -> None:
    df = pd.DataFrame(
        [
            {
                "condition": cond,
                "alert_precision": metrics_by_condition[cond]["alert_precision"],
                "alert_recall": metrics_by_condition[cond]["alert_recall"],
                "baseline_stability_rate": metrics_by_condition[cond]["baseline_stability_rate"],
                "false_positive_rate": metrics_by_condition[cond]["false_positive_rate"],
                "recovery_success_rate": metrics_by_condition[cond]["recovery_success_rate"],
                "peak_separation": metrics_by_condition[cond]["peak_separation"],
                "c_perturbation_alert_rate": metrics_by_condition[cond]["c_perturbation_alert_rate"],
            }
            for cond in CONDITIONS
        ]
    )

    print("\n" + "=" * 60)
    print("Benchmark metrics (precision/recall + stability)")
    print("=" * 60)
    print(df.to_string(index=False))

    print("\n" + "=" * 60)
    print("Node peak instability summary (perturbation phase)")
    print("=" * 60)
    for node in NODES:
        s_coh = next(x for x in summaries if x.node == node and x.condition == "coherent_time")
        s_dis = next(x for x in summaries if x.node == node and x.condition == "disturbed_time")
        print(f"{node}: coherent_peak={s_coh.peak_instability:.4f}  disturbed_peak={s_dis.peak_instability:.4f}")


def assert_benchmark_is_credible(metrics: Dict[str, Any]) -> None:
    """
    Hard assertions to keep the benchmark technically defensible.

    These floors are intentionally conservative: they enforce that
    - A/B remain nominal
    - C is the primary anomalous node (high recall with acceptable precision)
    - D does not behave like a target anomaly
    - Recovery is visible
    """

    coh = metrics["coherent_time"]
    dis = metrics["disturbed_time"]

    # Non-target precision should improve materially.
    # (The benchmark is designed so most alerts come from C during perturbation.)
    assert coh["alert_precision"] >= 0.40, f"coherent alert_precision too low: {coh['alert_precision']}"
    assert dis["alert_precision"] >= 0.40, f"disturbed alert_precision too low: {dis['alert_precision']}"

    # C recall should not collapse.
    assert coh["alert_recall"] >= 0.90, f"coherent alert_recall too low: {coh['alert_recall']}"
    assert dis["alert_recall"] >= 0.95, f"disturbed alert_recall too low: {dis['alert_recall']}"

    # A/B should remain nominal (stable + nominal interpretation).
    assert coh["baseline_stability_rate"] >= 0.75, f"coherent baseline_stability_rate too low: {coh['baseline_stability_rate']}"
    assert dis["baseline_stability_rate"] >= 0.75, f"disturbed baseline_stability_rate too low: {dis['baseline_stability_rate']}"

    # Recovery should restore nominal structure for C.
    assert coh["recovery_success_rate"] >= 0.80, f"coherent recovery_success_rate too low: {coh['recovery_success_rate']}"
    assert dis["recovery_success_rate"] >= 0.80, f"disturbed recovery_success_rate too low: {dis['recovery_success_rate']}"

    # C should separate from non-target peak instability during perturbation.
    assert coh["peak_separation"] > 0.2, f"coherent peak_separation too small: {coh['peak_separation']}"
    assert dis["peak_separation"] > 0.2, f"disturbed peak_separation too small: {dis['peak_separation']}"

    # False positives should be limited on non-C nodes during perturbation.
    assert coh["false_positive_rate"] <= 0.35, f"coherent false_positive_rate too high: {coh['false_positive_rate']}"
    assert dis["false_positive_rate"] <= 0.35, f"disturbed false_positive_rate too high: {dis['false_positive_rate']}"


# -----------------------------------------------------------------------------
# Main benchmark driver
# -----------------------------------------------------------------------------
def main() -> int:
    output_json_path = Path(OUTPUT_JSON)
    output_dir = Path(".")

    all_rows: List[Dict[str, Any]] = []
    node_condition_summaries: List[NodeConditionSummary] = []

    gal2_configured = bool(os.getenv("GAL2_API_KEY"))
    gal2_time_url = os.getenv("GAL2_TIME_URL", "https://api-v2.gal-2.com/time")
    gal2_used_any = False

    with tempfile.TemporaryDirectory(prefix="neraium_credible_multinode_") as tmp_dir:
        # Run for each condition (single run; deterministic because synthetic is deterministic)
        for condition in CONDITIONS:
            timestamps, gal2_used = timestamps_for_condition(condition, N_STEPS)
            if gal2_used:
                gal2_used_any = True

            for node in NODES:
                rows = run_node_condition(node, condition, tmp_dir, timestamps=timestamps)
                all_rows.extend(rows)

                summary = summarize_node_condition(rows)
                node_condition_summaries.append(summary)

        # Compute metrics across all rows.
        metrics_by_condition = compute_benchmark_metrics(all_rows)

        # Print for humans.
        print_node_state_distributions(node_condition_summaries)
        print_executive_summary(metrics_by_condition, node_condition_summaries)

        # Save CSVs
        df_timeseries = pd.DataFrame(all_rows)
        df_timeseries.to_csv(output_dir / OUTPUT_TIMESERIES_CSV, index=False)

        df_node_summary = pd.DataFrame(to_summary_rows(node_condition_summaries))
        df_node_summary.to_csv(output_dir / OUTPUT_NODE_SUMMARY_CSV, index=False)

        df_metrics = pd.DataFrame(
            [
                {
                    "condition": cond,
                    **{
                        k: v
                        for k, v in metrics_by_condition[cond].items()
                        if k not in {"tp", "fp", "fn"}
                    },
                    "tp": metrics_by_condition[cond]["tp"],
                    "fp": metrics_by_condition[cond]["fp"],
                    "fn": metrics_by_condition[cond]["fn"],
                }
                for cond in CONDITIONS
            ]
        )
        df_metrics.to_csv(output_dir / OUTPUT_METRICS_CSV, index=False)

        # Enforce credibility (after artifacts exist so failures remain diagnosable).
        assert_benchmark_is_credible(metrics_by_condition)

        # Save JSON result (reproducible summary + metrics)
        result_obj: Dict[str, Any] = {
            "description": "Credible 4-node SII structural perturbation benchmark (baseline/perturbation/recovery) with GAL-2 time substrate.",
            "nodes": NODES,
            "conditions": CONDITIONS,
            "t_start": T_START,
            "t_end": T_END,
            "phase_windows": {
                "baseline": [T_START, BASELINE_END],
                "perturbation": [PERT_START, PERT_END],
                "recovery": [RECOVERY_START, T_END - 1],
            },
            "engine_config": {
                "baseline_window": BASELINE_WINDOW,
                "recent_window": RECENT_WINDOW,
            },
            "gal2_api_configured": gal2_configured,
            "gal2_time_url": gal2_time_url if gal2_configured else None,
            "gal2_used_for_disturbed_time": gal2_used_any,
            "metrics_by_condition": metrics_by_condition,
            "artifacts": {
                "timeseries_csv": str(Path(OUTPUT_TIMESERIES_CSV).as_posix()),
                "node_summary_csv": str(Path(OUTPUT_NODE_SUMMARY_CSV).as_posix()),
                "metrics_csv": str(Path(OUTPUT_METRICS_CSV).as_posix()),
            },
            "seed": SEED,
        }

        # Stable JSON representation for node summaries
        result_obj["node_condition_summaries"] = [
            {
                "node": s.node,
                "condition": s.condition,
                "state_counts": s.state_counts,
                "interpreted_state_counts": s.interpreted_state_counts,
                "deviation_hit_count": s.deviation_hit_count,
                "deviation_hit_rate": s.deviation_hit_rate,
                "alert_hit_count": s.alert_hit_count,
                "alert_hit_rate": s.alert_hit_rate,
                "peak_instability": s.peak_instability,
                "mean_instability": s.mean_instability,
                "baseline_stable_rate": s.baseline_stable_rate,
                "recovery_nominal_rate": s.recovery_nominal_rate,
                "data_quality_gate_pass_rate": s.data_quality_gate_pass_rate,
            }
            for s in node_condition_summaries
        ]

        output_json_path.write_text(json.dumps(make_json_safe(result_obj), indent=2), encoding="utf-8")

    print(f"\nOutputs written:\n  - {output_json_path.as_posix()}\n  - {output_dir / OUTPUT_TIMESERIES_CSV}\n  - {output_dir / OUTPUT_NODE_SUMMARY_CSV}\n  - {output_dir / OUTPUT_METRICS_CSV}\n")
    return 0


def make_json_safe(obj: Any) -> Any:
    """Make objects JSON-serializable (numpy floats/ints)."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_safe(x) for x in obj]
    return str(obj)


if __name__ == "__main__":
    raise SystemExit(main())

