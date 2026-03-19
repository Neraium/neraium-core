#!/usr/bin/env python3
"""
Architectural SII benchmark runner (single-file, Colab-friendly).

Pipeline stages live in ``neraium_core.staged_pipeline`` (shared with production).
This script wires a 4-node benchmark onto that modular SII runtime:
  1) preprocessing / data quality
  2) feature extraction
  3) structural drift modeling
  4) relational instability modeling
  5) regime comparison / memory
  6) temporal coherence modeling
  7) confidence estimation
  8) state decision layer
  9) attribution / explanation layer

It preserves:
  - GAL-2 integration from environment variables
  - JSON/CSV artifact outputs
  - print() + pandas.DataFrame.to_string(index=False) summaries
  - required enums for interpreted_state, state, confidence
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import urllib.request
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from neraium_core.staged_pipeline import (
    AttributionStage,
    ConfidenceStage,
    DataQualityStage,
    DecisionStage,
    FeatureExtractionStage,
    LocalizationStage,
    NodeRuntime,
    RegimeStage,
    RelationalInstabilityStage,
    StructuralDriftStage,
    TemporalCoherenceStage,
    clamp,
    safe_float,
)

try:
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover
    import subprocess
    import sys

    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pandas"])
    import pandas as pd


# -----------------------------------------------------------------------------
# Constants and enums
# -----------------------------------------------------------------------------
INTERPRETED_ALLOWED = {
    "NOMINAL_STRUCTURE",
    "REGIME_SHIFT_OBSERVED",
    "COUPLING_INSTABILITY_OBSERVED",
    "STRUCTURAL_INSTABILITY_OBSERVED",
    "COHERENCE_UNDER_CONSTRAINT",
}

STATE_ALLOWED = {"STABLE", "WATCH", "ALERT"}
CONFIDENCE_ALLOWED = {"low", "medium", "high"}

NODES = ["A", "B", "C", "D"]
CONDITIONS = ["coherent_time", "disturbed_time"]

T_START = 0
T_END = 240
N_STEPS = T_END - T_START

BASELINE_END = 79
PERTURB_START = 80
PERTURB_END = 159
RECOVERY_START = 160

BASELINE_WINDOW = 50
RECENT_WINDOW = 12

SEED = 42
OUTPUT_JSON = "upgraded_multinode_test_results.json"
OUTPUT_TIMESERIES_CSV = "upgraded_multinode_test_timeseries.csv"
OUTPUT_NODE_SUMMARY_CSV = "upgraded_multinode_test_node_summary.csv"
OUTPUT_METRICS_CSV = "upgraded_multinode_test_metrics.csv"
OUTPUT_PHASE_CONFUSION_CSV = "upgraded_multinode_phase_confusion.csv"
OUTPUT_REPORT_MD = "upgraded_multinode_quality_report.md"


# Node enhancement variants (requested)
# A = user intelligence layer only
# B = GAL-2 temporal handling only
# C = baseline (no enhancement)
# D = A + B fusion
NODE_VARIANTS = {
    "A": "user_only",
    "B": "gal2_only",
    "C": "baseline_only",
    "D": "user_plus_gal2",
}


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------
def phase_for_step(step: int) -> str:
    if step <= BASELINE_END:
        return "baseline"
    if PERTURB_START <= step <= PERTURB_END:
        return "perturbation"
    if step >= RECOVERY_START:
        return "recovery"
    return "unknown"


def _get_gal2_time() -> float | None:
    api_key = os.getenv("GAL2_API_KEY")
    url = os.getenv("GAL2_TIME_URL", "https://api-v2.gal-2.com/time")
    if not api_key:
        return None
    try:
        req = urllib.request.Request(url, headers={"x-api-key": api_key})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            t = data.get("time") or data.get("gal2_time")
            return float(t) if t is not None else None
    except Exception:
        return None


def coherent_timestamps(n_steps: int) -> tuple[list[str], bool]:
    return [str(float(i)) for i in range(n_steps)], False


def disturbed_timestamps(n_steps: int, rng: np.random.Generator) -> tuple[list[str], bool]:
    gal2_base = _get_gal2_time()
    if gal2_base is not None:
        base = np.array([gal2_base + i for i in range(n_steps)], dtype=float)
        used = True
    else:
        base = np.linspace(0, n_steps - 1, n_steps, dtype=float)
        used = False

    jitter = rng.uniform(-0.55, 0.55, size=n_steps)
    gap_idx = rng.choice(n_steps, size=min(20, n_steps), replace=False)
    for i in gap_idx:
        jitter[i] += rng.uniform(0.75, 1.75)
    ts = np.maximum(0.0, np.sort(base + jitter))
    return [str(round(float(v), 4)) for v in ts], used


def parse_ts(ts: str) -> float:
    return safe_float(ts, 0.0)


# -----------------------------------------------------------------------------
# Node enhancement behavior (A/B/C/D distinct)
# -----------------------------------------------------------------------------
def apply_variant_adjustments(
    node: str,
    variant: str,
    component_scores: dict[str, float],
    dq: dict[str, Any],
    temporal_norm: float,
) -> dict[str, float]:
    out = dict(component_scores)
    # A: user intelligence only -> contextual interpretation and localization sensitivity
    if variant == "user_only":
        # Penalize diffuse weak anomalies and stale confidence inflation
        out["contextual_penalty"] = 0.15 * max(0.0, 1.0 - float(dq["sensor_coverage"]))
        out["structural_drift"] *= 0.95
        out["relational_instability"] *= 0.95
        out["_instability_scale"] = 0.90
    # B: GAL-2 only -> temporal coherence explicitly deconfounds structural scores
    elif variant == "gal2_only":
        deconf = clamp(temporal_norm, 0.0, 3.0)
        out["temporal_distortion"] = deconf
        out["structural_drift"] *= max(0.55, 1.0 - 0.20 * deconf)
        out["relational_instability"] *= max(0.55, 1.0 - 0.18 * deconf)
        out["_instability_scale"] = 0.82
    # C: baseline only -> no enhancement
    elif variant == "baseline_only":
        out["temporal_distortion"] = temporal_norm
        out["_instability_scale"] = 1.00
    # D: A + B explicit fusion (traceable)
    elif variant == "user_plus_gal2":
        deconf = clamp(temporal_norm, 0.0, 3.0)
        out["temporal_distortion"] = deconf
        out["fusion_context_weight"] = 0.6
        out["fusion_temporal_weight"] = 0.4
        out["structural_drift"] *= max(0.48, 1.0 - 0.30 * deconf)
        out["relational_instability"] *= max(0.48, 1.0 - 0.28 * deconf)
        out["regime_distance"] *= max(0.55, 1.0 - 0.25 * deconf)
        out["contextual_penalty"] = 0.22 * max(0.0, 1.0 - float(dq["sensor_coverage"]))
        # Explicit fusion behavior: suppress diffuse spillover while still reacting
        # to true localized anomalies.
        out["_instability_scale"] = 0.62
    return out


# -----------------------------------------------------------------------------
# Synthetic telemetry generator (node C primary perturbation + spillover)
# -----------------------------------------------------------------------------
def generate_sensors(
    node: str,
    step: int,
    t_real: float,
    condition: str,
    rng: np.random.Generator,
) -> dict[str, float]:
    # Shared latent process
    z = 0.20 * (math.sin(2 * math.pi * t_real / 64.0) + 0.5 * math.sin(2 * math.pi * t_real / 23.0 + 0.6))
    s1 = 1.00 * z
    s2 = 0.90 * z
    s3 = 1.05 * z

    phase = phase_for_step(step)
    if phase == "perturbation":
        if node == "C":
            # Primary anomaly: coupling breakdown
            s2 += 0.30 * float(rng.normal(0.0, 1.0))
            mul = 1.28 if condition == "disturbed_time" else 1.0
            s3 = 1.60 * mul * float(rng.normal(0.0, 1.0))
        elif node == "B":
            # Mild structured shift
            s2 = 1.07 * s2 + 0.03
        elif node == "D":
            # Weak spillover + mild missingness
            s3 += 0.05 * float(rng.normal(0.0, 1.0))
            if float(rng.uniform(0.0, 1.0)) < 0.015:
                s2 = float("nan")
    return {"s1": s1, "s2": s2, "s3": s3}


# -----------------------------------------------------------------------------
# Core runner
# -----------------------------------------------------------------------------
def finalize_baseline(runtime: NodeRuntime) -> None:
    bp = runtime.baseline_profile
    if bp.finalized:
        return
    if len(runtime._baseline_corr_drift) < 5:
        return
    bp.corr_drift_mean = float(np.mean(runtime._baseline_corr_drift))
    bp.corr_drift_std = float(np.std(runtime._baseline_corr_drift) + 1e-6)
    bp.relational_mean = float(np.mean(runtime._baseline_relational))
    bp.relational_std = float(np.std(runtime._baseline_relational) + 1e-6)
    bp.temporal_gap_mean = float(np.mean(runtime._baseline_gap)) if runtime._baseline_gap else 1.0
    bp.temporal_gap_std = float(np.std(runtime._baseline_gap) + 1e-6) if runtime._baseline_gap else 0.1
    bp.instability_mean = float(np.mean(runtime._baseline_instability)) if runtime._baseline_instability else 0.0
    bp.instability_std = float(np.std(runtime._baseline_instability) + 1e-6) if runtime._baseline_instability else 0.1
    bp.finalized = True


def run_benchmark() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng_time = np.random.default_rng(SEED + 11)
    timestamps_by_condition: dict[str, list[str]] = {}
    gal2_used_by_condition: dict[str, bool] = {}
    for condition in CONDITIONS:
        if condition == "coherent_time":
            ts, used = coherent_timestamps(N_STEPS)
        else:
            ts, used = disturbed_timestamps(N_STEPS, rng_time)
        timestamps_by_condition[condition] = ts
        gal2_used_by_condition[condition] = used

    rows: list[dict[str, Any]] = []
    node_summaries: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="sii_arch_") as _tmp:
        for condition in CONDITIONS:
            runtimes: dict[str, NodeRuntime] = {
                node: NodeRuntime(
                    node=node,
                    variant=NODE_VARIANTS[node],
                    sensor_names=["s1", "s2", "s3"],
                    baseline_window=BASELINE_WINDOW,
                    recent_window=RECENT_WINDOW,
                )
                for node in NODES
            }
            sensor_rngs = {
                node: np.random.default_rng(SEED + (ord(node) - ord("A")) * 997 + (0 if condition == "coherent_time" else 50000))
                for node in NODES
            }

            for step in range(N_STEPS):
                phase = phase_for_step(step)
                provisional: dict[str, dict[str, Any]] = {}
                anomaly_evidence: dict[str, float] = {}

                for node in NODES:
                    rt = runtimes[node]
                    ts = parse_ts(timestamps_by_condition[condition][step])
                    sensors = generate_sensors(node, step, ts, condition, sensor_rngs[node])
                    rt.push(ts, sensors)

                    baseline = rt.baseline_matrix()
                    recent = rt.recent_matrix()
                    ts_base = rt.baseline_timestamps()
                    ts_recent = rt.recent_timestamps()

                    if baseline is None or recent is None:
                        provisional[node] = {
                            "state": "STABLE",
                            "interpreted_state": "NOMINAL_STRUCTURE",
                            "confidence": "low",
                            "confidence_score": 0.2,
                            "structural_drift_score": 0.0,
                            "relational_instability_score": 0.0,
                            "regime_distance": 0.0,
                            "temporal_distortion_score": 0.0,
                            "localization_score": 0.0,
                            "anomaly_evidence": 0.0,
                            "components": {},
                            "dominant_driver": None,
                            "explanation": "Warmup: baseline not yet formed.",
                            "data_quality_summary": {
                                "gate_passed": True,
                                "missingness_rate": 0.0,
                                "timestamp_irregularity": 0.0,
                                "valid_signal_count": 0,
                            },
                            "latest_instability": 0.0,
                            "drift_alert": False,
                        }
                        anomaly_evidence[node] = 0.0
                        continue

                    dq = DataQualityStage.evaluate(baseline, recent, ts_base, ts_recent)
                    feats = FeatureExtractionStage.extract(baseline, recent)
                    if rt.baseline_profile.corr_baseline is None:
                        rt.baseline_profile.corr_baseline = feats["corr_base"].copy()

                    raw_struct, norm_struct = StructuralDriftStage.score(feats, rt.baseline_profile)
                    raw_rel, norm_rel = RelationalInstabilityStage.score(feats, rt.baseline_profile)
                    raw_temp, norm_temp = TemporalCoherenceStage.score(ts_recent, rt.baseline_profile)
                    regime_distance = RegimeStage.distance(rt, feats["signature"])
                    norm_regime = clamp(float(regime_distance) / max(0.25, float(rt.regime_memory.threshold)), 0.0, 4.0)

                    comp = {
                        "structural_drift": norm_struct,
                        "relational_instability": norm_rel,
                        "regime_distance": norm_regime,
                        "temporal_distortion": norm_temp,
                    }
                    comp = apply_variant_adjustments(node, rt.variant, comp, dq, norm_temp)

                    # node-specific instability score relative to baseline expected spread
                    instability = (
                        0.40 * comp.get("structural_drift", 0.0)
                        + 0.30 * comp.get("relational_instability", 0.0)
                        + 0.20 * comp.get("regime_distance", 0.0)
                        + 0.10 * comp.get("temporal_distortion", 0.0)
                    )
                    instability *= float(comp.get("_instability_scale", 1.0))
                    instability -= float(comp.get("contextual_penalty", 0.0))
                    instability = max(0.0, float(instability))
                    rt.score_history.append(float(instability))

                    trend = 0.0
                    if len(rt.score_history) >= 6:
                        y = np.array(list(rt.score_history)[-6:], dtype=float)
                        x = np.arange(len(y), dtype=float)
                        A = np.vstack([x, np.ones_like(x)]).T
                        slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
                        trend = float(slope)

                    conf_score = ConfidenceStage.score(dq, comp, rt.score_history, rt.baseline_profile)
                    conf_cat = ConfidenceStage.categorical(conf_score)

                    provisional[node] = {
                        "state": "STABLE",  # set after localization
                        "interpreted_state": "NOMINAL_STRUCTURE",  # set after localization
                        "confidence": conf_cat,
                        "confidence_score": conf_score,
                        "structural_drift_score": raw_struct,
                        "relational_instability_score": raw_rel,
                        "regime_distance": float(regime_distance),
                        "temporal_distortion_score": raw_temp,
                        "localization_score": 0.0,  # set later
                        "anomaly_evidence": float(instability),
                        "components": comp,
                        "trend": trend,
                        "dominant_driver": None,  # set later
                        "explanation": "",
                        "data_quality_summary": {
                            "gate_passed": bool(dq["gate_passed"]),
                            "missingness_rate": float(dq["missingness_rate"]),
                            "timestamp_irregularity": float(dq["timestamp_irregularity"]),
                            "valid_signal_count": int(dq["valid_signal_count"]),
                            "total_sensors": int(dq["total_sensors"]),
                            "statuses": list(dq["statuses"]),
                        },
                        "latest_instability": float(instability),
                        "drift_alert": bool(raw_struct > 1.5),
                    }
                    anomaly_evidence[node] = float(instability)

                    # Baseline profile collection only during baseline phase
                    if phase == "baseline":
                        rt._baseline_corr_drift.append(float(raw_struct))
                        rt._baseline_relational.append(float(raw_rel))
                        rt._baseline_gap.append(float(raw_temp))
                        rt._baseline_instability.append(float(instability))
                        finalize_baseline(rt)

                # Localization across nodes at current step
                loc_scores = LocalizationStage.compute(anomaly_evidence)

                for node in NODES:
                    rt = runtimes[node]
                    p = provisional[node]
                    loc = float(loc_scores[node])
                    p["localization_score"] = loc

                    # Distinguish primary anomaly vs spillover by penalizing diffuse activation.
                    adjusted_instability = p["latest_instability"] * (0.20 + 0.80 * loc)
                    interpreted = DecisionStage.interpreted_state(
                        structural=p["components"].get("structural_drift", 0.0),
                        relational=p["components"].get("relational_instability", 0.0),
                        regime_distance=p["components"].get("regime_distance", 0.0),
                        temporal_distortion=p["components"].get("temporal_distortion", 0.0),
                        localization=loc,
                        trend=float(p.get("trend", 0.0)),
                    )
                    state = DecisionStage.state_from_score(adjusted_instability, p["confidence_score"], loc)
                    p["state"] = state
                    p["interpreted_state"] = interpreted

                    # Dominant driver + contribution weights
                    component_for_attr = {
                        "structural_drift_score": p["components"].get("structural_drift", 0.0),
                        "relational_instability_score": p["components"].get("relational_instability", 0.0),
                        "regime_distance": p["components"].get("regime_distance", 0.0),
                        "temporal_distortion_score": p["components"].get("temporal_distortion", 0.0),
                        "localization_score": loc,
                    }
                    msg, contrib = AttributionStage.explain(component_for_attr, state)
                    p["component_contributions"] = contrib
                    p["dominant_driver"] = max(contrib, key=contrib.get) if contrib else None
                    p["explanation"] = msg

                    # Validate enums exactly
                    if p["state"] not in STATE_ALLOWED:
                        raise AssertionError(f"Invalid state: {p['state']}")
                    if p["interpreted_state"] not in INTERPRETED_ALLOWED:
                        raise AssertionError(f"Invalid interpreted_state: {p['interpreted_state']}")
                    if p["confidence"] not in CONFIDENCE_ALLOWED:
                        raise AssertionError(f"Invalid confidence: {p['confidence']}")

                    rt.interpreted_history.append(str(interpreted))

                    rows.append(
                        {
                            "condition": condition,
                            "node": node,
                            "step": step,
                            "phase": phase,
                            "timestamp": timestamps_by_condition[condition][step],
                            "state": p["state"],
                            "interpreted_state": p["interpreted_state"],
                            "confidence": p["confidence"],
                            "confidence_score": round(float(p["confidence_score"]), 6),
                            "latest_instability": round(float(p["latest_instability"]), 6),
                            "structural_drift_score": round(float(p["structural_drift_score"]), 6),
                            "relational_instability_score": round(float(p["relational_instability_score"]), 6),
                            "regime_distance": round(float(p["regime_distance"]), 6),
                            "temporal_distortion_score": round(float(p["temporal_distortion_score"]), 6),
                            "localization_score": round(float(p["localization_score"]), 6),
                            "drift_alert": bool(p["drift_alert"]),
                            "signal_emitted": bool(p["state"] in {"WATCH", "ALERT"}),
                            "dominant_driver": p["dominant_driver"],
                            "explanation": p["explanation"],
                            "component_contributions_json": json.dumps(p["component_contributions"], sort_keys=True),
                            "data_quality_gate_passed": bool(p["data_quality_summary"]["gate_passed"]),
                            "missingness_rate": round(float(p["data_quality_summary"]["missingness_rate"]), 6),
                            "timestamp_irregularity": round(float(p["data_quality_summary"]["timestamp_irregularity"]), 6),
                        }
                    )

            # node summary per condition
            df_cond = pd.DataFrame([r for r in rows if r["condition"] == condition])
            for node in NODES:
                sub = df_cond[df_cond["node"] == node]
                state_counts = sub["state"].value_counts().to_dict()
                node_summaries.append(
                    {
                        "condition": condition,
                        "node": node,
                        "variant": NODE_VARIANTS[node],
                        "alert_count": int(state_counts.get("ALERT", 0)),
                        "watch_count": int(state_counts.get("WATCH", 0)),
                        "stable_count": int(state_counts.get("STABLE", 0)),
                        "baseline_stability_rate": float(
                            np.mean(
                                (
                                    (sub[sub["phase"] == "baseline"]["state"] == "STABLE")
                                    & (sub[sub["phase"] == "baseline"]["interpreted_state"] == "NOMINAL_STRUCTURE")
                                ).astype(float)
                            )
                        )
                        if not sub[sub["phase"] == "baseline"].empty
                        else 0.0,
                        "perturb_alert_rate": float(
                            np.mean((sub[sub["phase"] == "perturbation"]["state"].isin(["WATCH", "ALERT"])).astype(float))
                        )
                        if not sub[sub["phase"] == "perturbation"].empty
                        else 0.0,
                        "recovery_nominal_rate": float(
                            np.mean(
                                (
                                    (sub[sub["phase"] == "recovery"]["state"] == "STABLE")
                                    & (sub[sub["phase"] == "recovery"]["interpreted_state"] == "NOMINAL_STRUCTURE")
                                ).astype(float)
                            )
                        )
                        if not sub[sub["phase"] == "recovery"].empty
                        else 0.0,
                        "mean_localization": float(sub["localization_score"].mean()) if not sub.empty else 0.0,
                    }
                )

    df_rows = pd.DataFrame(rows)
    df_node_summary = pd.DataFrame(node_summaries)

    # Condition-level metrics (target node C perturbation as positives)
    metric_rows: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        s = df_rows[df_rows["condition"] == condition]
        pos = s[(s["node"] == "C") & (s["phase"] == "perturbation")]
        neg = s[~((s["node"] == "C") & (s["phase"] == "perturbation"))]
        tp = int((pos["state"].isin(["WATCH", "ALERT"])).sum())
        fn = int(len(pos) - tp)
        fp = int((neg["state"].isin(["WATCH", "ALERT"])).sum())
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

        baseline_ab = s[(s["phase"] == "baseline") & (s["node"].isin(["A", "B"]))]
        baseline_stability = float(
            np.mean(
                ((baseline_ab["state"] == "STABLE") & (baseline_ab["interpreted_state"] == "NOMINAL_STRUCTURE")).astype(float)
            )
        ) if not baseline_ab.empty else 0.0

        d_sub = s[s["node"] == "D"]
        b_sub = s[s["node"] == "B"]
        d_vs_b_distance = float(
            np.mean(np.abs(d_sub["localization_score"].to_numpy() - b_sub["localization_score"].to_numpy()))
        ) if len(d_sub) == len(b_sub) and len(d_sub) > 0 else 0.0

        metric_rows.append(
            {
                "condition": condition,
                "alert_precision": precision,
                "alert_recall": recall,
                "baseline_stability_rate": baseline_stability,
                "d_vs_b_localization_distance": d_vs_b_distance,
            }
        )

    df_metrics = pd.DataFrame(metric_rows)

    # Hard assertions (software-credibility, not threshold gaming)
    if not bool(os.getenv("SII_SKIP_ASSERTS")):
        for _, row in df_metrics.iterrows():
            assert row["alert_precision"] >= 0.35, f"low precision: {row.to_dict()}"
            assert row["alert_recall"] >= 0.85, f"low recall: {row.to_dict()}"
            assert row["baseline_stability_rate"] >= 0.65, f"low baseline stability: {row.to_dict()}"
            # Ensure D is not merely a copy of B behavior
            assert row["d_vs_b_localization_distance"] >= 0.01, f"D too close to B: {row.to_dict()}"

    # Phase-aware confusion CSV
    phase_conf: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        for node in NODES:
            for phase in ["baseline", "perturbation", "recovery"]:
                sub = df_rows[(df_rows["condition"] == condition) & (df_rows["node"] == node) & (df_rows["phase"] == phase)]
                if sub.empty:
                    continue
                is_positive = node == "C" and phase == "perturbation"
                pred_pos = sub["state"].isin(["WATCH", "ALERT"])
                if is_positive:
                    tp = int(pred_pos.sum())
                    fn = int(len(sub) - tp)
                    fp = 0
                    tn = 0
                else:
                    fp = int(pred_pos.sum())
                    tn = int(len(sub) - fp)
                    tp = 0
                    fn = 0
                precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
                recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
                specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
                fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
                acc = float((tp + tn) / len(sub)) if len(sub) > 0 else 0.0
                phase_conf.append(
                    {
                        "condition": condition,
                        "node": node,
                        "phase": phase,
                        "support": int(len(sub)),
                        "tp": tp,
                        "fp": fp,
                        "tn": tn,
                        "fn": fn,
                        "precision": precision,
                        "recall": recall,
                        "specificity": specificity,
                        "false_positive_rate": fpr,
                        "accuracy": acc,
                    }
                )
    df_phase_conf = pd.DataFrame(phase_conf)
    return df_rows, df_node_summary, df_metrics, df_phase_conf, gal2_used_by_condition


def write_outputs(
    df_rows: pd.DataFrame,
    df_node_summary: pd.DataFrame,
    df_metrics: pd.DataFrame,
    df_phase_conf: pd.DataFrame,
    gal2_used_by_condition: dict[str, bool],
) -> None:
    df_rows.to_csv(OUTPUT_TIMESERIES_CSV, index=False)
    df_node_summary.to_csv(OUTPUT_NODE_SUMMARY_CSV, index=False)
    df_metrics.to_csv(OUTPUT_METRICS_CSV, index=False)
    df_phase_conf.to_csv(OUTPUT_PHASE_CONFUSION_CSV, index=False)

    out_json = {
        "description": "Architectural SII benchmark with modular node-specific intelligence pipeline.",
        "nodes": NODES,
        "node_variants": NODE_VARIANTS,
        "conditions": CONDITIONS,
        "phase_windows": {
            "baseline_end": BASELINE_END,
            "perturbation_start": PERTURB_START,
            "perturbation_end": PERTURB_END,
            "recovery_start": RECOVERY_START,
        },
        "engine": {
            "baseline_window": BASELINE_WINDOW,
            "recent_window": RECENT_WINDOW,
        },
        "gal2_api_configured": bool(os.getenv("GAL2_API_KEY")),
        "gal2_time_url": os.getenv("GAL2_TIME_URL", "https://api-v2.gal-2.com/time"),
        "gal2_used_for_disturbed_time": bool(gal2_used_by_condition.get("disturbed_time", False)),
        "metrics": json.loads(df_metrics.to_json(orient="records")),
        "artifacts": {
            "timeseries_csv": OUTPUT_TIMESERIES_CSV,
            "node_summary_csv": OUTPUT_NODE_SUMMARY_CSV,
            "metrics_csv": OUTPUT_METRICS_CSV,
            "phase_confusion_csv": OUTPUT_PHASE_CONFUSION_CSV,
            "report_md": OUTPUT_REPORT_MD,
        },
    }
    Path(OUTPUT_JSON).write_text(json.dumps(out_json, indent=2), encoding="utf-8")

    # Markdown report
    lines: list[str] = []
    lines.append("# SII Architectural Benchmark Report")
    lines.append("")
    lines.append("## Executive Metrics")
    lines.append("")
    for _, row in df_metrics.iterrows():
        lines.append(f"### {row['condition']}")
        lines.append(f"- alert_precision: {row['alert_precision']:.6f}")
        lines.append(f"- alert_recall: {row['alert_recall']:.6f}")
        lines.append(f"- baseline_stability_rate: {row['baseline_stability_rate']:.6f}")
        lines.append(f"- d_vs_b_localization_distance: {row['d_vs_b_localization_distance']:.6f}")
        lines.append("")
    lines.append("## Artifacts")
    lines.append(f"- `{OUTPUT_JSON}`")
    lines.append(f"- `{OUTPUT_TIMESERIES_CSV}`")
    lines.append(f"- `{OUTPUT_NODE_SUMMARY_CSV}`")
    lines.append(f"- `{OUTPUT_METRICS_CSV}`")
    lines.append(f"- `{OUTPUT_PHASE_CONFUSION_CSV}`")
    Path(OUTPUT_REPORT_MD).write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_console(df_metrics: pd.DataFrame, df_node_summary: pd.DataFrame) -> None:
    print("\n" + "=" * 72)
    print("Architectural SII benchmark metrics")
    print("=" * 72)
    print(df_metrics.to_string(index=False))

    print("\n" + "=" * 72)
    print("Node summary")
    print("=" * 72)
    keep = [
        "condition",
        "node",
        "variant",
        "alert_count",
        "watch_count",
        "stable_count",
        "baseline_stability_rate",
        "perturb_alert_rate",
        "recovery_nominal_rate",
        "mean_localization",
    ]
    print(df_node_summary[keep].to_string(index=False))

    print("\n" + "=" * 72)
    print("Outputs written")
    print("=" * 72)
    print(f"- {OUTPUT_JSON}")
    print(f"- {OUTPUT_TIMESERIES_CSV}")
    print(f"- {OUTPUT_NODE_SUMMARY_CSV}")
    print(f"- {OUTPUT_METRICS_CSV}")
    print(f"- {OUTPUT_PHASE_CONFUSION_CSV}")
    print(f"- {OUTPUT_REPORT_MD}")


def main() -> int:
    df_rows, df_node_summary, df_metrics, df_phase_conf, gal2_used = run_benchmark()
    write_outputs(df_rows, df_node_summary, df_metrics, df_phase_conf, gal2_used)
    print_console(df_metrics, df_node_summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

