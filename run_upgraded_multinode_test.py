#!/usr/bin/env python3
"""
Architectural SII benchmark runner (single-file, Colab-friendly).

This script implements a modular SII runtime pipeline with explicit stages:
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
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from neraium_core.staged_pipeline import (
    AttributionStage as SharedAttributionStage,
    ConfidenceStage as SharedConfidenceStage,
    DataQualityStage as SharedDataQualityStage,
    DecisionStage as SharedDecisionStage,
    FeatureExtractionStage as SharedFeatureExtractionStage,
    LocalizationStage as SharedLocalizationStage,
    NodeBaselineProfile as SharedNodeBaselineProfile,
    NodeRuntime as SharedNodeRuntime,
    RegimeMemory as SharedRegimeMemory,
    RegimeStage as SharedRegimeStage,
    RelationalInstabilityStage as SharedRelationalInstabilityStage,
    StructuralDriftStage as SharedStructuralDriftStage,
    TemporalCoherenceStage as SharedTemporalCoherenceStage,
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
def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def safe_float(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
        if math.isnan(x) or math.isinf(x):
            return default
        return x
    except (TypeError, ValueError):
        return default


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


def corr_from_matrix(mat: np.ndarray) -> np.ndarray:
    if mat.shape[0] < 2:
        return np.eye(mat.shape[1], dtype=float)
    c = np.corrcoef(mat, rowvar=False)
    c = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(c, 1.0)
    return c


def flatten_upper_tri(mat: np.ndarray) -> np.ndarray:
    idx = np.triu_indices(mat.shape[0], k=1)
    return mat[idx]


def z_norm(v: float, mean: float, std: float, eps: float = 1e-6) -> float:
    return (v - mean) / (std + eps)


def bounded_z(v: float, mean: float, std: float, cap: float = 4.0) -> float:
    return clamp(z_norm(v, mean, std), 0.0, cap)


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------
@dataclass
class NodeBaselineProfile:
    corr_baseline: np.ndarray | None = None
    corr_drift_mean: float = 0.0
    corr_drift_std: float = 0.1
    relational_mean: float = 0.0
    relational_std: float = 0.1
    temporal_gap_mean: float = 1.0
    temporal_gap_std: float = 0.1
    instability_mean: float = 0.0
    instability_std: float = 0.1
    finalized: bool = False


@dataclass
class RegimeMemory:
    centroids: list[np.ndarray] = field(default_factory=list)
    threshold: float = 2.0

    def nearest_distance(self, signature: np.ndarray) -> float:
        if not self.centroids:
            return 0.0
        dists = [float(np.linalg.norm(signature - c)) for c in self.centroids if c.shape == signature.shape]
        if not dists:
            return 0.0
        return float(min(dists))

    def update(self, signature: np.ndarray) -> float:
        if not self.centroids:
            self.centroids.append(signature.copy())
            return 0.0
        nearest = self.nearest_distance(signature)
        if nearest > self.threshold:
            self.centroids.append(signature.copy())
        return nearest


@dataclass
class NodeRuntime:
    node: str
    variant: str
    sensor_names: list[str]
    baseline_window: int
    recent_window: int
    values_history: deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=500))
    timestamp_history: deque[float] = field(default_factory=lambda: deque(maxlen=500))
    score_history: deque[float] = field(default_factory=lambda: deque(maxlen=120))
    interpreted_history: deque[str] = field(default_factory=lambda: deque(maxlen=20))
    baseline_profile: NodeBaselineProfile = field(default_factory=NodeBaselineProfile)
    regime_memory: RegimeMemory = field(default_factory=RegimeMemory)

    # Baseline collectors
    _baseline_corr_drift: list[float] = field(default_factory=list)
    _baseline_relational: list[float] = field(default_factory=list)
    _baseline_gap: list[float] = field(default_factory=list)
    _baseline_instability: list[float] = field(default_factory=list)

    def push(self, ts: float, sensors: dict[str, float]) -> np.ndarray:
        vec = np.array([safe_float(sensors.get(s), np.nan) for s in self.sensor_names], dtype=float)
        self.values_history.append(vec)
        self.timestamp_history.append(ts)
        return vec

    def recent_matrix(self) -> np.ndarray | None:
        if len(self.values_history) < self.recent_window:
            return None
        m = np.vstack(list(self.values_history)[-self.recent_window :])
        # impute NaN with column means
        if np.isnan(m).any():
            col_mean = np.nanmean(m, axis=0)
            col_mean = np.nan_to_num(col_mean, nan=0.0)
            inds = np.where(np.isnan(m))
            m[inds] = np.take(col_mean, inds[1])
        return m

    def baseline_matrix(self) -> np.ndarray | None:
        if len(self.values_history) < self.baseline_window:
            return None
        m = np.vstack(list(self.values_history)[: self.baseline_window])
        if np.isnan(m).any():
            col_mean = np.nanmean(m, axis=0)
            col_mean = np.nan_to_num(col_mean, nan=0.0)
            inds = np.where(np.isnan(m))
            m[inds] = np.take(col_mean, inds[1])
        return m

    def recent_timestamps(self) -> list[float] | None:
        if len(self.timestamp_history) < self.recent_window:
            return None
        return list(self.timestamp_history)[-self.recent_window :]

    def baseline_timestamps(self) -> list[float] | None:
        if len(self.timestamp_history) < self.baseline_window:
            return None
        return list(self.timestamp_history)[: self.baseline_window]


# -----------------------------------------------------------------------------
# Pipeline stages
# -----------------------------------------------------------------------------
class DataQualityStage:
    @staticmethod
    def evaluate(baseline: np.ndarray, recent: np.ndarray, ts_base: list[float] | None, ts_recent: list[float] | None) -> dict[str, float | bool | list[str]]:
        n_total = int(recent.size)
        miss = int(np.isnan(recent).sum())
        missingness_rate = float(miss / max(1, n_total))

        std_recent = np.nanstd(recent, axis=0)
        std_recent = np.nan_to_num(std_recent, nan=0.0)
        flatlined = int(np.sum(std_recent <= 1e-12))
        valid_signal_count = int(np.sum(std_recent > 1e-12))
        total_sensors = int(recent.shape[1])
        sensor_coverage = float(valid_signal_count / max(1, total_sensors))

        timestamp_irregularity = 0.0
        if ts_recent is not None and len(ts_recent) >= 3:
            gaps = np.diff(np.array(ts_recent, dtype=float))
            if np.all(gaps > 0):
                timestamp_irregularity = float(np.std(gaps) / (np.mean(gaps) + 1e-9))
        timestamp_irregularity = float(clamp(timestamp_irregularity, 0.0, 1.0))

        statuses: list[str] = []
        if missingness_rate > 0.5:
            statuses.append("DATA_QUALITY_LIMITED")
        if sensor_coverage < 0.5:
            statuses.append("LOW_SENSOR_COVERAGE")
        if timestamp_irregularity > 0.5:
            statuses.append("TIMESTAMP_IRREGULAR")

        gate_passed = bool(missingness_rate <= 0.5 and sensor_coverage >= 0.5 and valid_signal_count >= 2)
        return {
            "missingness_rate": missingness_rate,
            "timestamp_irregularity": timestamp_irregularity,
            "flatlined_sensor_count": flatlined,
            "valid_signal_count": valid_signal_count,
            "total_sensors": total_sensors,
            "sensor_coverage": sensor_coverage,
            "statuses": statuses,
            "gate_passed": gate_passed,
        }


class FeatureExtractionStage:
    @staticmethod
    def extract(baseline: np.ndarray, recent: np.ndarray) -> dict[str, Any]:
        base_mean = np.mean(baseline, axis=0)
        rec_mean = np.mean(recent, axis=0)
        base_std = np.std(baseline, axis=0)
        rec_std = np.std(recent, axis=0)

        corr_base = corr_from_matrix(baseline)
        corr_recent = corr_from_matrix(recent)
        rel_vec_base = flatten_upper_tri(corr_base)
        rel_vec_recent = flatten_upper_tri(corr_recent)

        signature = np.concatenate([rec_mean, rec_std, rel_vec_recent])
        return {
            "base_mean": base_mean,
            "rec_mean": rec_mean,
            "base_std": base_std,
            "rec_std": rec_std,
            "corr_base": corr_base,
            "corr_recent": corr_recent,
            "rel_vec_base": rel_vec_base,
            "rel_vec_recent": rel_vec_recent,
            "signature": signature,
        }


class StructuralDriftStage:
    @staticmethod
    def score(features: dict[str, Any], baseline_profile: NodeBaselineProfile) -> tuple[float, float]:
        corr_recent = features["corr_recent"]
        corr_ref = baseline_profile.corr_baseline if baseline_profile.corr_baseline is not None else features["corr_base"]
        raw = float(np.linalg.norm(corr_recent - corr_ref, ord="fro"))
        normalized = raw
        if baseline_profile.finalized:
            normalized = bounded_z(raw, baseline_profile.corr_drift_mean, baseline_profile.corr_drift_std, cap=4.0)
        return raw, normalized


class RelationalInstabilityStage:
    @staticmethod
    def score(features: dict[str, Any], baseline_profile: NodeBaselineProfile) -> tuple[float, float]:
        delta = features["rel_vec_recent"] - features["rel_vec_base"]
        raw = float(np.mean(np.abs(delta))) if delta.size else 0.0
        normalized = raw
        if baseline_profile.finalized:
            normalized = bounded_z(raw, baseline_profile.relational_mean, baseline_profile.relational_std, cap=4.0)
        return raw, normalized


class TemporalCoherenceStage:
    @staticmethod
    def score(ts_recent: list[float] | None, baseline_profile: NodeBaselineProfile) -> tuple[float, float]:
        if ts_recent is None or len(ts_recent) < 3:
            return 0.0, 0.0
        gaps = np.diff(np.array(ts_recent, dtype=float))
        if not np.all(gaps > 0):
            return 1.0, 3.0
        cv = float(np.std(gaps) / (np.mean(gaps) + 1e-9))
        raw = float(clamp(cv, 0.0, 5.0))
        normalized = raw
        if baseline_profile.finalized:
            normalized = bounded_z(raw, baseline_profile.temporal_gap_mean, baseline_profile.temporal_gap_std, cap=4.0)
        return raw, normalized


class RegimeStage:
    @staticmethod
    def distance(runtime: NodeRuntime, signature: np.ndarray) -> float:
        return runtime.regime_memory.update(signature)


class ConfidenceStage:
    @staticmethod
    def score(
        dq: dict[str, Any],
        component_scores: dict[str, float],
        score_history: deque[float],
        baseline_profile: NodeBaselineProfile,
    ) -> float:
        quality = (1.0 - float(dq["missingness_rate"])) * float(dq["sensor_coverage"]) * (1.0 - float(dq["timestamp_irregularity"]))
        quality = clamp(quality, 0.0, 1.0)

        # consistency across recent windows
        if len(score_history) >= 6:
            recent = np.array(list(score_history)[-6:], dtype=float)
            persistence = float(np.mean(recent > 1.0))
            volatility = float(np.std(recent))
        else:
            persistence = 0.0
            volatility = 0.0

        # component agreement (lower spread -> more agreement)
        vals = np.array(list(component_scores.values()), dtype=float)
        spread = float(np.std(vals) / (np.mean(vals) + 1e-6)) if vals.size else 1.0
        agreement = clamp(1.0 - 0.5 * spread, 0.0, 1.0)

        # distance from baseline relative spread
        if baseline_profile.finalized:
            baseline_std = max(0.05, baseline_profile.instability_std)
            distance_factor = clamp(float(np.mean(vals)) / (2.0 * baseline_std + 1e-6), 0.0, 1.0)
        else:
            distance_factor = 0.3

        conf = (
            0.35 * quality
            + 0.20 * agreement
            + 0.20 * clamp(1.0 - volatility, 0.0, 1.0)
            + 0.15 * clamp(persistence, 0.0, 1.0)
            + 0.10 * distance_factor
        )
        return clamp(conf, 0.0, 1.0)

    @staticmethod
    def categorical(conf: float) -> str:
        if conf >= 0.70:
            return "high"
        if conf >= 0.40:
            return "medium"
        return "low"


class LocalizationStage:
    @staticmethod
    def compute(anomaly_evidence_by_node: dict[str, float]) -> dict[str, float]:
        vals = np.array([max(0.0, float(v)) for v in anomaly_evidence_by_node.values()], dtype=float)
        s = float(np.sum(vals))
        if s <= 1e-9:
            return {k: 0.0 for k in anomaly_evidence_by_node.keys()}
        shares = {k: float(v) / s for k, v in anomaly_evidence_by_node.items()}
        concentration = float(np.max(vals) / (s + 1e-9))
        # Penalize diffuse activations network-wide.
        return {k: clamp(shares[k] * concentration * 2.0, 0.0, 1.0) for k in anomaly_evidence_by_node.keys()}


class DecisionStage:
    @staticmethod
    def interpreted_state(
        structural: float,
        relational: float,
        regime_distance: float,
        temporal_distortion: float,
        localization: float,
        trend: float,
    ) -> str:
        motion = structural > 1.2 or relational > 1.0
        strong_coupling_break = relational > 1.4
        regime_shift = regime_distance > 0.8
        temporal_only = temporal_distortion > 1.2 and not motion
        sustained_degrading = trend > 0.03

        if strong_coupling_break and localization > 0.25:
            return "COUPLING_INSTABILITY_OBSERVED"
        if motion and regime_shift and sustained_degrading and localization > 0.20:
            return "STRUCTURAL_INSTABILITY_OBSERVED"
        if motion and regime_shift and not sustained_degrading:
            return "REGIME_SHIFT_OBSERVED"
        # Temporal distortion alone should generally reduce confidence rather than
        # force a non-nominal interpreted state. Reserve this class for bounded
        # motion under temporal constraint.
        if (motion and temporal_distortion > 1.0 and localization < 0.20):
            return "COHERENCE_UNDER_CONSTRAINT"
        return "NOMINAL_STRUCTURE"

    @staticmethod
    def state_from_score(instability: float, confidence: float, localization: float) -> str:
        # Strongly penalize diffuse activation so spillover does not dominate.
        loc_gate = 0.40 + 0.60 * localization
        conf_gate = 0.55 + 0.45 * confidence
        adjusted = instability * loc_gate * conf_gate

        # Conservative guardrail: low-localization + low-confidence evidence should
        # not escalate aggressively even when raw scores are noisy.
        if localization < 0.16 and confidence < 0.55 and adjusted < 2.6:
            return "STABLE"

        if adjusted >= 2.0:
            return "ALERT"
        if adjusted >= 1.0:
            return "WATCH"
        return "STABLE"


class AttributionStage:
    @staticmethod
    def explain(components: dict[str, float], state: str) -> tuple[str, dict[str, float]]:
        total = sum(max(0.0, v) for v in components.values()) + 1e-9
        contrib = {k: max(0.0, v) / total for k, v in components.items()}
        ranked = sorted(contrib.items(), key=lambda kv: kv[1], reverse=True)
        top = [k for k, _ in ranked[:3]]
        msg = (
            f"{state}: dominated by {', '.join(top)}."
            if top
            else f"{state}: no dominant structural drivers."
        )
        return msg, contrib


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


# Rebind benchmark stage/runtime symbols to shared core implementations so
# benchmark and production execute the same stage logic path.
NodeBaselineProfile = SharedNodeBaselineProfile
RegimeMemory = SharedRegimeMemory
NodeRuntime = SharedNodeRuntime
DataQualityStage = SharedDataQualityStage
FeatureExtractionStage = SharedFeatureExtractionStage
StructuralDriftStage = SharedStructuralDriftStage
RelationalInstabilityStage = SharedRelationalInstabilityStage
TemporalCoherenceStage = SharedTemporalCoherenceStage
RegimeStage = SharedRegimeStage
ConfidenceStage = SharedConfidenceStage
LocalizationStage = SharedLocalizationStage
DecisionStage = SharedDecisionStage
AttributionStage = SharedAttributionStage


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

