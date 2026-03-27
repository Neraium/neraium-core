import math
import os
from collections import deque
from typing import Dict, List, Optional

import numpy as np


def _engine_debug_enabled() -> bool:
    return os.environ.get("NERAIUM_DEBUG_ENGINE", "0").strip().lower() in {"1", "true", "yes", "on"}

from neraium_core.stat_geometry import StatisticalGeometryLayer


class StructuralEngine:
    """
    Runtime structural monitoring engine for Neraium.

    This engine:
    - converts normalized telemetry frames into ordered numeric vectors
    - builds a healthy baseline from early frames
    - computes Mahalanobis drift from the baseline manifold
    - computes covariance / relational drift between baseline and recent windows
    - combines both into a structural drift score
    - tracks drift velocity for early instability estimation
    - emits a pilot-friendly event/result object for each processed frame
    """

    def __init__(
        self,
        baseline_window: int = 24,
        recent_window: int = 8,
        max_frames: int = 500,
        mahal_weight: float = 0.65,
        cov_weight: float = 0.35,
        smoothing_window: int = 3,
        enable_vector_smoothing: bool = True,
    ):
        if baseline_window < 2:
            raise ValueError("baseline_window must be at least 2")
        if recent_window < 2:
            raise ValueError("recent_window must be at least 2")
        if max_frames < (baseline_window + recent_window):
            raise ValueError("max_frames is too small for configured windows")
        if smoothing_window < 1:
            raise ValueError("smoothing_window must be at least 1")

        total_weight = mahal_weight + cov_weight
        if total_weight <= 0:
            raise ValueError("mahal_weight + cov_weight must be > 0")

        # Normalize weights defensively.
        self.mahal_weight = mahal_weight / total_weight
        self.cov_weight = cov_weight / total_weight

        self.baseline_window = baseline_window
        self.recent_window = recent_window
        self.smoothing_window = smoothing_window
        self.enable_vector_smoothing = enable_vector_smoothing

        self.frames = deque(maxlen=max_frames)
        self.sensor_order: List[str] = []
        self.latest_result: Optional[Dict] = None
        self.prev_drift: Optional[float] = None
        self._transition_pressure_history: deque[float] = deque(maxlen=60)
        self._shock_activity_history: deque[float] = deque(maxlen=60)
        self._structural_drift_history: deque[float] = deque(maxlen=60)
        self._state_history: deque[str] = deque(maxlen=60)
        self._frame_count: int = 0
        self._drift_history: deque[float] = deque(maxlen=160)
        self._velocity_history: deque[float] = deque(maxlen=160)
        self._evidence_history: deque[float] = deque(maxlen=160)
        self._onset_score_history: deque[float] = deque(maxlen=160)
        self._onset_candidate_history: deque[bool] = deque(maxlen=40)
        self._onset_candidate_streak: int = 0
        self._watch_evidence_history: deque[bool] = deque(maxlen=40)
        self._drift_excess_history: deque[float] = deque(maxlen=80)
        self._cumulative_drift_excess: float = 0.0
        self._sensor_shift_history: deque[float] = deque(maxlen=120)
        self._sensor_shift_streak: int = 0
        self._metric_history: dict[str, deque[float]] = {
            "transition_pressure": deque(maxlen=160),
            "lock_in_score": deque(maxlen=160),
            "pre_instability_score": deque(maxlen=160),
            "branching_factor": deque(maxlen=160),
            "state_contraction_score": deque(maxlen=160),
            "local_volume": deque(maxlen=160),
            "curvature": deque(maxlen=160),
            "directional_consistency": deque(maxlen=160),
            "recovery_margin": deque(maxlen=160),
        }
        self._entity_baseline_stability: deque[float] = deque(maxlen=240)
        self._early_warning_state_history: deque[str] = deque(maxlen=48)
        self.geometry_layer = StatisticalGeometryLayer(max_history=384, graph_window=24, stats_window=16)
        self._latest_geometry: Dict = {"available": False, "reason": "insufficient history"}
        self._latest_state_space_statistics: Dict = {"available": False, "reason": "insufficient history"}
        self._latest_state_graph: Dict = {"available": False, "reason": "insufficient history"}
        self._geometry_debug_frames_logged: int = 0
        self._early_warning_debug_frames_logged: int = 0

    def _vector_from_frame(self, frame: Dict) -> np.ndarray:
        """
        Convert the frame's sensor dictionary into a consistent ordered vector.
        Missing or invalid values are represented as NaN and handled downstream.
        """
        sensor_values = frame.get("sensor_values", {})

        if not isinstance(sensor_values, dict):
            raise ValueError("frame['sensor_values'] must be a dictionary")

        if not self.sensor_order:
            self.sensor_order = sorted(sensor_values.keys())

        values: List[float] = []

        for name in self.sensor_order:
            value = sensor_values.get(name)
            try:
                values.append(float(value))
            except (TypeError, ValueError):
                values.append(np.nan)

        return np.array(values, dtype=float)

    def _valid_matrix(self, frames: List[Dict]) -> Optional[np.ndarray]:
        """
        Build a matrix from stored frame vectors, dropping rows with any NaN values.
        """
        if not frames:
            return None

        X = np.vstack([f["_vector"] for f in frames])

        # Drop incomplete rows so missing telemetry does not become fake zeros.
        row_mask = ~np.isnan(X).any(axis=1)
        X = X[row_mask]

        if len(X) < 2:
            return None

        return X

    def _smooth_matrix(self, X: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """
        Apply light rolling smoothing across time for each sensor column.
        """
        if X is None or len(X) < 2 or self.smoothing_window <= 1:
            return X

        smoothed = X.copy()
        window = min(self.smoothing_window, len(X))
        kernel = np.ones(window, dtype=float) / float(window)

        for col in range(smoothed.shape[1]):
            smoothed[:, col] = np.convolve(smoothed[:, col], kernel, mode="same")

        return smoothed

    def _smooth_current_vector(self, vector: np.ndarray) -> np.ndarray:
        """
        Very light online smoothing against the previous vector to reduce spikes.
        """
        if not self.enable_vector_smoothing:
            return vector

        if len(self.frames) == 0:
            return vector

        prev = self.frames[-1]["_vector"]

        if prev.shape != vector.shape:
            return vector

        if np.isnan(prev).any() or np.isnan(vector).any():
            return vector

        return 0.5 * (vector + prev)

    def _regularize_covariance(self, cov: np.ndarray) -> np.ndarray:
        """
        Stabilize covariance via symmetry enforcement and eigenvalue flooring.
        """
        if cov.ndim == 0:
            cov = np.array([[float(cov)]], dtype=float)

        cov = 0.5 * (cov + cov.T)

        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.maximum(eigvals, 1e-6)

        return eigvecs @ np.diag(eigvals) @ eigvecs.T

    def _baseline_stats(self) -> Optional[Dict]:
        """
        Compute mean/covariance statistics from the baseline window.
        """
        if len(self.frames) < max(6, self.baseline_window):
            return None

        baseline_frames = list(self.frames)[: self.baseline_window]
        X = self._valid_matrix(baseline_frames)
        X = self._smooth_matrix(X)

        if X is None or len(X) < 2:
            return None

        mean = np.mean(X, axis=0)
        cov = np.cov(X, rowvar=False)
        cov = self._regularize_covariance(cov)
        inv_cov = np.linalg.inv(cov)

        return {
            "mean": mean,
            "cov": cov,
            "inv_cov": inv_cov,
        }

    def _mahalanobis(self, x: np.ndarray, mean: np.ndarray, inv_cov: np.ndarray) -> float:
        """
        Mahalanobis distance from current state to baseline manifold.
        """
        delta = x - mean
        raw = float(delta.T @ inv_cov @ delta)
        raw = max(raw, 0.0)
        return float(math.sqrt(raw))

    def _covariance_drift(self) -> float:
        """
        Measure relational / structural drift between baseline and recent windows.
        """
        if len(self.frames) < max(self.baseline_window + self.recent_window, 10):
            return 0.0

        baseline_frames = list(self.frames)[: self.baseline_window]
        recent_frames = list(self.frames)[-self.recent_window :]

        Xb = self._valid_matrix(baseline_frames)
        Xr = self._valid_matrix(recent_frames)

        Xb = self._smooth_matrix(Xb)
        Xr = self._smooth_matrix(Xr)

        if Xb is None or Xr is None:
            return 0.0

        cov_b = np.cov(Xb, rowvar=False)
        cov_r = np.cov(Xr, rowvar=False)

        cov_b = self._regularize_covariance(cov_b)
        cov_r = self._regularize_covariance(cov_r)

        return float(np.linalg.norm(cov_r - cov_b, ord="fro"))

    def _relational_stability(self, cov_drift: float) -> float:
        """
        Convert covariance drift into a bounded stability score.
        Higher drift means lower stability.
        """
        score = 1.0 / (1.0 + cov_drift)
        return max(0.0, min(1.0, float(score)))

    def _combined_drift(self, mahal: float, cov_drift: float) -> float:
        """
        Fuse manifold departure and structural deformation into one drift score.
        """
        return (self.mahal_weight * mahal) + (self.cov_weight * cov_drift)

    def _system_health(self, drift_score: float, stability_score: float) -> int:
        """
        Convert drift + stability into an operator-facing 0-100 health score.
        """
        health = 100.0 - min(drift_score * 18.0, 80.0)
        health += stability_score * 20.0
        health = max(0.0, min(100.0, health))
        return int(round(health))

    def _alert_state(self, drift_score: float) -> str:
        """
        Convert drift score into a simple alert state.
        """
        hist = list(self._drift_history)
        if len(hist) >= max(12, self.recent_window * 2):
            baseline = np.asarray(hist[:-3] if len(hist) > 9 else hist, dtype=float)
            watch_thr = float(np.quantile(baseline, 0.82))
            alert_thr = float(np.quantile(baseline, 0.93))
            watch_thr = max(1.1, watch_thr + 0.08)
            alert_thr = max(watch_thr + 0.18, alert_thr + 0.12)
        else:
            watch_thr, alert_thr = 1.8, 3.2
        if drift_score > alert_thr:
            return "ALERT"
        if drift_score > watch_thr:
            return "WATCH"
        return "STABLE"

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    @staticmethod
    def _trend(values: List[float]) -> float:
        if len(values) < 2:
            return 0.0
        return float(values[-1] - values[0]) / float(len(values) - 1)

    @staticmethod
    def _safe_float(value: object, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def _history_tail(self, key: str, size: int = 8) -> List[float]:
        hist = list(self._metric_history.get(key, deque()))
        return [float(v) for v in hist[-min(size, len(hist)) :]] if hist else []

    def _delta_from_history(self, key: str) -> float:
        tail = self._history_tail(key, size=2)
        if len(tail) < 2:
            return 0.0
        return float(tail[-1] - tail[-2])

    def _persistent_worsening(self, values: List[float], worsen_if_high: bool = True) -> float:
        if len(values) < 3:
            return 0.0
        diffs = [values[i] - values[i - 1] for i in range(1, len(values))]
        if not worsen_if_high:
            diffs = [-d for d in diffs]
        positive = [d for d in diffs if d > 0.0]
        ratio = float(len(positive)) / float(max(1, len(diffs)))
        avg = float(np.mean(positive)) if positive else 0.0
        return self._clamp01(0.65 * ratio + 0.35 * self._clamp01(avg / 0.05))

    def _compute_early_warning(
        self,
        *,
        drift_score: float,
        velocity: float,
        transition: Dict,
        geometry: Optional[Dict],
        state_space_statistics: Optional[Dict],
        state_graph: Optional[Dict],
        vector: np.ndarray,
    ) -> Dict:
        geometry = geometry or {}
        state_space_statistics = state_space_statistics or {}
        state_graph = state_graph or {}
        pressure = self._safe_float(transition.get("transition_pressure", 0.0))
        lock_in = self._safe_float(
            (self.latest_result or {}).get("experimental_analytics", {}).get("constraint_analysis", {}).get("lock_in_score", 0.0)
        )
        recovery_margin = self._clamp01(1.0 - lock_in)
        branching_factor = self._safe_float(state_graph.get("branching_factor", 0.0))
        contraction = self._safe_float(state_space_statistics.get("state_contraction_score", 0.0))
        local_volume = self._safe_float(state_space_statistics.get("local_volume", 0.0))
        curvature = self._safe_float(geometry.get("curvature", 0.0))
        directional_consistency = self._clamp01(self._safe_float(geometry.get("directional_consistency", 1.0), 1.0))

        self._metric_history["transition_pressure"].append(pressure)
        self._metric_history["lock_in_score"].append(lock_in)
        self._metric_history["branching_factor"].append(branching_factor)
        self._metric_history["state_contraction_score"].append(contraction)
        self._metric_history["local_volume"].append(local_volume)
        self._metric_history["curvature"].append(curvature)
        self._metric_history["directional_consistency"].append(directional_consistency)
        self._metric_history["recovery_margin"].append(recovery_margin)

        # Entity-relative baseline (only from historically stable frames).
        if drift_score <= 1.4 and pressure <= 0.46 and lock_in <= 0.34:
            self._entity_baseline_stability.append(float(drift_score))
        baseline_center = float(np.median(self._entity_baseline_stability)) if self._entity_baseline_stability else max(0.2, drift_score)
        baseline_spread = float(np.std(np.asarray(self._entity_baseline_stability, dtype=float))) if len(self._entity_baseline_stability) >= 6 else 0.12
        baseline_spread = max(0.08, baseline_spread)
        entity_relative_deviation = self._clamp01((drift_score - baseline_center) / (2.9 * baseline_spread + 1e-6))

        # Geometry-informed early strain.
        recent_vectors = [f.get("_vector") for f in list(self.frames)[-14:] if isinstance(f.get("_vector"), np.ndarray)]
        direction_variance = 0.0
        smoothness_instability = 0.0
        local_curvature_pre_drift = 0.0
        if len(recent_vectors) >= 5 and not np.isnan(vector).any():
            mat = np.vstack([np.asarray(v, dtype=float) for v in recent_vectors if not np.isnan(v).any()])
            if mat.shape[0] >= 5:
                deltas = np.diff(mat, axis=0)
                norms = np.linalg.norm(deltas, axis=1)
                valid = norms > 1e-9
                if np.any(valid):
                    unit = deltas[valid] / norms[valid][:, None]
                    direction_variance = self._clamp01(float(np.mean(np.var(unit, axis=0))) * 3.0)
                if norms.size >= 3:
                    smoothness_instability = self._clamp01(float(np.std(norms) / (np.mean(norms) + 1e-6)))
                accel = np.diff(deltas, axis=0) if deltas.shape[0] >= 2 else np.zeros((0, deltas.shape[1]))
                if accel.size > 0:
                    acc_norm = np.linalg.norm(accel, axis=1)
                    vel_norm = np.linalg.norm(deltas[1:], axis=1) + 1e-9
                    local_curvature_pre_drift = self._clamp01(float(np.mean(acc_norm / vel_norm)) * 0.35)

        pressure_tail = self._history_tail("transition_pressure", size=8)
        lock_tail = self._history_tail("lock_in_score", size=8)
        contraction_tail = self._history_tail("state_contraction_score", size=8)
        recovery_tail = self._history_tail("recovery_margin", size=8)
        branching_tail = self._history_tail("branching_factor", size=8)
        directional_tail = self._history_tail("directional_consistency", size=8)
        volume_tail = self._history_tail("local_volume", size=8)
        curvature_tail = self._history_tail("curvature", size=8)

        pressure_persist = self._persistent_worsening(pressure_tail, worsen_if_high=True)
        lock_persist = self._persistent_worsening(lock_tail, worsen_if_high=True)
        contraction_persist = self._persistent_worsening(contraction_tail, worsen_if_high=True)
        recovery_fall_persist = self._persistent_worsening(recovery_tail, worsen_if_high=False)
        directional_fall_persist = self._persistent_worsening(directional_tail, worsen_if_high=False)
        curvature_persist = self._persistent_worsening(curvature_tail, worsen_if_high=True)
        volume_regime_shift = self._clamp01(abs(self._trend(volume_tail)) / 0.04) if len(volume_tail) >= 3 else 0.0
        branch_pressure = self._clamp01(max(0.0, self._trend(branching_tail)) / 0.08) if len(branching_tail) >= 3 else 0.0

        short_hint = self._clamp01(
            0.21 * direction_variance
            + 0.16 * local_curvature_pre_drift
            + 0.15 * smoothness_instability
            + 0.14 * self._clamp01(1.0 - directional_consistency)
            + 0.16 * self._clamp01(curvature)
            + 0.18 * entity_relative_deviation
        )
        mid_confirmation = self._clamp01(
            0.22 * pressure_persist
            + 0.18 * contraction_persist
            + 0.16 * recovery_fall_persist
            + 0.14 * directional_fall_persist
            + 0.12 * curvature_persist
            + 0.10 * branch_pressure
            + 0.08 * volume_regime_shift
        )
        long_commitment = self._clamp01(
            0.24 * self._clamp01(pressure)
            + 0.20 * self._clamp01(lock_in)
            + 0.14 * self._clamp01(contraction)
            + 0.14 * self._clamp01(1.0 - recovery_margin)
            + 0.12 * self._clamp01(abs(velocity) / 0.2)
            + 0.16 * self._clamp01(drift_score / 4.5)
        )

        stability_erosion_score = self._clamp01(0.48 * short_hint + 0.52 * mid_confirmation)
        coherence_breakdown_score = self._clamp01(
            0.30 * directional_fall_persist
            + 0.24 * direction_variance
            + 0.18 * branch_pressure
            + 0.14 * volume_regime_shift
            + 0.14 * self._clamp01(1.0 - smoothness_instability)
        )
        pre_instability_score = self._clamp01(
            0.36 * stability_erosion_score
            + 0.27 * coherence_breakdown_score
            + 0.22 * short_hint
            + 0.15 * long_commitment
        )
        pre_commitment_score = self._clamp01(0.55 * mid_confirmation + 0.45 * long_commitment)
        structural_strain_score = self._clamp01(0.5 * direction_variance + 0.5 * local_curvature_pre_drift)
        deviation_acceleration_score = self._clamp01(max(0.0, self._delta_from_history("transition_pressure")) / 0.08)
        self._metric_history["pre_instability_score"].append(pre_instability_score)

        reasons: List[str] = []
        if stability_erosion_score >= 0.44:
            reasons.append("stability_erosion_persistent")
        if coherence_breakdown_score >= 0.42:
            reasons.append("coherence_breakdown_signals")
        if pressure_persist >= 0.5:
            reasons.append("transition_pressure_persistence")
        if recovery_fall_persist >= 0.45:
            reasons.append("recovery_margin_shrinking")
        if directional_fall_persist >= 0.45:
            reasons.append("directional_coherence_loss")
        if branch_pressure >= 0.42:
            reasons.append("branching_tension_rising")
        if not reasons:
            reasons.append("no_persistent_early_instability")

        if pre_instability_score >= 0.83 and pre_commitment_score >= 0.62:
            ew_state = "alert"
        elif pre_instability_score >= 0.58 and (mid_confirmation >= 0.42 or short_hint >= 0.55):
            ew_state = "emerging_instability"
        elif pre_instability_score >= 0.38 and stability_erosion_score >= 0.34:
            ew_state = "pre_instability"
        else:
            ew_state = "stable"
        self._early_warning_state_history.append(ew_state)

        return {
            "pre_instability_score": round(pre_instability_score, 4),
            "stability_erosion_score": round(stability_erosion_score, 4),
            "coherence_breakdown_score": round(coherence_breakdown_score, 4),
            "structural_strain_score": round(structural_strain_score, 4),
            "pre_commitment_score": round(pre_commitment_score, 4),
            "deviation_acceleration_score": round(deviation_acceleration_score, 4),
            "early_warning_state": ew_state,
            "early_warning_reasons": reasons[:6],
            "short_window_hint": round(short_hint, 4),
            "mid_window_confirmation": round(mid_confirmation, 4),
            "long_window_commitment": round(long_commitment, 4),
            "entity_relative_deviation": round(entity_relative_deviation, 4),
        }

    def _update_histories(self, *, drift_score: float, velocity: float, state: str, pressure: float) -> None:
        shock_activity = self._clamp01(abs(velocity) / 0.35)
        self._transition_pressure_history.append(float(pressure))
        self._shock_activity_history.append(float(shock_activity))
        self._structural_drift_history.append(float(drift_score))
        self._state_history.append(state)
        self._drift_history.append(float(drift_score))
        self._velocity_history.append(float(velocity))

    def _compute_base_metrics(
        self,
        frame: Dict,
        baseline: Optional[Dict],
        vector: np.ndarray,
    ) -> Dict:
        base = {
            "id": None,
            "event_type": "baseline_telemetry",
            "timestamp": frame["timestamp"],
            "site_id": frame["site_id"],
            "asset_id": frame["asset_id"],
            "state": "STABLE",
            "structural_drift_score": 0.0,
            "relational_stability_score": 1.0,
            "system_health": 100,
            "drift_alert": False,
            "lead_time_hours": None,
            "lead_time_confidence": 0.0,
            "drift_velocity": 0.0,
            "structural_driver": "baseline formation",
            "predicted_impact": "No near term operational disruption expected.",
            "explanation": "Initializing structural telemetry...",
            "mahalanobis_score": 0.0,
            "covariance_drift_score": 0.0,
            "latest_instability": 0.0,
            "confidence_score": 0.25,
            "interpreted_state": "BASELINE_FORMING",
            "regime_name": "INITIALIZING",
        }

        if baseline is None or np.isnan(vector).any():
            return base

        mahal = self._mahalanobis(vector, baseline["mean"], baseline["inv_cov"])
        cov_drift = self._covariance_drift()
        drift_score = self._combined_drift(mahal, cov_drift)
        stability_score = self._relational_stability(cov_drift)
        health = self._system_health(drift_score, stability_score)
        state = self._alert_state(drift_score)
        drift_alert = drift_score > 1.5

        velocity = 0.0 if self.prev_drift is None else (drift_score - self.prev_drift)
        self.prev_drift = drift_score

        lead_time_hours = None
        if velocity > 0.01:
            remaining = max(0.0, 4.5 - drift_score)
            lead_time_hours = round(min(240.0, remaining / velocity), 1)

        if state == "ALERT":
            event_type = "instability_escalation"
            impact = "Potential localized service disruption within 1 to 2 hours."
            driver = "cross-sensor structural divergence"
        elif state == "WATCH":
            event_type = "quality_observation"
            impact = "Early degradation detected. Maintenance window recommended."
            driver = "emerging relational drift"
        else:
            event_type = "flow_observation"
            impact = "No near term operational disruption expected."
            driver = "stable baseline telemetry"

        interpreted_state = "STABLE_FLOW"
        regime_name = "NOMINAL"
        if state == "WATCH":
            interpreted_state = "EMERGING_TRANSITION"
            regime_name = "DEGRADING"
        elif state == "ALERT":
            interpreted_state = "SUSTAINED_TRANSITION"
            regime_name = "INSTABILITY_REGIME"

        return {
            **base,
            "event_type": event_type,
            "state": state,
            "structural_drift_score": round(drift_score, 4),
            "relational_stability_score": round(stability_score, 4),
            "system_health": health,
            "drift_alert": drift_alert,
            "lead_time_hours": lead_time_hours,
            "lead_time_confidence": round(min(0.99, max(0.35, stability_score)), 2),
            "drift_velocity": round(velocity, 4),
            "structural_driver": driver,
            "predicted_impact": impact,
            "explanation": "Structural engine monitoring sensor relationships in real time.",
            "mahalanobis_score": round(mahal, 4),
            "covariance_drift_score": round(cov_drift, 4),
            "latest_instability": round(self._clamp01(drift_score / 4.5), 4),
            "confidence_score": round(self._clamp01(0.2 + 0.8 * stability_score), 4),
            "interpreted_state": interpreted_state,
            "regime_name": regime_name,
        }

    def _compute_transition_metrics(
        self,
        *,
        drift_score: float,
        velocity: float,
        stability_score: float,
        geometry: Optional[Dict] = None,
        state_space_statistics: Optional[Dict] = None,
        state_graph: Optional[Dict] = None,
    ) -> Dict:
        drift_n = self._clamp01(drift_score / 4.5)
        vel_n = self._clamp01(max(0.0, velocity) / 0.22)
        acc_n = 0.0
        if len(self._velocity_history) >= 2:
            acc_n = self._clamp01(max(0.0, velocity - self._velocity_history[-1]) / 0.12)
        disagreement = self._clamp01(abs(drift_n - vel_n))
        persistence = self._clamp01(
            float(np.mean([self._clamp01(v / 4.5) for v in list(self._drift_history)[-5:]]))
            if self._drift_history
            else 0.0
        )
        noise_sep = self._clamp01(max(0.0, (1.0 - stability_score) - 0.5 * self._clamp01(abs(velocity) / 0.3)))
        trend_alignment = self._clamp01(
            1.0 - min(1.0, abs(self._trend(list(self._drift_history)[-6:]) - self._trend(list(self._transition_pressure_history)[-6:])) / 0.18)
        ) if len(self._drift_history) >= 3 and len(self._transition_pressure_history) >= 3 else 0.5
        geometry_available = bool(isinstance(geometry, dict) and geometry.get("available", True) is not False)
        state_stats_available = bool(
            isinstance(state_space_statistics, dict) and state_space_statistics.get("available", True) is not False
        )
        state_graph_available = bool(isinstance(state_graph, dict) and state_graph.get("available", True) is not False)
        geom_curvature = self._clamp01(float((geometry or {}).get("curvature", 0.0))) if geometry_available else 0.0
        geom_direction = (
            self._clamp01(float((geometry or {}).get("directional_consistency", 0.0)))
            if geometry_available
            else 0.5
        )
        stat_local_volume = (
            self._clamp01(float((state_space_statistics or {}).get("local_volume", 0.0)))
            if state_stats_available
            else 0.0
        )
        graph_branching = (
            self._clamp01(float((state_graph or {}).get("branching_factor", 0.0)))
            if state_graph_available
            else 0.0
        )
        graph_entropy = (
            self._clamp01(float((state_graph or {}).get("transition_entropy", 0.0)))
            if state_graph_available
            else 0.0
        )
        evidence = self._clamp01(
            0.21 * persistence
            + 0.18 * drift_n
            + 0.16 * vel_n
            + 0.13 * acc_n
            + 0.09 * disagreement
            + 0.08 * noise_sep
            + 0.10 * trend_alignment
            + 0.03 * geom_curvature
            + 0.01 * (1.0 - geom_direction)
            + 0.01 * stat_local_volume
        )
        prior = list(self._evidence_history)
        if len(prior) >= 10:
            base = np.asarray(prior[:-4], dtype=float) if len(prior) > 14 else np.asarray(prior, dtype=float)
            threshold = float(np.quantile(base, 0.8))
            contrast = self._clamp01((evidence - threshold) / max(0.08, 2.0 * float(np.std(base))))
        else:
            contrast = 0.5 * evidence
        pressure = self._clamp01(0.52 * evidence + 0.42 * contrast + 0.04 * graph_branching + 0.02 * graph_entropy)
        self._evidence_history.append(float(evidence))
        recent_evidence = list(self._evidence_history)[-5:]
        evidence_persistence = self._clamp01(float(sum(1 for e in recent_evidence if e >= 0.55)) / max(1, len(recent_evidence)))
        if pressure >= 0.78 and evidence_persistence >= 0.45:
            transition_state = "SUSTAINED_TRANSITION"
        elif pressure >= 0.48 and evidence_persistence >= 0.32:
            transition_state = "EMERGING_TRANSITION"
        elif pressure >= 0.2:
            transition_state = "METASTABLE"
        else:
            transition_state = "NONE"
        return {"transition_pressure": round(pressure, 4), "transition_state": transition_state}

    def _rolling_slope(self, values: List[float]) -> float:
        if len(values) < 3:
            return 0.0
        x = np.arange(len(values), dtype=float)
        y = np.asarray(values, dtype=float)
        x_centered = x - x.mean()
        denom = float(np.sum(x_centered**2))
        if denom <= 1e-8:
            return 0.0
        return float(np.sum(x_centered * (y - y.mean())) / denom)

    def _baseline_reference(self) -> tuple[float, float]:
        drift_hist = list(self._drift_history)
        pressure_hist = list(self._transition_pressure_history)
        if not drift_hist:
            return 0.0, 0.1

        stable_pool = [
            float(d)
            for d, p in zip(drift_hist[-50:], pressure_hist[-50:], strict=False)
            if float(p) <= 0.45
        ]
        if len(stable_pool) < 6:
            tail = np.asarray(drift_hist[-50:], dtype=float)
            cutoff = float(np.quantile(tail, 0.55))
            stable_pool = [float(v) for v in tail if float(v) <= cutoff]
        if not stable_pool:
            stable_pool = [float(v) for v in drift_hist[-12:]]

        ref = np.asarray(stable_pool, dtype=float)
        center = float(np.median(ref))
        mad = float(np.median(np.abs(ref - center)))
        spread = max(0.08, 1.4826 * mad)
        return center, spread

    def _state_from_onset_evidence(
        self,
        *,
        drift_score: float,
        velocity: float,
        stability_score: float,
        transition_pressure: float,
        vector: np.ndarray,
        geometry: Optional[Dict] = None,
        state_space_statistics: Optional[Dict] = None,
        state_graph: Optional[Dict] = None,
    ) -> str:
        center, spread = self._baseline_reference()
        drift_dev = self._clamp01((float(drift_score) - center) / (3.5 * spread + 1e-6))
        drift_excess = max(0.0, float(drift_score) - (center + 1.25 * spread))
        self._cumulative_drift_excess = 0.9 * self._cumulative_drift_excess + drift_excess
        self._drift_excess_history.append(float(drift_excess))

        drift_tail = list(self._drift_history)[-8:] + [float(drift_score)]
        pressure_tail = list(self._transition_pressure_history)[-8:] + [float(transition_pressure)]
        velocity_tail = list(self._velocity_history)[-8:] + [float(velocity)]

        drift_slope_short = self._rolling_slope(drift_tail[-4:])
        drift_slope_mid = self._rolling_slope(drift_tail[-8:])
        pressure_slope = self._rolling_slope(pressure_tail[-6:])

        directional_consistency = self._clamp01(
            float(sum(1 for v in velocity_tail[-6:] if v >= 0.0)) / max(1, len(velocity_tail[-6:]))
        )
        drift_persistence = self._clamp01(float(sum(1 for d in drift_tail[-6:] if d > center + 1.1 * spread)) / 6.0)
        pressure_persistence = self._clamp01(float(sum(1 for p in pressure_tail[-6:] if p >= 0.5)) / 6.0)
        low_freq_alignment = self._clamp01(
            0.55 * self._clamp01(max(0.0, drift_slope_short) / 0.11)
            + 0.45 * self._clamp01(max(0.0, drift_slope_mid) / 0.08)
        )
        smoothness = self._clamp01(1.0 - float(np.std(np.diff(np.asarray(drift_tail[-7:], dtype=float)))) / 0.22)
        instability = self._clamp01(1.0 - float(stability_score))
        agreement = self._clamp01(
            0.4 * (1.0 if drift_slope_short > 0.0 and drift_slope_mid > 0.0 else 0.0)
            + 0.3 * (1.0 if pressure_slope > -0.003 else 0.0)
            + 0.3 * directional_consistency
        )
        geometry_available = bool(isinstance(geometry, dict) and geometry.get("available", True) is not False)
        state_stats_available = bool(
            isinstance(state_space_statistics, dict) and state_space_statistics.get("available", True) is not False
        )
        state_graph_available = bool(isinstance(state_graph, dict) and state_graph.get("available", True) is not False)
        geom_curvature = self._clamp01(float((geometry or {}).get("curvature", 0.0))) if geometry_available else 0.0
        geom_direction = (
            self._clamp01(float((geometry or {}).get("directional_consistency", 0.0)))
            if geometry_available
            else directional_consistency
        )
        state_contraction = (
            self._clamp01(float((state_space_statistics or {}).get("state_contraction_score", 0.0)))
            if state_stats_available
            else 0.0
        )
        graph_commitment = (
            self._clamp01(float((state_graph or {}).get("path_commitment_score", 0.0)))
            if state_graph_available
            else 0.0
        )

        onset_score = self._clamp01(
            0.21 * drift_dev
            + 0.16 * drift_persistence
            + 0.15 * pressure_persistence
            + 0.11 * low_freq_alignment
            + 0.11 * smoothness
            + 0.11 * instability
            + 0.08 * agreement
            + 0.04 * geom_curvature
            + 0.02 * (1.0 - geom_direction)
            + 0.01 * state_contraction
            + 0.01 * graph_commitment
        )

        prior_scores = list(self._onset_score_history)
        if len(prior_scores) >= 12:
            score_ref = np.asarray(prior_scores[-60:], dtype=float)
            onset_thr = float(np.quantile(score_ref, 0.76) + 0.025)
            watch_thr = float(np.quantile(score_ref, 0.68) + 0.02)
        else:
            onset_thr = 0.5
            watch_thr = 0.44

        alert_candidate = bool(
            onset_score >= onset_thr
            and drift_dev >= 0.35
            and drift_persistence >= 0.42
            and pressure_persistence >= 0.1
            and low_freq_alignment >= 0.34
            and directional_consistency >= 0.55
            and agreement >= 0.45
            and smoothness >= 0.26
        )

        self._onset_candidate_streak = self._onset_candidate_streak + 1 if alert_candidate else max(0, self._onset_candidate_streak - 1)
        self._onset_candidate_history.append(alert_candidate)
        self._onset_score_history.append(float(onset_score))
        watch_evidence = bool(
            onset_score >= watch_thr
            and drift_persistence >= 0.38
            and directional_consistency >= 0.5
            and smoothness >= 0.24
        )
        self._watch_evidence_history.append(watch_evidence)
        recent_candidates = list(self._onset_candidate_history)[-3:]
        sustained_watch = sum(1 for flag in list(self._watch_evidence_history)[-5:] if flag) >= 4

        sensor_shift = 0.0
        frame_vectors = [f.get("_vector") for f in list(self.frames)[-26:-1]]
        valid_vectors = [v for v in frame_vectors if isinstance(v, np.ndarray) and v.shape == vector.shape and not np.isnan(v).any()]
        if len(valid_vectors) >= 8 and not np.isnan(vector).any():
            hist = np.vstack(valid_vectors)
            med = np.median(hist, axis=0)
            mad = np.median(np.abs(hist - med), axis=0)
            z = np.abs(vector - med) / (mad + 1e-3)
            sensor_shift = float(np.mean(np.sort(z)[-min(6, len(z)):]))
        prior_shift = np.asarray(list(self._sensor_shift_history)[-25:], dtype=float)
        if prior_shift.size >= 8:
            shift_thr = float(np.quantile(prior_shift, 0.9) + 0.4 * np.std(prior_shift))
        else:
            shift_thr = 999.0
        sensor_break = sensor_shift >= shift_thr
        self._sensor_shift_streak = self._sensor_shift_streak + 1 if sensor_break else 0
        self._sensor_shift_history.append(float(sensor_shift))

        excess_tail = np.asarray(list(self._drift_excess_history)[-14:], dtype=float)
        if excess_tail.size >= 6:
            energy_threshold = float(
                52.0
                + 9.0 * float(np.mean(excess_tail))
                + 12.0 * float(np.std(excess_tail))
            )
        else:
            energy_threshold = 80.0
        energy_confirmed = bool(
            self._cumulative_drift_excess >= energy_threshold
            and sustained_watch
            and directional_consistency >= 0.5
        )
        confirmed_alert = (
            (self._onset_candidate_streak >= 2 and sum(1 for flag in recent_candidates if flag) >= 2)
            or energy_confirmed
            or (self._sensor_shift_streak >= 2 and drift_persistence >= 0.3 and directional_consistency >= 0.45)
        )

        if confirmed_alert:
            return "ALERT"
        if onset_score >= watch_thr and drift_persistence >= 0.34 and directional_consistency >= 0.5:
            return "WATCH"
        return "STABLE"

    def _analytics_unavailable(self, reason: str) -> Dict:
        return {"available": False, "reason": reason}

    def _compute_geometry_payload(self, frame: Dict, vector: np.ndarray) -> Dict:
        valid_vectors = [
            f.get("_vector")
            for f in list(self.frames)[-max(self.baseline_window + self.recent_window, 48) :]
            if isinstance(f.get("_vector"), np.ndarray)
            and f.get("_vector").shape == vector.shape
            and not np.isnan(f.get("_vector")).any()
        ]
        if len(valid_vectors) < 2:
            return {
                "geometry": {"available": False, "reason": "insufficient history"},
                "state_space_statistics": {"available": False, "reason": "insufficient history"},
                "state_graph": {"available": False, "reason": "insufficient history"},
            }
        matrix = np.vstack(valid_vectors)
        payload = self.geometry_layer.update(
            entity_id=str(frame.get("asset_id", "unknown")),
            matrix=matrix,
            representation_mode="combined",
        )
        if not isinstance(payload, dict):
            payload = {}
        payload.setdefault("geometry", {"available": False, "reason": "insufficient history"})
        payload.setdefault("state_space_statistics", {"available": False, "reason": "insufficient history"})
        payload.setdefault("state_graph", {"available": False, "reason": "insufficient history"})
        return payload

    def _compute_advanced_analytics(
        self,
        *,
        geometry: Optional[Dict] = None,
        state_space_statistics: Optional[Dict] = None,
        state_graph: Optional[Dict] = None,
        early_warning: Optional[Dict] = None,
    ) -> Dict:
        if len(self._structural_drift_history) < 3:
            unavailable = self._analytics_unavailable("insufficient history")
            return {
                "trajectory_analysis": unavailable,
                "branching_analysis": unavailable,
                "constraint_analysis": unavailable,
                "hierarchy_analysis": unavailable,
                "horizon_analysis": unavailable,
                "counterfactual_simulation": unavailable,
            }
        geometry = geometry or self._latest_geometry
        state_space_statistics = state_space_statistics or self._latest_state_space_statistics
        state_graph = state_graph or self._latest_state_graph
        early_warning = early_warning or {}
        geometry_available = bool(isinstance(geometry, dict) and geometry.get("available", True) is not False)
        state_stats_available = bool(
            isinstance(state_space_statistics, dict) and state_space_statistics.get("available", True) is not False
        )
        state_graph_available = bool(
            isinstance(state_graph, dict) and state_graph.get("available", True) is not False
        )

        pressure_tail = list(self._transition_pressure_history)[-6:]
        drift_tail = list(self._structural_drift_history)[-6:]
        shock_tail = list(self._shock_activity_history)[-6:]

        pressure_trend = self._trend(pressure_tail)
        drift_trend = self._trend(drift_tail)
        avg_pressure = float(np.mean(pressure_tail)) if pressure_tail else 0.0
        avg_shock = float(np.mean(shock_tail)) if shock_tail else 0.0

        if pressure_trend < -0.03 and drift_trend < -0.03:
            dominant_path = "STABILIZING"
        elif pressure_trend > 0.02 or drift_trend > 0.03:
            dominant_path = "DIVERGING"
        else:
            dominant_path = "METASTABLE"

        path_confidence = self._clamp01(0.35 + abs(pressure_trend) * 4.0 + abs(drift_trend) * 2.5)
        ew_pre = self._safe_float(early_warning.get("pre_instability_score", 0.0))
        ew_break = self._safe_float(early_warning.get("coherence_breakdown_score", 0.0))
        ew_erosion = self._safe_float(early_warning.get("stability_erosion_score", 0.0))
        if dominant_path == "METASTABLE" and ew_pre >= 0.58 and ew_erosion >= 0.5:
            dominant_path = "DIVERGING"
        trajectory = {
            "available": True,
            "dominant_path": dominant_path,
            "path_confidence": round(self._clamp01(path_confidence + 0.12 * ew_pre), 4),
            "trend_strength": round(abs(drift_trend) + abs(pressure_trend), 4),
        }

        drift_level = self._clamp01(float(np.mean(drift_tail)) / 4.5)
        path_directional_ambiguity = self._clamp01(1.0 - abs(pressure_trend - drift_trend) / 0.11)
        mixed_evidence = self._clamp01(1.0 - abs(avg_pressure - drift_level))
        geom_curvature = (
            self._clamp01(float((geometry or {}).get("curvature", 0.0)))
            if geometry_available
            else 0.0
        )
        graph_branching = (
            self._clamp01(float((state_graph or {}).get("branching_factor", 0.0)))
            if state_graph_available
            else 0.0
        )
        graph_entropy = (
            self._clamp01(float((state_graph or {}).get("transition_entropy", 0.0)))
            if state_graph_available
            else 0.0
        )
        decision_tension = self._clamp01(
            0.30 * avg_shock
            + 0.20 * self._clamp01(abs(pressure_trend) * 3.2)
            + 0.18 * path_directional_ambiguity
            + 0.18 * mixed_evidence
            + 0.08 * geom_curvature
            + 0.04 * graph_branching
            + 0.02 * graph_entropy
            + 0.08 * ew_break
        )
        commitment = self._clamp01(1.0 - (0.85 * decision_tension + 0.15 * path_directional_ambiguity))
        branch_count_estimate = 1
        if decision_tension >= 0.43 and mixed_evidence >= 0.32:
            branch_count_estimate = 2
        if decision_tension >= 0.68 and dominant_path == "METASTABLE" and path_directional_ambiguity >= 0.5:
            branch_count_estimate = 3
        branching = {
            "available": True,
            "is_branching": bool(
                decision_tension >= 0.45
                and (mixed_evidence >= 0.36 or path_directional_ambiguity >= 0.46)
                and (dominant_path != "DIVERGING" or mixed_evidence >= 0.5)
            ),
            "commitment": round(commitment, 4),
            "decision_tension": round(decision_tension, 4),
            "branch_count_estimate": branch_count_estimate,
        }

        persistence = self._clamp01(float(sum(1 for p in pressure_tail if p >= 0.58)) / max(1, len(pressure_tail)))
        sustained_pressure = self._clamp01(float(sum(1 for p in pressure_tail[-4:] if p >= 0.66)) / max(1, len(pressure_tail[-4:])))
        branch_collapse = self._clamp01(max(0.0, (1.0 - decision_tension) - 0.22))
        commitment_proxy = self._clamp01(0.55 * branch_collapse + 0.45 * (1.0 if dominant_path == "DIVERGING" else 0.3))
        state_contraction = (
            self._clamp01(float((state_space_statistics or {}).get("state_contraction_score", 0.0)))
            if state_stats_available
            else 0.0
        )
        path_commitment = (
            self._clamp01(float((state_graph or {}).get("path_commitment_score", 0.0)))
            if state_graph_available
            else 0.0
        )
        lock_raw = self._clamp01(
            0.21 * avg_pressure
            + 0.18 * drift_level
            + 0.15 * persistence
            + 0.08 * sustained_pressure
            + 0.08 * commitment_proxy
            + 0.09 * self._clamp01(max(0.0, pressure_trend) * 4.0)
            + 0.07 * self._clamp01(max(0.0, drift_trend) * 4.0)
            + 0.08 * state_contraction
            + 0.06 * path_commitment
        )
        convergence = self._clamp01(0.45 * persistence + 0.30 * sustained_pressure + 0.25 * branch_collapse)
        lock_in_score = self._clamp01(0.82 / (1.0 + math.exp(-5.2 * (lock_raw - 0.66))) * (0.9 + 0.2 * convergence))
        constraint = {
            "available": True,
            "lock_in_score": round(lock_in_score, 4),
            "point_of_no_return_risk": round(self._clamp01(lock_in_score * 1.15), 4),
            "recovery_margin": round(self._clamp01(1.0 - lock_in_score), 4),
        }

        origin_scope = "LOCAL"
        if lock_in_score > 0.7:
            origin_scope = "SYSTEMIC"
        elif lock_in_score > 0.45:
            origin_scope = "MULTI_SUBSYSTEM"
        hierarchy = {
            "available": True,
            "origin_scope": origin_scope,
            "propagation_risk": round(self._clamp01((lock_in_score * 0.7) + (avg_shock * 0.3)), 4),
        }

        horizon_bucket = "LONGER_HORIZON"
        horizon_before_adjustment = horizon_bucket
        pressure_persist = self._clamp01(float(sum(1 for p in pressure_tail if p >= 0.64)) / max(1, len(pressure_tail)))
        sustained_horizon_pressure = self._clamp01(float(sum(1 for p in pressure_tail[-4:] if p >= 0.72)) / max(1, len(pressure_tail[-4:])))
        accel = self._clamp01(max(0.0, self._trend([float(v) for v in list(self._velocity_history)[-6:]])) / 0.06)
        severity = 1.0 if dominant_path == "DIVERGING" else (0.55 if dominant_path == "METASTABLE" else 0.2)
        recovery_margin = self._clamp01(1.0 - lock_in_score)
        local_volume = (
            self._clamp01(float((state_space_statistics or {}).get("local_volume", 0.0)))
            if state_stats_available
            else 0.0
        )
        horizon_signal = self._clamp01(
            0.18 * avg_pressure
            + 0.20 * lock_in_score
            + 0.12 * pressure_persist
            + 0.10 * sustained_horizon_pressure
            + 0.14 * severity
            + 0.10 * accel
            + 0.08 * self._clamp01(max(0.0, pressure_trend) * 4.0)
            + 0.06 * self._clamp01(max(0.0, drift_trend) * 4.0)
            + 0.04 * local_volume
            + 0.10 * ew_erosion
            - 0.12 * recovery_margin
            - 0.08 * decision_tension
        )
        if (
            horizon_signal >= 0.9
            and pressure_persist >= 0.68
            and sustained_horizon_pressure >= 0.5
            and lock_in_score >= 0.74
            and severity >= 0.7
        ):
            horizon_bucket = "IMMINENT"
        elif horizon_signal >= 0.58:
            horizon_bucket = "NEAR_TERM"
        elif horizon_signal >= 0.4:
            horizon_bucket = "MID_TERM"
        early_warning_state = str(early_warning.get("early_warning_state", "stable"))
        ew_recent_states = list(self._early_warning_state_history)[-8:]
        ew_preinstability_count = sum(1 for state in ew_recent_states if state == "pre_instability")
        ew_emerging_count = sum(1 for state in ew_recent_states if state in {"pre_instability", "emerging_instability", "alert"})
        ew_consecutive_pre = 0
        for state in reversed(ew_recent_states):
            if state == "pre_instability":
                ew_consecutive_pre += 1
            else:
                break
        ew_persistent = ew_preinstability_count >= 3 or ew_consecutive_pre >= 2

        pre_tail = self._history_tail("pre_instability_score", size=6)
        lock_tail = self._history_tail("lock_in_score", size=6)
        pre_rising = len(pre_tail) >= 3 and self._trend(pre_tail[-4:]) > 0.01
        lock_rising = len(lock_tail) >= 3 and self._trend(lock_tail[-4:]) > 0.008
        coupled_rise = pre_rising and lock_rising

        if ew_persistent and horizon_bucket == "LONGER_HORIZON":
            horizon_bucket = "MID_TERM"
        if (
            coupled_rise
            and ew_emerging_count >= 4
            and horizon_bucket in {"LONGER_HORIZON", "MID_TERM"}
        ):
            horizon_bucket = "NEAR_TERM"

        horizon = {
            "available": True,
            "risk_horizon": horizon_bucket,
            "horizon_score": round(self._clamp01(horizon_signal), 4),
            "rationale": {
                "early_warning_state": early_warning_state,
                "early_warning_persistence_count": int(ew_preinstability_count),
                "early_warning_consecutive_pre_instability": int(ew_consecutive_pre),
                "early_warning_emerging_count": int(ew_emerging_count),
                "pre_instability_rising": bool(pre_rising),
                "lock_in_rising": bool(lock_rising),
                "horizon_before_adjustment": horizon_before_adjustment,
                "horizon_after_adjustment": horizon_bucket,
            },
        }

        baseline_path = "DIVERGING" if dominant_path == "DIVERGING" else "METASTABLE"
        counterfactual = {
            "available": True,
            "baseline_future": {
                "projected_path": baseline_path,
                "projected_lock_in_risk": round(lock_in_score, 4),
                "projected_horizon": horizon_bucket,
            },
            "counterfactuals": [
                {"scenario": "pressure_relief", "projected_path": "STABILIZING"},
                {"scenario": "status_quo", "projected_path": baseline_path},
                {"scenario": "continued_degradation", "projected_path": "DIVERGING"},
            ],
        }

        return {
            "trajectory_analysis": trajectory,
            "branching_analysis": branching,
            "constraint_analysis": constraint,
            "hierarchy_analysis": hierarchy,
            "horizon_analysis": horizon,
            "counterfactual_simulation": counterfactual,
        }

    def _assemble_result(self, base: Dict, transition: Dict, analytics: Dict, early_warning: Dict) -> Dict:
        result = {
            **base,
            "transition_pressure": transition["transition_pressure"],
            "transition_state": transition["transition_state"],
            "experimental_analytics": analytics,
            "early_warning": early_warning,
        }
        return result

    def process_frame(self, frame: Dict) -> Dict:
        """
        Process one normalized telemetry frame and return the latest structural event.
        Expected frame shape:
        {
            "timestamp": "...",
            "site_id": "...",
            "asset_id": "...",
            "sensor_values": {...}
        }
        """
        required = {"timestamp", "site_id", "asset_id", "sensor_values"}
        missing = [key for key in required if key not in frame]
        if missing:
            raise ValueError(f"Frame is missing required keys: {missing}")

        vector = self._vector_from_frame(frame)
        vector = self._smooth_current_vector(vector)
        stored = dict(frame)
        stored["_vector"] = vector
        self.frames.append(stored)

        baseline = self._baseline_stats()
        base = self._compute_base_metrics(frame=frame, baseline=baseline, vector=vector)
        geometry_payload = self._compute_geometry_payload(frame=frame, vector=vector)
        geometry_metrics = geometry_payload.get("geometry", {})
        state_space_statistics = geometry_payload.get("state_space_statistics", {})
        state_graph = geometry_payload.get("state_graph", {})
        self._latest_geometry = geometry_metrics if isinstance(geometry_metrics, dict) else {"available": False, "reason": "insufficient history"}
        self._latest_state_space_statistics = (
            state_space_statistics if isinstance(state_space_statistics, dict) else {"available": False, "reason": "insufficient history"}
        )
        self._latest_state_graph = state_graph if isinstance(state_graph, dict) else {"available": False, "reason": "insufficient history"}
        transition = self._compute_transition_metrics(
            drift_score=float(base["structural_drift_score"]),
            velocity=float(base["drift_velocity"]),
            stability_score=float(base["relational_stability_score"]),
            geometry=self._latest_geometry,
            state_space_statistics=self._latest_state_space_statistics,
            state_graph=self._latest_state_graph,
        )
        early_warning = self._compute_early_warning(
            drift_score=float(base["structural_drift_score"]),
            velocity=float(base["drift_velocity"]),
            transition=transition,
            geometry=self._latest_geometry,
            state_space_statistics=self._latest_state_space_statistics,
            state_graph=self._latest_state_graph,
            vector=vector,
        )
        early_erosion = self._safe_float(early_warning.get("stability_erosion_score", 0.0))
        pressure_boost = self._clamp01(early_erosion * 0.16)
        transition["transition_pressure"] = round(
            self._clamp01(self._safe_float(transition.get("transition_pressure", 0.0)) + pressure_boost), 4
        )
        pressure_now = float(transition["transition_pressure"])
        refined_state = self._state_from_onset_evidence(
            drift_score=float(base["structural_drift_score"]),
            velocity=float(base["drift_velocity"]),
            stability_score=float(base["relational_stability_score"]),
            transition_pressure=pressure_now,
            vector=vector,
            geometry=self._latest_geometry,
            state_space_statistics=self._latest_state_space_statistics,
            state_graph=self._latest_state_graph,
        )
        ew_state = str(early_warning.get("early_warning_state", "stable"))
        recent_ew = list(self._early_warning_state_history)[-6:]
        ew_emerging_persist = sum(1 for s in recent_ew if s in {"pre_instability", "emerging_instability", "alert"}) >= 3
        if refined_state == "STABLE" and ew_state in {"pre_instability", "emerging_instability"} and ew_emerging_persist:
            refined_state = "WATCH"
        if refined_state == "WATCH" and ew_state == "alert":
            refined_state = "ALERT"
        base["state"] = refined_state
        base["drift_alert"] = refined_state == "ALERT"
        if refined_state == "WATCH":
            base["event_type"] = "quality_observation"
            base["predicted_impact"] = "Early degradation detected. Maintenance window recommended."
            base["structural_driver"] = "persistent structural drift from local baseline"
            base["interpreted_state"] = "EMERGING_TRANSITION"
            base["regime_name"] = "DEGRADING"
        elif refined_state == "ALERT":
            base["event_type"] = "instability_escalation"
            base["predicted_impact"] = "Sustained transition pressure and directional drift detected."
            base["structural_driver"] = "confirmed multi-window structural transition"
            base["interpreted_state"] = "SUSTAINED_TRANSITION"
            base["regime_name"] = "INSTABILITY_REGIME"
        else:
            base["event_type"] = "flow_observation"
            base["predicted_impact"] = "No near term operational disruption expected."
            base["structural_driver"] = "stable baseline telemetry"
            base["interpreted_state"] = "STABLE_FLOW"
            base["regime_name"] = "NOMINAL"
        self._update_histories(
            drift_score=float(base["structural_drift_score"]),
            velocity=float(base["drift_velocity"]),
            state=str(base["state"]),
            pressure=float(transition["transition_pressure"]),
        )
        analytics = self._compute_advanced_analytics(
            geometry=self._latest_geometry,
            state_space_statistics=self._latest_state_space_statistics,
            state_graph=self._latest_state_graph,
            early_warning=early_warning,
        )
        analytics["early_warning"] = dict(early_warning)
        result = self._assemble_result(base=base, transition=transition, analytics=analytics, early_warning=early_warning)
        result["geometry"] = self._latest_geometry
        result["state_space_statistics"] = self._latest_state_space_statistics
        result["state_graph"] = self._latest_state_graph

        self._frame_count += 1
        if _engine_debug_enabled():
            if self._frame_count <= 3:
                print("DEBUG RESULT KEYS:", result.keys())
                print("DEBUG EXPERIMENTAL:", result["experimental_analytics"])
                print(
                    "DEBUG ADVANCED POPULATED:",
                    {name: bool(isinstance(payload, dict) and payload.get("available")) for name, payload in result["experimental_analytics"].items()},
                )
            geometry_live = bool(isinstance(self._latest_geometry, dict) and self._latest_geometry.get("available", True) is not False)
            if self._geometry_debug_frames_logged < 3 and (geometry_live or len(self.frames) >= self.geometry_layer.min_history):
                print("DEBUG GEOMETRY:", result.get("geometry"))
                print("DEBUG STATE SPACE:", result.get("state_space_statistics"))
                print("DEBUG STATE GRAPH:", result.get("state_graph"))
                self._geometry_debug_frames_logged += 1
            if self._early_warning_debug_frames_logged < 3 and len(self.frames) >= max(8, self.recent_window):
                horizon_rationale = ((analytics.get("horizon_analysis") or {}).get("rationale", {})) if isinstance(analytics, dict) else {}
                print(
                    "DEBUG EARLY WARNING:",
                    {
                        "pre_instability_score": early_warning.get("pre_instability_score"),
                        "stability_erosion_score": early_warning.get("stability_erosion_score"),
                        "coherence_breakdown_score": early_warning.get("coherence_breakdown_score"),
                        "transition_pressure_delta": round(self._delta_from_history("transition_pressure"), 6),
                        "curvature_delta": round(self._delta_from_history("curvature"), 6),
                        "local_volume_delta": round(self._delta_from_history("local_volume"), 6),
                        "early_warning_state": early_warning.get("early_warning_state"),
                        "early_warning_persistence_count": horizon_rationale.get("early_warning_persistence_count"),
                        "horizon_before_adjustment": horizon_rationale.get("horizon_before_adjustment"),
                        "horizon_after_adjustment": horizon_rationale.get("horizon_after_adjustment"),
                    },
                )
                self._early_warning_debug_frames_logged += 1

        self.latest_result = result
        return result


if __name__ == "__main__":
    # Minimal smoke test / example usage.
    engine = StructuralEngine()

    sample_frames = [
        {
            "timestamp": f"2026-03-13T00:{i:02d}:00Z",
            "site_id": "site-a",
            "asset_id": "asset-1",
            "sensor_values": {
                "temp": 50.0 + (i * 0.02),
                "pressure": 100.0 + (i * 0.03),
                "vibration": 1.0 + (i * 0.005),
            },
        }
        for i in range(40)
    ]

    for frame in sample_frames:
        output = engine.process_frame(frame)

    print(output)
