from __future__ import annotations

import math
from collections import deque
from typing import Any, Deque, Dict, List, Optional


def _is_nan(value: float) -> bool:
    return math.isnan(value)


def _has_nan(vector: List[float]) -> bool:
    return any(_is_nan(v) for v in vector)


def _mean_vector(rows: List[List[float]]) -> List[float]:
    return [sum(col) / len(rows) for col in zip(*rows)]


def _covariance(rows: List[List[float]]) -> List[List[float]]:
    n = len(rows)
    d = len(rows[0])
    mean = _mean_vector(rows)
    cov = [[0.0 for _ in range(d)] for _ in range(d)]
    denom = max(1, n - 1)
    for row in rows:
        delta = [row[i] - mean[i] for i in range(d)]
        for i in range(d):
            for j in range(d):
                cov[i][j] += (delta[i] * delta[j]) / denom
    return cov


def _regularize_covariance(cov: List[List[float]], ridge: float = 1e-6) -> List[List[float]]:
    size = len(cov)
    for i in range(size):
        cov[i][i] += ridge
    return cov


def _invert_matrix(matrix: List[List[float]]) -> List[List[float]]:
    n = len(matrix)
    a = [row[:] for row in matrix]
    inv = [[float(i == j) for j in range(n)] for i in range(n)]

    for i in range(n):
        pivot = a[i][i]
        if abs(pivot) < 1e-12:
            for j in range(i + 1, n):
                if abs(a[j][i]) > 1e-12:
                    a[i], a[j] = a[j], a[i]
                    inv[i], inv[j] = inv[j], inv[i]
                    pivot = a[i][i]
                    break
        if abs(pivot) < 1e-12:
            pivot = 1e-12
            a[i][i] = pivot

        scale = 1.0 / pivot
        for k in range(n):
            a[i][k] *= scale
            inv[i][k] *= scale

        for j in range(n):
            if j == i:
                continue
            factor = a[j][i]
            for k in range(n):
                a[j][k] -= factor * a[i][k]
                inv[j][k] -= factor * inv[i][k]

    return inv


def _quadratic_form(v: List[float], matrix: List[List[float]]) -> float:
    total = 0.0
    for i in range(len(v)):
        for j in range(len(v)):
            total += v[i] * matrix[i][j] * v[j]
    return total


def _frobenius_norm_diff(a: List[List[float]], b: List[List[float]]) -> float:
    total = 0.0
    for i in range(len(a)):
        for j in range(len(a[i])):
            diff = a[i][j] - b[i][j]
            total += diff * diff
    return math.sqrt(total)


class StructuralEngine:
    """Runtime structural monitoring engine."""

    def __init__(
        self,
        baseline_window: int = 24,
        recent_window: int = 8,
        max_frames: int = 500,
        mahal_weight: float = 0.65,
        cov_weight: float = 0.35,
        smoothing_window: int = 3,
        enable_vector_smoothing: bool = True,
    ) -> None:
        self.baseline_window = baseline_window
        self.recent_window = recent_window
        self.smoothing_window = smoothing_window
        self.enable_vector_smoothing = enable_vector_smoothing
        total_weight = mahal_weight + cov_weight
        self.mahal_weight = mahal_weight / total_weight
        self.cov_weight = cov_weight / total_weight
        self.frames: Deque[Dict[str, Any]] = deque(maxlen=max_frames)
        self.sensor_order: List[str] = []
        self.latest_result: Optional[Dict[str, Any]] = None
        self.prev_drift: Optional[float] = None

    def reset(self) -> None:
        self.frames.clear()
        self.sensor_order = []
        self.latest_result = None
        self.prev_drift = None

    def get_latest_result(self) -> Optional[Dict[str, Any]]:
        return self.latest_result

    def _vector_from_frame(self, frame: Dict[str, Any]) -> List[float]:
        sensor_values = frame.get("sensor_values", {})
        if not self.sensor_order:
            self.sensor_order = sorted(sensor_values.keys())

        values: List[float] = []
        for name in self.sensor_order:
            raw = sensor_values.get(name)
            try:
                values.append(float(raw))
            except (TypeError, ValueError):
                values.append(float("nan"))
        return values

    def _valid_matrix(self, frames: List[Dict[str, Any]]) -> List[List[float]]:
        rows = [f["_vector"] for f in frames]
        return [row for row in rows if not _has_nan(row)]

    def _smooth_current_vector(self, vector: List[float]) -> List[float]:
        if not self.enable_vector_smoothing or not self.frames:
            return vector
        prev = self.frames[-1]["_vector"]
        if len(prev) != len(vector) or _has_nan(prev) or _has_nan(vector):
            return vector
        return [(prev[i] + vector[i]) * 0.5 for i in range(len(vector))]

    def _baseline_stats(self) -> Optional[Dict[str, List[List[float]] | List[float]]]:
        if len(self.frames) < max(6, self.baseline_window):
            return None
        rows = self._valid_matrix(list(self.frames)[: self.baseline_window])
        if len(rows) < 2:
            return None
        mean = _mean_vector(rows)
        cov = _regularize_covariance(_covariance(rows))
        inv_cov = _invert_matrix(cov)
        return {"mean": mean, "cov": cov, "inv_cov": inv_cov}

    def _mahalanobis(self, x: List[float], mean: List[float], inv_cov: List[List[float]]) -> float:
        delta = [x[i] - mean[i] for i in range(len(x))]
        return math.sqrt(max(0.0, _quadratic_form(delta, inv_cov)))

    def _covariance_drift(self) -> float:
        if len(self.frames) < max(self.baseline_window + self.recent_window, 10):
            return 0.0
        base = self._valid_matrix(list(self.frames)[: self.baseline_window])
        recent = self._valid_matrix(list(self.frames)[-self.recent_window :])
        if len(base) < 2 or len(recent) < 2:
            return 0.0
        cov_b = _regularize_covariance(_covariance(base))
        cov_r = _regularize_covariance(_covariance(recent))
        return _frobenius_norm_diff(cov_r, cov_b)

    def process_frame(self, frame: Dict[str, Any]) -> Dict[str, Any]:
        vector = self._smooth_current_vector(self._vector_from_frame(frame))
        stored = dict(frame)
        stored["_vector"] = vector
        self.frames.append(stored)
        baseline = self._baseline_stats()

        if baseline is None or _has_nan(vector):
            result = {
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
            }
            self.latest_result = result
            return result

        mean = baseline["mean"]
        inv_cov = baseline["inv_cov"]
        mahal = self._mahalanobis(vector, mean, inv_cov)
        cov_drift = self._covariance_drift()
        drift_score = (self.mahal_weight * mahal) + (self.cov_weight * cov_drift)
        stability = 1.0 / (1.0 + cov_drift)
        health = int(
            round(max(0.0, min(100.0, 100.0 - min(drift_score * 18.0, 80.0) + stability * 20.0)))
        )
        state = "ALERT" if drift_score > 3.0 else "WATCH" if drift_score > 1.5 else "STABLE"

        velocity = 0.0 if self.prev_drift is None else drift_score - self.prev_drift
        self.prev_drift = drift_score
        lead_time = None
        if velocity > 0.01:
            lead_time = round(min(240.0, max(0.0, 4.5 - drift_score) / velocity), 1)

        result = {
            "id": None,
            "event_type": "instability_escalation"
            if state == "ALERT"
            else "quality_observation"
            if state == "WATCH"
            else "flow_observation",
            "timestamp": frame["timestamp"],
            "site_id": frame["site_id"],
            "asset_id": frame["asset_id"],
            "state": state,
            "structural_drift_score": round(drift_score, 4),
            "relational_stability_score": round(stability, 4),
            "system_health": health,
            "drift_alert": drift_score > 1.5,
            "lead_time_hours": lead_time,
            "lead_time_confidence": round(min(0.99, max(0.35, stability)), 2),
            "drift_velocity": round(velocity, 4),
            "structural_driver": "cross-sensor structural divergence"
            if state == "ALERT"
            else "emerging relational drift"
            if state == "WATCH"
            else "stable baseline telemetry",
            "predicted_impact": "Potential localized service disruption within 1 to 2 hours."
            if state == "ALERT"
            else "Early degradation detected. Maintenance window recommended."
            if state == "WATCH"
            else "No near term operational disruption expected.",
            "explanation": "Structural engine monitoring sensor relationships in real time.",
            "mahalanobis_score": round(mahal, 4),
            "covariance_drift_score": round(cov_drift, 4),
        }
        self.latest_result = result
        return result
