"""Structural drift engine for baseline-relative telemetry monitoring.

Rigorous/stronger components in this module:
- Baseline-relative multivariate state tracking.
- Mahalanobis distance from baseline manifold.
- Covariance (relational) drift between baseline and recent windows.
- Weighted fusion of manifold and relational drift.

Heuristic/approximate components in this module:
- Alert threshold bands.
- Health score conversion.
- Lead-time estimate derived from drift velocity.
- Human-facing text for driver/impact/explanation.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any

from neraium_core.models import StructuralResult


def _is_nan(value: float) -> bool:
    return math.isnan(value)


def _mean(values: list[float]) -> float:
    return sum(values) / float(len(values))


def _column_means(matrix: list[list[float]]) -> list[float]:
    return [_mean([row[i] for row in matrix]) for i in range(len(matrix[0]))]


def _covariance(matrix: list[list[float]]) -> list[list[float]]:
    rows = len(matrix)
    cols = len(matrix[0])
    means = _column_means(matrix)
    cov = [[0.0 for _ in range(cols)] for _ in range(cols)]
    denom = max(1, rows - 1)
    for i in range(cols):
        for j in range(cols):
            acc = 0.0
            for row in matrix:
                acc += (row[i] - means[i]) * (row[j] - means[j])
            cov[i][j] = acc / denom
    return cov


def _identity(n: int) -> list[list[float]]:
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def _invert(matrix: list[list[float]]) -> list[list[float]]:
    n = len(matrix)
    a = [row[:] for row in matrix]
    inv = _identity(n)

    for i in range(n):
        pivot = a[i][i]
        if abs(pivot) < 1e-12:
            for r in range(i + 1, n):
                if abs(a[r][i]) > 1e-12:
                    a[i], a[r] = a[r], a[i]
                    inv[i], inv[r] = inv[r], inv[i]
                    pivot = a[i][i]
                    break
        if abs(pivot) < 1e-12:
            pivot = 1e-6
            a[i][i] = pivot

        scale = 1.0 / pivot
        for c in range(n):
            a[i][c] *= scale
            inv[i][c] *= scale

        for r in range(n):
            if r == i:
                continue
            factor = a[r][i]
            for c in range(n):
                a[r][c] -= factor * a[i][c]
                inv[r][c] -= factor * inv[i][c]
    return inv


def _mat_vec_mul(matrix: list[list[float]], vector: list[float]) -> list[float]:
    return [sum(matrix[r][c] * vector[c] for c in range(len(vector))) for r in range(len(matrix))]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=False))


def _fro_norm(matrix: list[list[float]]) -> float:
    return math.sqrt(sum(cell * cell for row in matrix for cell in row))


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
        if baseline_window < 2:
            raise ValueError("baseline_window must be at least 2")
        if recent_window < 2:
            raise ValueError("recent_window must be at least 2")
        if max_frames < (baseline_window + recent_window):
            raise ValueError("max_frames is too small for configured windows")

        total_weight = mahal_weight + cov_weight
        if total_weight <= 0:
            raise ValueError("mahal_weight + cov_weight must be > 0")

        self.mahal_weight = mahal_weight / total_weight
        self.cov_weight = cov_weight / total_weight
        self.baseline_window = baseline_window
        self.recent_window = recent_window
        self.smoothing_window = smoothing_window
        self.enable_vector_smoothing = enable_vector_smoothing

        self.frames: deque[dict[str, Any]] = deque(maxlen=max_frames)
        self.sensor_order: list[str] = []
        self._latest_result: StructuralResult | None = None
        self.prev_drift: float | None = None

    def reset(self) -> None:
        self.frames.clear()
        self.sensor_order = []
        self._latest_result = None
        self.prev_drift = None

    def get_latest_result(self) -> StructuralResult | None:
        return self._latest_result

    def _vector_from_frame(self, frame: dict[str, Any]) -> list[float]:
        sensor_values = frame.get("sensor_values", {})
        if not isinstance(sensor_values, dict):
            raise ValueError("frame['sensor_values'] must be a dictionary")
        if not self.sensor_order:
            self.sensor_order = sorted(sensor_values.keys())

        vector: list[float] = []
        for name in self.sensor_order:
            value = sensor_values.get(name)
            try:
                vector.append(float(value))
            except (TypeError, ValueError):
                vector.append(float("nan"))
        return vector

    def _valid_matrix(self, frames: list[dict[str, Any]]) -> list[list[float]] | None:
        matrix = [frame["_vector"] for frame in frames]
        valid = [row for row in matrix if not any(_is_nan(v) for v in row)]
        return valid if len(valid) >= 2 else None

    def _smooth_matrix(self, matrix: list[list[float]] | None) -> list[list[float]] | None:
        if matrix is None or len(matrix) < 2 or self.smoothing_window <= 1:
            return matrix
        window = min(self.smoothing_window, len(matrix))
        smoothed = [row[:] for row in matrix]
        for c in range(len(matrix[0])):
            for r in range(len(matrix)):
                start = max(0, r - window + 1)
                chunk = [matrix[i][c] for i in range(start, r + 1)]
                smoothed[r][c] = _mean(chunk)
        return smoothed

    def _smooth_current_vector(self, vector: list[float]) -> list[float]:
        if not self.enable_vector_smoothing or not self.frames:
            return vector
        prev = self.frames[-1]["_vector"]
        if len(prev) != len(vector):
            return vector
        if any(_is_nan(v) for v in prev) or any(_is_nan(v) for v in vector):
            return vector
        return [(a + b) / 2.0 for a, b in zip(prev, vector, strict=False)]

    def _regularize_covariance(self, cov: list[list[float]]) -> list[list[float]]:
        n = len(cov)
        reg = [row[:] for row in cov]
        for i in range(n):
            if reg[i][i] < 1e-6:
                reg[i][i] = 1e-6
        return reg

    def _baseline_stats(self) -> dict[str, list[list[float]] | list[float]] | None:
        if len(self.frames) < max(6, self.baseline_window):
            return None
        baseline_frames = list(self.frames)[: self.baseline_window]
        matrix = self._smooth_matrix(self._valid_matrix(baseline_frames))
        if matrix is None:
            return None
        mean = _column_means(matrix)
        cov = self._regularize_covariance(_covariance(matrix))
        inv_cov = _invert(cov)
        return {"mean": mean, "inv_cov": inv_cov}

    def _mahalanobis(self, x: list[float], mean: list[float], inv_cov: list[list[float]]) -> float:
        delta = [a - b for a, b in zip(x, mean, strict=False)]
        return math.sqrt(max(0.0, _dot(delta, _mat_vec_mul(inv_cov, delta))))

    def _covariance_drift(self) -> float:
        if len(self.frames) < max(self.baseline_window + self.recent_window, 10):
            return 0.0

        baseline_frames = list(self.frames)[: self.baseline_window]
        recent_frames = list(self.frames)[-self.recent_window :]
        xb = self._smooth_matrix(self._valid_matrix(baseline_frames))
        xr = self._smooth_matrix(self._valid_matrix(recent_frames))
        if xb is None or xr is None:
            return 0.0

        cov_b = self._regularize_covariance(_covariance(xb))
        cov_r = self._regularize_covariance(_covariance(xr))
        diff = [[cov_r[i][j] - cov_b[i][j] for j in range(len(cov_b))] for i in range(len(cov_b))]
        return _fro_norm(diff)

    def process_frame(self, frame: dict[str, Any]) -> StructuralResult:
        required = {"timestamp", "site_id", "asset_id", "sensor_values"}
        missing = [key for key in required if key not in frame]
        if missing:
            raise ValueError(f"Frame is missing required keys: {missing}")

        vector = self._smooth_current_vector(self._vector_from_frame(frame))
        stored = dict(frame)
        stored["_vector"] = vector
        self.frames.append(stored)

        baseline = self._baseline_stats()
        if baseline is None or any(_is_nan(v) for v in vector):
            result = StructuralResult(
                id=None,
                event_type="baseline_telemetry",
                timestamp=frame["timestamp"],
                site_id=frame["site_id"],
                asset_id=frame["asset_id"],
                state="STABLE",
                structural_drift_score=0.0,
                relational_stability_score=1.0,
                system_health=100,
                drift_alert=False,
                lead_time_hours=None,
                lead_time_confidence=0.0,
                drift_velocity=0.0,
                structural_driver="baseline formation",
                predicted_impact="No near term operational disruption expected.",
                explanation="Initializing structural telemetry...",
                mahalanobis_score=0.0,
                covariance_drift_score=0.0,
            )
            self._latest_result = result
            return result

        mahal = self._mahalanobis(
            vector,
            baseline["mean"],  # type: ignore[arg-type]
            baseline["inv_cov"],  # type: ignore[arg-type]
        )
        cov_drift = self._covariance_drift()
        drift_score = (self.mahal_weight * mahal) + (self.cov_weight * cov_drift)
        stability = max(0.0, min(1.0, 1.0 / (1.0 + cov_drift)))
        health = int(round(max(0.0, min(100.0, (100.0 - min(drift_score * 18.0, 80.0)) + (stability * 20.0)))))

        state = "ALERT" if drift_score > 3.0 else "WATCH" if drift_score > 1.5 else "STABLE"
        velocity = 0.0 if self.prev_drift is None else drift_score - self.prev_drift
        self.prev_drift = drift_score

        lead_time_hours: float | None = None
        if velocity > 0.01:
            remaining = max(0.0, 4.5 - drift_score)
            lead_time_hours = round(min(240.0, remaining / velocity), 1)

        event_type = "flow_observation"
        impact = "No near term operational disruption expected."
        driver = "stable baseline telemetry"
        if state == "WATCH":
            event_type = "quality_observation"
            impact = "Early degradation detected. Maintenance window recommended."
            driver = "emerging relational drift"
        elif state == "ALERT":
            event_type = "instability_escalation"
            impact = "Potential localized service disruption within 1 to 2 hours."
            driver = "cross-sensor structural divergence"

        result = StructuralResult(
            id=None,
            event_type=event_type,
            timestamp=frame["timestamp"],
            site_id=frame["site_id"],
            asset_id=frame["asset_id"],
            state=state,
            structural_drift_score=round(drift_score, 4),
            relational_stability_score=round(stability, 4),
            system_health=health,
            drift_alert=drift_score > 1.5,
            lead_time_hours=lead_time_hours,
            lead_time_confidence=round(min(0.99, max(0.35, stability)), 2),
            drift_velocity=round(velocity, 4),
            structural_driver=driver,
            predicted_impact=impact,
            explanation="Structural engine monitoring sensor relationships in real time.",
            mahalanobis_score=round(mahal, 4),
            covariance_drift_score=round(cov_drift, 4),
        )
        self._latest_result = result
        return result
