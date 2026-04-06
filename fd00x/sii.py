"""
SII v2.0 core model.

This module provides a lightweight implementation of Systemic Infrastructure
Intelligence (SII) with five fundamental scoring layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


def _to_vector(readings: Mapping[str, float], sensors: Sequence[str]) -> np.ndarray:
    return np.asarray([float(readings.get(sensor, 0.0)) for sensor in sensors], dtype=float)


def _safe_corr_matrix(data: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Finite-safe correlation matrix that tolerates constant columns."""
    arr = np.asarray(data, dtype=float)
    if arr.ndim != 2:
        raise ValueError("data must be a 2D array")
    if arr.shape[0] < 2:
        return np.eye(arr.shape[1], dtype=float)

    centered = arr - np.mean(arr, axis=0, keepdims=True)
    std = np.std(centered, axis=0, ddof=1)
    valid = std > eps
    denom = np.where(valid, std, 1.0)
    z = centered / denom
    corr = (z.T @ z) / max(1, arr.shape[0] - 1)
    corr = np.asarray(corr, dtype=float)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr[~valid, :] = 0.0
    corr[:, ~valid] = 0.0
    np.fill_diagonal(corr, 1.0)
    return corr


@dataclass
class LayerWeights:
    l1: float = 0.26
    l2: float = 0.20
    l3: float = 0.20
    l4: float = 0.17
    l5: float = 0.17


@dataclass
class ScoreProgressionConfig:
    """Controls score progression spread and drift sensitivity."""

    baseline_fraction: float = 0.20
    drift_weight: float = 0.30
    variance_weight: float = 0.22
    alpha: float = 3.0
    raw_offset: float = 1.10


class SII:
    """Atomic five-layer SII engine."""

    def __init__(
        self,
        sensors: Iterable[str],
        baseline_data: np.ndarray,
        *,
        thresholds: Mapping[str, float] | None = None,
        layer_weights: LayerWeights | None = None,
        progression: ScoreProgressionConfig | None = None,
    ) -> None:
        self.sensors: List[str] = list(sensors)
        if len(self.sensors) == 0:
            raise ValueError("SII requires at least one sensor.")

        baseline_arr = np.asarray(baseline_data, dtype=float)
        if baseline_arr.ndim != 2 or baseline_arr.shape[1] != len(self.sensors):
            raise ValueError(
                f"baseline_data must have shape (n, {len(self.sensors)}), got {baseline_arr.shape}."
            )

        self.progression = progression or ScoreProgressionConfig()
        baseline_fraction = float(np.clip(self.progression.baseline_fraction, 0.15, 0.25))
        baseline_len = int(np.ceil(max(1, baseline_arr.shape[0]) * baseline_fraction))
        baseline_len = max(5, min(baseline_len, baseline_arr.shape[0]))
        self.baseline = baseline_arr[:baseline_len].copy()

        self.mean = self.baseline.mean(axis=0)
        self.std = np.clip(self.baseline.std(axis=0), 1e-8, None)
        self.var = self.std**2
        self.baseline_variance = float(max(np.mean(self.var), 1e-8))
        self.corr = _safe_corr_matrix(self.baseline)
        self.cov = np.cov(self.baseline, rowvar=False)
        self.weights = layer_weights or LayerWeights()
        self.thresholds: Dict[str, float] = {
            "healthy": 0.28,
            "elevated": 0.38,
            "caution": 0.52,
            "critical": 0.72,
        }
        if thresholds:
            self.thresholds.update({k.lower(): float(v) for k, v in thresholds.items()})

    def _layer_1_deviation(self, z: np.ndarray) -> float:
        return float(np.mean(np.abs(z)))

    def _layer_2_coupling_shift(self, z: np.ndarray) -> float:
        outer = np.outer(z, z)
        diff = np.abs(outer - self.corr)
        return float(np.mean(diff))

    def _layer_3_energy(self, z: np.ndarray) -> float:
        return float(np.sqrt(np.mean(z**2)))

    def _layer_4_cov_projection(self, z: np.ndarray) -> float:
        projected = self.cov @ z
        return float(np.linalg.norm(projected) / max(1, len(z)))

    def _layer_5_nonlinear_tension(self, z: np.ndarray) -> float:
        x = np.abs(np.tanh(z)) + np.abs(np.sin(z))
        return float(np.mean(x))

    def _variance_score(self, x: np.ndarray) -> float:
        sample_variance = float(np.var(x - self.mean))
        return float(np.log1p(sample_variance / self.baseline_variance))

    def _normalize_layers(self, layer_scores: Dict[str, float]) -> Dict[str, float]:
        return {
            "l1": float(_sigmoid(layer_scores["l1"] - 0.8)),
            "l2": float(_sigmoid(layer_scores["l2"] - 0.9)),
            "l3": float(_sigmoid(layer_scores["l3"] - 0.7)),
            "l4": float(_sigmoid(layer_scores["l4"] - 0.6)),
            "l5": float(_sigmoid(layer_scores["l5"] - 0.8)),
        }

    def _to_state(self, score: float) -> str:
        if not np.isfinite(score):
            return "insufficient_data"
        if score < self.thresholds["healthy"]:
            return "healthy"
        if score < self.thresholds["elevated"]:
            return "elevated"
        if score < self.thresholds["caution"]:
            return "caution"
        if score < self.thresholds["critical"]:
            return "caution"
        return "critical"

    def assess(
        self,
        readings: Mapping[str, float],
        *,
        current_cycle_index: int | None = None,
        total_cycles: int | None = None,
    ) -> Dict[str, object]:
        """
        Assess one sensor snapshot.

        Returns score, state, per-layer scores, and sensor-level importance.
        """

        x = _to_vector(readings, self.sensors)
        quality_flag: str | None = None
        if not np.all(np.isfinite(x)):
            x = np.where(np.isfinite(x), x, self.mean)
            quality_flag = "non_finite_input"
        z = (x - self.mean) / self.std

        raw_layers = {
            "l1": self._layer_1_deviation(z),
            "l2": self._layer_2_coupling_shift(z),
            "l3": self._layer_3_energy(z),
            "l4": self._layer_4_cov_projection(z),
            "l5": self._layer_5_nonlinear_tension(z),
        }
        layers = self._normalize_layers(raw_layers)
        atomic_score = (
            self.weights.l1 * layers["l1"]
            + self.weights.l2 * layers["l2"]
            + self.weights.l3 * layers["l3"]
            + self.weights.l4 * layers["l4"]
            + self.weights.l5 * layers["l5"]
        )
        variance_score = self._variance_score(x)
        if current_cycle_index is not None and total_cycles is not None and total_cycles > 1:
            drift = float(np.clip(current_cycle_index / float(total_cycles - 1), 0.0, 1.0))
        else:
            # Normalize fallback drift by baseline spread to avoid instability
            # when healthy means are near zero after centering/normalization.
            drift = float(
                np.clip(
                    np.mean(np.abs((x - self.mean) / np.clip(self.std, 1e-6, None))),
                    0.0,
                    1.0,
                )
            )
        raw_score = (
            float(atomic_score)
            + self.progression.variance_weight * variance_score
            + self.progression.drift_weight * drift
            - self.progression.raw_offset
        )
        final_score = float(_sigmoid(self.progression.alpha * raw_score))
        if not np.isfinite(final_score):
            final_score = 0.0
            quality_flag = quality_flag or "invalid_score"

        importance = np.abs(z)
        rank_idx = np.argsort(importance)[::-1]
        critical = [self.sensors[i] for i in rank_idx[: min(5, len(rank_idx))]]

        return {
            "score": float(np.clip(final_score, 0.0, 1.0)),
            "state": self._to_state(float(final_score)),
            "quality_flag": quality_flag,
            "layer_scores": layers,
            "raw_layer_scores": raw_layers,
            "atomic_score": float(atomic_score),
            "variance_score": float(variance_score),
            "drift": float(drift),
            "raw_score": float(raw_score),
            "sensor_importance": {self.sensors[i]: float(importance[i]) for i in range(len(self.sensors))},
            "critical_sensors": critical,
            "z_scores": {self.sensors[i]: float(z[i]) for i in range(len(self.sensors))},
        }

    def assess_sequence(self, timeline: Sequence[Mapping[str, float]]) -> List[Dict[str, object]]:
        outputs: List[Dict[str, object]] = []
        total = len(timeline)
        for idx, readings in enumerate(timeline):
            outputs.append(self.assess(readings, current_cycle_index=idx, total_cycles=total))
        if outputs:
            early = outputs[max(0, int(0.15 * (total - 1)))]["score"]
            mid = outputs[max(0, int(0.50 * (total - 1)))]["score"]
            late = outputs[max(0, int(0.85 * (total - 1)))]["score"]
            print("progression:", early, mid, late)
        return outputs


__all__ = ["LayerWeights", "ScoreProgressionConfig", "SII"]
