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


@dataclass
class LayerWeights:
    l1: float = 0.26
    l2: float = 0.20
    l3: float = 0.20
    l4: float = 0.17
    l5: float = 0.17


class SII:
    """Atomic five-layer SII engine."""

    def __init__(
        self,
        sensors: Iterable[str],
        baseline_data: np.ndarray,
        *,
        thresholds: Mapping[str, float] | None = None,
        layer_weights: LayerWeights | None = None,
    ) -> None:
        self.sensors: List[str] = list(sensors)
        if len(self.sensors) == 0:
            raise ValueError("SII requires at least one sensor.")

        self.baseline = np.asarray(baseline_data, dtype=float)
        if self.baseline.ndim != 2 or self.baseline.shape[1] != len(self.sensors):
            raise ValueError(
                f"baseline_data must have shape (n, {len(self.sensors)}), got {self.baseline.shape}."
            )

        self.mean = self.baseline.mean(axis=0)
        self.std = np.clip(self.baseline.std(axis=0), 1e-8, None)
        self.corr = np.corrcoef(self.baseline, rowvar=False)
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

    def _normalize_layers(self, layer_scores: Dict[str, float]) -> Dict[str, float]:
        return {
            "l1": float(_sigmoid(layer_scores["l1"] - 0.8)),
            "l2": float(_sigmoid(layer_scores["l2"] - 0.9)),
            "l3": float(_sigmoid(layer_scores["l3"] - 0.7)),
            "l4": float(_sigmoid(layer_scores["l4"] - 0.6)),
            "l5": float(_sigmoid(layer_scores["l5"] - 0.8)),
        }

    def _to_state(self, score: float) -> str:
        if score < self.thresholds["healthy"]:
            return "healthy"
        if score < self.thresholds["elevated"]:
            return "elevated"
        if score < self.thresholds["caution"]:
            return "caution"
        if score < self.thresholds["critical"]:
            return "warning"
        return "critical"

    def assess(self, readings: Mapping[str, float]) -> Dict[str, object]:
        """
        Assess one sensor snapshot.

        Returns score, state, per-layer scores, and sensor-level importance.
        """

        x = _to_vector(readings, self.sensors)
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

        importance = np.abs(z)
        rank_idx = np.argsort(importance)[::-1]
        critical = [self.sensors[i] for i in rank_idx[: min(5, len(rank_idx))]]

        return {
            "score": float(np.clip(atomic_score, 0.0, 1.0)),
            "state": self._to_state(float(atomic_score)),
            "layer_scores": layers,
            "raw_layer_scores": raw_layers,
            "sensor_importance": {self.sensors[i]: float(importance[i]) for i in range(len(self.sensors))},
            "critical_sensors": critical,
            "z_scores": {self.sensors[i]: float(z[i]) for i in range(len(self.sensors))},
        }


__all__ = ["LayerWeights", "SII"]
