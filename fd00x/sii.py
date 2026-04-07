"""
SII v2.0 core model.

This module provides a lightweight implementation of Systemic Infrastructure
Intelligence (SII) with five fundamental scoring layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional dependency
    pd = None


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


def _sanitize_scalar(value: float) -> float:
    return float(np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0))


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


def _extract_baseline_slice(data: np.ndarray) -> np.ndarray:
    total = int(data.shape[0])
    if total <= 0:
        return data
    baseline_len = int(np.ceil(total * 0.20))
    if total >= 30:
        baseline_len = max(30, baseline_len)
    baseline_len = max(1, min(total, baseline_len))
    return data[:baseline_len].copy()


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
    trend_weight: float = 0.25
    trend_window: int = 5
    progression_decay_floor: float = 0.985
    score_ema_alpha: float = 0.25
    state_hysteresis: float = 0.03


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

        if pd is not None and isinstance(baseline_data, pd.DataFrame):
            baseline_arr = baseline_data.to_numpy(dtype=float)
        else:
            baseline_arr = np.asarray(baseline_data, dtype=float)
        if baseline_arr.ndim != 2 or baseline_arr.shape[1] != len(self.sensors):
            raise ValueError(
                f"baseline_data must have shape (n, {len(self.sensors)}), got {baseline_arr.shape}."
            )

        self.progression = progression or ScoreProgressionConfig()
        self.baseline = _extract_baseline_slice(baseline_arr)

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
        return _sanitize_scalar(np.mean(np.abs(z)))

    def _layer_2_coupling_shift(self, z: np.ndarray) -> float:
        outer = np.outer(z, z)
        diff = np.abs(outer - self.corr)
        return _sanitize_scalar(np.mean(diff))

    def _layer_3_energy(self, z: np.ndarray) -> float:
        return _sanitize_scalar(np.sqrt(np.mean(z**2)))

    def _layer_4_cov_projection(self, z: np.ndarray) -> float:
        projected = self.cov @ z
        return _sanitize_scalar(np.linalg.norm(projected) / max(1, len(z)))

    def _layer_5_nonlinear_tension(self, z: np.ndarray) -> float:
        x = np.abs(np.tanh(z)) + np.abs(np.sin(z))
        return _sanitize_scalar(np.mean(x))

    def _variance_score(self, x: np.ndarray) -> float:
        sample_variance = _sanitize_scalar(np.var(x - self.mean))
        return _sanitize_scalar(np.log1p(sample_variance / self.baseline_variance))

    def _normalize_layers(self, layer_scores: Dict[str, float]) -> Dict[str, float]:
        norm = {
            "l1": _sanitize_scalar(_sigmoid(layer_scores["l1"] - 0.8)),
            "l2": _sanitize_scalar(_sigmoid(layer_scores["l2"] - 0.9)),
            "l3": _sanitize_scalar(_sigmoid(layer_scores["l3"] - 0.7)),
            "l4": _sanitize_scalar(_sigmoid(layer_scores["l4"] - 0.6)),
            "l5": _sanitize_scalar(_sigmoid(layer_scores["l5"] - 0.8)),
        }
        return {k: _sanitize_scalar(np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)) for k, v in norm.items()}

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
        z = np.nan_to_num((x - self.mean) / self.std, nan=0.0, posinf=0.0, neginf=0.0)

        raw_layers = {
            "l1": self._layer_1_deviation(z),
            "l2": self._layer_2_coupling_shift(z),
            "l3": self._layer_3_energy(z),
            "l4": self._layer_4_cov_projection(z),
            "l5": self._layer_5_nonlinear_tension(z),
        }
        raw_layers = {
            k: _sanitize_scalar(np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)) for k, v in raw_layers.items()
        }
        layers = self._normalize_layers(raw_layers)
        atomic_score = _sanitize_scalar(
            self.weights.l1 * layers["l1"]
            + self.weights.l2 * layers["l2"]
            + self.weights.l3 * layers["l3"]
            + self.weights.l4 * layers["l4"]
            + self.weights.l5 * layers["l5"]
        )
        variance_score = _sanitize_scalar(self._variance_score(x))
        if current_cycle_index is not None and total_cycles is not None and total_cycles > 1:
            life_frac = current_cycle_index / float(total_cycles - 1)
            drift = _sanitize_scalar(np.clip(np.mean(np.abs(z)) * life_frac, 0.0, 1.0))
        else:
            drift = _sanitize_scalar(
                np.clip(
                    np.mean(np.abs((x - self.mean) / np.clip(self.std, 1e-6, None))),
                    0.0,
                    1.0,
                )
            )
        raw_score = _sanitize_scalar(
            float(atomic_score)
            + self.progression.variance_weight * variance_score
            + self.progression.drift_weight * drift
            - self.progression.raw_offset
        )
        final_score = _sanitize_scalar(_sigmoid(self.progression.alpha * raw_score))
        if not np.isfinite(final_score):
            final_score = 0.0
            quality_flag = quality_flag or "invalid_score"

        importance = np.abs(z)
        rank_idx = np.argsort(importance)[::-1]
        critical = [self.sensors[i] for i in rank_idx[: min(5, len(rank_idx))]]

        return {
            "score": _sanitize_scalar(np.clip(final_score, 0.0, 1.0)),
            "state": self._to_state(float(final_score)),
            "quality_flag": quality_flag,
            "layer_scores": layers,
            "raw_layer_scores": raw_layers,
            "atomic_score": _sanitize_scalar(atomic_score),
            "variance_score": _sanitize_scalar(variance_score),
            "drift": _sanitize_scalar(drift),
            "raw_score": _sanitize_scalar(raw_score),
            "sensor_importance": {self.sensors[i]: float(importance[i]) for i in range(len(self.sensors))},
            "critical_sensors": critical,
            "z_scores": {self.sensors[i]: float(z[i]) for i in range(len(self.sensors))},
        }

    def _trend_component(self, history_scores: Sequence[float]) -> float:
        if len(history_scores) < 2:
            return 0.0
        k = max(1, int(self.progression.trend_window))
        early = np.asarray(history_scores[:k], dtype=float)
        recent = np.asarray(history_scores[-k:], dtype=float)
        trend = _sanitize_scalar(np.mean(recent) - np.mean(early))
        denom = max(float(np.mean(self.std)), 1e-6)
        normalized = _sanitize_scalar(trend / denom)
        bounded = float(np.clip(normalized, -1.0, 1.0))
        return _sanitize_scalar(self.progression.trend_weight * bounded)

    def _apply_progression_postprocessing(self, outputs: List[Dict[str, object]]) -> List[Dict[str, object]]:
        if not outputs:
            return outputs
        alpha = float(np.clip(self.progression.score_ema_alpha, 0.0, 1.0))
        decay_floor = float(np.clip(self.progression.progression_decay_floor, 0.0, 1.0))
        hysteresis = float(max(0.0, self.progression.state_hysteresis))
        smoothed_prev = _sanitize_scalar(outputs[0].get("score", 0.0))
        progression_prev = smoothed_prev
        states: List[str] = []
        raw_history: List[float] = []
        for idx, out in enumerate(outputs):
            raw = _sanitize_scalar(out.get("score", 0.0))
            raw_history.append(raw)
            trend_boost = self._trend_component(raw_history)
            raw_with_trend = _sanitize_scalar(np.clip(raw + trend_boost, 0.0, 1.0))
            if idx == 0:
                smoothed = raw_with_trend
            else:
                smoothed = _sanitize_scalar(alpha * raw_with_trend + (1.0 - alpha) * smoothed_prev)
            progression_score = _sanitize_scalar(max(smoothed, progression_prev * decay_floor))
            state = self._to_state(progression_score)
            if states and states[-1] in {"caution", "critical"} and state == "healthy":
                healthy_gate = self.thresholds["healthy"] - hysteresis
                if progression_score >= healthy_gate:
                    state = "elevated"

            out["trend_component"] = _sanitize_scalar(trend_boost)
            out["score_raw"] = raw
            out["score_smooth"] = _sanitize_scalar(smoothed)
            out["score"] = _sanitize_scalar(np.clip(progression_score, 0.0, 1.0))
            out["state"] = state

            smoothed_prev = smoothed
            progression_prev = progression_score
            states.append(state)
        return outputs

    def assess_sequence(self, timeline: Sequence[Mapping[str, float]]) -> List[Dict[str, object]]:
        outputs: List[Dict[str, object]] = []
        total = len(timeline)
        for idx, readings in enumerate(timeline):
            outputs.append(self.assess(readings, current_cycle_index=idx, total_cycles=total))
        outputs = self._apply_progression_postprocessing(outputs)
        if outputs:
            early = outputs[max(0, int(0.15 * (total - 1)))]["score"]
            mid = outputs[max(0, int(0.50 * (total - 1)))]["score"]
            late = outputs[max(0, int(0.85 * (total - 1)))]["score"]
            print("progression:", early, mid, late)
        return outputs


__all__ = ["LayerWeights", "ScoreProgressionConfig", "SII"]
