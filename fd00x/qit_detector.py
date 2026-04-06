"""QIT (Quantum-Information-Topological) hierarchical degradation detector.

This module provides a production-friendly detector that fuses five
mathematically distinct signals with progressive execution and automatic
fallback behavior.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import lzma
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np


_STATE_ORDER = ("GREEN", "AMBER", "YELLOW", "RED")


@dataclass
class QITConfig:
    """Configuration for the hierarchical QIT detector."""

    quantum_trigger: float = 0.25
    info_trigger: float = 0.40
    topological_interval: int = 50
    algorithmic_interval: int = 200

    amber_enter: float = 0.25
    yellow_enter: float = 0.45
    red_enter: float = 0.65
    amber_exit: float = 0.18
    yellow_exit: float = 0.36
    red_exit: float = 0.52

    enter_persistence: int = 3
    exit_persistence: int = 2

    weights: Mapping[str, float] = field(
        default_factory=lambda: {
            "quantum": 0.30,
            "information": 0.25,
            "free_energy": 0.20,
            "topological": 0.15,
            "algorithmic": 0.10,
        }
    )


class QITDetector:
    """Hierarchical detector combining quantum, information, and topology signals."""

    def __init__(
        self,
        sensors: Iterable[str],
        baseline_data: np.ndarray,
        config: Optional[QITConfig] = None,
    ) -> None:
        self.sensors = list(sensors)
        self.config = config or QITConfig()
        self._step = 0
        self._state = "GREEN"
        self._above_count = 0
        self._below_count = 0
        self._last_scores: Dict[str, float] = {
            "quantum": 0.0,
            "information": 0.0,
            "free_energy": 0.0,
            "topological": 0.0,
            "algorithmic": 0.0,
        }

        data = np.asarray(baseline_data, dtype=float)
        if data.ndim != 2:
            raise ValueError("baseline_data must be a 2D array (n_samples, n_features)")
        if data.shape[1] != len(self.sensors):
            raise ValueError("sensor count must match baseline_data feature dimension")

        self._mu = np.mean(data, axis=0)
        self._sigma = np.std(data, axis=0)
        self._sigma = np.where(self._sigma < 1e-8, 1e-8, self._sigma)
        self._cov = np.cov(data, rowvar=False)
        self._cov += np.eye(self._cov.shape[0]) * 1e-6
        self._precision = np.linalg.pinv(self._cov)

        baseline_centered = (data - self._mu) / self._sigma
        self._quantum_basis = self._build_quantum_basis(baseline_centered)
        self._quantum_ref = np.mean(np.exp(1j * baseline_centered), axis=0)
        self._entropy_ref = self._safe_entropy(np.abs(baseline_centered).mean(axis=0))

        self._window: List[np.ndarray] = []
        self._window_size = max(30, min(200, data.shape[0]))

        self._topology_ref = self._topology_signature(baseline_centered)
        self._algo_ref = self._algorithmic_complexity(data[-self._window_size :])

    def detect(self, sensor_values: Iterable[float]) -> Dict[str, Any]:
        x = np.asarray(list(sensor_values), dtype=float)
        if x.shape[0] != len(self.sensors):
            raise ValueError("sensor_values length must match detector sensors")

        self._step += 1
        self._window.append(x)
        if len(self._window) > self._window_size:
            self._window = self._window[-self._window_size :]

        z = (x - self._mu) / self._sigma

        scores: Dict[str, float] = {}
        evidence = 0.0

        scores["quantum"] = self._quantum_score(z)
        evidence += scores["quantum"]

        if scores["quantum"] > self.config.quantum_trigger:
            scores["information"] = self._information_geometry_score(z)
            evidence += scores["information"]
        else:
            scores["information"] = 0.0

        if scores["information"] > self.config.info_trigger:
            scores["free_energy"] = self._free_energy_score(z)
            evidence += scores["free_energy"]
        else:
            scores["free_energy"] = 0.0

        if self._step % max(1, self.config.topological_interval) == 0:
            scores["topological"] = self._topological_score()
        else:
            scores["topological"] = self._last_scores["topological"]
        evidence += scores["topological"]

        if self._step % max(1, self.config.algorithmic_interval) == 0:
            scores["algorithmic"] = self._algorithmic_score()
        else:
            scores["algorithmic"] = self._last_scores["algorithmic"]
        evidence += scores["algorithmic"]

        self._last_scores.update(scores)
        fused_total = self._fuse(scores)
        state, changed = self._update_state(fused_total)

        diagnostics = {
            "evidence": float(evidence),
            "state_changed": changed,
            "step": self._step,
            "layer_execution": {
                "quantum": True,
                "information": scores["quantum"] > self.config.quantum_trigger,
                "free_energy": scores["information"] > self.config.info_trigger,
                "topological": self._step % max(1, self.config.topological_interval) == 0,
                "algorithmic": self._step % max(1, self.config.algorithmic_interval) == 0,
            },
        }

        return {
            "scores": {k: float(np.clip(v, 0.0, 1.0)) for k, v in scores.items()},
            "fused": {"total": float(np.clip(fused_total, 0.0, 1.0)), "state": state},
            "can_alert": state == "RED",
            "diagnostics": diagnostics,
        }

    def _build_quantum_basis(self, data: np.ndarray) -> np.ndarray:
        try:
            _, _, vt = np.linalg.svd(data, full_matrices=False)
            rank = max(1, min(10, vt.shape[0]))
            return vt[:rank]
        except np.linalg.LinAlgError:
            return np.eye(data.shape[1], dtype=float)

    def _quantum_score(self, z: np.ndarray) -> float:
        phi = np.exp(1j * z)
        proj = np.real(np.abs(phi @ self._quantum_basis.T) ** 2).mean()
        ref_sim = np.abs(np.vdot(phi, self._quantum_ref)) ** 2
        denom = max(1e-8, np.vdot(phi, phi).real * np.vdot(self._quantum_ref, self._quantum_ref).real)
        kernel_sim = float(np.clip(ref_sim / denom, 0.0, 1.0))
        anomaly = 1.0 - kernel_sim
        score = 0.5 * float(np.clip(proj / max(1.0, self._quantum_basis.shape[0]), 0.0, 1.0)) + 0.5 * anomaly
        return float(np.clip(score, 0.0, 1.0))

    def _information_geometry_score(self, z: np.ndarray) -> float:
        delta = z
        d2 = float(delta.T @ self._precision @ delta)
        d = float(np.sqrt(max(d2, 0.0)))
        return float(1.0 - np.exp(-0.15 * d))

    def _free_energy_score(self, z: np.ndarray) -> float:
        energy = 0.5 * float(z.T @ z)
        entropy_gap = abs(self._safe_entropy(np.abs(z)) - self._entropy_ref)
        free_energy = energy + entropy_gap
        return float(1.0 - np.exp(-0.1 * free_energy))

    def _topology_signature(self, data: np.ndarray) -> np.ndarray:
        if data.shape[0] < 5:
            return np.array([0.0, 0.0, 0.0], dtype=float)
        sample = data[-min(64, data.shape[0]) :]
        pdists = []
        for i in range(sample.shape[0] - 1):
            d = np.linalg.norm(sample[i + 1 :] - sample[i], axis=1)
            pdists.append(d)
        flat = np.concatenate(pdists) if pdists else np.array([0.0])
        q = np.quantile(flat, [0.25, 0.5, 0.75])
        return q.astype(float)

    def _topological_score(self) -> float:
        data = np.asarray(self._window, dtype=float)
        if data.shape[0] < 5:
            return self._last_scores["topological"]
        z = (data - self._mu) / self._sigma
        cur = self._topology_signature(z)
        dist = float(np.linalg.norm(cur - self._topology_ref))
        return float(1.0 - np.exp(-dist))

    def _algorithmic_complexity(self, data: np.ndarray) -> float:
        payload = np.asarray(data, dtype=np.float32).tobytes()
        if not payload:
            return 0.0
        compressed = lzma.compress(payload, preset=3)
        return float(len(compressed) / max(1, len(payload)))

    def _algorithmic_score(self) -> float:
        data = np.asarray(self._window, dtype=float)
        if data.shape[0] < 10:
            return self._last_scores["algorithmic"]
        cur = self._algorithmic_complexity(data)
        delta = max(0.0, cur - self._algo_ref)
        return float(np.clip(delta / max(1e-6, self._algo_ref + 0.1), 0.0, 1.0))

    def _safe_entropy(self, values: np.ndarray) -> float:
        v = np.asarray(values, dtype=float)
        v = np.abs(v)
        total = float(np.sum(v))
        if total <= 0.0:
            return 0.0
        p = v / total
        p = np.clip(p, 1e-12, 1.0)
        return float(-np.sum(p * np.log(p)))

    def _fuse(self, scores: Mapping[str, float]) -> float:
        total_w = 0.0
        total = 0.0
        for name, weight in self.config.weights.items():
            val = float(scores.get(name, 0.0))
            w = float(max(0.0, weight))
            total += w * val
            total_w += w
        if total_w <= 0.0:
            return 0.0
        return total / total_w

    def _update_state(self, fused_total: float) -> tuple[str, bool]:
        prev = self._state
        desired = prev

        if fused_total >= self.config.red_enter:
            desired = "RED"
        elif fused_total >= self.config.yellow_enter:
            desired = "YELLOW"
        elif fused_total >= self.config.amber_enter:
            desired = "AMBER"
        else:
            desired = "GREEN"

        exit_thresholds = {
            "RED": self.config.red_exit,
            "YELLOW": self.config.yellow_exit,
            "AMBER": self.config.amber_exit,
            "GREEN": 0.0,
        }

        if _STATE_ORDER.index(desired) > _STATE_ORDER.index(prev):
            self._above_count += 1
            self._below_count = 0
            if self._above_count >= self.config.enter_persistence:
                self._state = desired
                self._above_count = 0
        elif _STATE_ORDER.index(desired) < _STATE_ORDER.index(prev):
            if fused_total < exit_thresholds.get(prev, 0.0):
                self._below_count += 1
            else:
                self._below_count = 0
            self._above_count = 0
            if self._below_count >= self.config.exit_persistence:
                self._state = desired
                self._below_count = 0
        else:
            self._above_count = 0
            self._below_count = 0

        return self._state, self._state != prev


def create_qit_detector(
    sensors: Iterable[str],
    baseline_data: np.ndarray,
    config: Optional[QITConfig] = None,
) -> QITDetector:
    """Factory for a QIT detector instance."""

    return QITDetector(sensors=sensors, baseline_data=baseline_data, config=config)
