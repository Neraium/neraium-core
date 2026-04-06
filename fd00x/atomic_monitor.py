"""Atomic layer monitor.

Implements a modular multi-layer detector operating on micro-time,
micro-dynamics, micro-structure, micro-state, and micro-topology.
Some methods are intentionally v1 approximations behind clean interfaces:
- directional dependence proxy is lagged-correlation based (TE placeholder)
- latent regime model is centroid occupancy (HDP-HMM placeholder)
- dynamic mode tracking is windowed DMD drift (incremental DMD placeholder)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .atomic_alerting import AtomicAlertStateMachine
from .atomic_baseline import AtomicBaseline, AtomicBaselineLearner
from .atomic_diagnostics import top_edges, top_sensor_indices
from .atomic_fusion import AtomicFusionEngine


@dataclass
class AtomicMonitorState:
    alert_level: str
    total_score: float
    component_scores: Dict[str, float]
    dominant_detector: str
    localization: Dict[str, object]


@dataclass
class AtomicUpdate:
    score: float
    components: Dict[str, float]
    alert_level: str
    localization: Dict[str, object]
    dominant_detector: str


class AtomicMonitor:
    """Streaming atomic detector with baseline learning + per-sample updates."""

    def __init__(self, sensors: List[str], config: dict) -> None:
        self.sensors = list(sensors)
        self.config = dict(config)
        self.window_size = int(self.config.get("window_size", 120))
        self.forgetting_factor = float(self.config.get("forgetting_factor", 0.98))
        self.sde_dt = float(self.config.get("sde_dt", 1.0))
        self.te_k = int(self.config.get("te_k", 3))
        self.latent_states = int(self.config.get("latent_state_count", 4))
        self.dmd_rank = int(self.config.get("dmd_rank", 4))
        self.compute_intervals = dict(self.config.get("compute_intervals", {
            "structure": 1,
            "state": 5,
            "topology": 10,
            "events": 1,
        }))
        self.weights = dict(self.config.get("detector_weights", {
            "micro_dynamics": 0.25,
            "micro_structure": 0.2,
            "micro_time": 0.15,
            "micro_state": 0.2,
            "micro_topology": 0.2,
        }))

        thresholds = self.config.get("alert_thresholds", {"green_yellow": 0.45, "yellow_red": 0.65})
        self.alert_machine = AtomicAlertStateMachine(
            thresholds.get("green_yellow", 0.45),
            thresholds.get("yellow_red", 0.65),
        )
        self.fusion = AtomicFusionEngine(self.weights)

        self.baseline: Optional[AtomicBaseline] = None
        self.buffer = np.empty((0, len(self.sensors)), dtype=float)
        self.counter = 0
        self.timings: List[float] = []
        self.component_history: List[Dict[str, float]] = []
        self.score_history: List[float] = []

        self.online_mu = np.zeros(len(self.sensors), dtype=float)
        self.online_sigma = np.ones(len(self.sensors), dtype=float)
        self.prev_x: Optional[np.ndarray] = None
        self.latest_structure_shift = np.zeros((len(self.sensors), len(self.sensors)), dtype=float)
        self.latest_sensor_shift = np.zeros(len(self.sensors), dtype=float)

    def learn_baseline(self, baseline_data: np.ndarray) -> None:
        learner = AtomicBaselineLearner(
            latent_states=self.latent_states,
            dmd_rank=self.dmd_rank,
            sde_dt=self.sde_dt,
            event_level_std=float(self.config.get("event_level_std", 1.0)),
        )
        self.baseline = learner.fit(np.asarray(baseline_data, dtype=float))
        self.online_mu = self.baseline.sde_mu.copy()
        self.online_sigma = np.maximum(self.baseline.sde_sigma.copy(), 1e-6)

    def update(self, x_t: np.ndarray, timestamp: Optional[float] = None) -> AtomicUpdate:
        if self.baseline is None:
            raise RuntimeError("learn_baseline must be called before update")
        if timestamp is None:
            timestamp = float(self.counter)
        x_t = np.asarray(x_t, dtype=float)

        t0 = float(self.counter)
        self.counter += 1
        self.buffer = np.vstack([self.buffer, x_t])
        if self.buffer.shape[0] > self.window_size:
            self.buffer = self.buffer[-self.window_size :]

        micro_dynamics, sensor_dyn = self._micro_dynamics(x_t)
        micro_time, sensor_time = self._micro_time()

        micro_structure = None
        edge_shift = np.zeros_like(self.latest_structure_shift)
        if self.counter % int(self.compute_intervals.get("structure", 1)) == 0:
            micro_structure, edge_shift = self._micro_structure()

        micro_state = None
        if self.counter % int(self.compute_intervals.get("state", 5)) == 0:
            micro_state = self._micro_state()

        micro_topology = None
        if self.counter % int(self.compute_intervals.get("topology", 10)) == 0:
            micro_topology = self._micro_topology()

        fused = self.fusion.fuse(
            {
                "micro_dynamics": micro_dynamics,
                "micro_structure": micro_structure,
                "micro_time": micro_time,
                "micro_state": micro_state,
                "micro_topology": micro_topology,
            }
        )
        level = self.alert_machine.update(fused.total_score, fused.dominant_detector, timestamp)

        self.latest_sensor_shift = 0.7 * sensor_dyn + 0.3 * sensor_time
        self.latest_structure_shift = edge_shift

        self.component_history.append(fused.component_scores)
        self.score_history.append(fused.total_score)
        self.timings.append(float(self.counter - t0))

        loc = self._localization(fused.dominant_detector)
        return AtomicUpdate(
            score=fused.total_score,
            components=fused.component_scores,
            alert_level=level,
            localization=loc,
            dominant_detector=fused.dominant_detector,
        )

    def get_state(self) -> AtomicMonitorState:
        comps = self.component_history[-1] if self.component_history else {}
        score = self.score_history[-1] if self.score_history else 0.0
        dominant = max(comps, key=comps.get) if comps else "none"
        return AtomicMonitorState(
            alert_level=self.alert_machine.level,
            total_score=score,
            component_scores=comps,
            dominant_detector=dominant,
            localization=self._localization(dominant),
        )

    def _micro_dynamics(self, x_t: np.ndarray) -> tuple[float, np.ndarray]:
        if self.prev_x is None:
            self.prev_x = x_t
            return 0.0, np.zeros(len(self.sensors), dtype=float)
        dx = (x_t - self.prev_x) / max(self.sde_dt, 1e-6)
        lam = self.forgetting_factor
        self.online_mu = lam * self.online_mu + (1 - lam) * dx
        centered = dx - self.online_mu
        self.online_sigma = np.sqrt(lam * self.online_sigma**2 + (1 - lam) * centered**2)
        self.prev_x = x_t

        z_mu = np.abs(self.online_mu - self.baseline.sde_mu) / np.maximum(self.baseline.sde_sigma, 1e-6)
        z_sigma = np.abs(self.online_sigma - self.baseline.sde_sigma) / np.maximum(self.baseline.sde_sigma, 1e-6)
        sensor = np.clip(0.5 * (z_mu + z_sigma), 0.0, 5.0)
        score = float(np.clip(np.mean(sensor) / 3.0, 0.0, 1.0))
        return score, sensor

    def _micro_structure(self) -> tuple[float, np.ndarray]:
        if self.buffer.shape[0] < 8:
            return 0.0, np.zeros_like(self.latest_structure_shift)
        corr = np.corrcoef(self.buffer.T)
        corr = np.nan_to_num(corr)
        dep_shift = np.abs(corr - self.baseline.dependence)

        directional = self._directional_proxy(self.buffer)
        dir_shift = np.abs(directional - self.baseline.directional_proxy)

        edge_shift = 0.6 * dep_shift + 0.4 * dir_shift
        score = float(np.clip(edge_shift.mean() * 2.0, 0.0, 1.0))
        return score, edge_shift

    def _directional_proxy(self, data: np.ndarray) -> np.ndarray:
        if data.shape[0] < 3:
            return np.zeros((data.shape[1], data.shape[1]), dtype=float)
        x = data[:-1]
        y = data[1:]
        n = data.shape[1]
        out = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                a = x[:, i]
                b = y[:, j]
                if a.std() < 1e-9 or b.std() < 1e-9:
                    out[i, j] = 0.0
                else:
                    out[i, j] = float(np.nan_to_num(np.corrcoef(a, b)[0, 1]))
        return out

    def _micro_state(self) -> float:
        x = self.buffer[-1]
        centers = self.baseline.latent_centroids
        d2 = ((centers - x[None, :]) ** 2).sum(axis=1)
        probs = np.exp(-d2 / max(np.median(d2) + 1e-6, 1e-6))
        probs = probs / max(probs.sum(), 1e-6)
        shift = np.abs(probs - self.baseline.latent_occupancy)
        return float(np.clip(shift.mean() * 2.0, 0.0, 1.0))

    def _micro_topology(self) -> float:
        if self.buffer.shape[0] < 10:
            return 0.0
        x = self.buffer[:-1].T
        y = self.buffer[1:].T
        u, s, vt = np.linalg.svd(x, full_matrices=False)
        r = min(self.dmd_rank, len(s))
        u_r = u[:, :r]
        s_inv = np.diag(1.0 / np.maximum(s[:r], 1e-9))
        a_tilde = u_r.T @ y @ vt[:r, :].T @ s_inv
        eigvals, _ = np.linalg.eig(a_tilde)
        pad = min(len(eigvals), len(self.baseline.mode_eigenvalues))
        drift = np.abs(np.sort_complex(eigvals)[:pad] - np.sort_complex(self.baseline.mode_eigenvalues)[:pad])
        return float(np.clip(np.mean(np.abs(drift)) * 2.0, 0.0, 1.0))

    def _micro_time(self) -> tuple[float, np.ndarray]:
        if self.buffer.shape[0] < 4:
            return 0.0, np.zeros(len(self.sensors), dtype=float)
        x = self.buffer
        mean = self.baseline.mean
        std = np.sqrt(np.diag(self.baseline.cov))
        std = np.where(std < 1e-6, 1.0, std)
        z = np.abs((x - mean) / std)
        level = float(self.config.get("event_level_std", 1.0))
        event = z > level
        sensor = np.zeros(len(self.sensors), dtype=float)
        for i in range(len(self.sensors)):
            transitions = np.where(np.diff(event[:, i].astype(int)) != 0)[0]
            rate = len(transitions) / max(len(x), 1)
            rate_diff = abs(rate - self.baseline.event_rate[i]) / max(self.baseline.event_rate[i] + 1e-3, 1e-3)
            if len(transitions) > 2:
                gaps = np.diff(transitions)
                cv = gaps.std() / max(gaps.mean(), 1e-6)
            else:
                cv = 1.0
            cv_diff = abs(cv - self.baseline.event_interval_cv[i]) / max(self.baseline.event_interval_cv[i] + 1e-3, 1e-3)
            sensor[i] = 0.5 * (rate_diff + cv_diff)
        return float(np.clip(sensor.mean() / 4.0, 0.0, 1.0)), np.clip(sensor, 0.0, 10.0)

    def _localization(self, dominant: str) -> Dict[str, object]:
        top_sensor_ids = top_sensor_indices(self.latest_sensor_shift, k=3)
        sensor_names = [self.sensors[i] for i in top_sensor_ids]
        shifted_edges = top_edges(self.latest_structure_shift, k=3)
        return {
            "top_contributing_sensors": sensor_names,
            "top_shifted_relationships": [
                {
                    "from": self.sensors[i],
                    "to": self.sensors[j],
                    "magnitude": float(v),
                }
                for i, j, v in shifted_edges
            ],
            "dominant_sub_detector": dominant,
        }
