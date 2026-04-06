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
    score: float                              # raw fused score (for downstream EMA in detector.py)
    components: Dict[str, float]              # normalized component scores
    alert_level: str
    localization: Dict[str, object]
    dominant_detector: str
    # Diagnostic fields (new)
    smoothed_score: float = 0.0               # EMA-smoothed fused score (drives alert state machine)
    raw_component_scores: Dict[str, float] = field(default_factory=dict)
    normalized_component_scores: Dict[str, float] = field(default_factory=dict)
    active_detector_count: int = 0


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

        thresholds = self.config.get("alert_thresholds", {"green_yellow": 0.60, "yellow_red": 0.80})
        self.alert_machine = AtomicAlertStateMachine(
            thresholds.get("green_yellow", 0.60),
            thresholds.get("yellow_red", 0.80),
        )
        self.fusion = AtomicFusionEngine(
            self.weights,
            activation_floor=float(self.config.get("fusion_activation_floor", 0.15)),
            min_active=int(self.config.get("fusion_min_active", 2)),
            downweight_factor=float(self.config.get("fusion_downweight_factor", 0.3)),
        )

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

        # Per-component baseline score statistics (populated during learn_baseline replay)
        self._component_score_stats: Dict[str, Dict[str, float]] = {}

        # EMA smoothing for fused score (alert state machine uses this)
        self._score_ema_alpha: float = float(self.config.get("score_ema_alpha", 0.3))
        self._smoothed_score: float = 0.0

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

        # Learn per-component score distributions via replay through baseline data
        self._component_score_stats = self._fit_component_baselines(
            np.asarray(baseline_data, dtype=float)
        )

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

        # --- Compute raw component scores ---
        micro_dynamics_raw, sensor_dyn = self._micro_dynamics(x_t)
        micro_time_raw, sensor_time = self._micro_time()

        micro_structure_raw = None
        edge_shift = np.zeros_like(self.latest_structure_shift)
        if self.counter % int(self.compute_intervals.get("structure", 1)) == 0:
            micro_structure_raw, edge_shift = self._micro_structure()

        micro_state_raw = None
        if self.counter % int(self.compute_intervals.get("state", 5)) == 0:
            micro_state_raw = self._micro_state()

        micro_topology_raw = None
        if self.counter % int(self.compute_intervals.get("topology", 10)) == 0:
            micro_topology_raw = self._micro_topology()

        raw_scores = {
            "micro_dynamics": micro_dynamics_raw,
            "micro_structure": micro_structure_raw,
            "micro_time": micro_time_raw,
            "micro_state": micro_state_raw,
            "micro_topology": micro_topology_raw,
        }

        # --- Normalize component scores against baseline scale ---
        normalized_scores = self._normalize_component_scores(raw_scores)

        # --- Fuse (includes per-component clipping + gating inside fusion engine) ---
        fused = self.fusion.fuse(normalized_scores)

        # --- EMA smoothing on fused score for alert state machine ---
        self._smoothed_score = (
            self._score_ema_alpha * fused.total_score
            + (1.0 - self._score_ema_alpha) * self._smoothed_score
        )

        # --- Alert state machine uses smoothed score ---
        level = self.alert_machine.update(
            self._smoothed_score, fused.dominant_detector, timestamp
        )

        self.latest_sensor_shift = 0.7 * sensor_dyn + 0.3 * sensor_time
        self.latest_structure_shift = edge_shift

        self.component_history.append(fused.component_scores)
        self.score_history.append(fused.total_score)
        self.timings.append(float(self.counter - t0))

        loc = self._localization(fused.dominant_detector)

        # Build raw scores dict (only non-None values)
        raw_scores_clean = {k: float(v) for k, v in raw_scores.items() if v is not None}

        return AtomicUpdate(
            score=fused.total_score,
            components=fused.component_scores,
            alert_level=level,
            localization=loc,
            dominant_detector=fused.dominant_detector,
            smoothed_score=self._smoothed_score,
            raw_component_scores=raw_scores_clean,
            normalized_component_scores=dict(fused.component_scores),
            active_detector_count=fused.active_count,
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

    # ─────────────────────────────────────────────────────────────────────────
    # Baseline replay: fit per-component healthy score distributions
    # ─────────────────────────────────────────────────────────────────────────

    def _fit_component_baselines(
        self, baseline_data: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """Replay baseline data to learn per-component healthy score distributions.

        Saves and restores all mutable runtime state so this call is side-effect
        free from the caller's perspective.
        """
        # --- Save current runtime state ---
        saved_buffer = self.buffer.copy()
        saved_counter = self.counter
        saved_prev_x = self.prev_x.copy() if self.prev_x is not None else None
        saved_online_mu = self.online_mu.copy()
        saved_online_sigma = self.online_sigma.copy()
        saved_comp_hist = list(self.component_history)
        saved_score_hist = list(self.score_history)
        saved_timings = list(self.timings)
        saved_alert_level = self.alert_machine.level
        saved_alert_history = list(self.alert_machine.history)
        saved_sensor_shift = self.latest_sensor_shift.copy()
        saved_structure_shift = self.latest_structure_shift.copy()
        saved_smoothed = self._smoothed_score

        # --- Reset for replay ---
        self.buffer = np.empty((0, len(self.sensors)), dtype=float)
        self.counter = 0
        self.prev_x = None
        self.online_mu = self.baseline.sde_mu.copy()
        self.online_sigma = np.maximum(self.baseline.sde_sigma.copy(), 1e-6)
        self.component_history = []
        self.score_history = []
        self.timings = []
        self.alert_machine.level = "GREEN"
        self.alert_machine.history = []
        self.latest_sensor_shift = np.zeros(len(self.sensors), dtype=float)
        self.latest_structure_shift = np.zeros(
            (len(self.sensors), len(self.sensors)), dtype=float
        )
        self._smoothed_score = 0.0

        # --- Replay and collect raw component scores ---
        store: Dict[str, List[float]] = {k: [] for k in self.weights}
        for x_t in baseline_data:
            x_t = np.asarray(x_t, dtype=float)
            self.counter += 1
            self.buffer = np.vstack([self.buffer, x_t])
            if self.buffer.shape[0] > self.window_size:
                self.buffer = self.buffer[-self.window_size :]

            dyn, sensor_dyn = self._micro_dynamics(x_t)
            time_s, sensor_time = self._micro_time()

            struct_s = None
            edge_shift = np.zeros_like(self.latest_structure_shift)
            if self.counter % int(self.compute_intervals.get("structure", 1)) == 0:
                struct_s, edge_shift = self._micro_structure()

            state_s = None
            if self.counter % int(self.compute_intervals.get("state", 5)) == 0:
                state_s = self._micro_state()

            topo_s = None
            if self.counter % int(self.compute_intervals.get("topology", 10)) == 0:
                topo_s = self._micro_topology()

            self.latest_sensor_shift = 0.7 * sensor_dyn + 0.3 * sensor_time
            self.latest_structure_shift = edge_shift

            store["micro_dynamics"].append(dyn)
            store["micro_time"].append(time_s)
            if struct_s is not None:
                store["micro_structure"].append(struct_s)
            if state_s is not None:
                store["micro_state"].append(state_s)
            if topo_s is not None:
                store["micro_topology"].append(topo_s)

        # --- Restore state ---
        self.buffer = saved_buffer
        self.counter = saved_counter
        self.prev_x = saved_prev_x
        self.online_mu = saved_online_mu
        self.online_sigma = saved_online_sigma
        self.component_history = saved_comp_hist
        self.score_history = saved_score_hist
        self.timings = saved_timings
        self.alert_machine.level = saved_alert_level
        self.alert_machine.history = saved_alert_history
        self.latest_sensor_shift = saved_sensor_shift
        self.latest_structure_shift = saved_structure_shift
        self._smoothed_score = saved_smoothed

        # --- Compute robust distribution stats ---
        stats: Dict[str, Dict[str, float]] = {}
        for name, vals in store.items():
            if len(vals) >= 5:
                arr = np.asarray(vals, dtype=float)
                median = float(np.median(arr))
                mad = float(np.median(np.abs(arr - median)))
                stats[name] = {
                    "median": median,
                    "mad": max(mad, 1e-6),
                    "p90": float(np.percentile(arr, 90)),
                }
            else:
                # Insufficient data: treat any score as anomalous by setting median=0, mad=1
                stats[name] = {"median": 0.0, "mad": 1.0, "p90": 1.0}

        return stats

    # ─────────────────────────────────────────────────────────────────────────
    # Per-component score normalization
    # ─────────────────────────────────────────────────────────────────────────

    def _normalize_component_scores(
        self, raw: Dict[str, Optional[float]]
    ) -> Dict[str, Optional[float]]:
        """Normalize each component score relative to its healthy baseline distribution.

        Only amplifies scores that **exceed** the p90 of healthy behaviour — scores
        within the healthy band map to 0.  The p90 is used as the zero-point; a
        score equal to 2×p90 maps to 1.0.

        Formula:
            normalized = clip( max(raw - p90, 0) / max(p90, 1e-6), 0, 1 )

        This guarantees that normal operating transients (warm-up, operating-condition
        shifts within the healthy regime) do not drive the fused anomaly score, while
        genuine degradation — which pushes a component above its healthy maximum —
        produces a clear positive normalized score.

        Components with no baseline stats pass through clipped to [0, 1].
        """
        out: Dict[str, Optional[float]] = {}
        for k, v in raw.items():
            if v is None:
                out[k] = None
                continue
            stats = self._component_score_stats.get(k)
            if not stats:
                # No baseline stats yet: pass through clipped
                out[k] = float(np.clip(v, 0.0, 1.0))
                continue
            raw_val = float(v)
            p90 = stats["p90"]
            # Excess above the healthy p90, normalized so that 2×p90 → 1.0
            excess = max(raw_val - p90, 0.0)
            normalized = float(np.clip(excess / max(p90, 1e-6), 0.0, 1.0))
            out[k] = normalized
        return out

    # ─────────────────────────────────────────────────────────────────────────
    # Component detectors
    # ─────────────────────────────────────────────────────────────────────────

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
        # Require enough samples to represent operating-condition variability.
        # FD004 cycles through multiple conditions; a very small window produces
        # misleadingly high correlation drift during warm-up.
        min_samples = max(8, self.window_size // 4)
        if self.buffer.shape[0] < min_samples:
            return 0.0, np.zeros_like(self.latest_structure_shift)

        # Compute correlation matrix manually to avoid NaN from zero-variance columns.
        # np.corrcoef internally divides by std and will produce NaN for constant columns;
        # by computing manually we can intercept and zero-out those columns.
        window = self.buffer
        n, d = window.shape
        # ddof=1 std to match Pearson correlation denominator
        std = window.std(axis=0, ddof=1) if n > 1 else window.std(axis=0)
        zero_var = std < 1e-8
        std_safe = np.where(zero_var, 1.0, std)
        centered = window - window.mean(axis=0)
        normed = centered / std_safe          # shape (n, d)
        corr = (normed.T @ normed) / max(n - 1, 1)
        # Zero-out rows/cols for channels with no variance (undefined correlation)
        corr[zero_var, :] = 0.0
        corr[:, zero_var] = 0.0
        np.fill_diagonal(corr, 1.0)
        corr = np.clip(corr, -1.0, 1.0)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

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
