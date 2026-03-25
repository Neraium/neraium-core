from __future__ import annotations

from collections import Counter, deque
import json
from datetime import datetime, timezone
from typing import Dict, List, Optional

import os
import numpy as np

from neraium_core.causal import causal_metrics, granger_causality_matrix
from neraium_core.causal_attribution import causal_attribution
from neraium_core.causal_graph import (
    causal_graph_metrics,
    causal_propagation_spread,
    causal_root_cause_chains,
)
from neraium_core.branching import derive_branching_analysis
from neraium_core.data_quality import (
    compute_data_quality,
    data_quality_summary,
    impute_missing_simple,
    should_use_degraded_analytics,
)
from neraium_core.decision_layer import decision_output
from neraium_core.directional import directional_metrics, lagged_correlation_matrix
from neraium_core.early_warning import early_warning_metrics
from neraium_core.entropy import interaction_entropy
from neraium_core.experimental_analytics.hierarchy_analysis import analyze_hierarchy_cascade
from neraium_core.experimental_analytics.constraint_analysis import analyze_constraint_lock_in
from neraium_core.experimental_analytics.trajectory_analysis import classify_trajectory_path
from neraium_core.experimental_analytics.horizon_analysis import estimate_risk_horizon
from neraium_core.experimental_analytics.counterfactual_simulation import simulate_counterfactual_futures
from neraium_core.forecast_models import forecast_next, time_to_threshold_ar1
from neraium_core.forecasting import instability_trend, time_to_instability
from neraium_core.geometry import (
    correlation_matrix,
    normalize_window,
    signal_structural_importance,
    structural_drift,
)
from neraium_core.graph import graph_metrics, thresholded_adjacency
from neraium_core.regime import build_regime_signature, assign_regime, update_regime_library
from neraium_core.regime_store import RegimeStore
from neraium_core.scoring import canonicalize_components, canonicalize_weights, composite_instability_score_normalized
from neraium_core.spectral import dominant_mode_loading, spectral_gap, spectral_radius
from neraium_core.staged_pipeline import (
    AttributionStage,
    DecisionStage,
    FeatureExtractionStage,
    NodeBaselineProfile,
    RelationalInstabilityStage,
    StructuralDriftStage,
    TemporalCoherenceStage,
    flatten_upper_tri,
)
from neraium_core.subsystems import subsystem_spectral_measures


# How slowly the rolling baseline adapts (only when nominal); avoid absorbing instability.
DEFAULT_BASELINE_ADAPTATION_ALPHA = 0.92
# Composite below this and nominal state required to update rolling baseline.
BASELINE_UPDATE_MAX_COMPOSITE = 0.85
# Number of recent interpreted states to compute classification stability.
CLASSIFICATION_STABILITY_WINDOW = 15
TRANSITION_MEMORY_WINDOW = 8
TRANSITION_EMERGING_THRESHOLD = 0.85
TRANSITION_SUSTAINED_THRESHOLD = 1.15

# Calibrate alert thresholds from early nominal scores to reduce false positives
# and prevent early single-sample spikes from triggering alerts.
MIN_BASELINE_SAMPLES_FOR_CALIBRATION = 28
ALERT_PERSISTENCE_WINDOW = 3
MIN_CONSECUTIVE_WATCH = 2
MIN_CONSECUTIVE_ALERT = 2


def _vector_from_sensor_values(sensor_values: Dict[str, object], order: List[str]) -> np.ndarray:
    """Build a fixed-order vector; missing keys become NaN (new sensors mid-stream)."""
    values: list[float] = []
    for name in order:
        v = sensor_values.get(name)
        try:
            values.append(float(v) if v is not None else np.nan)
        except (TypeError, ValueError):
            values.append(np.nan)
    return np.asarray(values, dtype=float)


def _env_enabled(var_name: str, *, default: str = "1") -> bool:
    """Feature toggle helper that treats 0/false/no/off as disabled."""
    v = os.environ.get(var_name, default)
    if v is None:
        return True
    return str(v).strip().lower() not in {"0", "false", "no", "off"}


class StructuralEngine:
    def __init__(
        self,
        baseline_window: int = 50,
        recent_window: int = 12,
        window_stride: int = 1,
        regime_store_path: str = "regime_library.json",
        baseline_adaptation_alpha: float = DEFAULT_BASELINE_ADAPTATION_ALPHA,
    ):
        self.baseline_window = baseline_window
        self.recent_window = recent_window
        self.window_stride = max(1, window_stride)
        self.frames = deque(maxlen=500)
        self.sensor_order: List[str] = []
        self.latest_result: Optional[Dict] = None
        self.score_history: deque[float] = deque(maxlen=120)
        self.baseline_adaptation_alpha = baseline_adaptation_alpha
        # Rolling baseline: updated only when system is nominal and composite low.
        self._rolling_baseline_corr: Optional[np.ndarray] = None
        # Recent interpreted states for classification stability.
        self._state_history: deque[str] = deque(maxlen=CLASSIFICATION_STABILITY_WINDOW)
        self._stage_baseline_profile = NodeBaselineProfile()
        self._transition_pressure_history: deque[float] = deque(maxlen=TRANSITION_MEMORY_WINDOW)
        self._regime_history: deque[str | None] = deque(maxlen=TRANSITION_MEMORY_WINDOW)
        self._adjacency_history: deque[np.ndarray] = deque(maxlen=TRANSITION_MEMORY_WINDOW)
        self._dominant_mode_history: deque[np.ndarray] = deque(maxlen=TRANSITION_MEMORY_WINDOW)
        self._stable_manifold_corr: Optional[np.ndarray] = None
        self._low_transition_activity_streak: int = 0
        self._last_corr_recent: Optional[np.ndarray] = None
        self._transition_pressure_ema: float = 0.0
        self._prev_stable_novelty: float = 0.0
        self._prev_regime_novelty: float = 0.0
        self._corr_jump_history: deque[float] = deque(maxlen=24)
        self._edge_flip_history: deque[float] = deque(maxlen=24)
        self._spectral_jump_history: deque[float] = deque(maxlen=24)
        self._prev_spectral_radius: float | None = None
        self._shock_boost_steps_remaining: int = 0
        self._shock_refractory_steps: int = 0
        self._subsystem_instability_history: deque[float] = deque(maxlen=TRANSITION_MEMORY_WINDOW)
        self._regime_novelty_history: deque[float] = deque(maxlen=TRANSITION_MEMORY_WINDOW)
        self._shock_activity_history: deque[float] = deque(maxlen=TRANSITION_MEMORY_WINDOW)
        self._structural_drift_history: deque[float] = deque(maxlen=TRANSITION_MEMORY_WINDOW)
        self.transition_aware_enabled: bool = _env_enabled("NERAIUM_TRANSITION_AWARE", default="1")

        # Drift-score threshold calibration (watch/alert).
        self._drift_score_history: deque[float] = deque(maxlen=120)
        self._baseline_drift_score_samples: deque[float] = deque(maxlen=256)
        self._drift_watch_alert_thresholds: tuple[float, float] | None = None

        # Composite-score threshold calibration for decision-layer emission.
        self._baseline_composite_score_samples: deque[float] = deque(maxlen=256)
        self._composite_watch_alert_thresholds: tuple[float, float] | None = None

        # Debug helpers: print first alert reasoning once per engine instance.
        self._first_alert_logged: bool = False
        self._experimental_analytics_debug_logged: bool = False

        self.regime_store = RegimeStore(regime_store_path)
        persisted = self.regime_store.load()
        self.regime_signatures: list[dict[str, object]] = list(persisted.get("regimes", []))
        self.regime_baselines: dict[str, dict[str, object]] = dict(persisted.get("baselines", {}))

        # Baseline management: lock prevents rolling update; metadata for UI/API.
        self.baseline_locked: bool = False
        self._baseline_set_at: Optional[str] = None  # ISO timestamp when baseline was last established
        self._baseline_coverage_samples: int = 0  # Number of samples in baseline window

    def reset_baseline(self) -> None:
        """Clear rolling baseline and calibration state so baseline is recomputed from window."""
        self._rolling_baseline_corr = None
        self._baseline_set_at = None
        self._baseline_coverage_samples = 0
        self._drift_watch_alert_thresholds = None
        self._composite_watch_alert_thresholds = None
        self._baseline_drift_score_samples.clear()
        self._baseline_composite_score_samples.clear()
        self._last_corr_recent = None
        self._transition_pressure_ema = 0.0
        self._low_transition_activity_streak = 0
        self._prev_stable_novelty = 0.0
        self._prev_regime_novelty = 0.0
        self._corr_jump_history.clear()
        self._edge_flip_history.clear()
        self._spectral_jump_history.clear()
        self._prev_spectral_radius = None
        self._shock_boost_steps_remaining = 0
        self._shock_refractory_steps = 0
        self._subsystem_instability_history.clear()
        self._regime_novelty_history.clear()
        self._shock_activity_history.clear()
        self._structural_drift_history.clear()

    def lock_baseline(self, locked: bool = True) -> None:
        """Lock or unlock baseline. When locked, rolling baseline stops adapting."""
        self.baseline_locked = bool(locked)

    def get_baseline_info(self) -> Dict[str, object]:
        """Return baseline metadata for UI/API: when set, coverage, locked state."""
        mode = "rolling" if self._rolling_baseline_corr is not None else "fixed"
        return {
            "baseline_set_at": self._baseline_set_at,
            "baseline_coverage_samples": self._baseline_coverage_samples,
            "baseline_window_config": self.baseline_window,
            "baseline_locked": self.baseline_locked,
            "baseline_mode": mode,
        }

    def _persist_regime_state(self) -> None:
        self.regime_store.save(
            {
                "regimes": self.regime_signatures,
                "baselines": self.regime_baselines,
            }
        )

    def _analytics_unavailable_payload(self, reason: str) -> dict[str, object]:
        return {
            "trajectory_analysis": {
                "available": False,
                "reason": reason,
            },
            "branching_analysis": {
                "available": False,
                "reason": reason,
            },
            "constraint_analysis": {
                "available": False,
                "reason": reason,
            },
            "hierarchy_analysis": {
                "available": False,
                "reason": reason,
            },
            "horizon_analysis": {
                "available": False,
                "reason": reason,
            },
            "counterfactual_simulation": {
                "available": False,
                "reason": reason,
            },
        }

    def _debug_print_experimental_analytics_once(self, result: Dict) -> None:
        if self._experimental_analytics_debug_logged:
            return
        print(json.dumps(result.get("experimental_analytics", {}), indent=2)[:1500])
        self._experimental_analytics_debug_logged = True

    def _persistence_features(self) -> dict[str, float]:
        """
        Lightweight persistence/hysteresis helpers derived from composite history.

        This does not change analytics; it provides decision-layer context so
        transient motion does not escalate into persistent instability.
        """
        values_arr = np.asarray(list(self.score_history), dtype=float)
        n = int(values_arr.size)
        if n == 0:
            return {
                "history_len": 0.0,
                "rolling_mean": 0.0,
                "rolling_std": 0.0,
                "consecutive_elevated": 0.0,
                "consecutive_high": 0.0,
            }

        window = values_arr[-min(n, 12) :]
        rolling_mean = float(np.mean(window)) if window.size else 0.0
        rolling_std = float(np.std(window)) if window.size else 0.0

        consecutive_elevated = 0
        consecutive_high = 0
        for i in range(n - 1, -1, -1):
            v = float(values_arr[i])
            if v >= 1.5:
                consecutive_elevated += 1
            else:
                break
        for i in range(n - 1, -1, -1):
            v = float(values_arr[i])
            if v >= 2.5:
                consecutive_high += 1
            else:
                break

        return {
            "history_len": float(n),
            "rolling_mean": float(rolling_mean),
            "rolling_std": float(rolling_std),
            "consecutive_elevated": float(consecutive_elevated),
            "consecutive_high": float(consecutive_high),
        }

    def _vector_from_frame(self, frame: Dict) -> np.ndarray:
        """Project sensor_values into a vector; grow ``sensor_order`` when new keys appear.

        Previously only the first frame's keys were used, so adding channels later was ignored
        (stuck at e.g. four nodes). Merging keys and rebuilding buffered vectors fixes that.
        """
        sensor_values = frame.get("sensor_values") or {}
        incoming = sorted(sensor_values.keys())
        if not self.sensor_order:
            self.sensor_order = list(incoming)
        else:
            merged = sorted(set(self.sensor_order) | set(incoming))
            if merged != self.sensor_order:
                self.sensor_order = merged
                for f in self.frames:
                    sv = f.get("sensor_values") or {}
                    f["_vector"] = _vector_from_sensor_values(sv, self.sensor_order)
        return _vector_from_sensor_values(sensor_values, self.sensor_order)

    def _get_recent_window(self, frames_list: list[dict] | None = None) -> Optional[np.ndarray]:
        """Use ``frames_list`` when available to avoid repeated ``list(deque)`` copies per step."""
        fl = frames_list if frames_list is not None else list(self.frames)
        if len(fl) < self.recent_window:
            return None

        vectors = np.stack([f["_vector"] for f in fl[-self.recent_window :]], axis=0)
        vectors = vectors[:: self.window_stride]

        if vectors.shape[0] < 2:
            return None

        return vectors

    def _get_baseline_window(self, frames_list: list[dict] | None = None) -> Optional[np.ndarray]:
        fl = frames_list if frames_list is not None else list(self.frames)
        if len(fl) < self.baseline_window:
            return None

        vectors = np.stack([f["_vector"] for f in fl[: self.baseline_window]], axis=0)
        vectors = vectors[:: self.window_stride]

        if vectors.shape[0] < 2:
            return None

        return vectors

    def _get_recent_timestamps(self, frames_list: list[dict] | None = None) -> Optional[list[float]]:
        fl = frames_list if frames_list is not None else list(self.frames)
        if len(fl) < self.recent_window:
            return None
        ts_vals: list[float] = []
        for f in fl[-self.recent_window :]:
            try:
                ts_vals.append(float(f.get("timestamp")))
            except (TypeError, ValueError):
                continue
        return ts_vals if len(ts_vals) >= 2 else None

    def _get_baseline_timestamps(self, frames_list: list[dict] | None = None) -> Optional[list[float]]:
        fl = frames_list if frames_list is not None else list(self.frames)
        if len(fl) < self.baseline_window:
            return None
        ts_vals: list[float] = []
        for f in fl[: self.baseline_window]:
            try:
                ts_vals.append(float(f.get("timestamp")))
            except (TypeError, ValueError):
                continue
        return ts_vals if len(ts_vals) >= 2 else None

    def _system_health(self, drift_score: float, stability_score: float) -> int:
        health = 100.0 - min(drift_score * 20.0, 85.0)
        health += stability_score * 20.0
        return int(round(max(0.0, min(100.0, health))))

    def _alert_state(self, drift_score: float) -> str:
        # Until we have nominal calibration samples, suppress early alerts
        # to avoid false positives driven by unstable correlation estimates.
        if self._drift_watch_alert_thresholds is None:
            return "STABLE"

        watch_thr, alert_thr = self._drift_watch_alert_thresholds

        # Persistence requirement: require consecutive drift-score elevations.
        window = list(self._drift_score_history)[-ALERT_PERSISTENCE_WINDOW:]
        consec_watch = 0
        consec_alert = 0
        for v in reversed(window):
            # Use strict comparison to avoid borderline numerical equality
            # (e.g. calibrated nominal drift floors).
            if v > alert_thr:
                consec_alert += 1
                consec_watch += 1
            elif v > watch_thr:
                consec_watch += 1
            else:
                break

        if consec_alert >= MIN_CONSECUTIVE_ALERT:
            return "ALERT"
        if consec_watch >= MIN_CONSECUTIVE_WATCH:
            return "WATCH"
        return "STABLE"

    def _drift_alert(self, drift_score: float) -> bool:
        return self._alert_state(drift_score) == "ALERT"

    @staticmethod
    def _clamp01(value: float) -> float:
        return float(max(0.0, min(1.0, value)))

    def _transition_metrics(
        self,
        *,
        drift_score: float,
        corr_recent: np.ndarray,
        regime_name: str | None,
        regime_distance: float | None,
        adjacency: np.ndarray,
        dominant_mode: np.ndarray,
        spectral_radius_value: float,
    ) -> dict[str, float]:
        drift_hist = list(self._drift_score_history)
        drift_delta = 0.0
        drift_accel = 0.0
        if len(drift_hist) >= 2:
            drift_delta = max(0.0, float(drift_hist[-1] - drift_hist[-2]))
        if len(drift_hist) >= 3:
            prev_delta = float(drift_hist[-2] - drift_hist[-3])
            drift_accel = max(0.0, drift_delta - prev_delta)
        short_horizon_drift = self._clamp01(0.72 * drift_delta + 0.28 * max(0.0, drift_score - 0.55))
        acceleration = self._clamp01(drift_accel * 2.4 + max(0.0, drift_delta - 0.09))

        stable_novelty = 0.0
        if self._stable_manifold_corr is not None and self._stable_manifold_corr.shape == corr_recent.shape:
            stable_novelty = self._clamp01(
                structural_drift(corr_recent, self._stable_manifold_corr, norm="fro") / 1.8
            )

        regime_novelty = 0.0
        if regime_distance is not None:
            regime_novelty = self._clamp01(float(regime_distance) / 2.0)
        previous_regime = self._regime_history[-1] if self._regime_history else None
        if previous_regime is not None and regime_name is not None and previous_regime != regime_name:
            regime_novelty = max(regime_novelty, 0.85)

        edge_flip_rate = 0.0
        if self._adjacency_history:
            prev_adj = self._adjacency_history[-1]
            if prev_adj.shape == adjacency.shape and adjacency.size:
                prev_bin = prev_adj > 0
                curr_bin = adjacency > 0
                denom = float(np.sum(np.triu(np.ones_like(curr_bin, dtype=bool), k=1)))
                flips = float(np.sum(np.triu(np.logical_xor(prev_bin, curr_bin), k=1)))
                if denom > 0.0:
                    edge_flip_rate = self._clamp01(flips / denom)

        dominant_mode_rotation = 0.0
        if self._dominant_mode_history and dominant_mode.size:
            prev_mode = self._dominant_mode_history[-1]
            if prev_mode.shape == dominant_mode.shape and prev_mode.size:
                prev_norm = np.linalg.norm(prev_mode)
                curr_norm = np.linalg.norm(dominant_mode)
                if prev_norm > 1e-9 and curr_norm > 1e-9:
                    alignment = abs(float(np.dot(prev_mode, dominant_mode) / (prev_norm * curr_norm)))
                    dominant_mode_rotation = self._clamp01(1.0 - alignment)

        corr_jump_raw = 0.0
        corr_spike = 0.0
        if self._last_corr_recent is not None and self._last_corr_recent.shape == corr_recent.shape:
            corr_jump_raw = structural_drift(corr_recent, self._last_corr_recent, norm="fro")
            corr_spike = self._clamp01(max(0.0, corr_jump_raw - 0.06) / 0.26)

        stable_novelty_delta = max(0.0, stable_novelty - float(self._prev_stable_novelty))
        regime_novelty_delta = max(0.0, regime_novelty - float(self._prev_regime_novelty))
        novelty_spike = self._clamp01(2.2 * stable_novelty_delta + 1.6 * regime_novelty_delta)
        spike_term = self._clamp01(0.65 * corr_spike + 0.35 * novelty_spike)

        dynamic_activity = float(np.mean([acceleration, spike_term, edge_flip_rate, dominant_mode_rotation]))
        novelty_activation = 0.08 + 0.92 * dynamic_activity

        # Emphasize active transition kinetics (acceleration/spikes) over static
        # degraded level components. Context terms keep broad-domain sensitivity
        # but with lower persistence bias.
        impulse = (
            0.46 * acceleration
            + 0.34 * spike_term
            + 0.08 * edge_flip_rate
            + 0.07 * dominant_mode_rotation
            + 0.05 * short_horizon_drift
        )
        context = (
            0.04 * stable_novelty * novelty_activation
            + 0.03 * regime_novelty * novelty_activation
        )
        raw_pressure = (impulse + context) * 2.85
        self._transition_pressure_ema = 0.20 * self._transition_pressure_ema + 0.80 * raw_pressure

        # Strongly decay transition pressure when no active change is observed,
        # so chronically bad-but-steady states do not remain transition-hot.
        quasi_steady = (
            drift_delta < 0.03
            and acceleration < 0.05
            and spike_term < 0.08
            and edge_flip_rate < 0.09
            and dominant_mode_rotation < 0.09
        )
        if quasi_steady:
            self._low_transition_activity_streak += 1
        else:
            self._low_transition_activity_streak = 0
        streak_decay = 0.74 ** min(self._low_transition_activity_streak, 8)
        activity_gate = 0.12 + 1.05 * dynamic_activity
        pressure = self._transition_pressure_ema * activity_gate * streak_decay
        if acceleration < 0.08 and spike_term < 0.1:
            residual_cap = 0.22 * (short_horizon_drift + edge_flip_rate + dominant_mode_rotation)
            pressure = min(pressure, residual_cap)
        pressure = max(pressure, 0.85 * spike_term)

        spectral_jump_raw = 0.0
        if self._prev_spectral_radius is not None:
            spectral_jump_raw = abs(float(spectral_radius_value) - float(self._prev_spectral_radius))
        spectral_jump_norm = self._clamp01(max(0.0, spectral_jump_raw - 0.04) / 0.22)

        corr_hist = list(self._corr_jump_history)
        edge_hist = list(self._edge_flip_history)
        spectral_hist = list(self._spectral_jump_history)

        def _is_shock(
            current: float,
            history: list[float],
            *,
            floor: float,
            sigma: float = 2.5,
        ) -> bool:
            if current < floor:
                return False
            if len(history) < 6:
                return current >= (floor * 1.35)
            arr = np.asarray(history, dtype=float)
            mu = float(np.mean(arr))
            std = float(np.std(arr))
            return current > (mu + sigma * max(std, 1e-4))

        corr_shock = _is_shock(corr_jump_raw, corr_hist, floor=0.22)
        edge_shock = _is_shock(edge_flip_rate, edge_hist, floor=0.24)
        spectral_shock = _is_shock(spectral_jump_raw, spectral_hist, floor=0.10)
        shock_triggered = (
            self._shock_refractory_steps <= 0
            and corr_shock
            and (edge_shock or spectral_shock)
        )
        shock_boost = 0.0
        if shock_triggered:
            self._shock_boost_steps_remaining = 2
            self._shock_refractory_steps = 3
        if self._shock_boost_steps_remaining > 0:
            shock_strength = self._clamp01(
                0.52 * corr_spike + 0.26 * edge_flip_rate + 0.22 * spectral_jump_norm
            )
            shock_boost = 0.35 + 0.85 * shock_strength
            self._shock_boost_steps_remaining -= 1
        if self._shock_refractory_steps > 0:
            self._shock_refractory_steps -= 1

        pressure = pressure + shock_boost

        self._prev_stable_novelty = stable_novelty
        self._prev_regime_novelty = regime_novelty
        self._corr_jump_history.append(float(corr_jump_raw))
        self._edge_flip_history.append(float(edge_flip_rate))
        self._spectral_jump_history.append(float(spectral_jump_raw))
        self._last_corr_recent = np.array(corr_recent, dtype=float, copy=True)
        self._prev_spectral_radius = float(spectral_radius_value)

        return {
            "short_horizon_drift": float(short_horizon_drift),
            "consecutive_step_acceleration": float(acceleration),
            "correlation_spike": float(corr_spike),
            "novelty_spike": float(novelty_spike),
            "corr_jump_raw": float(corr_jump_raw),
            "spectral_jump_raw": float(spectral_jump_raw),
            "shock_triggered": float(1.0 if shock_triggered else 0.0),
            "shock_boost": float(shock_boost),
            "stable_manifold_novelty": float(stable_novelty),
            "regime_novelty": float(regime_novelty),
            "graph_edge_flip_rate": float(edge_flip_rate),
            "dominant_mode_rotation": float(dominant_mode_rotation),
            "transition_pressure": float(pressure),
        }

    def _transition_state(self, pressure: float) -> str:
        self._transition_pressure_history.append(float(pressure))
        history = list(self._transition_pressure_history)
        if not history:
            return "NONE"
        recent = history[-min(4, len(history)) :]
        sustained = sum(1 for v in recent if v >= TRANSITION_SUSTAINED_THRESHOLD)
        emerging = sum(1 for v in recent if v >= TRANSITION_EMERGING_THRESHOLD)
        current = float(recent[-1])
        weighted_recent = float(
            np.average(
                np.asarray(recent, dtype=float),
                weights=np.linspace(0.6, 1.0, num=len(recent), dtype=float),
            )
        )
        if (current >= TRANSITION_SUSTAINED_THRESHOLD and sustained >= 2) or sustained >= 3:
            return "SUSTAINED_TRANSITION"
        if (current >= TRANSITION_EMERGING_THRESHOLD and emerging >= 2) or weighted_recent >= (
            TRANSITION_EMERGING_THRESHOLD * 1.03
        ):
            return "EMERGING_TRANSITION"
        return "NONE"

    def _counterfactual_guidance(
        self,
        *,
        sensor_names: list[str],
        corr_baseline: np.ndarray,
        corr_recent: np.ndarray,
        adjacency: np.ndarray,
        graph: dict[str, float],
        subsystem: dict[str, object],
        spectral: dict[str, object],
        transition_metrics: dict[str, float],
    ) -> dict[str, object]:
        """
        Read-only counterfactual intervention analysis.

        This layer explains which structural relationships are most associated
        with current instability and which directional shifts (toward baseline
        structure) would likely reduce transition pressure.
        """
        n = int(corr_recent.shape[0]) if corr_recent.ndim == 2 else 0
        if n < 2:
            return {
                "available": False,
                "reason": "insufficient_relational_structure",
                "top_structural_break_contributors": [],
                "top_stabilizing_directions": [],
                "reversibility": {
                    "classification": "REVERSIBLE",
                    "scores": {
                        "persistence": 0.0,
                        "regime_novelty": 0.0,
                        "fragmentation": 0.0,
                        "drift_trend": 0.0,
                        "locked_in_index": 0.0,
                    },
                    "observation": (
                        "Current transition appears reversible with limited structural evidence "
                        "of persistent lock-in."
                    ),
                },
            }

        delta = np.nan_to_num(np.asarray(corr_recent, dtype=float) - np.asarray(corr_baseline, dtype=float))
        abs_delta = np.abs(delta)
        max_edge_delta = float(np.max(abs_delta)) if abs_delta.size else 0.0
        node_drift = np.sum(abs_delta, axis=1)
        node_drift_norm = node_drift / (float(np.max(node_drift)) + 1e-9)

        dominant_mode = np.asarray(spectral.get("dominant_eigenvector", []), dtype=float)
        if dominant_mode.size == n:
            mode_abs = np.abs(dominant_mode)
            mode_abs = mode_abs / (float(np.max(mode_abs)) + 1e-9)
        else:
            mode_abs = np.zeros(n, dtype=float)
        mode_rotation = self._clamp01(float(transition_metrics.get("dominant_mode_rotation", 0.0)))

        clusters = subsystem.get("clusters") if isinstance(subsystem, dict) else []
        cluster_map: dict[int, int] = {}
        if isinstance(clusters, list):
            for ci, members in enumerate(clusters):
                if isinstance(members, list):
                    for idx in members:
                        try:
                            ii = int(idx)
                        except (TypeError, ValueError):
                            continue
                        if 0 <= ii < n:
                            cluster_map[ii] = ci
        subsystem_instability = self._clamp01(float(subsystem.get("max_instability", 0.0)) / 2.0)

        contributors: list[dict[str, object]] = []
        for i in range(n):
            for j in range(i + 1, n):
                base_ij = float(corr_baseline[i, j])
                curr_ij = float(corr_recent[i, j])
                d_ij = curr_ij - base_ij
                d_abs = abs(d_ij)
                if d_abs < 1e-8:
                    continue

                flip = 1.0 if (np.sign(base_ij) != np.sign(curr_ij) and abs(base_ij) > 0.05 and abs(curr_ij) > 0.05) else 0.0
                weakened = abs(curr_ij) < abs(base_ij)
                edge_delta_norm = d_abs / (max_edge_delta + 1e-9)
                edge_node_term = 0.5 * float(node_drift_norm[i] + node_drift_norm[j])
                edge_mode_term = 0.5 * float(mode_abs[i] + mode_abs[j]) * mode_rotation
                same_cluster = cluster_map.get(i) == cluster_map.get(j) if cluster_map else False
                edge_subsystem_term = subsystem_instability if same_cluster else 0.35 * subsystem_instability

                score = (
                    0.48 * edge_delta_norm
                    + 0.20 * flip
                    + 0.18 * edge_node_term
                    + 0.08 * edge_mode_term
                    + 0.06 * edge_subsystem_term
                )
                score = float(self._clamp01(score))

                rel_a = sensor_names[i]
                rel_b = sensor_names[j]
                change_label = "weakened_coupling" if weakened else "strengthened_coupling"
                if flip > 0.5:
                    change_label = "sign_reversal"
                observation = (
                    f"Instability is most associated with {change_label.replace('_', ' ')} between {rel_a} and {rel_b}."
                )
                direction_text = (
                    f"Recovery would likely require restoring baseline coherence between {rel_a} and {rel_b}."
                    if weakened
                    else f"Recovery would likely require reducing divergence from baseline coupling between {rel_a} and {rel_b}."
                )
                if flip > 0.5:
                    direction_text = (
                        f"Recovery would likely require restoring sign-consistent coupling between {rel_a} and {rel_b} toward baseline."
                    )

                contributors.append(
                    {
                        "relationship": f"{rel_a} <-> {rel_b}",
                        "sensor_a": rel_a,
                        "sensor_b": rel_b,
                        "contributor_score": round(score, 4),
                        "change_type": change_label,
                        "observed_change": round(float(d_ij), 4),
                        "observation": observation,
                        "stabilizing_direction": direction_text,
                        "evidence": {
                            "baseline_correlation": round(base_ij, 4),
                            "current_correlation": round(curr_ij, 4),
                            "absolute_delta": round(d_abs, 4),
                            "edge_flip": bool(flip > 0.5),
                            "mode_rotation_weight": round(float(edge_mode_term), 4),
                        },
                    }
                )

        contributors.sort(key=lambda item: float(item.get("contributor_score", 0.0)), reverse=True)
        top_contributors = contributors[:5]

        transition_pressure = self._clamp01(float(transition_metrics.get("transition_pressure", 0.0)) / 1.35)
        top_directions: list[dict[str, object]] = []
        for rank, item in enumerate(top_contributors, start=1):
            relief = self._clamp01(float(item.get("contributor_score", 0.0)) * (0.55 + 0.45 * transition_pressure))
            top_directions.append(
                {
                    "rank": rank,
                    "relationship": item.get("relationship"),
                    "directional_change": item.get("stabilizing_direction"),
                    "estimated_transition_relief": round(float(relief), 4),
                    "observation": (
                        "This directional structural shift is associated with lower drift and transition pressure in the current geometry."
                    ),
                }
            )

        if isinstance(clusters, list) and clusters:
            largest = max(clusters, key=lambda members: len(members) if isinstance(members, list) else 0)
            if isinstance(largest, list) and len(largest) >= 2:
                names = [sensor_names[int(i)] for i in largest if isinstance(i, (int, np.integer)) and 0 <= int(i) < n]
                if names:
                    top_directions.append(
                        {
                            "rank": len(top_directions) + 1,
                            "relationship": ", ".join(names),
                            "directional_change": (
                                f"Recovery would likely require restoring baseline connectivity coherence within subsystem [{', '.join(names)}]."
                            ),
                            "estimated_transition_relief": round(float(0.4 + 0.4 * subsystem_instability), 4),
                            "observation": (
                                "Subsystem fragmentation appears associated with current instability; improving internal coherence is directionally stabilizing."
                            ),
                        }
                    )

        pressure_hist = list(self._transition_pressure_history)
        recent_press = pressure_hist[-min(5, len(pressure_hist)) :] if pressure_hist else []
        persistence = self._clamp01((float(np.mean(recent_press)) if recent_press else 0.0) / TRANSITION_SUSTAINED_THRESHOLD)
        novelty = self._clamp01(
            max(
                float(transition_metrics.get("regime_novelty", 0.0)),
                float(transition_metrics.get("stable_manifold_novelty", 0.0)),
            )
        )
        density = self._clamp01(float(graph.get("density", 0.0)) / 0.7)
        connectivity = self._clamp01(float(graph.get("connectivity", 0.0)))
        subsystem_count = float(subsystem.get("subsystem_count", 0.0))
        subsystem_frag = self._clamp01((subsystem_count - 1.0) / 3.0)
        fragmentation = self._clamp01(0.42 * (1.0 - connectivity) + 0.33 * (1.0 - density) + 0.25 * subsystem_frag)

        drift_hist = list(self._drift_score_history)
        trend_vals: list[float] = []
        if len(drift_hist) >= 2:
            tail = drift_hist[-min(5, len(drift_hist)) :]
            trend_vals = [max(0.0, float(tail[k] - tail[k - 1])) for k in range(1, len(tail))]
        drift_trend = self._clamp01(3.2 * (float(np.mean(trend_vals)) if trend_vals else 0.0))

        locked_in_index = self._clamp01(
            0.33 * persistence + 0.25 * novelty + 0.24 * fragmentation + 0.18 * drift_trend
        )
        if locked_in_index >= 0.68 or (persistence >= 0.75 and novelty >= 0.65 and fragmentation >= 0.55):
            reversibility = "LOCKED_IN"
            reversibility_observation = (
                "Current transition appears structurally locked in: persistent transition pressure, regime novelty, and fragmentation are jointly elevated."
            )
        elif locked_in_index <= 0.38 and persistence < 0.55 and fragmentation < 0.5:
            reversibility = "REVERSIBLE"
            reversibility_observation = (
                "Current transition appears structurally reversible: persistence and fragmentation remain limited."
            )
        else:
            reversibility = "METASTABLE"
            reversibility_observation = (
                "Current transition appears metastable: partial recovery is plausible but structural pressure remains active."
            )

        return {
            "available": True,
            "top_structural_break_contributors": top_contributors,
            "top_stabilizing_directions": top_directions[:5],
            "reversibility": {
                "classification": reversibility,
                "scores": {
                    "persistence": round(float(persistence), 4),
                    "regime_novelty": round(float(novelty), 4),
                    "fragmentation": round(float(fragmentation), 4),
                    "drift_trend": round(float(drift_trend), 4),
                    "locked_in_index": round(float(locked_in_index), 4),
                },
                "observation": reversibility_observation,
            },
        }

    def process_frame(self, frame: Dict) -> Dict:
        vector = self._vector_from_frame(frame)

        stored = dict(frame)
        stored["_vector"] = vector
        self.frames.append(stored)

        result = {
            "timestamp": frame["timestamp"],
            "site_id": frame["site_id"],
            "asset_id": frame["asset_id"],
            "state": "STABLE",
            "structural_drift_score": 0.0,
            "relational_stability_score": 1.0,
            "system_health": 100,
            "drift_alert": False,
            "sensor_relationships": self.sensor_order,
            "regime_name": None,
            "regime_distance": None,
            "regime_drift": 0.0,
            "latest_drift": 0.0,
            "latest_instability": 0.0,
            "relational_instability_score": 0.0,
            "temporal_distortion_score": 0.0,
            "localization_score": 0.0,
            "causal_attribution": {"top_drivers": [], "driver_scores": {}},
            "dominant_driver": None,
            "explanation": "Warmup: awaiting sufficient window history.",
            "baseline_mode": None,
            "data_quality_summary": {},
            "active_sensor_count": 0,
            "missing_sensor_count": 0,
            "transition_pressure": 0.0,
            "transition_state": "NONE",
            "experimental_analytics": self._analytics_unavailable_payload("warmup"),
        }

        # Skip deque→list snapshot during warmup (saves O(n) per frame until windows fill).
        if len(self.frames) < self.baseline_window or len(self.frames) < self.recent_window:
            self.latest_result = result
            self._debug_print_experimental_analytics_once(result)
            return result

        frames_list = list(self.frames)
        baseline_window = self._get_baseline_window(frames_list)
        recent_window = self._get_recent_window(frames_list)

        if baseline_window is None or recent_window is None:
            self.latest_result = result
            self._debug_print_experimental_analytics_once(result)
            return result

        ts_baseline = self._get_baseline_timestamps(frames_list)
        ts_recent = self._get_recent_timestamps(frames_list)

        data_quality_report = compute_data_quality(
            baseline_window,
            recent_window,
            sensor_names=self.sensor_order,
            timestamps_baseline=ts_baseline,
            timestamps_recent=ts_recent,
        )
        result["data_quality"] = data_quality_report.to_dict()
        dq_summary = data_quality_summary(data_quality_report)
        result["data_quality_summary"] = dq_summary
        result["active_sensor_count"] = dq_summary["valid_signal_count"]
        result["missing_sensor_count"] = dq_summary["missing_sensor_count"]

        use_degraded = (not data_quality_report.gate_passed) and should_use_degraded_analytics(
            data_quality_report
        )
        # Optional imputation when gate failed but we still want meaningful degraded output.
        if not data_quality_report.gate_passed and use_degraded:
            baseline_window = impute_missing_simple(baseline_window, method="column_mean")
            recent_window = impute_missing_simple(recent_window, method="column_mean")

        z_baseline, baseline_mean, baseline_std = normalize_window(baseline_window)
        z_recent, recent_mean, recent_std = normalize_window(recent_window)

        valid_mask = (np.nan_to_num(recent_std) > 1e-12) | (np.nan_to_num(baseline_std) > 1e-12)
        valid_signal_count = int(np.sum(valid_mask))

        warning = early_warning_metrics(np.nan_to_num(recent_window, nan=0.0))

        signature = build_regime_signature(recent_mean, recent_std)
        self.regime_signatures = update_regime_library(signature, self.regime_signatures)
        assigned_regime = assign_regime(signature, self.regime_signatures)

        regime_name = assigned_regime["name"] if assigned_regime else None
        regime_distance = float(assigned_regime["distance"]) if assigned_regime else None

        analytics: dict[str, object] = {
            **self._analytics_unavailable_payload("pending_multivariate_processing"),
            "early_warning": warning,
            "relational_metrics_skipped": valid_signal_count < 2,
            "regime_signature": {
                "current": [float(v) for v in signature],
                "nearest": assigned_regime,
                "assigned_name": regime_name,
                "library_size": len(self.regime_signatures),
            },
        }

        components = canonicalize_components(
            {
                "drift": 0.0,
                "regime_drift": 0.0,
                "early_warning": warning["variance"] + max(0.0, warning["lag1_autocorrelation"]),
            }
        )

        if valid_signal_count >= 2:
            z_base_valid = z_baseline[:, valid_mask]
            z_recent_valid = z_recent[:, valid_mask]
            stage_features = FeatureExtractionStage.extract(z_base_valid, z_recent_valid)

            corr_baseline = correlation_matrix(z_base_valid)
            corr_recent = correlation_matrix(z_recent_valid)

            # Adaptive baseline: use rolling baseline when available to avoid static reference.
            baseline_corr_used = corr_baseline
            baseline_mode = "fixed"
            if (
                self._rolling_baseline_corr is not None
                and self._rolling_baseline_corr.shape == corr_recent.shape
            ):
                baseline_corr_used = self._rolling_baseline_corr
                baseline_mode = "rolling"

            self._stage_baseline_profile.corr_baseline = np.array(baseline_corr_used, dtype=float, copy=True)
            stage_structural_raw, _ = StructuralDriftStage.score(stage_features, self._stage_baseline_profile)
            stage_relational_raw, _ = RelationalInstabilityStage.score(stage_features, self._stage_baseline_profile)
            temporal_raw, _ = TemporalCoherenceStage.score(ts_recent, self._stage_baseline_profile)
            # Preserve production sensitivity by keeping legacy drift geometry while
            # binding stage outputs into runtime diagnostics.
            drift_score = structural_drift(corr_recent, baseline_corr_used, norm="fro")
            drift_score = float(drift_score)
            self._drift_score_history.append(drift_score)
            self._structural_drift_history.append(drift_score)
            if self._drift_watch_alert_thresholds is None:
                self._baseline_drift_score_samples.append(drift_score)
                if len(self._baseline_drift_score_samples) >= MIN_BASELINE_SAMPLES_FOR_CALIBRATION:
                    watch_thr = float(np.percentile(list(self._baseline_drift_score_samples), 82.0))
                    alert_thr = float(np.percentile(list(self._baseline_drift_score_samples), 93.5))
                    if alert_thr < watch_thr:
                        watch_thr, alert_thr = alert_thr, watch_thr
                    self._drift_watch_alert_thresholds = (watch_thr, alert_thr)
            rel_delta_legacy = flatten_upper_tri(corr_recent) - flatten_upper_tri(baseline_corr_used)
            relational_raw = float(np.mean(np.abs(rel_delta_legacy))) if rel_delta_legacy.size else 0.0
            relational_raw = max(relational_raw, stage_relational_raw, 0.5 * stage_structural_raw)
            stability_score = 1.0 / (1.0 + drift_score)

            regime_drift = 0.0
            if regime_name is not None:
                if regime_name not in self.regime_baselines:
                    self.regime_baselines[regime_name] = {
                        "signature": signature.tolist(),
                        "correlation": corr_recent.tolist(),
                        "count": 1,
                    }
                else:
                    regime_corr = np.asarray(self.regime_baselines[regime_name]["correlation"], dtype=float)
                    regime_drift = structural_drift(corr_recent, regime_corr, norm="fro")
                    # Regime-specific baseline: EMA update so we gradually adapt inside stable regime.
                    alpha = 0.88
                    updated = alpha * regime_corr + (1.0 - alpha) * corr_recent
                    self.regime_baselines[regime_name]["correlation"] = updated.tolist()
                    self.regime_baselines[regime_name]["count"] = int(
                        self.regime_baselines[regime_name].get("count", 0)
                    ) + 1

                self._persist_regime_state()

            signal_importance = signal_structural_importance(corr_recent)
            adjacency = thresholded_adjacency(corr_recent, threshold=0.6)
            graph = graph_metrics(adjacency, corr=corr_recent)

            directional = directional_metrics(lagged_correlation_matrix(z_recent_valid, lag=1))

            causal_matrix = granger_causality_matrix(z_recent_valid)
            causal = causal_metrics(causal_matrix)
            causal_graph = causal_graph_metrics(causal_matrix, threshold=0.1)

            valid_sensor_names = [self.sensor_order[i] for i in range(len(valid_mask)) if valid_mask[i]]
            causal_prop = None
            dominant_causal_source = None
            causal_chains = None
            if _env_enabled("NERAIUM_CAUSAL_INTELLIGENCE", default="1"):
                try:
                    causal_prop = causal_propagation_spread(
                        causal_matrix,
                        threshold=0.1,
                        max_steps=2,
                        top_k=3,
                    )
                    top_sources = causal_prop.get("top_sources") if isinstance(causal_prop, dict) else None
                    if top_sources:
                        top_idx = int(top_sources[0])
                        if 0 <= top_idx < len(valid_sensor_names):
                            dominant_causal_source = valid_sensor_names[top_idx]
                except Exception:
                    causal_prop = None

            if _env_enabled("NERAIUM_CAUSAL_ROOT_CAUSE_CHAINS", default="1"):
                try:
                    causal_chains = causal_root_cause_chains(
                        causal_matrix,
                        valid_sensor_names,
                        threshold=0.1,
                        max_depth=3,
                        chain_count=2,
                    )
                except Exception:
                    causal_chains = None
            attr = causal_attribution(
                baseline_corr_used,
                corr_recent,
                causal_matrix,
                valid_sensor_names,
                top_k=10,
            )
            result["causal_attribution"] = attr
            result["dominant_driver"] = attr["top_drivers"][0] if attr["top_drivers"] else None
            if dominant_causal_source is not None:
                result["dominant_causal_source"] = dominant_causal_source
            if causal_chains:
                result["causal_root_cause_chains"] = causal_chains
                analytics["causal_root_cause_chains"] = causal_chains
                try:
                    best = max(causal_chains, key=lambda x: float(x.get("chain_score", 0.0)))
                    chain_nodes = best.get("chain_nodes") if isinstance(best, dict) else None
                    if isinstance(chain_nodes, list) and chain_nodes:
                        result["root_cause_narrative"] = " -> ".join([str(n) for n in chain_nodes])
                except Exception:
                    pass

            subsystem = subsystem_spectral_measures(corr_recent)

            spectral = {
                "radius": spectral_radius(corr_recent),
                "gap": spectral_gap(corr_recent),
                **dominant_mode_loading(corr_recent),
            }

            entropy_score = float(interaction_entropy(corr_recent))
            raw_components = {
                "drift": drift_score,
                "relational_drift": relational_raw,
                "regime_drift": regime_drift,
                "transition_pressure": 0.0,
                "spectral": spectral["radius"],
                "directional": max(
                    float(directional.get("divergence", 0.0)),
                    float(causal.get("causal_divergence", 0.0)),
                ),
                "entropy": entropy_score,
                "subsystem_instability": float(subsystem["max_instability"]),
                "temporal_distortion": temporal_raw,
            }

            # Merge order matters: preserve early_warning computed from the
            # latest signal window, while ensuring freshly computed relational
            # drift / regime drift / spectral / divergence / entropy /
            # subsystem instability are not clobbered by stale base defaults.
            base_components = components
            raw_canonical = canonicalize_components(raw_components)
            raw_canonical["early_warning"] = float(base_components.get("early_warning", 0.0))

            base_components.update(raw_canonical)
            components = base_components

            result.update(
                {
                    "structural_drift_score": round(drift_score, 4),
                    "relational_stability_score": round(stability_score, 4),
                    "system_health": self._system_health(drift_score, stability_score),
                    "state": self._alert_state(drift_score),
                    "drift_alert": self._drift_alert(drift_score),
                    "regime_name": regime_name,
                    "regime_distance": round(regime_distance, 4) if regime_distance is not None else None,
                    "regime_drift": round(float(regime_drift), 4),
                    "latest_drift": round(float(drift_score), 4),
                    "baseline_mode": baseline_mode,
                }
            )
            regime_memory_state = {
                "regime_name": regime_name,
                "library_size": len(self.regime_signatures),
                "baseline_count": (
                    int(self.regime_baselines.get(regime_name, {}).get("count", 0))
                    if regime_name
                    else None
                ),
            }
            result["regime_memory_state"] = regime_memory_state

            dominant_mode = np.asarray(spectral.get("dominant_eigenvector", []), dtype=float)
            transition_metrics = {
                "short_horizon_drift": 0.0,
                "consecutive_step_acceleration": 0.0,
                "stable_manifold_novelty": 0.0,
                "regime_novelty": 0.0,
                "graph_edge_flip_rate": 0.0,
                "dominant_mode_rotation": 0.0,
                "transition_pressure": 0.0,
            }
            if self.transition_aware_enabled:
                transition_metrics = self._transition_metrics(
                    drift_score=drift_score,
                    corr_recent=corr_recent,
                    regime_name=regime_name,
                    regime_distance=regime_distance,
                    adjacency=np.asarray(adjacency, dtype=float),
                    dominant_mode=dominant_mode,
                    spectral_radius_value=float(spectral.get("radius", 0.0)),
                )
                transition_pressure = float(transition_metrics["transition_pressure"])
                transition_state = self._transition_state(transition_pressure)
                raw_components["transition_pressure"] = transition_pressure
                components["transition_pressure"] = transition_pressure
                result["transition_pressure"] = round(transition_pressure, 4)
                result["transition_state"] = transition_state
                self._regime_history.append(regime_name)
                self._adjacency_history.append(np.asarray(adjacency, dtype=float))
                if dominant_mode.size:
                    self._dominant_mode_history.append(dominant_mode)
            else:
                result["transition_pressure"] = 0.0
                result["transition_state"] = "NONE"

            counterfactual_guidance = self._counterfactual_guidance(
                sensor_names=valid_sensor_names,
                corr_baseline=baseline_corr_used,
                corr_recent=corr_recent,
                adjacency=np.asarray(adjacency, dtype=float),
                graph=graph,
                subsystem=subsystem,
                spectral=spectral,
                transition_metrics=transition_metrics,
            )
            reversibility = counterfactual_guidance.get("reversibility", {}) if isinstance(counterfactual_guidance, dict) else {}
            reversibility_scores = reversibility.get("scores", {}) if isinstance(reversibility, dict) else {}

            self._subsystem_instability_history.append(float(subsystem.get("max_instability", 0.0)))
            self._regime_novelty_history.append(float(transition_metrics.get("regime_novelty", 0.0)))
            self._shock_activity_history.append(float(transition_metrics.get("shock_triggered", 0.0)))
            trajectory_analysis = classify_trajectory_path(
                drift_history=list(self._drift_score_history),
                transition_pressure_history=list(self._transition_pressure_history),
                subsystem_instability_history=list(self._subsystem_instability_history),
                regime_novelty_history=list(self._regime_novelty_history),
                shock_activity_history=list(self._shock_activity_history),
                reversibility_classification=str(reversibility.get("classification", "")),
                reversibility_score=float(reversibility_scores.get("locked_in_index", 0.0)),
            )
            hierarchy_analysis = analyze_hierarchy_cascade(
                sensor_names=valid_sensor_names,
                subsystem=subsystem,
                graph=graph,
                causal_propagation=causal_prop,
                counterfactual_guidance=counterfactual_guidance,
                transition=transition_metrics,
            )
            constraint_analysis = analyze_constraint_lock_in(
                transition_pressure_history=list(self._transition_pressure_history),
                shock_activity_history=list(self._shock_activity_history),
                structural_drift_score=float(drift_score),
                regime_novelty=float(transition_metrics.get("regime_novelty", 0.0)),
                regime_distance=float(regime_distance) if regime_distance is not None else None,
                subsystem_instability=float(subsystem.get("max_instability", 0.0)),
                reversibility_classification=str(reversibility.get("classification", "")),
                reversibility_score=float(reversibility_scores.get("locked_in_index", 0.0)),
                trajectory_analysis=trajectory_analysis,
            )
            branching_analysis = derive_branching_analysis(trajectory_analysis)
            horizon_analysis = estimate_risk_horizon(
                transition_pressure_history=list(self._transition_pressure_history),
                shock_activity_history=list(self._shock_activity_history),
                structural_drift_history=list(self._drift_score_history),
                trajectory_analysis=trajectory_analysis,
                branching_analysis=branching_analysis,
                constraint_analysis=constraint_analysis,
            )
            counterfactual_simulation = simulate_counterfactual_futures(
                transition_pressure_history=list(self._transition_pressure_history),
                shock_activity_history=list(self._shock_activity_history),
                structural_drift_history=list(self._drift_score_history),
                trajectory_analysis=trajectory_analysis,
                branching_analysis=branching_analysis,
                constraint_analysis=constraint_analysis,
                hierarchy_analysis=hierarchy_analysis,
                horizon_analysis=horizon_analysis,
            )

            analytics.update(
                {
                    "valid_sensor_names": valid_sensor_names,
                    "correlation_geometry": {
                        "baseline": corr_baseline.tolist(),
                        "current": corr_recent.tolist(),
                    },
                    "signal_structural_importance": [float(v) for v in signal_importance],
                    "graph": graph,
                    "directional": directional,
                    "causal": causal,
                    "causal_graph": causal_graph,
                    "causal_propagation": causal_prop,
                    "causal_root_cause_chains": causal_chains,
                    "subsystems": subsystem,
                    "spectral": spectral,
                    "entropy": entropy_score,
                    "regime_drift": float(regime_drift),
                    "transition": transition_metrics,
                    "counterfactual_guidance": counterfactual_guidance,
                    "trajectory_analysis": trajectory_analysis,
                    "branching_analysis": branching_analysis,
                    "hierarchy_analysis": hierarchy_analysis,
                    "constraint_analysis": constraint_analysis,
                    "horizon_analysis": horizon_analysis,
                    "counterfactual_simulation": counterfactual_simulation,
                }
            )
            result["reversibility_classification"] = (
                counterfactual_guidance.get("reversibility", {}).get("classification")
                if isinstance(counterfactual_guidance, dict)
                else None
            )
        else:
            result["regime_memory_state"] = {
                "regime_name": regime_name,
                "library_size": len(self.regime_signatures),
                "baseline_count": None,
            }
            result["transition_pressure"] = 0.0
            result["transition_state"] = "NONE"
            analytics["counterfactual_guidance"] = {
                "available": False,
                "reason": "relational_metrics_skipped",
                "top_structural_break_contributors": [],
                "top_stabilizing_directions": [],
                "reversibility": {
                    "classification": "REVERSIBLE",
                    "scores": {
                        "persistence": 0.0,
                        "regime_novelty": 0.0,
                        "fragmentation": 0.0,
                        "drift_trend": 0.0,
                        "locked_in_index": 0.0,
                    },
                    "observation": (
                        "Counterfactual guidance unavailable because multivariate relational metrics were skipped."
                    ),
                },
            }
            analytics["trajectory_analysis"] = {
                "dominant_path": "METASTABLE",
                "path_scores": {"stabilizing": 0.3333, "metastable": 0.3334, "diverging": 0.3333},
                "rationale": {
                    "observation": "Trajectory analysis unavailable because multivariate relational metrics were skipped.",
                },
            }
            analytics["hierarchy_analysis"] = {
                "available": False,
                "reason": "relational_metrics_skipped",
                "origin_scope": "LOCAL",
                "origin_subsystem": None,
                "propagation_risk": "LOW",
                "cascade_direction": [],
                "localized_vs_global_score": 0.0,
                "rationale": {
                    "observation": "Hierarchy analysis unavailable because multivariate relational metrics were skipped.",
                },
            }
            analytics["constraint_analysis"] = {
                "available": False,
                "reason": "relational_metrics_skipped",
                "recovery_margin": None,
                "lock_in_score": None,
                "commitment_to_failure": "MODERATE",
                "point_of_no_return_risk": "ELEVATED",
                "rationale": {
                    "observation": "Constraint analysis unavailable because multivariate relational metrics were skipped.",
                },
            }
            analytics["branching_analysis"] = {
                "available": False,
                "reason": "relational_metrics_skipped",
            }
            analytics["horizon_analysis"] = {
                "available": False,
                "reason": "relational_metrics_skipped",
            }
            analytics["counterfactual_simulation"] = {
                "available": False,
                "reason": "relational_metrics_skipped",
            }

        # Per-component confidence: down-weight or fully suppress evidence when the
        # data quality gate indicates unreliable inputs. Production alerts should
        # be driven by Tier-1 components only.
        tier1_components = {"relational_drift", "regime_drift", "spectral", "early_warning"}
        if self.transition_aware_enabled:
            tier1_components.add("transition_pressure")

        # Evidence quality in [0, 1]
        missingness_factor = max(0.0, 1.0 - float(data_quality_report.missingness_rate))
        variability_factor = max(0.0, min(1.0, float(data_quality_report.variability_coverage)))
        coverage_factor = max(0.0, min(1.0, float(data_quality_report.sensor_coverage)))
        sample_factor = 0.0
        if data_quality_report.total_sensors > 0:
            sample_factor = float(data_quality_report.valid_signal_count) / float(max(1, data_quality_report.total_sensors))
        sample_factor = max(0.0, min(1.0, sample_factor))

        evidence_conf = (
            missingness_factor
            * (0.4 + 0.6 * variability_factor)
            * (0.4 + 0.6 * coverage_factor)
            * (0.5 + 0.5 * sample_factor)
        )
        if not bool(data_quality_report.gate_passed):
            evidence_conf *= 0.25
        if use_degraded:
            evidence_conf *= 0.5  # Explicit degraded confidence when using fallback analytics
        evidence_conf = max(0.0, min(1.0, evidence_conf))

        correlation_ready = valid_signal_count >= 2

        # Classification stability: how consistent recent interpreted states have been.
        state_history_list = list(self._state_history)
        if len(state_history_list) >= 2:
            counts = Counter(state_history_list)
            most_common_count = max(counts.values()) if counts else 0
            classification_stability = float(most_common_count) / float(len(state_history_list))
        else:
            classification_stability = 1.0

        # Metric disagreement: high std across components slightly reduces confidence.
        comp_vals = [float(components.get(k, 0.0)) for k in tier1_components if k in components]
        if comp_vals:
            arr_c = np.asarray(comp_vals, dtype=float)
            mean_c = float(np.mean(arr_c))
            std_c = float(np.std(arr_c))
            disagreement = std_c / (mean_c + 1e-6)
            disagreement_factor = max(0.7, 1.0 - disagreement * 0.15)
        else:
            disagreement_factor = 1.0

        stabilized_confidence = evidence_conf * (0.6 + 0.4 * classification_stability) * disagreement_factor
        stabilized_confidence = max(0.0, min(1.0, stabilized_confidence))

        # Surface an uncertainty block for operator trust.
        # This is intended to answer: "how sure are we" + "what limited the evidence".
        uncertainty: dict[str, object] = {
            "confidence_score": round(float(stabilized_confidence), 4),
            "evidence_confidence": round(float(evidence_conf), 4),
            "gate_passed": bool(data_quality_report.gate_passed),
            "data_quality_summary": dict(dq_summary),
            "classification_stability": round(float(classification_stability), 4),
            "what_could_change": [],
        }

        try:
            missing_count = int(dq_summary.get("missing_sensor_count", 0))
            if missing_count > 0:
                uncertainty["what_could_change"].append(
                    "Reducing missing/flatlined sensors can increase evidence quality."
                )
            if not bool(dq_summary.get("gate_passed", True)):
                uncertainty["what_could_change"].append(
                    "Improving telemetry reliability to pass the data-quality gate can raise confidence."
                )
        except Exception:
            pass

        # Regime baseline confidence depends on how much history exists for the
        # assigned regime. If we don't yet have baseline correlation samples,
        # the regime drift evidence is treated as unreliable.
        regime_count = 0
        if regime_name is not None:
            entry = self.regime_baselines.get(regime_name)
            if isinstance(entry, dict):
                try:
                    regime_count = int(entry.get("count", 0) or 0)
                except (TypeError, ValueError):
                    regime_count = 0

        regime_factor = min(1.0, float(regime_count) / 5.0) if regime_count > 0 else 0.0

        component_confidence: dict[str, float] = {k: 0.0 for k in components.keys()}

        # Tier-1
        component_confidence["relational_drift"] = evidence_conf if correlation_ready else 0.0
        component_confidence["spectral"] = evidence_conf if correlation_ready else 0.0
        component_confidence["early_warning"] = evidence_conf
        component_confidence["regime_drift"] = evidence_conf * regime_factor if correlation_ready else 0.0
        if self.transition_aware_enabled:
            component_confidence["transition_pressure"] = evidence_conf if correlation_ready else 0.0

        # Suppress non-Tier-1 components explicitly (keeps production composite Tier-1 only)
        for k in list(component_confidence.keys()):
            if k not in tier1_components:
                component_confidence[k] = 0.0

        analytics["component_confidence"] = component_confidence

        # Confidence-weighted composite: use confidence as a scaling on component weights
        # so that unreliable evidence doesn't dilute the Tier-1 score.
        base_weights = canonicalize_weights()
        weights_for_composite: dict[str, float] = {}
        for k, w in base_weights.items():
            weights_for_composite[k] = float(w) * float(component_confidence.get(k, 0.0))

        components_for_decision = {
            k: float(v) * float(component_confidence.get(k, 0.0)) if k in component_confidence else float(v)
            for k, v in components.items()
        }

        composite = composite_instability_score_normalized(components, weights=weights_for_composite)
        composite = float(composite)
        self.score_history.append(composite)

        # Calibrate decision thresholds from early nominal composite history.
        if self._composite_watch_alert_thresholds is None:
            self._baseline_composite_score_samples.append(composite)
            if len(self._baseline_composite_score_samples) >= MIN_BASELINE_SAMPLES_FOR_CALIBRATION:
                watch_thr = float(np.percentile(list(self._baseline_composite_score_samples), 82.0))
                alert_thr = float(np.percentile(list(self._baseline_composite_score_samples), 93.5))
                if alert_thr < watch_thr:
                    watch_thr, alert_thr = alert_thr, watch_thr
                self._composite_watch_alert_thresholds = (watch_thr, alert_thr)

        persistence = self._persistence_features()

        forecast = {
            "method": "regression+ar1",
            "trend": float(instability_trend(self.score_history)),
            "time_to_instability": time_to_instability(self.score_history),
            "ar1_next": forecast_next(self.score_history),
            "ar1_time_to_instability": time_to_threshold_ar1(self.score_history),
            "persistence": persistence,
        }

        # Temporal foresight upgrade: observational scenario projections.
        # These are "what-if" time-to-threshold estimates derived from the same
        # AR(1) forecast, with selected component magnitudes scaled.
        if _env_enabled("NERAIUM_TEMPORAL_SCENARIOS", default="1"):
            try:
                scenario_defs = [
                    {
                        "scenario": "structural_drift_up_12pct",
                        "scale": {"relational_drift": 1.12, "regime_drift": 1.08, "early_warning": 1.05},
                    },
                    {
                        "scenario": "coupling_breakdown_up_10pct",
                        "scale": {"directional_divergence": 1.10, "spectral": 1.10},
                    },
                    {"scenario": "interaction_entropy_up_10pct", "scale": {"entropy": 1.10}},
                ]

                threshold = 1.5
                score_series = list(self.score_history)
                projections: list[dict[str, object]] = []
                for sc in scenario_defs:
                    scen_components = dict(components)
                    for k, factor in sc["scale"].items():
                        if k in scen_components:
                            scen_components[k] = float(scen_components[k]) * float(factor)

                    scen_score = float(
                        composite_instability_score_normalized(
                            scen_components, weights=weights_for_composite
                        )
                    )
                    scen_series = list(score_series)
                    if scen_series:
                        scen_series[-1] = scen_score
                    tti = time_to_threshold_ar1(scen_series, threshold=threshold, max_steps=200)
                    projections.append(
                        {
                            "scenario": sc["scenario"],
                            "projected_composite_score": scen_score,
                            "projected_time_to_instability_steps": tti,
                        }
                    )

                forecast["scenario_projections"] = projections
            except Exception:
                pass

        decision = decision_output(
            composite_score=float(composite),
            components=components_for_decision,
            forecast=forecast,
            confidence_score=stabilized_confidence,
            classification_stability=classification_stability,
            watch_threshold=(
                float(self._composite_watch_alert_thresholds[0])
                if self._composite_watch_alert_thresholds is not None
                else None
            ),
            alert_threshold=(
                float(self._composite_watch_alert_thresholds[1])
                if self._composite_watch_alert_thresholds is not None
                else None
            ),
            min_history_for_alerts=MIN_BASELINE_SAMPLES_FOR_CALIBRATION,
        )
        result.update(decision)
        if self.transition_aware_enabled:
            transition_pressure = float(result.get("transition_pressure", components.get("transition_pressure", 0.0)))
            transition_state = str(result.get("transition_state", "NONE"))
            state_rank = {"STABLE": 0, "WATCH": 1, "ALERT": 2}
            current_state = str(result.get("state", "STABLE"))
            target_state = current_state
            if transition_state == "SUSTAINED_TRANSITION" and transition_pressure >= TRANSITION_SUSTAINED_THRESHOLD:
                target_state = "ALERT"
            elif transition_state == "EMERGING_TRANSITION" and transition_pressure >= TRANSITION_EMERGING_THRESHOLD:
                target_state = "WATCH"
            if state_rank.get(target_state, 0) > state_rank.get(current_state, 0):
                result["state"] = target_state
                result["drift_alert"] = target_state == "ALERT"

        result["uncertainty"] = uncertainty
        stage_interpreted = DecisionStage.interpreted_state(
            structural=float(components.get("drift", 0.0)),
            relational=float(components.get("relational_drift", 0.0)),
            regime_distance=float(components.get("regime_drift", 0.0)),
            temporal_distortion=float(components.get("temporal_distortion", 0.0)),
            localization=1.0,
            trend=float(forecast.get("trend", 0.0)),
        )
        if (
            str(result.get("interpreted_state", "NOMINAL_STRUCTURE")) == "NOMINAL_STRUCTURE"
            and stage_interpreted != "NOMINAL_STRUCTURE"
        ):
            result["interpreted_state"] = stage_interpreted
        elif str(result.get("interpreted_state", "NOMINAL_STRUCTURE")) == "NOMINAL_STRUCTURE":
            # Single-node runtime fallback: preserve legacy structural/coupling detection
            # semantics when multi-node localization context is unavailable.
            rel = float(components.get("relational_drift", 0.0))
            drf = float(components.get("drift", 0.0))
            if rel > 0.9:
                result["interpreted_state"] = "COUPLING_INSTABILITY_OBSERVED"
            elif drf > 1.1:
                result["interpreted_state"] = "STRUCTURAL_INSTABILITY_OBSERVED"
        result["confidence_score"] = round(stabilized_confidence, 4)
        result["latest_instability"] = round(float(composite), 4)
        result["relational_instability_score"] = round(float(components.get("relational_drift", 0.0)), 4)
        result["temporal_distortion_score"] = round(float(components.get("temporal_distortion", data_quality_report.timestamp_irregularity)), 4)
        result["localization_score"] = 0.0

        self._state_history.append(decision.get("interpreted_state", "NOMINAL_STRUCTURE"))

        # Rolling baseline: update only when nominal, composite low, and not locked.
        transition_blocks_baseline = (
            self.transition_aware_enabled
            and (
                str(result.get("transition_state", "NONE")) != "NONE"
                or float(result.get("transition_pressure", 0.0)) >= TRANSITION_EMERGING_THRESHOLD
            )
        )
        if (
            valid_signal_count >= 2
            and not self.baseline_locked
            and decision.get("interpreted_state") == "NOMINAL_STRUCTURE"
            and not transition_blocks_baseline
            and float(composite) < BASELINE_UPDATE_MAX_COMPOSITE
        ):
            if self._rolling_baseline_corr is None or self._rolling_baseline_corr.shape != corr_recent.shape:
                self._rolling_baseline_corr = np.array(corr_recent, dtype=float, copy=True)
                self._baseline_set_at = datetime.now(timezone.utc).isoformat()
                self._baseline_coverage_samples = self.baseline_window
            else:
                alpha = self.baseline_adaptation_alpha
                self._rolling_baseline_corr = alpha * self._rolling_baseline_corr + (1.0 - alpha) * corr_recent

            if self._stable_manifold_corr is None or self._stable_manifold_corr.shape != corr_recent.shape:
                self._stable_manifold_corr = np.array(corr_recent, dtype=float, copy=True)
            else:
                manifold_alpha = min(0.985, max(0.90, self.baseline_adaptation_alpha + 0.03))
                self._stable_manifold_corr = (
                    manifold_alpha * self._stable_manifold_corr + (1.0 - manifold_alpha) * corr_recent
                )

        analytics["composite_instability"] = round(float(composite), 4)
        analytics["forecasting"] = forecast
        analytics["components"] = components
        explain_components = {
            "structural_drift_score": float(result.get("structural_drift_score", 0.0)),
            "relational_instability_score": float(result.get("relational_instability_score", 0.0)),
            "regime_distance": float(result.get("regime_distance", 0.0) or 0.0),
            "temporal_distortion_score": float(result.get("temporal_distortion_score", 0.0)),
        }
        msg, contrib = AttributionStage.explain(explain_components, str(result.get("state", "STABLE")))
        result["explanation"] = msg
        analytics["component_contributions"] = contrib
        trajectory_analysis = analytics.get("trajectory_analysis") if isinstance(analytics, dict) else None
        branching_analysis = analytics.get("branching_analysis") if isinstance(analytics, dict) else None
        constraint_analysis = analytics.get("constraint_analysis") if isinstance(analytics, dict) else None
        hierarchy_analysis = analytics.get("hierarchy_analysis") if isinstance(analytics, dict) else None
        horizon_analysis = analytics.get("horizon_analysis") if isinstance(analytics, dict) else None
        counterfactual_simulation = (
            analytics.get("counterfactual_simulation") if isinstance(analytics, dict) else None
        )
        result["dominant_driver"] = (
            max(contrib.items(), key=lambda item: item[1])[0]
            if contrib
            else result.get("dominant_driver")
        )
        result["component_confidence"] = component_confidence

        for key, payload in self._analytics_unavailable_payload("missing inputs").items():
            existing = analytics.get(key)
            analytics[key] = existing if isinstance(existing, dict) and existing else payload

        trajectory_analysis = analytics["trajectory_analysis"]
        branching_analysis = analytics["branching_analysis"]
        constraint_analysis = analytics["constraint_analysis"]
        hierarchy_analysis = analytics["hierarchy_analysis"]
        horizon_analysis = analytics["horizon_analysis"]
        counterfactual_simulation = analytics["counterfactual_simulation"]

        result["experimental_analytics"] = analytics
        result["experimental_analytics"] = {
            "trajectory_analysis": trajectory_analysis,
            "branching_analysis": branching_analysis,
            "constraint_analysis": constraint_analysis,
            "hierarchy_analysis": hierarchy_analysis,
            "horizon_analysis": horizon_analysis,
            "counterfactual_simulation": counterfactual_simulation,
            **analytics,
        }
        print("DEBUG EXP ANALYTICS:", result.get("experimental_analytics"))
        self._debug_print_experimental_analytics_once(result)

        debug_verbose = os.environ.get("NERAIUM_DEBUG_SII_VERBOSE", "0").strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
            "",
        }
        if debug_verbose:
            drift_thr = self._drift_watch_alert_thresholds
            comp_thr = self._composite_watch_alert_thresholds
            causal_prop = analytics.get("causal_propagation") if isinstance(analytics.get("causal_propagation"), dict) else {}
            causal_prop_top_sources = causal_prop.get("top_sources")
            causal_prop_spread = causal_prop.get("spread_scores")
            causal_prop_top_pairs = causal_prop.get("top_pairs")
            graph = analytics.get("graph")
            causal_graph = analytics.get("causal_graph")

            drift_score_tail = list(self._drift_score_history)[-ALERT_PERSISTENCE_WINDOW:]
            consec_watch = 0
            consec_alert = 0
            if isinstance(drift_thr, tuple) and len(drift_thr) == 2:
                watch_thr, alert_thr = drift_thr
                for v in reversed(drift_score_tail):
                    if v > alert_thr:
                        consec_alert += 1
                        consec_watch += 1
                    elif v > watch_thr:
                        consec_watch += 1
                    else:
                        break

            print(
                "[NERAIUM_DEBUG_SII_VERBOSE]"
                f" state={result.get('state')} drift_score={float(result.get('latest_drift', 0.0)):.6g}"
                f" drift_thr={drift_thr}"
                f" drift_persist=(watch={consec_watch}, alert={consec_alert})"
                f" composite={float(result.get('latest_instability', 0.0)):.6g}"
                f" comp_thr={comp_thr}"
                f" signal_emitted={result.get('signal_emitted', None)}"
                f" top_sources={causal_prop_top_sources}"
                f" spread_scores={(causal_prop_spread if causal_prop_spread is not None else [])[:3]}"
                f" graph_summary={graph}"
                f" causal_graph_summary={(causal_graph if isinstance(causal_graph, dict) else {})}"
            )

            # One-time first alert reasoning.
            if result.get("state") in {"WATCH", "ALERT"} and not self._first_alert_logged:
                print(
                    "[NERAIUM_DEBUG_SII_VERBOSE][first_alert]"
                    f" state={result.get('state')} latest_drift={float(result.get('latest_drift', 0.0)):.6g}"
                    f" drift_thr={drift_thr} drift_score_tail={drift_score_tail}"
                    f" drift_persist=(watch={consec_watch}, alert={consec_alert})"
                    f" composite_thr={comp_thr}"
                )
                self._first_alert_logged = True

        try:
            trajectory_analysis = classify_trajectory_path(
                transition_pressure_history=list(self._transition_pressure_history),
                shock_activity_history=list(self._shock_activity_history),
                structural_drift_history=list(self._structural_drift_history),
            )
        except Exception as e:
            trajectory_analysis = {"available": False, "reason": str(e)}

        try:
            branching_analysis = derive_branching_analysis(trajectory_analysis)
            if branching_analysis is None:
                branching_analysis = {"available": False, "reason": "empty branching analysis"}
        except Exception as e:
            branching_analysis = {"available": False, "reason": str(e)}

        try:
            constraint_analysis = analyze_constraint_lock_in(
                transition_pressure_history=list(self._transition_pressure_history),
                shock_activity_history=list(self._shock_activity_history),
                structural_drift_score=float(result.get("structural_drift_score", 0.0)),
            )
        except Exception as e:
            constraint_analysis = {"available": False, "reason": str(e)}

        try:
            hierarchy_analysis = analyze_hierarchy_cascade(
                sensor_names=list((frame.get("sensor_values") or {}).keys()),
                subsystem=result.get("subsystem", {}),
            )
        except Exception as e:
            hierarchy_analysis = {"available": False, "reason": str(e)}

        try:
            horizon_analysis = estimate_risk_horizon(
                trajectory_analysis=trajectory_analysis,
                constraint_analysis=constraint_analysis,
            )
        except Exception as e:
            horizon_analysis = {"available": False, "reason": str(e)}

        try:
            counterfactual_simulation = simulate_counterfactual_futures(
                transition_pressure_history=list(self._transition_pressure_history),
                shock_activity_history=list(self._shock_activity_history),
                structural_drift_history=list(self._structural_drift_history),
                trajectory_analysis=trajectory_analysis,
                branching_analysis=branching_analysis,
                constraint_analysis=constraint_analysis,
                hierarchy_analysis=hierarchy_analysis,
                horizon_analysis=horizon_analysis,
            )
        except Exception as e:
            counterfactual_simulation = {"available": False, "reason": str(e)}

        result["experimental_analytics"] = {
            "trajectory_analysis": trajectory_analysis,
            "branching_analysis": branching_analysis,
            "constraint_analysis": constraint_analysis,
            "hierarchy_analysis": hierarchy_analysis,
            "horizon_analysis": horizon_analysis,
            "counterfactual_simulation": counterfactual_simulation,
        }
        print("DEBUG ANALYTICS:", result["experimental_analytics"])

        self.latest_result = result

        return result
