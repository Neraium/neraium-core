from __future__ import annotations

from collections import Counter, deque
import json
from datetime import datetime, timezone
from typing import Dict, List, Optional

import os
import numpy as np

from neraium_core.causal import (
    causal_metrics,
    generate_hypotheses,
    generate_validation_plan,
    granger_causality_matrix,
    rank_actions,
    run_counterfactual_checks,
    score_hypotheses,
)
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
from neraium_core.explanation_layer import build_explanation_text
from neraium_core.directional import directional_metrics, lagged_correlation_matrix
from neraium_core.early_warning import early_warning_metrics
from neraium_core.entropy import interaction_entropy
from neraium_core.experimental_analytics.hierarchy_analysis import analyze_hierarchy_cascade
from neraium_core.experimental_analytics.constraint_analysis import analyze_constraint_lock_in
from neraium_core.experimental_analytics.directional_evolution import derive_directional_evolution_features
from neraium_core.experimental_analytics.path_prototypes import derive_path_prototypes
from neraium_core.experimental_analytics.trajectory_analysis import classify_trajectory_path
from neraium_core.experimental_analytics.trajectory_shape_features import derive_trajectory_shape_features
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
from neraium_core.robustness import (
    classify_drift_noise,
    compute_multi_scale_states,
    compute_sensitivity,
    compute_stability_metrics,
    generate_structural_explanations,
)
from neraium_core.structural_upgrade import (
    build_evidence_block,
    derive_path_prototype_summary,
    update_episode_memory,
)
from neraium_core.scoring import canonicalize_components, canonicalize_weights, composite_instability_score_normalized
from neraium_core.spectral import dominant_mode_loading, spectral_gap, spectral_radius
from neraium_core.context_invariant_representation import (
    RepresentationWeights,
    TemporalRepresentationConfig,
    build_temporal_representation,
)
from neraium_core.temporal_features import derive_temporal_rate_features
from neraium_core.stat_geometry import StatisticalGeometryLayer
from neraium_core.temporal_quality import derive_temporal_quality_signals
from neraium_core.detection.readiness import compute_engine_readiness
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
from neraium_core.realtime.buffer import HistoryRingBuffer, TimestampDequeBuffer, VectorDequeBuffer
from neraium_core.math.probabilistic_engine import MonteCarloSampler, StructuralUncertaintyTracker
from neraium_core.math.verification_engine import run_all_checks
from neraium_core.tetrahedral_state import compute_tetrahedral_state
from neraium_core.engine_stages.stage_boundaries import structural_engine_stage_groups


# How slowly the rolling baseline adapts (only when nominal); avoid absorbing instability.
DEFAULT_BASELINE_ADAPTATION_ALPHA = 0.92
# Composite below this and nominal state required to update rolling baseline.
BASELINE_UPDATE_MAX_COMPOSITE = 0.85
# Number of recent interpreted states to compute classification stability.
CLASSIFICATION_STABILITY_WINDOW = 15
TRANSITION_MEMORY_WINDOW = 8
TRANSITION_EMERGING_THRESHOLD = 0.85
TRANSITION_SUSTAINED_THRESHOLD = 1.15

# Fast mode keeps the public output contract but thins/defers non-critical heavy analytics.
FAST_MODE_GEOMETRY_UPDATE_INTERVAL = 3
FAST_MODE_GEOMETRY_DOWNSAMPLE_STEP = 2

# Calibrate alert thresholds from early nominal scores to reduce false positives
# and prevent early single-sample spikes from triggering alerts.
MIN_BASELINE_SAMPLES_FOR_CALIBRATION = 28
# Locked FD004 policy defaults are centralized here for Phase-1 compatibility.
LOCKED_FD004_POLICY_DEFAULTS = {
    "drift_smoothing_window": 25,
    "watch_quantile": 0.65,
    "alert_quantile": 0.85,
    "watch_persistence": 5,
    "alert_persistence": 3,
    "fast_trigger_multiplier": 1.25,
    "alert_latch_enabled": True,
    "unlatch_ratio": 0.75,
}
DEFAULT_DRIFT_SMOOTHING_WINDOW = LOCKED_FD004_POLICY_DEFAULTS["drift_smoothing_window"]
DEFAULT_WATCH_QUANTILE = LOCKED_FD004_POLICY_DEFAULTS["watch_quantile"]
DEFAULT_ALERT_QUANTILE = LOCKED_FD004_POLICY_DEFAULTS["alert_quantile"]
DEFAULT_WATCH_PERSISTENCE = LOCKED_FD004_POLICY_DEFAULTS["watch_persistence"]
DEFAULT_ALERT_PERSISTENCE = LOCKED_FD004_POLICY_DEFAULTS["alert_persistence"]
DEFAULT_FAST_TRIGGER_MULTIPLIER = LOCKED_FD004_POLICY_DEFAULTS["fast_trigger_multiplier"]
DEFAULT_ALERT_LATCH_ENABLED = LOCKED_FD004_POLICY_DEFAULTS["alert_latch_enabled"]
DEFAULT_UNLATCH_RATIO = LOCKED_FD004_POLICY_DEFAULTS["unlatch_ratio"]


def _to_epoch_seconds(value: object) -> float:
    if isinstance(value, datetime):
        dt = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
        return float(dt.timestamp())
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        raw = value.strip()
        try:
            return float(raw)
        except ValueError:
            if raw.endswith("Z"):
                raw = raw[:-1] + "+00:00"
            try:
                dt = datetime.fromisoformat(raw)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return float(dt.timestamp())
            except ValueError:
                return 0.0
    return 0.0


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


def _incremental_windows_enabled() -> bool:
    """When True, skip full ``list(frames)`` scans and cache frozen baseline matrices."""
    v = os.environ.get("NERAIUM_INCREMENTAL", "1")
    return str(v).strip().lower() not in {"0", "false", "no", "off"}


_FRAME_DEBUG_ENV_KEYS = (
    "NERAIUM_FRAME_DEBUG",
    "NERAIUM_DEBUG_SII",
    "NERAIUM_DEBUG_SII_VERBOSE",
    "NERAIUM_DEBUG_RAW_FEATURES",
    "NERAIUM_DEBUG_EXP_ANALYTICS",
    "NERAIUM_DEBUG_GEOMETRY",
)


def _effective_frame_debug(explicit: bool) -> bool:
    """True when per-frame stdout / debug prints are allowed (default: off).

    ``explicit`` comes from :class:`StructuralEngine` ``frame_debug=...``. Any legacy
    env var that previously toggled a subsection also enables the master gate so
    existing notebooks keep working.
    """
    if explicit:
        return True
    for key in _FRAME_DEBUG_ENV_KEYS:
        v = os.environ.get(key, "0")
        if str(v).strip().lower() not in {"0", "false", "no", "off", ""}:
            return True
    return False


class StructuralEngine:
    """Geometric structural drift engine with policy driven by drift state-machine.

    Architecture notes:
      - Core structural layer (primary): geometric / temporal processing
        over normalized baseline-vs-recent windows.
      - Policy layer (primary for state outputs): drift smoothing + threshold
        calibration + watch/alert persistence state-machine.
      - Auxiliary / legacy compatibility layer: diagnostics (including optional
        Mahalanobis-style signals) + composite-score compatibility payloads.
      - Backward-compatible flat output fields are intentionally preserved.

    Public API compatibility:
      - ``process_frame(frame)`` remains the public entry point.
      - ``policy_state`` and ``state`` are derived from structural drift policy.
      - Legacy fields and aliases remain populated for existing consumers.
    """
    def __init__(
        self,
        baseline_window: int = 50,
        recent_window: int = 12,
        window_stride: int = 1,
        regime_store_path: str = "regime_library.json",
        baseline_adaptation_alpha: float = DEFAULT_BASELINE_ADAPTATION_ALPHA,
        representation_mode: str = "combined",
        reference_strategy: str = "robust",
        context_diagnostics_enabled: bool = True,
        feature_weights: dict[str, float] | None = None,
        frame_debug: bool = False,
        drift_smoothing_window: int = DEFAULT_DRIFT_SMOOTHING_WINDOW,
        watch_quantile: float = DEFAULT_WATCH_QUANTILE,
        alert_quantile: float = DEFAULT_ALERT_QUANTILE,
        watch_persistence: int = DEFAULT_WATCH_PERSISTENCE,
        alert_persistence: int = DEFAULT_ALERT_PERSISTENCE,
        fast_trigger_multiplier: float = DEFAULT_FAST_TRIGGER_MULTIPLIER,
        alert_latch_enabled: bool = DEFAULT_ALERT_LATCH_ENABLED,
        unlatch_ratio: float = DEFAULT_UNLATCH_RATIO,
        drift_threshold_quantiles: tuple[float, float] | None = None,
        drift_persistence_length: int | None = None,
    ):
        self.baseline_window = baseline_window
        self.recent_window = recent_window
        self.window_stride = max(1, window_stride)
        self.frames = deque(maxlen=500)
        self.sensor_order: List[str] = []
        self.latest_result: Optional[Dict] = None
        self.score_history: deque[float] = deque(maxlen=120)
        self.baseline_adaptation_alpha = baseline_adaptation_alpha
        base_weights = RepresentationWeights()
        merged_weights = RepresentationWeights(
            raw_weight=float((feature_weights or {}).get("raw_weight", base_weights.raw_weight)),
            residual_weight=float((feature_weights or {}).get("residual_weight", base_weights.residual_weight)),
            delta_weight=float((feature_weights or {}).get("delta_weight", base_weights.delta_weight)),
            slope_weight=float((feature_weights or {}).get("slope_weight", base_weights.slope_weight)),
            drift_weight=float((feature_weights or {}).get("drift_weight", base_weights.drift_weight)),
            second_diff_weight=float((feature_weights or {}).get("second_diff_weight", base_weights.second_diff_weight)),
        )
        self.representation_config = TemporalRepresentationConfig(
            mode=representation_mode,
            reference_strategy=reference_strategy,
            enable_diagnostics=bool(context_diagnostics_enabled),
            weights=merged_weights,
        )
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
        self._temporal_consistency_history: deque[float] = deque(maxlen=TRANSITION_MEMORY_WINDOW)
        self._tetrahedral_position_history: deque[list[float]] = deque(maxlen=64)
        self._signal_instability_history: deque[float] = deque(maxlen=24)
        self._shape_change_history: deque[float] = deque(maxlen=24)
        self._spectral_shift_history: deque[float] = deque(maxlen=24)
        self._coherence_loss_history: deque[float] = deque(maxlen=24)
        self._raw_debug_frames_logged: int = 0
        self._episode_history: list[dict[str, object]] = []
        self._current_episode: dict[str, object] = {
            "current_episode_type": "onset",
            "episode_index": 0,
            "episode_start": 0,
            "episode_duration": 0,
            "episode_transition_reason": "initialization",
            "episode_history": [],
        }
        self.transition_aware_enabled: bool = _env_enabled("NERAIUM_TRANSITION_AWARE", default="1")
        self.fast_mode: bool = os.getenv("NERAIUM_FAST_MODE", "0") == "1"
        self._fast_geometry_update_interval: int = FAST_MODE_GEOMETRY_UPDATE_INTERVAL if self.fast_mode else 1
        self._fast_geometry_downsample_step: int = FAST_MODE_GEOMETRY_DOWNSAMPLE_STEP if self.fast_mode else 1
        self._fast_geometry_payload_cache: dict[str, object] | None = None
        # Extra frames after windows first fill before EMERGING/SUSTAINED labels are trusted.
        _stab = os.environ.get("NERAIUM_TRANSITION_STABILIZATION_MARGIN") or os.environ.get(
            "NERAIUM_TRANSITION_WARMUP_MARGIN", "8"
        )
        self.transition_stabilization_margin_frames: int = int(str(_stab).strip() or "8")
        self.transition_classification_min_history: int = int(
            os.environ.get("NERAIUM_TRANSITION_MIN_HISTORY", "6").strip() or "6"
        )
        if self.fast_mode:
            self.geometry_layer = StatisticalGeometryLayer(max_history=192, graph_window=12, stats_window=8)
        else:
            self.geometry_layer = StatisticalGeometryLayer(max_history=384, graph_window=24, stats_window=16)
        self._frame_debug: bool = _effective_frame_debug(frame_debug)
        if drift_smoothing_window < 1:
            raise ValueError("drift_smoothing_window must be >= 1")
        if drift_threshold_quantiles is not None:
            watch_quantile, alert_quantile = drift_threshold_quantiles
        if drift_persistence_length is not None:
            watch_persistence = int(drift_persistence_length)
        if watch_persistence < 1:
            raise ValueError("watch_persistence must be >= 1")
        if alert_persistence < 1:
            raise ValueError("alert_persistence must be >= 1")
        if fast_trigger_multiplier <= 1.0:
            raise ValueError("fast_trigger_multiplier must be > 1.0")
        if not (0.0 < unlatch_ratio < 1.0):
            raise ValueError("unlatch_ratio must be in (0, 1)")
        watch_q, alert_q = float(watch_quantile), float(alert_quantile)
        if not (0.0 < float(watch_q) < 1.0) or not (0.0 < float(alert_q) < 1.0):
            raise ValueError("watch_quantile/alert_quantile must be in (0, 1)")
        if float(watch_q) >= float(alert_q):
            raise ValueError("watch_quantile must be < alert_quantile")
        self.drift_smoothing_window = int(drift_smoothing_window)
        self.watch_quantile = float(watch_q)
        self.alert_quantile = float(alert_q)
        self.drift_threshold_quantiles = (self.watch_quantile, self.alert_quantile)
        self.watch_persistence = int(watch_persistence)
        self.alert_persistence = int(alert_persistence)
        self.drift_persistence_length = self.watch_persistence
        self.fast_trigger_multiplier = float(fast_trigger_multiplier)
        self.alert_latch_enabled = bool(alert_latch_enabled)
        self.unlatch_ratio = float(unlatch_ratio)

        # Drift-score threshold calibration (watch/alert).
        self._drift_score_history: deque[float] = deque(maxlen=120)
        self._drift_smooth_history: deque[float] = deque(maxlen=120)
        self._baseline_drift_score_samples: deque[float] = deque(maxlen=256)
        self._drift_watch_alert_thresholds: tuple[float, float] | None = None
        self._drift_smoothing_buffer: deque[float] = deque(maxlen=self.drift_smoothing_window)
        self._drift_smoothing_sum: float = 0.0
        self._watch_counter: int = 0
        self._alert_counter: int = 0
        self._alert_latched: bool = False
        self._current_alert_state: str = "STABLE"

        # Composite-score threshold calibration for legacy compatibility outputs.
        self._baseline_composite_score_samples: deque[float] = deque(maxlen=256)
        self._composite_watch_alert_thresholds: tuple[float, float] | None = None

        # Debug helpers: print first alert reasoning once per engine instance.
        self._first_alert_logged: bool = False
        self._experimental_analytics_debug_logged: bool = False
        self._geometry_debug_frames_logged: int = 0
        self._last_geometry_debug_branching_factor: float | None = None

        self.regime_store = RegimeStore(regime_store_path)
        persisted = self.regime_store.load()
        self.regime_signatures: list[dict[str, object]] = list(persisted.get("regimes", []))
        self.regime_baselines: dict[str, dict[str, object]] = dict(persisted.get("baselines", {}))

        # Baseline management: lock prevents rolling update; metadata for UI/API.
        self.baseline_locked: bool = False
        self._baseline_set_at: Optional[str] = None  # ISO timestamp when baseline was last established
        self._baseline_coverage_samples: int = 0  # Number of samples in baseline window

        # Incremental windows: frozen first baseline_window rows + rolling recent deque.
        self._recent_vector_buffer = VectorDequeBuffer(recent_window)
        self._recent_ts_buffer = TimestampDequeBuffer(recent_window)
        self._baseline_matrix_cache: np.ndarray | None = None
        self._baseline_ts_cached: list[float] | None = None
        self._sensor_schema_dirty: bool = False
        self._history_ring = HistoryRingBuffer(500)

        # --- Math engine integrations ---
        # Structural uncertainty tracker: tracks posterior over structural events.
        self._structural_uncertainty = StructuralUncertaintyTracker()
        # Monte Carlo sampler: bootstraps 90 % confidence interval on composite score.
        self._mc_sampler = MonteCarloSampler()
        # Rolling component history for MC bootstrapping (last 50 frames).
        self._component_history: deque[dict[str, float]] = deque(maxlen=50)

        # Startup: verify scoring invariants once (log warning, never raise).
        try:
            _vreport = run_all_checks()
            if not _vreport.all_passed:
                import logging as _vlog
                _vlog.getLogger(__name__).warning(
                    "Engine math verification detected issues at startup: %s",
                    _vreport.summary(),
                )
        except Exception:
            pass

    @staticmethod
    def stage_groups() -> list[dict[str, object]]:
        """Return static stage boundary metadata for extraction planning."""
        return [dict(group) for group in structural_engine_stage_groups()]

    def _default_result_payload(self, frame: Dict) -> Dict[str, object]:
        """Warmup-safe output contract with backward-compatible fields.

        Output assembly responsibility:
          - Primary policy/state fields are always present.
          - Legacy aliases remain available even before full analytics are ready.
          - Nested diagnostic payload slots are pre-populated for stable schemas.
        """
        return {
            "timestamp": frame["timestamp"],
            "site_id": frame["site_id"],
            "asset_id": frame["asset_id"],
            "state": "STABLE",
            "policy_state": "STABLE",
            "policy_watch": False,
            "policy_alert": False,
            "structural_drift_score": 0.0,
            "structural_drift_score_smoothed": 0.0,
            "drift_smooth": 0.0,
            "relational_stability_score": 1.0,
            "system_health": 100,
            "drift_alert": False,
            "sensor_relationships": self.sensor_order,
            "regime_name": None,
            "regime_distance": None,
            "regime_drift": 0.0,
            "latest_drift": 0.0,
            "latest_drift_smoothed": 0.0,
            "watch_threshold": None,
            "alert_threshold": None,
            "latest_instability": 0.0,
            "relational_instability_score": 0.0,
            "temporal_distortion_score": 0.0,
            "localization_score": 0.0,
            "attribution": {"top_drivers": [], "driver_scores": {}},
            "causal_analysis": {
                "hypotheses": [],
                "top_hypothesis": None,
                "counterfactual": {
                    "counterfactual_checks": [],
                    "robustness": 0.0,
                    "interpretation": "Causal analysis unavailable during warmup.",
                },
                "validation_plan": [],
                "recommended_sequence": [],
                "best_next_action": None,
                "status": {"available": False, "reason": "warmup"},
            },
            "dominant_driver": None,
            "explanation": "Warmup: awaiting sufficient window history.",
            "baseline_mode": None,
            "data_quality_summary": {},
            "active_sensor_count": 0,
            "missing_sensor_count": 0,
            "transition_pressure": 0.0,
            "transition_state": "NONE",
            "experimental_analytics": self._analytics_unavailable_payload("warmup"),
            "robustness": {},
            "sensitivity": {},
            "explanations": {},
            "multi_scale": {},
            "drift_noise": {},
            "geometry": {"available": False, "reason": "insufficient history"},
            "state_space_statistics": {"available": False, "reason": "insufficient history"},
            "state_graph": {"available": False, "reason": "insufficient history"},
            "geometry_explanations": {"available": False, "reason": "insufficient history"},
            "tetrahedral_state": self._safe_default_tetrahedral_payload(),
        }

    def _safe_default_tetrahedral_payload(self) -> Dict[str, object]:
        """Return a deterministic tetrahedral payload for warmup/unavailable paths."""
        try:
            return compute_tetrahedral_state(
                structural_drift_score=0.0,
                relational_instability_score=0.0,
                transition_pressure=0.0,
                temporal_consistency_score=1.0,
                history_positions=list(self._tetrahedral_position_history),
            )
        except Exception:
            return {
                "weights": {
                    "structural_drift_score": 0.25,
                    "relational_instability_score": 0.25,
                    "transition_pressure": 0.25,
                    "temporal_inconsistency": 0.25,
                },
                "position": [0.0, 0.0, 0.0],
                "nearest_vertex": "STRUCTURAL",
                "interpreted_label": "STRUCTURAL_STRESS_BUILDING",
                "nearest_face": "RELATIONAL_TRANSITION_TEMPORAL",
                "edge_alignment": 0.0,
                "speed": 0.0,
                "curvature": 0.0,
                "state_label": "BALANCED",
                "movement_summary": "stationary",
            }

    def _enforce_policy_contract(self, result: Dict[str, object]) -> None:
        """Ensure policy_* fields are sourced from structural drift policy layer only."""
        policy_state = str(self._current_alert_state or "STABLE")
        result["policy_state"] = policy_state
        result["policy_watch"] = policy_state == "WATCH"
        result["policy_alert"] = policy_state == "ALERT"
        # Backward-compat single-field consumers still read ``state``.
        result["state"] = policy_state

    @staticmethod
    def _extract_auxiliary_mahalanobis(frame: Dict[str, object]) -> float | None:
        """Compatibility read for md/mahalanobis aliases (auxiliary path only)."""
        aux_md = frame.get(
            "mahalanobis_score",
            frame.get("mahalanobis_distance", frame.get("md_signal", frame.get("md"))),
        )
        return float(aux_md) if isinstance(aux_md, (int, float)) else None

    def _attach_architecture_outputs(self, result: Dict[str, object], frame: Dict[str, object]) -> None:
        """Package explicit architectural outputs without changing primary behavior."""
        policy_state = str(result.get("policy_state", "STABLE"))
        result["core_structural_outputs"] = {
            "structural_drift_score": float(result.get("structural_drift_score", 0.0) or 0.0),
            "structural_drift_score_smoothed": float(result.get("structural_drift_score_smoothed", 0.0) or 0.0),
            "transition_pressure": float(result.get("transition_pressure", 0.0) or 0.0),
            "transition_state": str(result.get("transition_state", "NONE")),
            "regime_name": result.get("regime_name"),
            "regime_distance": result.get("regime_distance"),
            "regime_drift": float(result.get("regime_drift", 0.0) or 0.0),
        }
        result["policy_outputs"] = {
            "policy_state": policy_state,
            "policy_watch": bool(result.get("policy_watch", False)),
            "policy_alert": bool(result.get("policy_alert", False)),
        }
        result["auxiliary_diagnostics"] = {
            "mahalanobis_score": self._extract_auxiliary_mahalanobis(frame),
            "drift_noise": result.get("drift_noise", {}),
            "uncertainty": result.get("uncertainty", {}),
            "geometry": result.get("geometry", {}),
            "state_space_statistics": result.get("state_space_statistics", {}),
            "state_graph": result.get("state_graph", {}),
        }
        result["legacy_scoring"] = {
            "composite_instability_score": float(result.get("latest_instability", 0.0) or 0.0),
            "component_confidence": result.get("component_confidence", {}),
            "legacy_module": "neraium_core.scoring",
            "primary_policy_source": "structural_drift_state_machine",
        }

    def reset_baseline(self) -> None:
        """Clear rolling baseline and calibration state so baseline is recomputed from window."""
        self._rolling_baseline_corr = None
        self._baseline_set_at = None
        self._baseline_coverage_samples = 0
        self._drift_watch_alert_thresholds = None
        self._composite_watch_alert_thresholds = None
        self._baseline_drift_score_samples.clear()
        self._baseline_composite_score_samples.clear()
        self._drift_smooth_history.clear()
        self._drift_smoothing_buffer.clear()
        self._drift_smoothing_sum = 0.0
        self._watch_counter = 0
        self._alert_counter = 0
        self._alert_latched = False
        self._current_alert_state = "STABLE"
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
        self._temporal_consistency_history.clear()
        self._tetrahedral_position_history.clear()
        self._signal_instability_history.clear()
        self._shape_change_history.clear()
        self._spectral_shift_history.clear()
        self._coherence_loss_history.clear()
        self._raw_debug_frames_logged = 0
        self._structural_uncertainty.reset()
        self._component_history.clear()

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

    def _fast_mode_geometry_payload(
        self,
        *,
        frame: Dict[str, object],
        z_recent_valid: np.ndarray,
    ) -> dict[str, object]:
        """Fast-mode geometry throttle: reuse cached payloads between sampled updates."""
        should_refresh = (
            self._fast_geometry_payload_cache is None
            or (len(self.frames) % max(1, self._fast_geometry_update_interval) == 0)
        )
        if should_refresh:
            matrix = z_recent_valid
            if (
                isinstance(matrix, np.ndarray)
                and matrix.ndim == 2
                and matrix.shape[0] >= 6
                and self._fast_geometry_downsample_step > 1
            ):
                matrix = matrix[:: self._fast_geometry_downsample_step]
            self._fast_geometry_payload_cache = self.geometry_layer.update(
                entity_id=str(frame.get("asset_id", "unknown")),
                matrix=matrix,
                representation_mode=self.representation_config.resolved_mode(),
            )
        payload = self._fast_geometry_payload_cache
        return payload if isinstance(payload, dict) else {"available": False, "reason": "fast_mode_unavailable"}

    def _apply_fast_mode_payload_downgrades(self, result: Dict[str, object]) -> None:
        """Documented fast-mode simplifications while preserving stable return schema keys."""
        exp = result.get("experimental_analytics")
        if not isinstance(exp, dict):
            result["experimental_analytics"] = self._analytics_unavailable_payload("fast_mode")

        result.setdefault("drift_noise", {"available": False, "reason": "fast_mode"})
        result.setdefault(
            "multi_scale",
            {
                "short_term_state": "fast_mode",
                "mid_term_state": "fast_mode",
                "long_term_state": "fast_mode",
                "scale_conflict": 0.0,
                "scale_alignment": 1.0,
                "scale_conflict_reason": "fast_mode",
            },
        )
        result.setdefault("robustness", {})
        result.setdefault("sensitivity", {"top_drivers": [], "feature_contributions": {}})

    def process_stream_frame(self, frame: Dict) -> Dict:
        """Streaming-compatible alias that preserves deterministic batch semantics."""
        return self.process_frame(frame)

    def reset_stream(self) -> None:
        """Reset in-memory stream state while preserving constructor configuration."""
        self.frames.clear()
        self._history_ring.clear()
        self.sensor_order = []
        self.latest_result = None
        self.score_history.clear()
        self._structural_uncertainty.reset()
        self._component_history.clear()
        self._state_history.clear()
        self._transition_pressure_history.clear()
        self._regime_history.clear()
        self._adjacency_history.clear()
        self._dominant_mode_history.clear()
        self._drift_score_history.clear()
        self._drift_smooth_history.clear()
        self._shock_activity_history.clear()
        self._structural_drift_history.clear()
        self._episode_history = []
        self._current_episode = {
            "current_episode_type": "onset",
            "episode_index": 0,
            "episode_start": 0,
            "episode_duration": 0,
            "episode_transition_reason": "reset_stream",
            "episode_history": [],
        }
        self._first_alert_logged = False
        self._experimental_analytics_debug_logged = False
        self._geometry_debug_frames_logged = 0
        self._last_geometry_debug_branching_factor = None
        self.reset_baseline()

    def snapshot_state(self) -> Dict[str, object]:
        """Serializable snapshot for incremental/long-running deployments."""
        return {
            "sensor_order": list(self.sensor_order),
            "frames": list(self.frames),
            "score_history": list(self.score_history),
            "state_history": list(self._state_history),
            "transition_pressure_history": list(self._transition_pressure_history),
            "drift_score_history": list(self._drift_score_history),
            "drift_smooth_history": list(self._drift_smooth_history),
            "shock_activity_history": list(self._shock_activity_history),
            "structural_drift_history": list(self._structural_drift_history),
            "tetrahedral_position_history": list(self._tetrahedral_position_history),
            "watch_counter": int(self._watch_counter),
            "alert_counter": int(self._alert_counter),
            "alert_latched": bool(self._alert_latched),
            "current_alert_state": self._current_alert_state,
            "rolling_baseline_corr": self._rolling_baseline_corr.tolist() if isinstance(self._rolling_baseline_corr, np.ndarray) else None,
            "stable_manifold_corr": self._stable_manifold_corr.tolist() if isinstance(self._stable_manifold_corr, np.ndarray) else None,
            "baseline_locked": bool(self.baseline_locked),
            "baseline_set_at": self._baseline_set_at,
            "baseline_coverage_samples": int(self._baseline_coverage_samples),
            "current_episode": dict(self._current_episode),
            "episode_history": list(self._episode_history),
        }

    def restore_state(self, state: Dict[str, object]) -> None:
        """Restore snapshot produced by ``snapshot_state``."""
        self.sensor_order = list(state.get("sensor_order", []))
        self.frames = deque(list(state.get("frames", [])), maxlen=500)
        self.score_history = deque(list(state.get("score_history", [])), maxlen=120)
        self._state_history = deque(list(state.get("state_history", [])), maxlen=CLASSIFICATION_STABILITY_WINDOW)
        self._transition_pressure_history = deque(list(state.get("transition_pressure_history", [])), maxlen=TRANSITION_MEMORY_WINDOW)
        self._drift_score_history = deque(list(state.get("drift_score_history", [])), maxlen=120)
        self._drift_smooth_history = deque(list(state.get("drift_smooth_history", [])), maxlen=120)
        self._shock_activity_history = deque(list(state.get("shock_activity_history", [])), maxlen=TRANSITION_MEMORY_WINDOW)
        self._structural_drift_history = deque(list(state.get("structural_drift_history", [])), maxlen=TRANSITION_MEMORY_WINDOW)
        self._tetrahedral_position_history = deque(list(state.get("tetrahedral_position_history", [])), maxlen=64)
        self._watch_counter = int(state.get("watch_counter", 0))
        self._alert_counter = int(state.get("alert_counter", 0))
        self._alert_latched = bool(state.get("alert_latched", False))
        self._current_alert_state = str(state.get("current_alert_state", "STABLE"))
        self._drift_smoothing_buffer.clear()
        self._drift_smoothing_sum = 0.0
        for value in list(self._drift_score_history)[-self.drift_smoothing_window :]:
            self._drift_smoothing_buffer.append(float(value))
            self._drift_smoothing_sum += float(value)
        rbc = state.get("rolling_baseline_corr")
        smc = state.get("stable_manifold_corr")
        self._rolling_baseline_corr = np.asarray(rbc, dtype=float) if isinstance(rbc, list) else None
        self._stable_manifold_corr = np.asarray(smc, dtype=float) if isinstance(smc, list) else None
        self.baseline_locked = bool(state.get("baseline_locked", False))
        self._baseline_set_at = state.get("baseline_set_at")
        self._baseline_coverage_samples = int(state.get("baseline_coverage_samples", 0))
        self._current_episode = dict(state.get("current_episode", self._current_episode))
        self._episode_history = list(state.get("episode_history", []))
        self._history_ring.rebuild_from_frames(list(self.frames))

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
        if not self._frame_debug:
            return
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
        incoming = sorted(
            name
            for name, value in sensor_values.items()
            if self._is_real_numeric_value(value)
        )
        if not self.sensor_order:
            self.sensor_order = list(incoming)
        else:
            merged = sorted(set(self.sensor_order) | set(incoming))
            if merged != self.sensor_order:
                self.sensor_order = merged
                for f in self.frames:
                    sv = f.get("sensor_values") or {}
                    f["_vector"] = _vector_from_sensor_values(sv, self.sensor_order)
                self._sensor_schema_dirty = True
        return _vector_from_sensor_values(sensor_values, self.sensor_order)

    @staticmethod
    def _is_real_numeric_value(value: object) -> bool:
        """True for finite non-bool numeric values observed in real telemetry frames."""
        if isinstance(value, bool):
            return False
        try:
            fv = float(value)
        except (TypeError, ValueError):
            return False
        return bool(np.isfinite(fv))

    @staticmethod
    def _is_valid_window_matrix(m: np.ndarray | None, *, min_rows: int = 2) -> bool:
        """Shared cache/window readiness gate: non-empty rows and non-zero feature width."""
        if m is None:
            return False
        if not isinstance(m, np.ndarray):
            return False
        if m.ndim != 2:
            return False
        rows, cols = int(m.shape[0]), int(m.shape[1])
        return rows >= int(min_rows) and cols >= 1

    @classmethod
    def _windows_ready(cls, baseline: np.ndarray | None, recent: np.ndarray | None) -> bool:
        if not cls._is_valid_window_matrix(baseline) or not cls._is_valid_window_matrix(recent):
            return False
        return bool(baseline is not None and recent is not None and baseline.shape[1] == recent.shape[1])

    def _extract_windows_from_chronological(self, m: np.ndarray) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Baseline / recent slices from a chronological (oldest→newest) matrix."""
        if m.ndim != 2:
            return None, None
        if m.shape[0] < self.baseline_window or m.shape[0] < self.recent_window:
            return None, None
        bl = m[: self.baseline_window][:: self.window_stride]
        rc = m[-self.recent_window :][:: self.window_stride]
        if not self._windows_ready(bl, rc):
            return None, None
        return bl, rc

    def _get_recent_timestamps(self, frames_list: list[dict] | None = None) -> Optional[list[float]]:
        """Timestamps for the trailing ``recent_window`` frames.

        When ``frames_list`` is omitted, indexes ``self.frames`` (a deque) directly to
        avoid ``list(self.frames)`` allocations on the post-warmup hot path.
        """
        if frames_list is not None:
            fl = frames_list
            if len(fl) < self.recent_window:
                return None
            frame_iter = fl[-self.recent_window :]
        else:
            n = len(self.frames)
            if n < self.recent_window:
                return None
            frame_iter = (self.frames[j] for j in range(-self.recent_window, 0))
        ts_vals: list[float] = []
        for f in frame_iter:
            try:
                ts_vals.append(float(f.get("timestamp")))
            except (TypeError, ValueError):
                continue
        return ts_vals if len(ts_vals) >= 2 else None

    def _get_baseline_timestamps(self, frames_list: list[dict] | None = None) -> Optional[list[float]]:
        if frames_list is not None:
            fl = frames_list
            if len(fl) < self.baseline_window:
                return None
            frame_iter = fl[: self.baseline_window]
        else:
            n = len(self.frames)
            if n < self.baseline_window:
                return None
            bw = min(self.baseline_window, n)
            frame_iter = (self.frames[j] for j in range(bw))
        ts_vals: list[float] = []
        for f in frame_iter:
            try:
                ts_vals.append(float(f.get("timestamp")))
            except (TypeError, ValueError):
                continue
        return ts_vals if len(ts_vals) >= 2 else None

    def _invalidate_window_caches(self) -> None:
        self._baseline_matrix_cache = None
        self._baseline_ts_cached = None
        self._recent_vector_buffer.clear()
        self._recent_ts_buffer.clear()

    def _rebuild_incremental_buffers_after_schema_change(self) -> None:
        """After sensor-order merge, repopulate rolling buffers from the current deque."""
        for f in self.frames:
            v = f.get("_vector")
            if v is None:
                continue
            try:
                tsv = float(f.get("timestamp"))
            except (TypeError, ValueError):
                tsv = 0.0
            self._recent_vector_buffer.append(np.asarray(v, dtype=float))
            self._recent_ts_buffer.append(tsv)
        if len(self.frames) >= self.baseline_window:
            m = self._history_ring.chronological_matrix()
            vecs = m[: self.baseline_window][:: self.window_stride]
            if self._is_valid_window_matrix(vecs):
                self._baseline_matrix_cache = np.asarray(vecs, dtype=np.float64, order="C")
                fl = list(self.frames)
                ts_b: list[float] = []
                for f in fl[: self.baseline_window]:
                    try:
                        ts_b.append(float(f.get("timestamp")))
                    except (TypeError, ValueError):
                        ts_b.append(0.0)
                self._baseline_ts_cached = ts_b

    def _refresh_baseline_matrix_cache(self) -> None:
        """Snapshot first baseline_window frames once the deque has filled them."""
        if self._baseline_matrix_cache is not None:
            return
        if len(self.frames) < self.baseline_window:
            return
        m = self._history_ring.chronological_matrix()
        vecs = m[: self.baseline_window][:: self.window_stride]
        if self._is_valid_window_matrix(vecs):
            self._baseline_matrix_cache = np.asarray(vecs, dtype=np.float64, order="C")
            fl = list(self.frames)
            ts_b: list[float] = []
            for f in fl[: self.baseline_window]:
                try:
                    ts_b.append(float(f.get("timestamp")))
                except (TypeError, ValueError):
                    ts_b.append(0.0)
            self._baseline_ts_cached = ts_b

    def _materialize_strided_recent(self) -> np.ndarray | None:
        m = self._recent_vector_buffer.to_matrix()
        if m is None:
            return None
        vectors = m[:: self.window_stride]
        return vectors if self._is_valid_window_matrix(vectors) else None

    def _system_health(self, drift_score: float, stability_score: float) -> int:
        health = 100.0 - min(drift_score * 20.0, 85.0)
        health += stability_score * 20.0
        return int(round(max(0.0, min(100.0, health))))

    def _update_drift_state_machine(self, drift_score: float) -> tuple[str, float]:
        # O(1) rolling mean for drift smoothing to avoid frame-level overhead.
        raw = float(drift_score)
        if self._drift_smoothing_buffer.maxlen is None or len(self._drift_smoothing_buffer) < self._drift_smoothing_buffer.maxlen:
            self._drift_smoothing_buffer.append(raw)
            self._drift_smoothing_sum += raw
        else:
            dropped = float(self._drift_smoothing_buffer.popleft())
            self._drift_smoothing_sum -= dropped
            self._drift_smoothing_buffer.append(raw)
            self._drift_smoothing_sum += raw

        smooth = raw
        if self._drift_smoothing_buffer:
            smooth = float(self._drift_smoothing_sum / len(self._drift_smoothing_buffer))
        self._drift_smooth_history.append(smooth)

        # Until we have nominal calibration samples, suppress early alerts
        # to avoid false positives driven by unstable correlation estimates.
        if self._drift_watch_alert_thresholds is None:
            self._current_alert_state = "STABLE"
            return self._current_alert_state, smooth

        watch_thr, alert_thr = self._drift_watch_alert_thresholds
        if smooth > alert_thr:
            self._alert_counter += 1
        else:
            self._alert_counter = max(0, self._alert_counter - 1)

        if smooth > watch_thr:
            self._watch_counter += 1
        else:
            self._watch_counter = max(0, self._watch_counter - 1)

        if self._alert_counter >= self.alert_persistence:
            self._alert_latched = True
        if smooth > (alert_thr * self.fast_trigger_multiplier):
            self._alert_latched = True
            self._alert_counter = max(self._alert_counter, self.alert_persistence)
        if not self.alert_latch_enabled and smooth <= alert_thr:
            self._alert_latched = False

        if self._alert_latched and smooth < (watch_thr * self.unlatch_ratio):
            self._alert_latched = False
            self._alert_counter = 0

        if self._alert_latched:
            self._current_alert_state = "ALERT"
        elif self._alert_counter >= self.alert_persistence:
            self._current_alert_state = "ALERT"
        elif self._watch_counter >= self.watch_persistence:
            self._current_alert_state = "WATCH"
        else:
            self._current_alert_state = "STABLE"
        return self._current_alert_state, smooth

    def _alert_state(self, drift_score: float) -> str:
        return self._current_alert_state

    def _drift_alert(self, drift_score: float) -> bool:
        return self._current_alert_state == "ALERT"

    @staticmethod
    def _clamp01(value: float) -> float:
        return float(max(0.0, min(1.0, value)))

    @staticmethod
    def _signal_feature_value(delta: dict[str, float], pattern: str) -> float:
        vals = [float(v) for k, v in delta.items() if pattern in k]
        if not vals:
            return 0.0
        return float(np.mean(vals))

    def _derive_signal_degradation(self, rich_features: dict[str, object] | None) -> dict[str, object]:
        payload = rich_features if isinstance(rich_features, dict) else {}
        delta = payload.get("delta", {}) if isinstance(payload.get("delta"), dict) else {}
        change_summary = payload.get("change_summary", {}) if isinstance(payload.get("change_summary"), dict) else {}

        volatility_erosion = self._clamp01(
            0.45 * max(0.0, self._signal_feature_value(delta, "rolling_energy"))
            + 0.20 * max(0.0, self._signal_feature_value(delta, "rolling_local_volatility"))
            + 0.20 * max(0.0, self._signal_feature_value(delta, "mean_abs_first_diff"))
            + 0.15 * max(0.0, self._signal_feature_value(delta, "mean_abs_second_diff"))
        )
        spectral_shift = self._clamp01(
            0.26 * abs(self._signal_feature_value(delta, "spectral_centroid"))
            + 0.18 * abs(self._signal_feature_value(delta, "spectral_spread"))
            + 0.16 * max(0.0, self._signal_feature_value(delta, "spectral_entropy"))
            + 0.20 * abs(self._signal_feature_value(delta, "low_high_frequency_energy_ratio"))
            + 0.20 * abs(self._signal_feature_value(delta, "dominant_frequency_ratio"))
        )
        shape_change = self._clamp01(
            0.22 * max(0.0, self._signal_feature_value(delta, "kurtosis"))
            + 0.15 * abs(self._signal_feature_value(delta, "crest_factor"))
            + 0.15 * max(0.0, self._signal_feature_value(delta, "roughness"))
            + 0.14 * max(0.0, self._signal_feature_value(delta, "percentile_spread"))
            + 0.14 * max(0.0, self._signal_feature_value(delta, "local_jitter_score"))
            + 0.10 * abs(self._signal_feature_value(delta, "skewness"))
        )
        coherence_loss = self._clamp01(
            0.45 * max(0.0, self._signal_feature_value(delta, "sync_loss"))
            + 0.35 * max(0.0, self._signal_feature_value(delta, "decoupling_index"))
            + 0.20 * max(0.0, self._signal_feature_value(delta, "relative_drift"))
        )
        consistency_breakdown = self._clamp01(float(change_summary.get("feature_consistency_breakdown", 0.0)))
        feature_drift = self._clamp01(float(change_summary.get("feature_drift_magnitude", 0.0)))
        volatility_drift = self._clamp01(float(change_summary.get("feature_volatility_drift", 0.0)))
        signal_instability = self._clamp01(
            0.33 * volatility_erosion + 0.28 * shape_change + 0.19 * feature_drift + 0.20 * consistency_breakdown
        )
        coherence_loss = self._clamp01(0.75 * coherence_loss + 0.25 * consistency_breakdown)
        volatility_erosion = self._clamp01(0.7 * volatility_erosion + 0.3 * volatility_drift)

        self._signal_instability_history.append(signal_instability)
        self._shape_change_history.append(shape_change)
        self._spectral_shift_history.append(spectral_shift)
        self._coherence_loss_history.append(coherence_loss)

        drift_accel = 0.0
        if len(self._signal_instability_history) >= 3:
            vals = list(self._signal_instability_history)
            drift_accel = max(0.0, (vals[-1] - vals[-2]) - (vals[-2] - vals[-3]))
        trend_persistence = 0.0
        if len(self._signal_instability_history) >= 5:
            tail = np.asarray(list(self._signal_instability_history)[-5:], dtype=float)
            trend_persistence = float(np.mean(np.diff(tail) > 0.0))
        signal_instability = self._clamp01(signal_instability + 0.08 * drift_accel + 0.08 * trend_persistence)

        total = 0.30 * signal_instability + 0.23 * shape_change + 0.25 * spectral_shift + 0.12 * volatility_erosion + 0.10 * coherence_loss
        if total >= 0.7:
            state = "ELEVATED"
        elif total >= 0.4:
            state = "EMERGING"
        else:
            state = "NOMINAL"

        drivers = [
            ("signal_instability_score", signal_instability),
            ("spectral_shift_score", spectral_shift),
            ("shape_change_score", shape_change),
            ("volatility_erosion_score", volatility_erosion),
            ("coherence_loss_score", coherence_loss),
        ]
        drivers.sort(key=lambda x: x[1], reverse=True)

        return {
            "signal_instability_score": round(float(signal_instability), 4),
            "energy_instability_score": round(float(signal_instability), 4),
            "spectral_shift_score": round(float(spectral_shift), 4),
            "shape_change_score": round(float(shape_change), 4),
            "shape_instability_score": round(float(shape_change), 4),
            "volatility_erosion_score": round(float(volatility_erosion), 4),
            "coherence_loss_score": round(float(coherence_loss), 4),
            "signal_degradation_state": state,
            "top_signal_drivers": [
                {"metric": k, "score": round(float(v), 4)} for k, v in drivers[:3]
            ],
            "top_degradation_drivers": [
                {"metric": k, "score": round(float(v), 4)} for k, v in drivers[:3]
            ],
            "composite_signal_degradation": round(float(total), 4),
        }

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
        temporal_features: dict[str, object] | None = None,
        temporal_quality: dict[str, float] | None = None,
    ) -> dict[str, float]:
        drift_hist = list(self._drift_score_history)
        drift_delta = 0.0
        drift_accel = 0.0
        if len(drift_hist) >= 2:
            drift_delta = max(0.0, float(drift_hist[-1] - drift_hist[-2]))
        if len(drift_hist) >= 3:
            prev_delta = float(drift_hist[-2] - drift_hist[-3])
            drift_accel = max(0.0, drift_delta - prev_delta)
        time_accel = float((temporal_features or {}).get("drift_acceleration", 0.0))
        time_vel = float((temporal_features or {}).get("drift_velocity", 0.0))
        temporal_consistency = float((temporal_quality or {}).get("temporal_consistency_score", 1.0))
        gap_irregularity = float((temporal_quality or {}).get("timestamp_gap_irregularity", 0.0))
        short_horizon_drift = self._clamp01(0.66 * drift_delta + 0.20 * max(0.0, drift_score - 0.55) + 0.14 * time_vel)
        acceleration = self._clamp01(drift_accel * 2.0 + max(0.0, drift_delta - 0.09) + 0.7 * time_accel)

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
        time_gate = 0.75 + 0.35 * temporal_consistency + 0.30 * gap_irregularity
        pressure = self._transition_pressure_ema * activity_gate * streak_decay * time_gate
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
            "temporal_consistency": float(self._clamp01(temporal_consistency)),
            "timestamp_gap_irregularity": float(self._clamp01(gap_irregularity)),
            "timing_velocity": float(self._clamp01(time_vel)),
            "timing_acceleration": float(self._clamp01(time_accel)),
        }

    def _transition_state(self, pressure: float, temporal_consistency: float = 1.0) -> str:
        """Classify transition from recent pressure history. Caller must append ``pressure`` to history first."""
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
        consistency_boost = 0.85 + 0.30 * self._clamp01(float(temporal_consistency))
        weighted_current = current * consistency_boost
        weighted_recent = weighted_recent * consistency_boost
        if (weighted_current >= TRANSITION_SUSTAINED_THRESHOLD and sustained >= 2) or sustained >= 3:
            return "SUSTAINED_TRANSITION"
        if (weighted_current >= TRANSITION_EMERGING_THRESHOLD and emerging >= 2) or weighted_recent >= (
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
        sensor_values = frame.get("sensor_values") or {}

        history_transition_len_before = len(self._transition_pressure_history)
        history_shock_len_before = len(self._shock_activity_history)
        history_drift_len_before = len(self._structural_drift_history)

        stored = dict(frame)
        stored["_vector"] = vector
        self.frames.append(stored)

        try:
            ts_val = float(frame["timestamp"])
        except (TypeError, ValueError):
            ts_val = 0.0
        try:
            ts_ring = float(frame["timestamp"])
        except (TypeError, ValueError):
            ts_ring = float(len(self.frames) - 1)
        if self._sensor_schema_dirty:
            self._invalidate_window_caches()
            self._history_ring.rebuild_from_frames(list(self.frames))
            self._rebuild_incremental_buffers_after_schema_change()
            self._sensor_schema_dirty = False
        else:
            self._history_ring.append(vector, ts_ring)
            if _incremental_windows_enabled():
                self._recent_vector_buffer.append(vector)
                self._recent_ts_buffer.append(ts_val)
                self._refresh_baseline_matrix_cache()

        result = self._default_result_payload(frame)
        temporal_quality: dict[str, object] = {}
        temporal_features: dict[str, object] = {}

        # Skip deque→list snapshot during warmup (saves O(n) per frame until windows fill).
        can_process_full_frame = False
        baseline_window = None
        recent_window = None
        chronological_M: np.ndarray | None = None
        if len(self.frames) >= self.baseline_window and len(self.frames) >= self.recent_window:
            if _incremental_windows_enabled() and self._is_valid_window_matrix(self._baseline_matrix_cache):
                ir = self._materialize_strided_recent()
                if self._windows_ready(self._baseline_matrix_cache, ir):
                    baseline_window = self._baseline_matrix_cache
                    recent_window = ir
            if not self._windows_ready(baseline_window, recent_window):
                chronological_M = self._history_ring.chronological_matrix()
                baseline_window, recent_window = self._extract_windows_from_chronological(chronological_M)
            can_process_full_frame = self._windows_ready(baseline_window, recent_window)

        if can_process_full_frame:
            if chronological_M is not None:
                history_matrix = chronological_M
            else:
                history_matrix = self._history_ring.chronological_matrix()
            history_ts = self._history_ring.chronological_timestamps()
            rep = build_temporal_representation(history_matrix, self.representation_config, timestamps=history_ts)
            transformed_history = rep.transformed
            baseline_window = np.asarray(
                transformed_history[: self.baseline_window][:: self.window_stride],
                dtype=float,
            )
            recent_window = np.asarray(
                transformed_history[-self.recent_window :][:: self.window_stride],
                dtype=float,
            )
            ts_baseline = self._get_baseline_timestamps(None)
            ts_recent = self._get_recent_timestamps(None)
            temporal_quality = derive_temporal_quality_signals(ts_recent)

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
            temporal_features = derive_temporal_rate_features(recent_window=z_recent, timestamps=ts_recent)

            valid_mask = (np.nan_to_num(recent_std) > 1e-12) | (np.nan_to_num(baseline_std) > 1e-12)
            valid_signal_count = int(np.sum(valid_mask))
            valid_signal_count = min(valid_signal_count, len(self.sensor_order))

            warning = early_warning_metrics(np.nan_to_num(recent_window, nan=0.0))

            signature = build_regime_signature(recent_mean, recent_std)
            self.regime_signatures = update_regime_library(signature, self.regime_signatures)
            assigned_regime = assign_regime(signature, self.regime_signatures)

            regime_name = assigned_regime["name"] if assigned_regime else None
            regime_distance = float(assigned_regime["distance"]) if assigned_regime else None
            coherence_margin = float(assigned_regime.get("coherence_margin", 0.0)) if assigned_regime else 0.0
            regime_novelty = max(0.0, -coherence_margin)

            analytics: dict[str, object] = {
                **self._analytics_unavailable_payload("pending_multivariate_processing"),
                "early_warning": warning,
                "relational_metrics_skipped": valid_signal_count < 2,
                "representation": {
                    "mode": self.representation_config.resolved_mode(),
                    "reference_strategy": self.representation_config.resolved_strategy(),
                    "weights": {
                        "raw_weight": float(self.representation_config.weights.raw_weight),
                        "residual_weight": float(self.representation_config.weights.residual_weight),
                        "delta_weight": float(self.representation_config.weights.delta_weight),
                        "slope_weight": float(self.representation_config.weights.slope_weight),
                        "drift_weight": float(self.representation_config.weights.drift_weight),
                        "second_diff_weight": float(self.representation_config.weights.second_diff_weight),
                    },
                },
                "regime_signature": {
                    "current": [float(v) for v in signature],
                    "nearest": assigned_regime,
                    "assigned_name": regime_name,
                    "library_size": len(self.regime_signatures),
                    "coherence_margin": round(float(coherence_margin), 6),
                },
            }
            if self.representation_config.enable_diagnostics:
                analytics["context_diagnostics"] = dict(rep.diagnostics)

            components = canonicalize_components(
                {
                    "drift": 0.0,
                    "regime_drift": regime_novelty,
                    "early_warning": warning["variance"] + max(0.0, warning["lag1_autocorrelation"]),
                }
            )

            if valid_signal_count >= 2:
                z_base_valid = z_baseline[:, valid_mask]
                z_recent_valid = z_recent[:, valid_mask]
                stage_features = FeatureExtractionStage.extract(z_base_valid, z_recent_valid)
                rich_signal = stage_features.get("rich_signal_features", {})
                signal_degradation = self._derive_signal_degradation(
                    rich_signal if isinstance(rich_signal, dict) else {}
                )
                result["signal_degradation"] = signal_degradation

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
                        watch_thr = float(np.quantile(list(self._baseline_drift_score_samples), self.watch_quantile))
                        alert_thr = float(np.quantile(list(self._baseline_drift_score_samples), self.alert_quantile))
                        if alert_thr < watch_thr:
                            watch_thr, alert_thr = alert_thr, watch_thr
                        self._drift_watch_alert_thresholds = (watch_thr, alert_thr)
                alert_state, smoothed_drift_score = self._update_drift_state_machine(drift_score)
                rel_delta_legacy = flatten_upper_tri(corr_recent) - flatten_upper_tri(baseline_corr_used)
                relational_raw = float(np.mean(np.abs(rel_delta_legacy))) if rel_delta_legacy.size else 0.0
                relational_raw = max(relational_raw, stage_relational_raw, 0.5 * stage_structural_raw)
                stability_score = 1.0 / (1.0 + drift_score)

                regime_drift = 0.0
                if regime_name is not None:
                    persist_regime_state = False
                    if regime_name not in self.regime_baselines:
                        self.regime_baselines[regime_name] = {
                            "signature": signature.tolist(),
                            "correlation": corr_recent.tolist(),
                            "count": 1,
                        }
                        persist_regime_state = True
                    else:
                        regime_corr = np.asarray(self.regime_baselines[regime_name]["correlation"], dtype=float)
                        if regime_corr.shape == corr_recent.shape:
                            regime_drift = structural_drift(corr_recent, regime_corr, norm="fro")
                            # Regime-specific baseline: EMA update so we gradually adapt inside stable regime.
                            alpha = 0.88
                            updated = alpha * regime_corr + (1.0 - alpha) * corr_recent
                            self.regime_baselines[regime_name]["correlation"] = updated.tolist()
                            regime_count = int(
                                self.regime_baselines[regime_name].get("count", 0)
                            ) + 1
                            self.regime_baselines[regime_name]["count"] = regime_count
                            # Persist periodically to avoid high write/serialization overhead while
                            # preserving durability over long-lived streams.
                            persist_regime_state = (regime_count % 16) == 0
                        else:
                            self.regime_baselines[regime_name] = {
                                "signature": signature.tolist(),
                                "correlation": corr_recent.tolist(),
                                "count": 1,
                            }
                            regime_drift = 0.0
                            persist_regime_state = True

                    if persist_regime_state:
                        self._persist_regime_state()

                signal_importance = signal_structural_importance(corr_recent)
                adjacency = thresholded_adjacency(corr_recent, threshold=0.6)
                graph = graph_metrics(adjacency, corr=corr_recent)

                directional = directional_metrics(lagged_correlation_matrix(z_recent_valid, lag=1))

                causal_matrix = granger_causality_matrix(z_recent_valid)
                causal = causal_metrics(causal_matrix)
                causal_graph = causal_graph_metrics(causal_matrix, threshold=0.1)

                valid_sensor_names = [rep.feature_names[i] for i in range(len(valid_mask)) if valid_mask[i]]
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
                result["attribution"] = {
                    "top_drivers": attr.get("top_drivers", []),
                    "driver_scores": attr.get("driver_scores", {}),
                }
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
                    "relational_drift": relational_raw + 0.12 * float(signal_degradation.get("coherence_loss_score", 0.0)),
                    "regime_drift": regime_drift,
                    "transition_pressure": 0.07 * float(signal_degradation.get("signal_instability_score", 0.0))
                    + 0.06 * float(signal_degradation.get("volatility_erosion_score", 0.0)),
                    "spectral": spectral["radius"] + 0.25 * float(signal_degradation.get("spectral_shift_score", 0.0)),
                    "directional": max(
                        float(directional.get("divergence", 0.0)),
                        float(causal.get("causal_divergence", 0.0)),
                    ) + 0.10 * float(signal_degradation.get("shape_change_score", 0.0)),
                    "entropy": entropy_score + 0.08 * float(signal_degradation.get("shape_change_score", 0.0)),
                    "subsystem_instability": float(subsystem["max_instability"]),
                    "temporal_distortion": temporal_raw + 0.12 * float(signal_degradation.get("signal_instability_score", 0.0)),
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
                        "structural_drift_score_smoothed": round(float(smoothed_drift_score), 4),
                        "drift_smooth": round(float(smoothed_drift_score), 4),
                        "relational_stability_score": round(stability_score, 4),
                        "system_health": self._system_health(drift_score, stability_score),
                        # Backward-compat precedence: `state` intentionally reflects
                        # policy state so existing consumers stay single-field.
                        "state": alert_state,
                        "drift_alert": alert_state == "ALERT",
                        "policy_state": alert_state,
                        "policy_watch": alert_state == "WATCH",
                        "policy_alert": alert_state == "ALERT",
                        "regime_name": regime_name,
                        "regime_distance": round(regime_distance, 4) if regime_distance is not None else None,
                        "regime_drift": round(float(regime_drift), 4),
                        "latest_drift": round(float(drift_score), 4),
                        "latest_drift_smoothed": round(float(smoothed_drift_score), 4),
                        "baseline_mode": baseline_mode,
                        "context_dominance_score": round(float(rep.diagnostics.get("context_dominance_score", 0.0)), 4),
                        "dynamic_signal_strength": round(float(rep.diagnostics.get("dynamic_signal_strength", 0.0)), 4),
                        "early_separation_flag": bool(rep.diagnostics.get("early_separation_flag", False)),
                    }
                )
                if self._drift_watch_alert_thresholds is not None:
                    watch_thr, alert_thr = self._drift_watch_alert_thresholds
                    result["drift_thresholds"] = {
                        "watch": round(float(watch_thr), 6),
                        "alert": round(float(alert_thr), 6),
                    }
                    result["watch_threshold"] = round(float(watch_thr), 6)
                    result["alert_threshold"] = round(float(alert_thr), 6)
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
                        temporal_features=temporal_features,
                        temporal_quality=temporal_quality,
                    )
                    transition_pressure = float(transition_metrics["transition_pressure"])
                    raw_components["transition_pressure"] = transition_pressure
                    components["transition_pressure"] = transition_pressure
                    result["transition_pressure"] = round(transition_pressure, 4)
                    result["transition_state"] = "NONE"
                    self._regime_history.append(regime_name)
                    self._adjacency_history.append(np.asarray(adjacency, dtype=float))
                    if dominant_mode.size:
                        self._dominant_mode_history.append(dominant_mode)
                else:
                    result["transition_pressure"] = 0.0
                    result["transition_state"] = "NONE"

                if self.fast_mode:
                    geometry_payload = self._fast_mode_geometry_payload(
                        frame=frame,
                        z_recent_valid=z_recent_valid,
                    )
                else:
                    geometry_payload = self.geometry_layer.update(
                        entity_id=str(frame.get("asset_id", "unknown")),
                        matrix=z_recent_valid,
                        representation_mode=self.representation_config.resolved_mode(),
                    )
                geometry_metrics = geometry_payload.get("geometry", {}) if isinstance(geometry_payload, dict) else {}
                state_space_statistics = (
                    geometry_payload.get("state_space_statistics", {}) if isinstance(geometry_payload, dict) else {}
                )
                state_graph = geometry_payload.get("state_graph", {}) if isinstance(geometry_payload, dict) else {}
                geometry_available = bool(
                    isinstance(geometry_metrics, dict) and geometry_metrics.get("available", True) is not False
                )
                state_space_av = bool(
                    isinstance(state_space_statistics, dict) and state_space_statistics.get("available", True) is not False
                )
                state_graph_av = bool(
                    isinstance(state_graph, dict) and state_graph.get("available", True) is not False
                )
                if self.transition_aware_enabled:
                    base_transition_pressure = float(result.get("transition_pressure", 0.0))
                    final_transition_pressure = base_transition_pressure
                    if geometry_available:
                        geom_curvature = self._clamp01(float(geometry_metrics.get("curvature", 0.0)))
                        geom_directional_consistency = self._clamp01(
                            float(geometry_metrics.get("directional_consistency", 0.0))
                        )
                        state_contraction = self._clamp01(
                            float(state_space_statistics.get("state_contraction_score", 0.0))
                        )
                        state_expansion = self._clamp01(float(state_space_statistics.get("state_expansion_score", 0.0)))
                        geometry_pressure_term = self._clamp01(
                            0.32 * geom_curvature
                            + 0.24 * (1.0 - geom_directional_consistency)
                            + 0.22 * state_expansion
                            + 0.22 * state_contraction
                        )
                        adjusted_transition_pressure = max(
                            0.0, base_transition_pressure * (0.98 + 0.08 * geometry_pressure_term)
                        )
                        transition_metrics["geometry_transition_term"] = float(geometry_pressure_term)
                        transition_metrics["transition_pressure_pre_geometry"] = float(base_transition_pressure)
                        transition_metrics["transition_pressure"] = float(adjusted_transition_pressure)
                        final_transition_pressure = float(adjusted_transition_pressure)
                        result["transition_pressure"] = round(float(final_transition_pressure), 4)
                        components["transition_pressure"] = float(final_transition_pressure)
                        raw_components["transition_pressure"] = float(final_transition_pressure)
                    else:
                        transition_metrics["geometry_transition_term"] = 0.0
                    self._transition_pressure_history.append(float(final_transition_pressure))
                    rd_gate = compute_engine_readiness(
                        frame_count=len(self.frames),
                        baseline_window=self.baseline_window,
                        recent_window=self.recent_window,
                        transition_pressure_history_len=len(self._transition_pressure_history),
                        warmup_margin_frames=self.transition_stabilization_margin_frames,
                        min_transition_history=self.transition_classification_min_history,
                        geometry_available=geometry_available,
                        geometry_reason=str(geometry_metrics.get("reason", "")) if isinstance(geometry_metrics, dict) else None,
                        state_space_available=state_space_av,
                        state_space_reason=str(state_space_statistics.get("reason", ""))
                        if isinstance(state_space_statistics, dict)
                        else None,
                        state_graph_available=state_graph_av,
                        state_graph_reason=str(state_graph.get("reason", "")) if isinstance(state_graph, dict) else None,
                    )
                    if not rd_gate.transition_classification_ready:
                        result["transition_state"] = "WARMUP"
                    else:
                        result["transition_state"] = self._transition_state(
                            float(final_transition_pressure),
                            temporal_consistency=float(temporal_quality.get("temporal_consistency_score", 1.0)),
                        )

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
                if not self.fast_mode:
                    directional_evolution = derive_directional_evolution_features(
                        recent_window=z_recent_valid,
                        feature_names=valid_sensor_names,
                        timestamps=ts_recent,
                    )
                    trajectory_shape = derive_trajectory_shape_features(
                        recent_window=z_recent_valid,
                        timestamps=ts_recent,
                    )
                    path_prototypes = derive_path_prototypes(
                        directional_evolution=directional_evolution,
                        trajectory_shape=trajectory_shape,
                        geometry=geometry_metrics,
                        state_graph=state_graph,
                        top_k=3,
                    )
                    trajectory_analysis = classify_trajectory_path(
                        drift_history=list(self._drift_score_history),
                        transition_pressure_history=list(self._transition_pressure_history),
                        subsystem_instability_history=list(self._subsystem_instability_history),
                        regime_novelty_history=list(self._regime_novelty_history),
                        shock_activity_history=list(self._shock_activity_history),
                        reversibility_classification=str(reversibility.get("classification", "")),
                        reversibility_score=float(reversibility_scores.get("locked_in_index", 0.0)),
                        directional_evolution=directional_evolution,
                        trajectory_shape=trajectory_shape,
                        path_prototypes=path_prototypes,
                        temporal_quality=temporal_quality,
                        temporal_features=temporal_features,
                        geometry=geometry_metrics,
                        state_space_statistics=state_space_statistics,
                        state_graph=state_graph,
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
                        temporal_quality=temporal_quality,
                        temporal_features=temporal_features,
                        state_space_statistics=state_space_statistics if isinstance(state_space_statistics, dict) else None,
                        state_graph=state_graph if isinstance(state_graph, dict) else None,
                    )
                    branching_analysis = derive_branching_analysis(
                        trajectory_analysis,
                        temporal_quality=temporal_quality,
                        temporal_features=temporal_features,
                        geometry=geometry_metrics if isinstance(geometry_metrics, dict) else None,
                        state_graph=state_graph if isinstance(state_graph, dict) else None,
                    )
                    horizon_analysis = estimate_risk_horizon(
                        transition_pressure_history=list(self._transition_pressure_history),
                        shock_activity_history=list(self._shock_activity_history),
                        structural_drift_history=list(self._drift_score_history),
                        trajectory_analysis=trajectory_analysis,
                        branching_analysis=branching_analysis,
                        constraint_analysis=constraint_analysis,
                        temporal_quality=temporal_quality,
                        temporal_features=temporal_features,
                        geometry=geometry_metrics if isinstance(geometry_metrics, dict) else None,
                        state_space_statistics=state_space_statistics if isinstance(state_space_statistics, dict) else None,
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
                        temporal_quality=temporal_quality,
                        temporal_features=temporal_features,
                    )
                else:
                    trajectory_analysis = {"available": False, "reason": "fast_mode"}
                    branching_analysis = {"available": False, "reason": "fast_mode"}
                    hierarchy_analysis = {"available": False, "reason": "fast_mode"}
                    constraint_analysis = {"available": False, "reason": "fast_mode"}
                    horizon_analysis = {"available": False, "reason": "fast_mode"}
                    counterfactual_simulation = {"available": False, "reason": "fast_mode"}

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
                        "temporal_quality": temporal_quality,
                        "temporal_features": temporal_features,
                        "trajectory_analysis": trajectory_analysis,
                        "branching_analysis": branching_analysis,
                        "hierarchy_analysis": hierarchy_analysis,
                        "constraint_analysis": constraint_analysis,
                        "horizon_analysis": horizon_analysis,
                        "counterfactual_simulation": counterfactual_simulation,
                        "geometry": geometry_metrics,
                        "state_space_statistics": state_space_statistics,
                        "state_graph": state_graph,
                        "geometry_explanations": geometry_payload.get("geometry_explanations", {}),
                        "fleet_geometry": geometry_payload.get("fleet_geometry", {}),
                        "state_space": geometry_payload.get("state_space", {}),
                        "signal_features": rich_signal,
                        "signal_degradation": signal_degradation,
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
                result["signal_degradation"] = {
                    "signal_instability_score": 0.0,
                    "energy_instability_score": 0.0,
                    "spectral_shift_score": 0.0,
                    "shape_change_score": 0.0,
                    "shape_instability_score": 0.0,
                    "volatility_erosion_score": 0.0,
                    "coherence_loss_score": 0.0,
                    "signal_degradation_state": "NOMINAL",
                    "top_signal_drivers": [],
                    "top_degradation_drivers": [],
                    "composite_signal_degradation": 0.0,
                }
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
                analytics["geometry"] = {"available": False, "reason": "insufficient history"}
                analytics["state_space_statistics"] = {"available": False, "reason": "insufficient history"}
                analytics["state_graph"] = {"available": False, "reason": "insufficient history"}
                analytics["geometry_explanations"] = {"available": False, "reason": "insufficient history"}
                analytics["fleet_geometry"] = {}
                analytics["state_space"] = {}

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

            # --- Math engine: structural uncertainty posteriors ---
            # Attach uncertainty over structural events (not score labels).
            self._component_history.append(
                {k: float(v) for k, v in components.items() if isinstance(v, (int, float))}
            )
            try:
                _struct_est = self._structural_uncertainty.update(
                    {
                        "relational_drift": float(components.get("relational_drift", 0.0) or 0.0),
                        "spectral": float(components.get("spectral", 0.0) or 0.0),
                        "operator_drift": float(components.get("regime_drift", 0.0) or 0.0),
                        "regime_drift": float(components.get("regime_drift", 0.0) or 0.0),
                        "transition_pressure": float(components.get("transition_pressure", 0.0) or 0.0),
                        "coherence_margin": float(component_confidence.get("relational_drift", 0.0) or 0.0),
                    }
                )
                result["structural_uncertainty"] = {
                    "posterior": {
                        k: round(float(v), 4) for k, v in _struct_est.posterior_means.items()
                    },
                    "ci_90": {
                        k: [round(float(ci[0]), 4), round(float(ci[1]), 4)]
                        for k, ci in _struct_est.credible_intervals_90.items()
                    },
                    "sample_count": int(_struct_est.sample_count),
                }
            except Exception:
                pass

            # --- Math engine: Monte Carlo confidence bounds ---
            # Bootstrap 90 % / 50 % CI on composite score from recent component history.
            if len(self._component_history) >= 10:
                try:
                    _mc = self._mc_sampler.bootstrap(
                        list(self._component_history),
                        n_samples=300,
                        seed=42,
                    )
                    result["score_confidence"] = {
                        "ci_90": [round(_mc.p5, 4), round(_mc.p95, 4)],
                        "ci_50": [round(_mc.p25, 4), round(_mc.p75, 4)],
                        "mean": round(_mc.mean, 4),
                        "std": round(_mc.std, 4),
                        "probability_watch": round(_mc.probability_watch, 4),
                        "probability_alert": round(_mc.probability_alert, 4),
                        "probability_critical": round(_mc.probability_critical, 4),
                    }
                except Exception:
                    pass

            # Calibrate decision thresholds from early nominal composite history.
            if self._composite_watch_alert_thresholds is None:
                self._baseline_composite_score_samples.append(composite)
                if len(self._baseline_composite_score_samples) >= MIN_BASELINE_SAMPLES_FOR_CALIBRATION:
                    watch_thr = float(np.percentile(list(self._baseline_composite_score_samples), 82.0))
                    alert_thr = float(np.percentile(list(self._baseline_composite_score_samples), 93.5))
                    if alert_thr < watch_thr:
                        watch_thr, alert_thr = alert_thr, watch_thr
                    self._composite_watch_alert_thresholds = (watch_thr, alert_thr)

            if self.fast_mode:
                forecast = {
                    "method": "disabled_fast_mode",
                    "trend": 0.0,
                    "time_to_instability": None,
                    "ar1_next": None,
                    "ar1_time_to_instability": None,
                    "persistence": {},
                }
            else:
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
                debug_prints=self._frame_debug,
            )
            result.update(decision)
            if self.transition_aware_enabled:
                transition_pressure = float(result.get("transition_pressure", components.get("transition_pressure", 0.0)))
                transition_state = str(result.get("transition_state", "NONE"))
                state_rank = {"STABLE": 0, "WATCH": 1, "ALERT": 2}
                current_state = str(result.get("state", "STABLE"))
                target_state = current_state
                if transition_state == "WARMUP":
                    target_state = current_state
                elif transition_state == "SUSTAINED_TRANSITION" and transition_pressure >= TRANSITION_SUSTAINED_THRESHOLD:
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
            result["temporal_consistency_score"] = round(float(temporal_quality.get("temporal_consistency_score", 0.0)), 4)
            result["ordering_stability_score"] = round(float(temporal_quality.get("ordering_stability_score", 0.0)), 4)
            result["timestamp_gap_irregularity"] = round(float(temporal_quality.get("timestamp_gap_irregularity", 0.0)), 4)
            result["alignment_confidence"] = round(float(temporal_quality.get("alignment_confidence", 0.0)), 4)
            result["effective_sampling_density"] = round(float(temporal_quality.get("effective_sampling_density", 0.0)), 4)
            result["localization_score"] = 0.0
            self._temporal_consistency_history.append(float(temporal_quality.get("temporal_consistency_score", 0.0)))

            tetrahedral_payload = self._safe_default_tetrahedral_payload()
            if isinstance(analytics, dict):
                reversibility_block = analytics.get("counterfactual_guidance", {}).get("reversibility", {}) if isinstance(analytics.get("counterfactual_guidance"), dict) else {}
                reversibility_scores = reversibility_block.get("scores", {}) if isinstance(reversibility_block, dict) else {}
                try:
                    tetrahedral_payload = compute_tetrahedral_state(
                        structural_drift_score=float(result.get("structural_drift_score", 0.0) or 0.0),
                        relational_instability_score=float(result.get("relational_instability_score", 0.0) or 0.0),
                        transition_pressure=float(result.get("transition_pressure", 0.0) or 0.0),
                        temporal_consistency_score=float(result.get("temporal_consistency_score", 0.0) or 0.0),
                        history_positions=list(self._tetrahedral_position_history),
                        regime_drift=float(result.get("regime_drift", 0.0) or 0.0),
                        reversibility=(
                            float(reversibility_scores.get("locked_in_index", 0.0))
                            if isinstance(reversibility_scores, dict)
                            else None
                        ),
                        geometry_curvature=(
                            float((analytics.get("geometry") or {}).get("curvature", 0.0))
                            if isinstance(analytics.get("geometry"), dict)
                            else None
                        ),
                    )
                except Exception:
                    tetrahedral_payload = self._safe_default_tetrahedral_payload()
            position = tetrahedral_payload.get("position")
            if isinstance(position, list) and len(position) == 3:
                self._tetrahedral_position_history.append([float(v) for v in position])
            result["tetrahedral_state"] = tetrahedral_payload
            analytics["tetrahedral_state"] = tetrahedral_payload

            self._state_history.append(decision.get("interpreted_state", "NOMINAL_STRUCTURE"))

            # Rolling baseline: update only when nominal, composite low, and not locked.
            _ts_baseline = str(result.get("transition_state", "NONE"))
            transition_blocks_baseline = (
                self.transition_aware_enabled
                and _ts_baseline != "WARMUP"
                and (
                    _ts_baseline != "NONE"
                    or float(result.get("transition_pressure", 0.0)) >= TRANSITION_EMERGING_THRESHOLD
                )
            )
            if (
                valid_signal_count >= 2
                and not self.baseline_locked
                and decision.get("interpreted_state") in {"NOMINAL_STRUCTURE", "COUPLING_INSTABILITY_OBSERVED"}
                and not transition_blocks_baseline
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
            analytics["temporal_quality"] = temporal_quality
            analytics["temporal_features"] = temporal_features
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

            recommended_action = None
            recs = result.get("response_recommendations")
            if isinstance(recs, list) and recs:
                first_rec = recs[0]
                if isinstance(first_rec, dict):
                    recommended_action = str(first_rec.get("action", "") or "").strip() or None

            result["explanation_text"] = build_explanation_text(
                current_decision=str(result.get("interpreted_state", "NOMINAL_STRUCTURE")),
                attribution=result.get("attribution") if isinstance(result.get("attribution"), dict) else None,
                risk=result.get("risk_level"),
                confidence=result.get("confidence"),
                recommended_action=recommended_action,
            )
            result["component_confidence"] = component_confidence
            result["geometry"] = analytics.get("geometry", {}) if isinstance(analytics, dict) else {}
            result["state_space_statistics"] = analytics.get("state_space_statistics", {}) if isinstance(analytics, dict) else {}
            result["state_graph"] = analytics.get("state_graph", {}) if isinstance(analytics, dict) else {}
            result["geometry_explanations"] = analytics.get("geometry_explanations", {}) if isinstance(analytics, dict) else {}

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
            result["experimental_analytics"]["tetrahedral_state"] = tetrahedral_payload
            debug_raw_features = os.environ.get("NERAIUM_DEBUG_RAW_FEATURES", "0").strip().lower() not in {
                "0",
                "false",
                "no",
                "off",
                "",
            }
            if self._frame_debug and debug_raw_features and self._raw_debug_frames_logged < 3:
                sigf = analytics.get("signal_features", {}) if isinstance(analytics, dict) else {}
                delta = sigf.get("delta", {}) if isinstance(sigf, dict) else {}
                change_summary = sigf.get("change_summary", {}) if isinstance(sigf, dict) else {}
                key_time = {
                    "mean_abs_first_diff": round(float(self._signal_feature_value(delta, "mean_abs_first_diff")), 6),
                    "std_first_diff": round(float(self._signal_feature_value(delta, "std_first_diff")), 6),
                    "mean_abs_second_diff": round(float(self._signal_feature_value(delta, "mean_abs_second_diff")), 6),
                    "rolling_local_volatility": round(float(self._signal_feature_value(delta, "rolling_local_volatility")), 6),
                    "roughness": round(float(self._signal_feature_value(delta, "roughness")), 6),
                }
                key_freq = {
                    "spectral_centroid": round(float(self._signal_feature_value(delta, "spectral_centroid")), 6),
                    "spectral_entropy": round(float(self._signal_feature_value(delta, "spectral_entropy")), 6),
                    "low_high_frequency_energy_ratio": round(float(self._signal_feature_value(delta, "low_high_frequency_energy_ratio")), 6),
                    "dominant_frequency_ratio": round(float(self._signal_feature_value(delta, "dominant_frequency_ratio")), 6),
                }
                sig_deg = result.get("signal_degradation", {}) if isinstance(result.get("signal_degradation"), dict) else {}
                print("DEBUG RAW FEATURES:", {"time_domain": key_time, "frequency_domain": key_freq, "change_summary": change_summary})
                print(
                    "DEBUG RAW SIGNAL SCORES:",
                    {
                        "signal_instability_score": sig_deg.get("signal_instability_score"),
                        "shape_change_score": sig_deg.get("shape_change_score"),
                        "spectral_shift_score": sig_deg.get("spectral_shift_score"),
                        "volatility_erosion_score": sig_deg.get("volatility_erosion_score"),
                        "coherence_loss_score": sig_deg.get("coherence_loss_score"),
                        "state": sig_deg.get("signal_degradation_state"),
                    },
                )
                print(
                    "DEBUG RAW->GEOMETRY ACTIVATION:",
                    {
                        "geometry_path_length_available": (result.get("geometry", {}) or {}).get("path_length") is not None,
                        "state_space_local_volume_available": (result.get("state_space_statistics", {}) or {}).get("local_volume") is not None,
                        "state_graph_branching_factor_available": (result.get("state_graph", {}) or {}).get("branching_factor") is not None,
                        "horizon": ((result.get("experimental_analytics", {}) or {}).get("horizon_analysis", {}) or {}).get("risk_horizon"),
                    },
                )
                self._raw_debug_frames_logged += 1
            if self._frame_debug and os.environ.get("NERAIUM_DEBUG_EXP_ANALYTICS", "0").strip().lower() not in {
                "0",
                "false",
                "no",
                "off",
                "",
            }:
                print("DEBUG EXP ANALYTICS:", result.get("experimental_analytics"))
                self._debug_print_experimental_analytics_once(result)
            if (
                self._frame_debug
                and os.environ.get("NERAIUM_DEBUG_GEOMETRY", "0").strip().lower()
                not in {"0", "false", "no", "off", ""}
                and can_process_full_frame
                and valid_signal_count >= 2
                and self._geometry_debug_frames_logged < 3
            ):
                state_space = result.get("state_space_statistics") if isinstance(result.get("state_space_statistics"), dict) else {}
                state_graph = result.get("state_graph") if isinstance(result.get("state_graph"), dict) else {}
                branching_factor = float(state_graph.get("branching_factor", 0.0) or 0.0)
                branching_intensity = float(state_graph.get("branching_intensity", 0.0) or 0.0)
                transition_entropy = float(state_graph.get("transition_entropy", 0.0) or 0.0)
                graph_divergence_score = float(state_graph.get("graph_divergence_score", 0.0) or 0.0)
                anisotropy = float(state_space.get("anisotropy", 1.0) or 1.0)
                directional_consistency = float((result.get("geometry") or {}).get("directional_consistency", 1.0) or 1.0)
                directional_inconsistency = float(state_graph.get("directional_inconsistency", 1.0 - directional_consistency) or 0.0)
                raw_divergence_score = float(state_graph.get("raw_divergence_score", 0.0) or 0.0)
                if self._last_geometry_debug_branching_factor is None:
                    branching_delta = 0.0
                else:
                    branching_delta = branching_factor - self._last_geometry_debug_branching_factor

                if branching_delta > 1e-4:
                    branching_trend = "increasing"
                elif branching_delta < -1e-4:
                    branching_trend = "decreasing"
                else:
                    branching_trend = "flat"

                print("DEBUG GEOMETRY:", result.get("geometry"))
                print(
                    "DEBUG STATE SPACE:",
                    {
                        "covariance_trace": state_space.get("covariance_trace"),
                        "covariance_logdet": state_space.get("covariance_logdet"),
                        "local_volume": state_space.get("local_volume"),
                    },
                )
                print(
                    "DEBUG BRANCH INPUTS:",
                    {
                        "transition_entropy": round(transition_entropy, 6),
                        "graph_divergence_score": round(graph_divergence_score, 6),
                        "anisotropy": round(anisotropy, 6),
                        "directional_inconsistency": round(directional_inconsistency, 6),
                        "raw_combined_divergence_score": round(raw_divergence_score, 6),
                        "final_normalized_branching_factor": round(branching_factor, 6),
                        "branching_intensity": round(branching_intensity, 6),
                    },
                )
                print(
                    "DEBUG BRANCHING:",
                    {
                        "branching_factor": round(branching_factor, 6),
                        "branching_intensity": round(branching_intensity, 6),
                        "transition_entropy": round(transition_entropy, 6),
                        "graph_divergence_score": round(graph_divergence_score, 6),
                        "branching_factor_delta": round(branching_delta, 6),
                        "branching_trend": branching_trend,
                        "branching_reason": (
                            "entropy/divergence pressure rising"
                            if branching_trend == "increasing"
                            and (transition_entropy > 0.35 or graph_divergence_score > 0.35)
                            else (
                                "trajectory consolidating; uncertainty pressure is not rising"
                                if branching_trend in {"decreasing", "flat"}
                                and (transition_entropy < 0.3 and graph_divergence_score < 0.3)
                                else "mixed indicators; inspect entropy/divergence vs commitment"
                            )
                        ),
                    },
                )
                transition_used = bool(
                    (analytics.get("transition") or {}).get("geometry_transition_term", 0.0)
                    if isinstance(analytics, dict)
                    else False
                )
                branching_diag = ((analytics.get("branching_analysis") or {}).get("diagnostics", {})) if isinstance(analytics, dict) else {}
                lockin_rationale = ((analytics.get("constraint_analysis") or {}).get("rationale", {})) if isinstance(analytics, dict) else {}
                print("DEBUG GEOMETRY USED transition_pressure:", transition_used)
                print(
                    "DEBUG GEOMETRY USED branching:",
                    {
                        "angular_divergence": branching_diag.get("angular_divergence"),
                        "state_graph_branching_factor": branching_diag.get("state_graph_branching_factor"),
                        "state_graph_transition_entropy": branching_diag.get("state_graph_transition_entropy"),
                    },
                )
                print(
                    "DEBUG GEOMETRY USED lock_in:",
                    {
                        "path_commitment_score": lockin_rationale.get("path_commitment_score"),
                        "state_contraction_score": lockin_rationale.get("state_contraction_score"),
                    },
                )
                horizon_rationale = ((analytics.get("horizon_analysis") or {}).get("rationale", {})) if isinstance(analytics, dict) else {}
                branching_analysis_payload = (analytics.get("branching_analysis") or {}) if isinstance(analytics, dict) else {}
                print(
                    "DEBUG HORIZON INPUTS:",
                    {
                        "transition_persistence": horizon_rationale.get("transition_persistence"),
                        "lock_in_score": horizon_rationale.get("lock_in_score"),
                        "recovery_margin": horizon_rationale.get("recovery_margin"),
                        "contraction_score": horizon_rationale.get("state_contraction_score"),
                        "path_commitment": lockin_rationale.get("path_commitment_score"),
                        "final_horizon_class": (analytics.get("horizon_analysis") or {}).get("risk_horizon"),
                    },
                )
                print(
                    "DEBUG BRANCHING REFINEMENT:",
                    {
                        "transition_entropy": branching_diag.get("state_graph_transition_entropy"),
                        "graph_divergence_score": branching_diag.get("state_graph_graph_divergence_score"),
                        "anisotropy": branching_diag.get("anisotropy_normalized"),
                        "counterfactual_spread": branching_diag.get("counterfactual_spread"),
                        "final_branching_factor": state_graph.get("branching_factor"),
                        "final_branch_count_estimate": branching_analysis_payload.get("branch_count_estimate"),
                    },
                )
                self._last_geometry_debug_branching_factor = branching_factor
                self._geometry_debug_frames_logged += 1

            debug_verbose = os.environ.get("NERAIUM_DEBUG_SII_VERBOSE", "0").strip().lower() not in {
                "0",
                "false",
                "no",
                "off",
                "",
            }
            if self._frame_debug and debug_verbose:
                drift_thr = self._drift_watch_alert_thresholds
                comp_thr = self._composite_watch_alert_thresholds
                causal_prop = analytics.get("causal_propagation") if isinstance(analytics.get("causal_propagation"), dict) else {}
                causal_prop_top_sources = causal_prop.get("top_sources")
                causal_prop_spread = causal_prop.get("spread_scores")
                causal_prop_top_pairs = causal_prop.get("top_pairs")
                graph = analytics.get("graph")
                causal_graph = analytics.get("causal_graph")

                print(
                    "[NERAIUM_DEBUG_SII_VERBOSE]"
                    f" state={result.get('state')} drift_score={float(result.get('latest_drift', 0.0)):.6g}"
                    f" drift_thr={drift_thr}"
                    f" drift_persist=(watch={self._watch_counter}, alert={self._alert_counter}, latched={self._alert_latched})"
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

        if self.fast_mode:
            transition_pressure_value = float(result.get("transition_pressure", 0.0) or 0.0)
            if len(self._transition_pressure_history) == history_transition_len_before:
                self._transition_pressure_history.append(transition_pressure_value)
            if len(self._shock_activity_history) == history_shock_len_before:
                self._shock_activity_history.append(0.0)
            if len(self._structural_drift_history) == history_drift_len_before:
                self._structural_drift_history.append(float(result.get("structural_drift_score", 0.0) or 0.0))

            geo_r = result.get("geometry") if isinstance(result.get("geometry"), dict) else {}
            ss_r = result.get("state_space_statistics") if isinstance(result.get("state_space_statistics"), dict) else {}
            sg_r = result.get("state_graph") if isinstance(result.get("state_graph"), dict) else {}
            rd_final = compute_engine_readiness(
                frame_count=len(self.frames),
                baseline_window=self.baseline_window,
                recent_window=self.recent_window,
                transition_pressure_history_len=len(self._transition_pressure_history),
                warmup_margin_frames=self.transition_stabilization_margin_frames,
                min_transition_history=self.transition_classification_min_history,
                geometry_available=geo_r.get("available") is not False if geo_r else None,
                geometry_reason=str(geo_r.get("reason", "")) if geo_r else None,
                state_space_available=ss_r.get("available") is not False if ss_r else None,
                state_space_reason=str(ss_r.get("reason", "")) if ss_r else None,
                state_graph_available=sg_r.get("available") is not False if sg_r else None,
                state_graph_reason=str(sg_r.get("reason", "")) if sg_r else None,
            )
            result["readiness"] = rd_final.as_dict()
            result["engine_ready"] = rd_final.ready
            result["confidence_score"] = round(float(result.get("confidence_score", 0.0) or 0.0), 4)
            self._apply_fast_mode_payload_downgrades(result)
            self.latest_result = dict(result)
            return result

        drift_noise = classify_drift_noise(list(self._drift_score_history))
        result["drift_noise"] = drift_noise
        matrix_for_scale = recent_window if isinstance(recent_window, np.ndarray) else None
        if matrix_for_scale is None and len(self.frames) >= 3:
            matrix_for_scale = self._history_ring.chronological_tail_matrix(min(24, len(self.frames)))
        multi_scale = compute_multi_scale_states(np.nan_to_num(matrix_for_scale, nan=0.0)) if isinstance(matrix_for_scale, np.ndarray) else {
            "short_term_state": "insufficient_data",
            "mid_term_state": "insufficient_data",
            "long_term_state": "insufficient_data",
            "scale_conflict": 0.0,
            "scale_alignment": 1.0,
            "scale_conflict_reason": "insufficient_data",
        }
        result["multi_scale"] = multi_scale

        exp = result.get("experimental_analytics", {}) if isinstance(result.get("experimental_analytics"), dict) else {}
        trajectory = exp.get("trajectory_analysis", {}) if isinstance(exp, dict) else {}
        branching = exp.get("branching_analysis", {}) if isinstance(exp, dict) else {}
        constraint = exp.get("constraint_analysis", {}) if isinstance(exp, dict) else {}
        horizon = exp.get("horizon_analysis", {}) if isinstance(exp, dict) else {}
        counterfactual = exp.get("counterfactual_simulation", {}) if isinstance(exp, dict) else {}
        counterfactual_spread = float(counterfactual.get("scenario_spread", 0.0) or 0.0) if isinstance(counterfactual, dict) else 0.0
        stability = compute_stability_metrics(
            drift_history=list(self._drift_score_history),
            transition_history=list(self._transition_pressure_history),
            state_history=list(self._state_history),
            trajectory_label=str(trajectory.get("trajectory_label", "unknown")) if isinstance(trajectory, dict) else "unknown",
            branch_count=float(branching.get("branch_count_estimate", 1.0) or 1.0) if isinstance(branching, dict) else 1.0,
            lock_in_score=float(constraint.get("lock_in_score", 0.0) or 0.0) if isinstance(constraint, dict) else 0.0,
            horizon_steps=float(horizon.get("horizon_steps", 0.0) or 0.0) if isinstance(horizon, dict) else 0.0,
            counterfactual_spread=counterfactual_spread,
        )
        result["robustness"] = {"stability": stability}
        result["stability"] = stability

        if isinstance(recent_window, np.ndarray) and isinstance(baseline_window, np.ndarray):
            result["sensitivity"] = compute_sensitivity(
                baseline_window=baseline_window,
                recent_window=recent_window,
                feature_names=list(self.sensor_order),
                trajectory_score=float(stability.get("trajectory_stability_score", 0.0)),
                lock_in_score=float((constraint.get("lock_in_score", 0.0) if isinstance(constraint, dict) else 0.0) or 0.0),
                branching_score=float((branching.get("branch_entropy", 0.0) if isinstance(branching, dict) else 0.0) or 0.0),
            )
        else:
            result["sensitivity"] = {
                "top_drivers": [],
                "feature_contributions": {},
                "trajectory_drivers": {},
                "lock_in_drivers": {},
                "branching_drivers": {},
            }
        existing_attribution = result.get("attribution", {}) if isinstance(result.get("attribution"), dict) else {}
        result["attribution"] = {
            "top_drivers": existing_attribution.get("top_drivers", result["sensitivity"].get("top_drivers", [])),
            "driver_scores": existing_attribution.get("driver_scores", {}),
            "trajectory_drivers": result["sensitivity"].get("trajectory_drivers", {}),
            "branching_drivers": result["sensitivity"].get("branching_drivers", {}),
            "lock_in_drivers": result["sensitivity"].get("lock_in_drivers", {}),
            "horizon_drivers": result["sensitivity"].get("horizon_drivers", {}),
            "counterfactual_drivers": result["sensitivity"].get("counterfactual_drivers", {}),
            "group_contributions": result["sensitivity"].get("group_contributions", {}),
        }
        structural_signals = {
            "structural_drift_score": float(result.get("structural_drift_score", 0.0) or 0.0),
            "relational_instability_score": float(result.get("relational_instability_score", 0.0) or 0.0),
            "regime_distance": (
                float(result.get("regime_distance", 0.0))
                if result.get("regime_distance") is not None
                else 0.0
            ),
            "regime_drift": float(result.get("regime_drift", 0.0) or 0.0),
            "transition_pressure": float(result.get("transition_pressure", 0.0) or 0.0),
            "localization_score": float(result.get("localization_score", 0.0) or 0.0),
            "temporal_distortion_score": float(result.get("temporal_distortion_score", 0.0) or 0.0),
        }
        regime_memory = {
            **(result.get("regime_memory_state", {}) if isinstance(result.get("regime_memory_state"), dict) else {}),
            "regime_distance": structural_signals["regime_distance"],
            "regime_name": result.get("regime_name"),
        }
        result["regime_memory"] = regime_memory
        risk_assessment = {
            "risk_level": result.get("risk_level", "LOW"),
            "trend": result.get("trend", "UNKNOWN"),
            "latest_instability": float(result.get("latest_instability", 0.0) or 0.0),
            "confidence_score": float(result.get("confidence_score", 0.0) or 0.0),
        }
        result["risk_assessment"] = risk_assessment
        result["operator_guidance"] = {
            "operator_message": result.get("operator_message"),
            "response_recommendations": result.get("response_recommendations", []),
        }
        hypothesis_candidates = generate_hypotheses(
            attribution=result.get("attribution", {}),
            structural_signals=structural_signals,
            regime_memory=regime_memory,
            risk_assessment=risk_assessment,
        )
        scored_hypotheses = score_hypotheses(
            hypotheses=hypothesis_candidates,
            regime_memory=regime_memory,
            risk_assessment=risk_assessment,
            structural_signals=structural_signals,
        )
        top_hypothesis = scored_hypotheses.get("top_hypothesis")
        counterfactual = run_counterfactual_checks(
            top_hypothesis=top_hypothesis if isinstance(top_hypothesis, dict) else None,
            attribution=result.get("attribution", {}),
            structural_signals=structural_signals,
        )
        validation_plan = generate_validation_plan(
            ranked_hypotheses=(
                scored_hypotheses.get("ranked_hypotheses")
                if isinstance(scored_hypotheses.get("ranked_hypotheses"), list)
                else []
            ),
            attribution=result.get("attribution", {}),
            risk_assessment=risk_assessment,
            counterfactual=counterfactual,
        )
        ranked_actions = rank_actions(validation_plan=validation_plan, risk_assessment=risk_assessment)
        causal_available = can_process_full_frame and bool(scored_hypotheses.get("ranked_hypotheses"))
        causal_reason = "ok" if causal_available else ("warmup" if not can_process_full_frame else "insufficient_evidence")
        result["causal_analysis"] = {
            "hypotheses": scored_hypotheses.get("ranked_hypotheses", []),
            "top_hypothesis": top_hypothesis,
            "counterfactual": counterfactual,
            "validation_plan": validation_plan,
            "recommended_sequence": ranked_actions.get("recommended_sequence", []),
            "best_next_action": ranked_actions.get("best_next_action"),
            "status": {"available": bool(causal_available), "reason": causal_reason},
        }

        result["path_prototypes"] = derive_path_prototype_summary(
            trajectory=trajectory if isinstance(trajectory, dict) else {},
            branching=branching if isinstance(branching, dict) else {},
            constraint=constraint if isinstance(constraint, dict) else {},
            drift_noise=drift_noise,
            multi_scale=multi_scale if isinstance(multi_scale, dict) else {},
        )

        self._current_episode, self._episode_history = update_episode_memory(
            current_step=len(self.frames),
            previous_episode=self._current_episode,
            episode_history=self._episode_history,
            transition_pressure=float(result.get("transition_pressure", 0.0) or 0.0),
            drift_type=str(drift_noise.get("interpreted_change_type", "noise")),
            lock_in_score=float(constraint.get("lock_in_score", 0.0) or 0.0) if isinstance(constraint, dict) else 0.0,
            horizon_category=str(horizon.get("horizon_category", "unknown")) if isinstance(horizon, dict) else "unknown",
            branch_count=float(branching.get("branch_count_estimate", 1.0) or 1.0) if isinstance(branching, dict) else 1.0,
            multi_scale=multi_scale if isinstance(multi_scale, dict) else {},
        )
        result["episodes"] = dict(self._current_episode)

        result["explanations"] = generate_structural_explanations(
            trajectory_label=str(trajectory.get("trajectory_label", "insufficient_data")) if isinstance(trajectory, dict) else "insufficient_data",
            branching_count=float(branching.get("branch_count_estimate", 1.0) or 1.0) if isinstance(branching, dict) else 1.0,
            lock_in_score=float(constraint.get("lock_in_score", 0.0) or 0.0) if isinstance(constraint, dict) else 0.0,
            horizon_label=str(horizon.get("horizon_category", "unknown")) if isinstance(horizon, dict) else "unknown",
            counterfactual_spread=counterfactual_spread,
            drift_type=str(drift_noise.get("interpreted_change_type", "noise")),
            stability=stability,
            path_prototype=str((result.get("path_prototypes", {}) or {}).get("dominant_prototype", "unknown")),
        )
        sig = result.get("signal_degradation", {}) if isinstance(result.get("signal_degradation"), dict) else {}
        if isinstance(result.get("explanations"), dict):
            result["explanations"]["signal_degradation"] = (
                "signal_instability={:.2f}; shape_change={:.2f}; spectral_shift={:.2f}; volatility_erosion={:.2f}; coherence_loss={:.2f}".format(
                    float(sig.get("signal_instability_score", 0.0)),
                    float(sig.get("shape_change_score", 0.0)),
                    float(sig.get("spectral_shift_score", 0.0)),
                    float(sig.get("volatility_erosion_score", 0.0)),
                    float(sig.get("coherence_loss_score", 0.0)),
                )
            )

        result["evidence"] = build_evidence_block(
            frame_count=len(self.frames),
            temporal_quality=temporal_quality if isinstance(temporal_quality, dict) else {},
            stability=stability,
            attribution=result["attribution"],
            explanation_count=len(result.get("explanations", {})),
        )
        result["fleet_comparison"] = {
            "peer_relative_fragility": {str(result.get("asset_id", "entity")): 0.0},
            "rankings": {},
            "outlier_entities": [],
            "comparison_summary": "single_entity_context",
        }

        transition_pressure_value = float(result.get("transition_pressure", 0.0) or 0.0)
        if len(self._transition_pressure_history) == history_transition_len_before:
            self._transition_pressure_history.append(transition_pressure_value)

        geo_r = result.get("geometry") if isinstance(result.get("geometry"), dict) else {}
        ss_r = result.get("state_space_statistics") if isinstance(result.get("state_space_statistics"), dict) else {}
        sg_r = result.get("state_graph") if isinstance(result.get("state_graph"), dict) else {}
        rd_final = compute_engine_readiness(
            frame_count=len(self.frames),
            baseline_window=self.baseline_window,
            recent_window=self.recent_window,
            transition_pressure_history_len=len(self._transition_pressure_history),
            warmup_margin_frames=self.transition_stabilization_margin_frames,
            min_transition_history=self.transition_classification_min_history,
            geometry_available=geo_r.get("available") is not False if geo_r else None,
            geometry_reason=str(geo_r.get("reason", "")) if geo_r else None,
            state_space_available=ss_r.get("available") is not False if ss_r else None,
            state_space_reason=str(ss_r.get("reason", "")) if ss_r else None,
            state_graph_available=sg_r.get("available") is not False if sg_r else None,
            state_graph_reason=str(sg_r.get("reason", "")) if sg_r else None,
        )
        result["readiness"] = rd_final.as_dict()
        result["engine_ready"] = rd_final.ready
        result["engine_stabilization_progress"] = rd_final.stabilization_progress
        result["engine_warmup_progress"] = rd_final.stabilization_progress
        result["engine_min_history_required"] = rd_final.min_history_required
        result["transition_outputs_actionable"] = rd_final.transition_classification_ready
        result["engine_transition_detectable"] = rd_final.transition_classification_ready
        result["neraium"] = {
            "readiness": rd_final.as_dict(),
            "transition_outputs_actionable": rd_final.transition_classification_ready,
        }

        # Final policy enforcement and architecture-specific payload packaging.
        self._enforce_policy_contract(result)
        self._attach_architecture_outputs(result, frame)
        if len(self._shock_activity_history) == history_shock_len_before:
            self._shock_activity_history.append(0.0)
        if len(self._structural_drift_history) == history_drift_len_before:
            self._structural_drift_history.append(float(result.get("structural_drift_score", 0.0) or 0.0))

        self.latest_result = result

        return result
