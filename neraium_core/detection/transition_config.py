"""Configuration for platform-wide structural transition detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


FallbackMode = Literal["max_slope", "none", "regime_shift"]


@dataclass
class TransitionDetectionConfig:
    """
    Reusable detection parameters for any time-ordered structural replay dataframe.

    Index / row trimming (offline replay, applied in order):
    - ``warmup_fraction``: drop the first ``floor(n * warmup_fraction)`` rows after sort.
    - ``warmup_index_floor``: if the timeline column is numeric, keep rows with index >= this value;
      ignored if ``warmup_fraction`` is set. (Legacy name in some scripts: “min cycle”.)
    - ``min_history``: require at least this many rows after trimming or return no detection.

    Signals:
    - ``primary_signal``: explicit column to use; if None, first match from ``signal_priority``.
    - ``signal_mode``: ``single`` or ``composite`` (weighted normalized blend).
    - ``composite_weights``: column -> weight for composite mode (missing columns skipped).
    """

    # Default matches common replay CSVs; set explicitly for your timeline (e.g. ``timestamp_step``).
    index_column: str = "cycle"
    primary_signal: str | None = None
    signal_priority: tuple[str, ...] = (
        "transition_pressure",
        "structural_drift_score",
        "latest_instability",
        "early_warning_pre_instability_score",
        "constraint_analysis_lock_in_score",
    )
    signal_mode: Literal["single", "composite"] = "single"
    use_composite_signal: bool = False
    composite_weights: dict[str, float] | None = None

    min_history: int = 2
    warmup_index_floor: float = 30.0
    warmup_fraction: float | None = None
    # Extra rows dropped after index/fraction trim (suppresses early activation band in replay).
    stabilization_row_margin: int = 0
    smoothing_window: int = 5
    threshold: float = 0.5
    sustained_points: int = 2
    # Minimum consecutive points above threshold (>= sustained_points when stricter).
    confirmation_points: int = 2
    # Ignore threshold crossings in the first N rows after warmup (suppress activation spike).
    signal_artifact_window: int = 0
    fallback_mode: FallbackMode = "max_slope"
    require_monotonic_confirmation: bool = False
    use_relative_thresholds: bool = True
    ensemble_agreement: bool = False
    relative_threshold_low: float = 0.35
    relative_threshold_high: float = 0.65
    verbose: bool = False
    # Grouped evaluation (summarize_* helpers); not used by detect_transition.
    entity_group_column: str = "entity_id"
