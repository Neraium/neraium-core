"""Configuration for platform-wide structural transition detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


FallbackMode = Literal["max_slope", "none", "regime_shift"]


@dataclass
class TransitionDetectionConfig:
    """
    Reusable detection parameters for any time-ordered structural replay dataframe.

    Warmup handling (applied in order):
    - ``warmup_fraction``: drop the first ``floor(n * warmup_fraction)`` rows after sort.
    - ``warmup_cycles``: if the index column is numeric, keep rows with index >= this value;
      ignored if ``warmup_fraction`` is set.
    - ``min_history``: require at least this many rows after warmup or return no detection.

    Signals:
    - ``primary_signal``: explicit column to use; if None, first match from ``signal_priority``.
    - ``signal_mode``: ``single`` or ``composite`` (weighted normalized blend).
    - ``composite_weights``: column -> weight for composite mode (missing columns skipped).
    """

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
    composite_weights: dict[str, float] | None = None

    min_history: int = 2
    warmup_cycles: float = 30.0
    warmup_fraction: float | None = None
    smoothing_window: int = 5
    threshold: float = 0.5
    sustained_points: int = 2
    fallback_mode: FallbackMode = "max_slope"
    require_monotonic_confirmation: bool = False
    use_relative_thresholds: bool = True
    ensemble_agreement: bool = False
    relative_threshold_low: float = 0.35
    relative_threshold_high: float = 0.65
    verbose: bool = False
    # Grouped evaluation (summarize_* helpers); not used by detect_transition.
    entity_group_column: str = "unit"
