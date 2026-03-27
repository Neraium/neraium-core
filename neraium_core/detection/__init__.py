"""Platform-wide structural transition detection and evaluation."""

from neraium_core.detection.composite_signal import composite_structural_pressure
from neraium_core.detection.readiness import EngineReadinessState, compute_engine_readiness, readiness_dict_for_partial_stream
from neraium_core.detection.transition_config import TransitionDetectionConfig
from neraium_core.detection.transition_detector import (
    detect_transition,
    detect_transition_from_signal,
    detect_transition_from_signals,
    summarize_transition_detection,
)
from neraium_core.detection.transition_evaluation import (
    aggregate_detection_summary,
    export_detection_tables,
    summarize_entity_transitions,
)
from neraium_core.detection.transition_plots import (
    plot_normalized_detection_histogram,
    plot_raw_position_histogram,
    plot_signal_vs_outcome,
    plot_timeseries_with_transition,
    save_figure,
)

__all__ = [
    "EngineReadinessState",
    "TransitionDetectionConfig",
    "aggregate_detection_summary",
    "composite_structural_pressure",
    "compute_engine_readiness",
    "detect_transition",
    "detect_transition_from_signal",
    "detect_transition_from_signals",
    "export_detection_tables",
    "readiness_dict_for_partial_stream",
    "plot_normalized_detection_histogram",
    "plot_raw_position_histogram",
    "plot_signal_vs_outcome",
    "plot_timeseries_with_transition",
    "save_figure",
    "summarize_entity_transitions",
    "summarize_transition_detection",
]
