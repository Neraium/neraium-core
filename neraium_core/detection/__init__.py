"""Platform-wide structural transition detection and evaluation."""

from neraium_core.detection.transition_config import TransitionDetectionConfig
from neraium_core.detection.transition_detector import (
    detect_transition,
    detect_transition_from_signal,
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
    "TransitionDetectionConfig",
    "aggregate_detection_summary",
    "detect_transition",
    "detect_transition_from_signal",
    "export_detection_tables",
    "plot_normalized_detection_histogram",
    "plot_raw_position_histogram",
    "plot_signal_vs_outcome",
    "plot_timeseries_with_transition",
    "save_figure",
    "summarize_entity_transitions",
    "summarize_transition_detection",
]
