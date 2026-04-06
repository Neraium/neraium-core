"""
fd00x — Structural early-warning evaluation framework for CMAPSS turbofan data.

This package upgrades the FD00x experiment scripts into a reproducible
evaluation framework for structural drift / relationship-change detection.

Module layout
-------------
config      Tunable parameters, named presets (default_trusted / balanced / strict)
detector    Core structural drift detector, corrected persistence warning logic
evaluation  Dataset loading, metrics, baselines, tuning, scoring
plotting    Diagnostic plots (lazy matplotlib import)
experiment  CLI entry point and workflow runner

Quick start
-----------
From the repository root::

    python -m fd00x.experiment --mode evaluate --dataset FD004
    python -m fd00x.experiment --mode all --dataset FD001 FD002 FD003 FD004
    python -m fd00x.experiment --preset default_trusted --dataset FD004
    python -m fd00x.experiment --mode tune --dataset FD004

Key public API::

    from fd00x.config import DetectorConfig, PRESETS
    from fd00x.detector import StructuralDriftDetector, find_warning_index
    from fd00x.evaluation import (
        load_cmapss_dataset,
        evaluate_detector,
        compute_aggregate_metrics,
        compare_baselines,
        run_tuning,
    )
"""

from .config import ALL_DATASETS, DEFAULT_PRESET, PRESETS, DetectorConfig
from .detector import StructuralDriftDetector, find_warning_index
from .qit_detector import QITConfig, QITDetector, create_qit_detector

__all__ = [
    "DetectorConfig",
    "PRESETS",
    "DEFAULT_PRESET",
    "ALL_DATASETS",
    "StructuralDriftDetector",
    "find_warning_index",
    "QITConfig",
    "QITDetector",
    "create_qit_detector",
]
