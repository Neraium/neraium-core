"""
fd00x — Structural early-warning evaluation framework for CMAPSS turbofan data.

Pure structural detection without experimental layers.

Uses:
- Raw structural drift: covariance, correlation, Mahalanobis distance
- Trajectory acceleration: 2nd derivative (degradation curvature)
- Relational instability: correlation breakdown (systemic degradation)
- Multi-signal confirmation: combine structural + amplitude signals

No RUL in detection logic. No future data leakage.

Module layout
-------------
config              Tunable parameters, named presets
detector            Core structural drift detector (no QIT layer)
structural_signals  Reusable structural change detection
settings            SII-ML preset settings
sii                 Core SII with five-layer atomic scoring
sii_ml              ML-enhanced SII
evaluation          Dataset loading, metrics, baselines, tuning
plotting            Diagnostic plots
experiment          CLI entry point and workflow runner

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
from .structural_signals import StructuralSignalDetector, create_structural_detector
from .settings import get_optimal_config
from .sii import LayerWeights, SII
from .sii_ml import SIIML, create_siiml

__all__ = [
    "DetectorConfig",
    "PRESETS",
    "DEFAULT_PRESET",
    "ALL_DATASETS",
    "StructuralDriftDetector",
    "find_warning_index",
    "StructuralSignalDetector",
    "create_structural_detector",
    "LayerWeights",
    "SII",
    "SIIML",
    "create_siiml",
    "get_optimal_config",
]
