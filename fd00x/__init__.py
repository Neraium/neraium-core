"""
fd00x — Structural early-warning evaluation framework for CMAPSS turbofan data.

This package upgrades the FD00x experiment scripts into a reproducible
evaluation framework for structural drift / relationship-change detection.

Module layout
-------------
config              Tunable parameters, named presets (default_trusted / balanced / strict)
detector            Core structural drift detector with multi-signal confirmation
structural_signals  Reusable structural change detection (acceleration, correlation breakdown)
settings            SII-ML preset settings (conservative / balanced / aggressive)
sii                 Core SII with five-layer atomic scoring
sii_ml              ML-enhanced SII (attention + graph learning + neural booster)
evaluation          Dataset loading, metrics, baselines, tuning, scoring
plotting            Diagnostic plots (lazy matplotlib import)
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
from .qit_detector import QITConfig, QITDetector, create_qit_detector
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
    "QITConfig",
    "QITDetector",
    "create_qit_detector",
    "LayerWeights",
    "SII",
    "SIIML",
    "create_siiml",
    "get_optimal_config",
]
