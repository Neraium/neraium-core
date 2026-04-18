"""Neraium Intelligence Stack — Five-layer structural state transition detection.

The Intelligence Stack is a layered, mathematically interpretable architecture
for online detection of degradation and state transitions in multivariate sensor systems.

Five layers:
1. Structural Geometry: Mahalanobis, covariance drift, correlation drift
2. Relational Instability: Sensor correlation breakdown detection
3. Trajectory Dynamics: EMA, velocity, acceleration, change-point detection
4. Regime Transition: Online clustering, transition confidence, persistence
5. Evidence Fusion: Multi-signal confirmation gating, interpretable alerts

All layers are causal (no future data), interpretable, and online-safe.

Quick start:

    from neraium_core.intelligence_stack import StructuralDriftDetector, DetectorConfig

    config = DetectorConfig()
    detector = StructuralDriftDetector(config)

    # Fit reference from healthy data
    ref = detector.fit_reference(healthy_data)

    # Score unit
    scores = detector.score_unit(unit_data, ref)
    print(f"Warning at cycle: {scores.warning_index}")

See INTELLIGENCE_STACK.md for architecture details.
"""

from .config import ALL_DATASETS, DEFAULT_PRESET, PRESETS, DetectorConfig
from .detector import StructuralDriftDetector, find_warning_index
from .regime_transition import RegimeTransitionLayer, RegimeTransitionOutput
from .structural_signals import StructuralSignalDetector

__all__ = [
    "DetectorConfig",
    "PRESETS",
    "DEFAULT_PRESET",
    "ALL_DATASETS",
    "StructuralDriftDetector",
    "find_warning_index",
    "StructuralSignalDetector",
    "RegimeTransitionLayer",
    "RegimeTransitionOutput",
]
