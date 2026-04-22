"""Regime Transition Modeling layer of the Intelligence Stack.

Online-safe regime assignment and transition detection using structural features.
No future leakage; computes fully causal per-cycle outputs.

Regime transitions indicate state changes — shifts from healthy operation through
degradation stages to failure. Transitions are detected by sudden changes in the
structural feature vector and tracked with persistence metrics.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class RegimeTransitionOutput:
    """Per-cycle regime transition evidence."""

    regime_id: int
    regime_label: str
    transition_score: float
    transition_detected: bool
    regime_persistence: int
    regime_persistence_cycles: int
    regime_instability_score: float = 0.0  # Frontier improvement: early instability signal
    transition_confidence: float = 0.0  # Frontier improvement: confidence in transition
    regime_wobble: bool = False  # Frontier improvement: unstable state assignment


class RegimeTransitionLayer:
    """Online regime assignment and transition detection.

    Uses structural features (Mahalanobis, covariance drift, correlation drift)
    to assign latent regime labels and detect transitions with confidence scores.

    Online-safe: no future data required; works on streaming input.
    """

    def __init__(
        self,
        n_regimes: int = 3,
        transition_threshold: float = 1.5,
        persistence_window: int = 5,
        feature_smoothing: float = 0.1,
        verbose: bool = False,
    ) -> None:
        """Initialize regime transition layer.

        Args:
            n_regimes: Number of latent regimes (typically 2-3: healthy, degrading, critical)
            transition_threshold: Z-score threshold for detecting regime transitions
            persistence_window: Minimum cycles in regime for high confidence
            feature_smoothing: EMA alpha for feature vector smoothing
            verbose: Debug output
        """
        self.n_regimes = n_regimes
        self.transition_threshold = float(transition_threshold)
        self.persistence_window = int(persistence_window)
        self.feature_smoothing = float(feature_smoothing)
        self.verbose = verbose

        # Regime labels for interpretability
        self.regime_labels = {
            0: "healthy",
            1: "early_degradation",
            2: "advanced_degradation",
            3: "critical",
        }

        # Online state
        self._regime_centroids: Optional[np.ndarray] = None
        self._regime_counts: Optional[np.ndarray] = None
        self._feature_history: list[np.ndarray] = []
        self._regime_sequence: list[int] = []
        self._persistence_counter: int = 0
        self._current_regime: int = 0
        self._feature_mean: Optional[np.ndarray] = None
        self._feature_std: Optional[np.ndarray] = None

        # Frontier improvements: regime instability tracking
        self._regime_assignment_history: list[int] = []  # Track regime assignments for wobble detection
        self._feature_distance_history: list[float] = []  # Track feature distance to assigned regime

    def reset(self) -> None:
        """Reset online state for new unit."""
        self._regime_centroids = None
        self._regime_counts = None
        self._feature_history = []
        self._regime_sequence = []
        self._persistence_counter = 0
        self._current_regime = 0
        self._feature_mean = None
        self._feature_std = None
        self._regime_assignment_history = []
        self._feature_distance_history = []

    def initialize_regimes(self, features: np.ndarray) -> None:
        """Initialize regime centroids from early structural features.

        Called after reference fitting to set initial regime locations.

        Args:
            features: [n_cycles, 3] array of [mahal, cov_drift, corr_drift]
        """
        features = np.asarray(features, dtype=float)
        if features.ndim != 2 or features.shape[1] != 3:
            raise ValueError("Features must be [n_cycles, 3] shape")

        # Use early healthy data to initialize first regime centroid
        healthy_window = min(20, features.shape[0] // 3)
        healthy_features = features[:healthy_window]

        # Initialize centroids: spread along the feature space
        self._feature_mean = healthy_features.mean(axis=0)
        self._feature_std = np.maximum(healthy_features.std(axis=0), 1e-6)

        # Initialize centroids: first is healthy mean, others spread
        self._regime_centroids = np.zeros((self.n_regimes, 3), dtype=float)
        self._regime_centroids[0] = self._feature_mean.copy()

        # Spread other centroids away from healthy (represents degradation direction)
        if self.n_regimes > 1:
            degradation_direction = self._feature_std * 2.0
            for i in range(1, self.n_regimes):
                self._regime_centroids[i] = self._feature_mean + degradation_direction * float(i)

        self._regime_counts = np.ones(self.n_regimes, dtype=float)

    def process_cycle(
        self,
        mahal_distance: float,
        cov_drift: float,
        corr_drift: float,
    ) -> RegimeTransitionOutput:
        """Process one cycle: update regime, detect transitions.

        Args:
            mahal_distance: Mahalanobis distance from reference mean
            cov_drift: Frobenius norm of covariance change
            corr_drift: Frobenius norm of correlation change

        Returns:
            RegimeTransitionOutput with regime ID, transition flag, persistence, and
            frontier improvements: regime instability, transition confidence, regime wobble
        """
        # Construct feature vector
        features = np.array([mahal_distance, cov_drift, corr_drift], dtype=float)

        # Lazy initialization if not done yet
        if self._regime_centroids is None:
            self.initialize_regimes(np.array([features]))

        # Normalize features for regime assignment
        if self._feature_std is None:
            self._feature_std = np.ones(3, dtype=float)

        normalized_features = (features - self._feature_mean) / (self._feature_std + 1e-6)

        # Assign regime: nearest centroid
        distances = np.linalg.norm(
            self._regime_centroids - normalized_features[np.newaxis, :],
            axis=1,
        )
        new_regime = int(np.argmin(distances))

        # Clip to valid range
        new_regime = np.clip(new_regime, 0, self.n_regimes - 1)

        # Frontier improvement 1: Regime instability score
        # Measure closeness of second-nearest regime (uncertainty in assignment)
        sorted_distances = np.sort(distances)
        distance_to_nearest = float(sorted_distances[0])
        distance_to_second = float(sorted_distances[1]) if len(sorted_distances) > 1 else float(sorted_distances[0])

        # Instability: low margin between nearest and second-nearest regimes
        regime_separation = distance_to_second - distance_to_nearest
        regime_instability = 1.0 - np.exp(-max(0, regime_separation))  # Lower separation = higher instability

        # Frontier improvement 2: Transition confidence
        # Confidence grows with persistence; drops if regime changes
        transition_detected = new_regime != self._current_regime

        if transition_detected:
            self._persistence_counter = 1
            transition_confidence = 0.0  # Just transitioned, low confidence
        else:
            self._persistence_counter += 1
            # Confidence grows: 0 at t=1, 0.7 at t=5, 0.95 at t=15
            transition_confidence = 1.0 - np.exp(-0.25 * self._persistence_counter)

        # Frontier improvement 3: Regime wobble detection
        # Track regime assignments in last few cycles; wobble = high variability
        self._regime_assignment_history.append(new_regime)
        self._feature_distance_history.append(distance_to_nearest)

        wobble_window = 5
        regime_wobble = False
        if len(self._regime_assignment_history) >= wobble_window:
            recent_regimes = self._regime_assignment_history[-wobble_window:]
            unique_regimes = len(set(recent_regimes))
            # Wobble: more than 2 regime changes in 5 cycles
            wobble_transitions = sum(1 for i in range(1, len(recent_regimes)) if recent_regimes[i] != recent_regimes[i - 1])
            regime_wobble = wobble_transitions >= 2

        # Detect transition
        transition_score = float(distances[new_regime])

        # Update regime centroid (online learning, slow)
        if self._regime_counts is not None:
            alpha = 1.0 / (1.0 + self._regime_counts[new_regime])
            self._regime_centroids[new_regime] = (
                (1.0 - alpha) * self._regime_centroids[new_regime]
                + alpha * normalized_features
            )
            self._regime_counts[new_regime] += 1.0

        # Record history
        self._feature_history.append(features.copy())
        self._regime_sequence.append(new_regime)
        self._current_regime = new_regime

        # Regime label for interpretability
        regime_label = self.regime_labels.get(new_regime, f"regime_{new_regime}")

        return RegimeTransitionOutput(
            regime_id=new_regime,
            regime_label=regime_label,
            transition_score=transition_score,
            transition_detected=transition_detected and self._persistence_counter == 1,
            regime_persistence=self._persistence_counter,
            regime_persistence_cycles=self._persistence_counter,
            regime_instability_score=float(regime_instability),
            transition_confidence=float(transition_confidence),
            regime_wobble=regime_wobble,
        )


def compute_structural_features(
    mahal: np.ndarray,
    cov_shift: np.ndarray,
    corr_shift: np.ndarray,
) -> np.ndarray:
    """Stack structural features into [n_cycles, 3] array for regime layer.

    Args:
        mahal: [n_cycles] Mahalanobis distances
        cov_shift: [n_cycles] covariance drift scores
        corr_shift: [n_cycles] correlation drift scores

    Returns:
        [n_cycles, 3] feature matrix
    """
    mahal = np.asarray(mahal, dtype=float)
    cov_shift = np.asarray(cov_shift, dtype=float)
    corr_shift = np.asarray(corr_shift, dtype=float)

    if not (len(mahal) == len(cov_shift) == len(corr_shift)):
        raise ValueError("All feature arrays must have same length")

    return np.column_stack([mahal, cov_shift, corr_shift])
