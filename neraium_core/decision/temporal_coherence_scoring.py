"""Temporal coherence scoring: distinguish persistent structural changes from noise.

Measures signal coherence over time windows to determine if observed changes
are genuine structural degradation vs transient fluctuations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TemporalCoherence:
    """Coherence metrics for a signal over time."""

    metric_name: str
    window_scores: list[float] = field(default_factory=list)  # Coherence per window
    overall_coherence: float = 0.0  # 0-1, higher = more persistent
    signal_strength: float = 0.0  # vs noise [0, 1]
    frames_coherent: int = 0  # Consecutive frames showing same pattern
    coherence_type: str = "noisy"  # "increasing" | "steady" | "decreasing" | "noisy"


@dataclass
class TemporalCoherenceProfile:
    """Full temporal coherence picture for a frame."""

    coherences: dict[str, TemporalCoherence] = field(default_factory=dict)
    overall_system_coherence: float = 0.0  # Aggregate across all metrics
    primary_signal_type: str = "noisy"  # Most common coherence type
    is_coherent_overall: bool = False  # Is degradation signal coherent enough to trust?


def _compute_window_coherence(values: list[float], window_size: int = 3) -> list[float]:
    """Compute coherence score for sliding windows.

    Coherence within a window = how consistent are values relative to variation.
    High coherence = values stay in narrow band.
    Low coherence = values jump around.

    Returns:
        List of coherence scores [0, 1] for each window
    """
    if len(values) < window_size:
        return []

    scores = []
    for i in range(len(values) - window_size + 1):
        window = values[i : i + window_size]
        if not window:
            continue

        mean = sum(window) / len(window)
        variance = sum((x - mean) ** 2 for x in window) / len(window)
        std_dev = variance ** 0.5

        # Coherence: inverse of normalized variance
        # If std_dev is small, coherence is high
        max_possible_std = max(window) - min(window) if max(window) > min(window) else 1.0
        normalized_std = std_dev / max_possible_std if max_possible_std > 0 else 0.0

        # Coherence = 1 - normalized_std (inverted)
        coherence = max(0.0, 1.0 - normalized_std)
        scores.append(coherence)

    return scores


def _compute_signal_to_noise(
    values: list[float],
    baseline: Optional[list[float]] = None,
    window_size: int = 5,
) -> float:
    """Estimate signal-to-noise ratio.

    If baseline provided, SNR = variance(signal) / variance(baseline noise)
    Otherwise, SNR from autocorrelation strength.

    Returns:
        SNR in [0, 1]
    """
    if not values or len(values) < 3:
        return 0.0

    recent = values[-window_size:] if len(values) >= window_size else values

    if baseline:
        signal_var = sum((x - sum(recent) / len(recent)) ** 2 for x in recent) / len(recent)
        noise_var = sum((x - sum(baseline) / len(baseline)) ** 2 for x in baseline) / len(baseline)

        if noise_var > 1e-6:
            snr = min(1.0, signal_var / noise_var)
        else:
            snr = 0.5
    else:
        # Estimate from autocorrelation
        if len(recent) < 2:
            return 0.5

        mean_val = sum(recent) / len(recent)
        variance = sum((x - mean_val) ** 2 for x in recent) / len(recent)

        if variance < 0.01:
            snr = 0.3  # Low variance = low signal
        elif variance > 0.5:
            snr = 0.8  # High variance = strong signal
        else:
            snr = 0.5 + (variance - 0.01) / 0.98 * 0.3

    return max(0.0, min(1.0, snr))


def _classify_coherence_type(
    window_scores: list[float],
    recent_window_count: int = 3,
) -> str:
    """Classify coherence trend: increasing, steady, decreasing, or noisy."""
    if not window_scores or len(window_scores) < 2:
        return "noisy"

    recent = window_scores[-recent_window_count:] if len(window_scores) >= recent_window_count else window_scores

    if not recent:
        return "noisy"

    mean_recent = sum(recent) / len(recent)

    if mean_recent < 0.3:
        return "noisy"

    # Check trend
    if len(recent) >= 2:
        trend = recent[-1] - recent[0]

        if trend > 0.15:
            return "increasing"
        elif trend < -0.15:
            return "decreasing"
        else:
            return "steady"

    return "steady"


class TemporalCoherenceScorer:
    """Scores temporal coherence of key metrics over time."""

    HISTORY_WINDOW = 20  # Track last 20 frames
    COHERENCE_THRESHOLD = 0.6  # Above this = coherent signal

    def __init__(self):
        self.metric_histories: dict[str, list[float]] = {}
        self.coherence_results: dict[str, TemporalCoherence] = {}

    def update_metrics(
        self,
        metrics: dict[str, float],
    ) -> TemporalCoherenceProfile:
        """Update coherence scores with new metric values.

        Args:
            metrics: Dict of metric_name -> value [0, 1]

        Returns:
            TemporalCoherenceProfile with coherence for each metric
        """
        # Update histories
        for metric_name, value in metrics.items():
            if metric_name not in self.metric_histories:
                self.metric_histories[metric_name] = []

            self.metric_histories[metric_name].append(float(value))

            # Keep window
            if len(self.metric_histories[metric_name]) > self.HISTORY_WINDOW:
                self.metric_histories[metric_name] = self.metric_histories[metric_name][
                    -self.HISTORY_WINDOW :
                ]

        # Compute coherence for each metric
        coherences: dict[str, TemporalCoherence] = {}
        for metric_name, history in self.metric_histories.items():
            coherence = self._score_metric_coherence(metric_name, history)
            coherences[metric_name] = coherence
            self.coherence_results[metric_name] = coherence

        # Build profile
        profile = TemporalCoherenceProfile(coherences=coherences)
        profile.overall_system_coherence = self._aggregate_coherence(coherences)
        profile.primary_signal_type = self._get_primary_signal_type(coherences)
        profile.is_coherent_overall = profile.overall_system_coherence > self.COHERENCE_THRESHOLD

        return profile

    def _score_metric_coherence(self, metric_name: str, history: list[float]) -> TemporalCoherence:
        """Score coherence for a single metric."""
        window_scores = _compute_window_coherence(history, window_size=3)
        overall = sum(window_scores) / len(window_scores) if window_scores else 0.0
        signal_strength = _compute_signal_to_noise(history)
        coherence_type = _classify_coherence_type(window_scores)

        # Count consecutive frames showing same coherence direction
        frames_coherent = 0
        if window_scores and len(window_scores) >= 2:
            recent = window_scores[-5:]
            mean_recent = sum(recent) / len(recent)
            frames_coherent = sum(1 for s in reversed(window_scores) if s >= mean_recent * 0.9)

        return TemporalCoherence(
            metric_name=metric_name,
            window_scores=window_scores,
            overall_coherence=overall,
            signal_strength=signal_strength,
            frames_coherent=frames_coherent,
            coherence_type=coherence_type,
        )

    def _aggregate_coherence(self, coherences: dict[str, TemporalCoherence]) -> float:
        """Aggregate coherence across all metrics."""
        if not coherences:
            return 0.0

        scores = [c.overall_coherence for c in coherences.values()]
        return sum(scores) / len(scores) if scores else 0.0

    def _get_primary_signal_type(self, coherences: dict[str, TemporalCoherence]) -> str:
        """Determine primary coherence type across all metrics."""
        types = [c.coherence_type for c in coherences.values()]

        if not types:
            return "noisy"

        # Count each type
        type_counts = {}
        for t in types:
            type_counts[t] = type_counts.get(t, 0) + 1

        # Return most common
        return max(type_counts, key=type_counts.get)

    def get_coherence_narrative(self, profile: TemporalCoherenceProfile) -> str:
        """Generate human-readable coherence narrative."""
        if profile.overall_system_coherence > 0.75:
            strength = "very strong"
        elif profile.overall_system_coherence > 0.6:
            strength = "strong"
        elif profile.overall_system_coherence > 0.4:
            strength = "moderate"
        else:
            strength = "weak"

        return f"Coherence {strength} ({profile.overall_system_coherence:.2f}): {profile.primary_signal_type} pattern"
