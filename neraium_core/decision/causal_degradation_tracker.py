"""Causal degradation tracker: Track which relationships are degrading and their propagation.

Understands system as interconnected relationships and tracks:
- Which relationships are actively degrading
- Degradation velocity per relationship
- Propagation: when one breaks, which others follow?
- Causal evidence trail
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RelationshipDegradation:
    """Track degradation of a single relationship."""

    signal_pair: tuple[str, str]  # ("pressure", "flow")
    baseline_correlation: float  # Original correlation [0, 1]
    current_correlation: float  # Current correlation [0, 1]
    degradation_velocity: float  # Rate of change per frame
    frames_degrading: int  # Consecutive frames showing degradation
    confidence: float  # Confidence in this degradation [0, 1]
    is_critical: bool  # Does this relationship matter for stability?
    reversible: bool = True  # Can it recover?
    affected_subsystems: list[str] = field(default_factory=list)


@dataclass
class DegradationMap:
    """Full degradation picture for current frame."""

    degradations_active: list[RelationshipDegradation] = field(default_factory=list)
    root_degradations: list[RelationshipDegradation] = field(default_factory=list)
    predicted_next_breaks: list[tuple[tuple[str, str], float]] = field(default_factory=list)
    root_cause_candidate: Optional[tuple[str, str]] = None
    degradation_count: int = 0
    critical_degradation_count: int = 0
    is_cascading: bool = False  # Multiple relationships breaking in sequence?


class CausalDegradationTracker:
    """Tracks relationship degradation and propagation patterns."""

    HISTORY_WINDOW = 20
    CRITICAL_SIGNALS = {"pressure", "flow", "vibration", "temperature"}
    CORRELATION_THRESHOLD = 0.3  # Baseline
    DEGRADATION_THRESHOLD = 0.15  # Min drop to count as degradation

    def __init__(self):
        self.relationship_histories: dict[tuple[str, str], list[float]] = {}
        self.degradation_history: list[DegradationMap] = []
        self.previous_active_degradations: set[tuple[str, str]] = set()

    def analyze_relationships(
        self,
        sii_output: dict,
        relational_instability: float,
    ) -> DegradationMap:
        """Analyze relationships from SII output to identify degradations.

        Args:
            sii_output: SII structural output with attribution data
            relational_instability: Overall relational instability metric [0, 1]

        Returns:
            DegradationMap with current degradation analysis
        """
        degradation_map = DegradationMap()

        # Extract attribution data (relationship information)
        attribution = sii_output.get("attribution", {})
        if not isinstance(attribution, dict):
            return degradation_map

        # Build relationship pairs from available sensor info
        sensor_relationships = sii_output.get("sensor_relationships", [])
        top_drivers = attribution.get("top_drivers", [])

        if not sensor_relationships or not top_drivers:
            return degradation_map

        # Identify probable relationships (top drivers likely decouple from others)
        degradations = self._detect_relationship_degradations(
            top_drivers=top_drivers,
            sensor_relationships=sensor_relationships,
            relational_instability=relational_instability,
        )

        degradation_map.degradations_active = degradations
        degradation_map.degradation_count = len(degradations)
        degradation_map.critical_degradation_count = sum(1 for d in degradations if d.is_critical)

        # Identify root degradations (likely initiators)
        degradation_map.root_degradations = self._identify_root_degradations(degradations)
        if degradation_map.root_degradations:
            degradation_map.root_cause_candidate = degradation_map.root_degradations[0].signal_pair

        # Detect cascading failures
        degradation_map.is_cascading = self._detect_cascading_failure(degradations)

        # Predict next likely breaks
        degradation_map.predicted_next_breaks = self._predict_next_breaks(
            active_degradations=degradations,
            all_sensors=sensor_relationships,
        )

        # Track history
        self.degradation_history.append(degradation_map)
        if len(self.degradation_history) > self.HISTORY_WINDOW:
            self.degradation_history = self.degradation_history[-self.HISTORY_WINDOW :]

        self.previous_active_degradations = {d.signal_pair for d in degradations}

        return degradation_map

    def _detect_relationship_degradations(
        self,
        top_drivers: list[str],
        sensor_relationships: list[str],
        relational_instability: float,
    ) -> list[RelationshipDegradation]:
        """Detect degrading relationships from attribution data."""
        degradations: list[RelationshipDegradation] = []

        # For each top driver, assume it's decoupling from others
        for driver in top_drivers[:3]:  # Focus on top 3
            for sensor in sensor_relationships:
                if sensor == driver:
                    continue

                pair = tuple(sorted([driver, sensor]))
                is_critical = any(s in pair for s in self.CRITICAL_SIGNALS)

                # Estimate degradation from relational instability + driver prominence
                estimated_degradation = min(1.0, relational_instability * (0.5 + len(top_drivers) * 0.15))

                if estimated_degradation > self.DEGRADATION_THRESHOLD:
                    degradation = RelationshipDegradation(
                        signal_pair=pair,
                        baseline_correlation=0.7,  # Assume normal baseline
                        current_correlation=max(0.0, 0.7 - estimated_degradation),
                        degradation_velocity=estimated_degradation / max(1, self._get_frames_degrading(pair)),
                        frames_degrading=self._get_frames_degrading(pair),
                        confidence=min(0.9, 0.4 + relational_instability),
                        is_critical=is_critical,
                        affected_subsystems=list(pair),
                    )
                    degradations.append(degradation)

        return degradations

    def _get_frames_degrading(self, signal_pair: tuple[str, str]) -> int:
        """Count consecutive frames this relationship has been degrading."""
        count = 0
        for dmap in reversed(self.degradation_history[-5:]):
            if any(d.signal_pair == signal_pair for d in dmap.degradations_active):
                count += 1
            else:
                break
        return max(1, count)

    def _identify_root_degradations(
        self, degradations: list[RelationshipDegradation]
    ) -> list[RelationshipDegradation]:
        """Identify likely root cause degradations (initiators, not consequences)."""
        if not degradations:
            return []

        # Root degradations: high confidence, critical signals, high velocity
        roots = sorted(
            degradations,
            key=lambda d: (d.confidence * d.is_critical * d.degradation_velocity),
            reverse=True,
        )

        return roots[:2]  # Return top 2 likely roots

    def _detect_cascading_failure(self, degradations: list[RelationshipDegradation]) -> bool:
        """Detect if multiple relationships breaking in sequence (cascade)."""
        if len(degradations) < 2:
            return False

        # Check if some are new while others have been degrading
        frame_counts = [d.frames_degrading for d in degradations]

        # Cascade: varying frame counts suggests sequence (not all appeared at once)
        if frame_counts and max(frame_counts) - min(frame_counts) >= 2:
            return True

        return False

    def _predict_next_breaks(
        self,
        active_degradations: list[RelationshipDegradation],
        all_sensors: list[str],
    ) -> list[tuple[tuple[str, str], float]]:
        """Predict which relationships are likely to break next.

        Returns:
            List of (signal_pair, probability) tuples
        """
        predictions: list[tuple[tuple[str, str], float]] = []

        # Sensors already involved in degradations
        involved_sensors = set()
        for d in active_degradations:
            involved_sensors.update(d.signal_pair)

        # Sensors not yet involved but might couple with degrading ones
        for sensor in all_sensors:
            if sensor in involved_sensors:
                continue

            # Probability of degradation = avg degradation confidence of coupled sensors
            coupled_degradations = [d for d in active_degradations if sensor in d.affected_subsystems or any(s in all_sensors for s in d.signal_pair)]

            if coupled_degradations:
                prob = sum(d.confidence for d in coupled_degradations) / len(coupled_degradations)
                for d in coupled_degradations:
                    pair = tuple(sorted([sensor, list(d.signal_pair)[0]]))
                    predictions.append((pair, prob * 0.7))  # Discount for prediction

        # Sort by probability
        predictions.sort(key=lambda x: x[1], reverse=True)

        return predictions[:3]

    def get_degradation_narrative(self, degradation_map: DegradationMap) -> str:
        """Generate human-readable narrative of degradations."""
        if not degradation_map.degradations_active:
            return "No relationship degradation detected."

        critical = degradation_map.critical_degradation_count
        total = degradation_map.degradation_count

        narrative = f"{total} relationship(s) degrading ({critical} critical)"

        if degradation_map.root_cause_candidate:
            sig1, sig2 = degradation_map.root_cause_candidate
            narrative += f", root: {sig1}-{sig2} coupling loss"

        if degradation_map.is_cascading:
            narrative += ", cascading failure pattern detected"

        if degradation_map.predicted_next_breaks:
            next_pair = degradation_map.predicted_next_breaks[0][0]
            narrative += f", next likely: {next_pair[0]}-{next_pair[1]}"

        return narrative
