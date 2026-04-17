"""Pattern memory: match against known prior behaviors.

Uses simple vector similarity (cosine) to find historical patterns.
"""

from __future__ import annotations

from typing import Any
from neraium_core.decision.models import PatternMatch


class PatternMemory:
    """Simple in-memory pattern store with cosine similarity matching."""

    def __init__(self):
        self.patterns: list[dict[str, Any]] = []

    def add_pattern(
        self,
        pattern_id: str,
        features: list[float],
        outcome: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Store a pattern for future matching.

        Args:
            pattern_id: Unique identifier (e.g., "asset_001:run_42")
            features: Vector of [drift, relational, shock, regime_distance, ...]
            outcome: What happened ("self_resolved", "escalated", "cyclic", etc.)
            metadata: Additional context (time_to_outcome, severity, etc.)
        """
        self.patterns.append({
            "id": pattern_id,
            "features": list(features),
            "outcome": outcome,
            "metadata": metadata or {},
        })

    def find_match(
        self,
        current_features: list[float],
        top_k: int = 1,
        min_similarity: float = 0.7,
    ) -> PatternMatch | None:
        """Find closest historical pattern.

        Args:
            current_features: Current state vector
            top_k: Return top-k matches (returns best)
            min_similarity: Only return if similarity >= this

        Returns:
            PatternMatch if found, else None
        """
        if not self.patterns:
            return None

        scores = []
        for pattern in self.patterns:
            sim = cosine_similarity(current_features, pattern["features"])
            if sim >= min_similarity:
                scores.append((sim, pattern))

        if not scores:
            return None

        best_sim, best_pattern = max(scores, key=lambda x: x[0])

        metadata = best_pattern.get("metadata", {})
        return PatternMatch(
            pattern_id=best_pattern["id"],
            similarity=best_sim,
            prior_outcome=best_pattern["outcome"],
            time_to_outcome_hours=metadata.get("time_to_outcome_hours"),
            confidence=best_sim,
        )


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Returns [0, 1] where 1 = identical direction.
    """
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0

    try:
        dot = sum(a * b for a, b in zip(v1, v2))
        mag1 = (sum(a * a for a in v1)) ** 0.5
        mag2 = (sum(b * b for b in v2)) ** 0.5

        if mag1 < 1e-10 or mag2 < 1e-10:
            return 0.0

        return dot / (mag1 * mag2)
    except (ValueError, ZeroDivisionError):
        return 0.0


def build_feature_vector(
    *,
    drift_score: float,
    relational_instability: float,
    shock_activity: float,
    regime_distance: float = 0.5,
    system_phase_encoded: float = 0.5,
) -> list[float]:
    """Build a feature vector for pattern matching.

    Simple approach: normalize key metrics into [0, 1].
    """
    return [
        min(1.0, max(0.0, drift_score)),
        min(1.0, max(0.0, relational_instability)),
        min(1.0, max(0.0, shock_activity)),
        min(1.0, max(0.0, regime_distance)),
        min(1.0, max(0.0, system_phase_encoded)),
    ]
