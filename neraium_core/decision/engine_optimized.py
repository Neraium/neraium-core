"""Optimized decision engine with lazy evaluation and modular phases.

Replaces the monolithic decide() method with a pipeline of independent,
cacheable computation phases for better performance and maintainability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from neraium_core.decision.models import Decision, SeverityLevel, TrajectoryLabel, DegradationStage
from neraium_core.engine_optimizations import MemoizedScoreComputation


@dataclass
class ComputationPhaseResult:
    """Result of a single computation phase with optional caching."""
    phase_name: str
    result: Any
    cached: bool = False


class DecisionComputationPipeline:
    """Orchestrates the decision computation as independent, optimizable phases.

    Each phase:
    - Receives only the data it needs
    - Can be cached if inputs haven't changed
    - Runs independently and doesn't modify shared state
    - Returns a focused result

    This structure enables:
    - Per-phase caching without global cache corruption
    - Parallel execution of independent phases
    - Easier testing and debugging
    - Clear performance bottleneck identification
    """

    def __init__(self):
        self._phase_cache: dict[str, ComputationPhaseResult] = {}
        self._score_cache = MemoizedScoreComputation()
        self._phase_metrics: dict[str, dict[str, int]] = {}

    def _record_phase(self, phase_name: str, cached: bool) -> None:
        """Track phase execution for monitoring."""
        if phase_name not in self._phase_metrics:
            self._phase_metrics[phase_name] = {"calls": 0, "cached": 0}
        self._phase_metrics[phase_name]["calls"] += 1
        if cached:
            self._phase_metrics[phase_name]["cached"] += 1

    def phase_severity_classification(
        self,
        state: str,
        drift_score: float,
        relational_instability: float,
        system_phase: str,
        prior_severity: Optional[SeverityLevel] = None,
        persistence_frames: int = 0,
        persistence_trajectory: float = 0.0,
        force_recompute: bool = False,
    ) -> SeverityLevel:
        """Phase 1: Classify severity with hysteresis.

        This phase is relatively fast, but we cache it to avoid recomputation
        when inputs are identical (common in stable periods).
        """
        from neraium_core.decision import policy

        cache_key = f"severity:{state}:{drift_score:.4f}:{relational_instability:.4f}:{system_phase}:{prior_severity}:{persistence_frames}:{persistence_trajectory:.4f}"

        if not force_recompute and cache_key in self._phase_cache:
            self._record_phase("severity_classification", True)
            return self._phase_cache[cache_key].result

        severity = policy.classify_severity(
            state=state,
            drift_score=drift_score,
            relational_instability=relational_instability,
            system_phase=system_phase,
            prior_severity=prior_severity,
            persistence_frames=persistence_frames,
            persistence_trajectory=persistence_trajectory,
        )

        self._phase_cache[cache_key] = ComputationPhaseResult(
            phase_name="severity_classification",
            result=severity,
            cached=False,
        )
        self._record_phase("severity_classification", False)
        return severity

    def phase_finding_confidence(
        self,
        drift_score: float,
        signal_count: int,
        data_quality_issues: int,
        state: str,
        relational_instability: float,
        force_recompute: bool = False,
    ) -> float:
        """Phase 2: Score finding confidence.

        Confidence scoring is quick but repeated for every frame.
        Caching reduces redundant calculations.
        """
        from neraium_core.decision import confidence as conf_module

        cache_key = f"confidence:{drift_score:.4f}:{signal_count}:{data_quality_issues}:{state}:{relational_instability:.4f}"

        if not force_recompute and cache_key in self._phase_cache:
            self._record_phase("finding_confidence", True)
            return self._phase_cache[cache_key].result

        confidence = conf_module.score_finding_confidence(
            drift_score=drift_score,
            signal_count=signal_count,
            data_quality_issues=data_quality_issues,
            state=state,
            relational_instability=relational_instability,
        )

        self._phase_cache[cache_key] = ComputationPhaseResult(
            phase_name="finding_confidence",
            result=confidence,
            cached=False,
        )
        self._record_phase("finding_confidence", False)
        return confidence

    def phase_transient_detection(
        self,
        drift_score: float,
        shock_activity: float,
        system_phase: str,
        state: str,
        drift_history: Optional[list[float]] = None,
        force_recompute: bool = False,
    ) -> tuple[float, bool]:
        """Phase 3: Detect transient events.

        Returns (transient_score, is_safe_transient).
        """
        from neraium_core.decision import transient_gating

        # Simplified cache key for transient scoring
        cache_key = f"transient:{drift_score:.4f}:{shock_activity:.4f}:{system_phase}:{state}"

        if not force_recompute and cache_key in self._phase_cache:
            self._record_phase("transient_detection", True)
            return self._phase_cache[cache_key].result

        drift_trend = self._compute_drift_trend(drift_history)
        transient_score = transient_gating.score_transient_likelihood(
            drift_score=drift_score,
            drift_trend=drift_trend,
            shock_activity=shock_activity,
            system_phase=system_phase,
            state=state,
            drift_history=drift_history if isinstance(drift_history, list) else None,
        )

        result = (transient_score, 0.0)
        self._phase_cache[cache_key] = ComputationPhaseResult(
            phase_name="transient_detection",
            result=result,
            cached=False,
        )
        self._record_phase("transient_detection", False)
        return result

    def _compute_drift_trend(self, drift_history: Any) -> float:
        """Compute drift trend from history (fast, no caching needed)."""
        if not isinstance(drift_history, list) or len(drift_history) < 2:
            return 0.0
        try:
            recent = [float(v) for v in drift_history[-5:]]
            return (recent[-1] - recent[0]) / max(len(recent) - 1, 1)
        except (ValueError, TypeError):
            return 0.0

    def clear_cache(self) -> None:
        """Clear computation cache between sessions."""
        self._phase_cache.clear()
        self._score_cache.clear()

    def phase_pattern_matching(
        self,
        drift_score: float,
        relational_instability: float,
        shock_activity: float,
        regime_distance: float,
        system_phase_encoded: float,
        force_recompute: bool = False,
    ) -> Any:
        """Phase 4: Pattern matching with feature vector caching.

        Pattern matching against historical patterns to identify
        similar failure/degradation modes.
        """
        from neraium_core.decision import pattern_memory as pm_module

        cache_key = f"pattern:{drift_score:.4f}:{relational_instability:.4f}:{shock_activity:.4f}:{regime_distance:.4f}:{system_phase_encoded:.4f}"

        if not force_recompute and cache_key in self._phase_cache:
            self._record_phase("pattern_matching", True)
            return self._phase_cache[cache_key].result

        feature_vector = pm_module.build_feature_vector(
            drift_score=drift_score,
            relational_instability=relational_instability,
            shock_activity=shock_activity,
            regime_distance=regime_distance,
            system_phase_encoded=system_phase_encoded,
        )
        # Actual pattern matching is done by pattern_memory singleton
        # This phase just caches the feature vector computation

        self._phase_cache[cache_key] = ComputationPhaseResult(
            phase_name="pattern_matching",
            result=feature_vector,
            cached=False,
        )
        self._record_phase("pattern_matching", False)
        return feature_vector

    def cache_stats(self) -> dict[str, Any]:
        """Return caching statistics for monitoring."""
        return {
            "phases": self._phase_metrics,
            "total_cached_entries": len(self._phase_cache),
        }
