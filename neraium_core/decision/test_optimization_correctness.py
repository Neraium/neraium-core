"""Test that optimizations don't degrade decision accuracy.

Validates that cached and optimized code paths produce identical
results to the original implementation.
"""

from __future__ import annotations

import unittest
from typing import Any

from neraium_core.decision.engine_optimized import DecisionComputationPipeline
from neraium_core.decision import policy, confidence as conf_module, transient_gating


class TestOptimizationCorrectness(unittest.TestCase):
    """Verify optimized computation phases match original logic."""

    def setUp(self):
        self.pipeline = DecisionComputationPipeline()

    def test_severity_classification_matches_direct(self):
        """Verify cached severity classification equals direct call."""
        params = {
            "state": "DEGRADING",
            "drift_score": 0.65,
            "relational_instability": 0.42,
            "system_phase": "degrading",
            "prior_severity": None,
            "persistence_frames": 0,
            "persistence_trajectory": 0.0,
        }

        # Direct call
        direct = policy.classify_severity(**params)

        # Pipeline (first call - miss)
        pipeline_miss = self.pipeline.phase_severity_classification(**params)

        # Pipeline (second call - hit)
        pipeline_hit = self.pipeline.phase_severity_classification(**params)

        # All should be equal
        self.assertEqual(direct, pipeline_miss)
        self.assertEqual(direct, pipeline_hit)

    def test_confidence_scoring_matches_direct(self):
        """Verify cached confidence scoring equals direct call."""
        params = {
            "drift_score": 0.75,
            "signal_count": 12,
            "data_quality_issues": 1,
            "state": "DEGRADING",
            "relational_instability": 0.55,
        }

        # Direct call
        direct = conf_module.score_finding_confidence(**params)

        # Pipeline (first call - miss)
        pipeline_miss = self.pipeline.phase_finding_confidence(**params)

        # Pipeline (second call - hit)
        pipeline_hit = self.pipeline.phase_finding_confidence(**params)

        # All should be equal (within floating point tolerance)
        self.assertAlmostEqual(direct, pipeline_miss, places=6)
        self.assertAlmostEqual(direct, pipeline_hit, places=6)

    def test_transient_detection_matches_direct(self):
        """Verify cached transient detection equals direct call."""
        params = {
            "drift_score": 0.45,
            "shock_activity": 0.2,
            "system_phase": "stable",
            "state": "STABLE",
            "drift_history": [0.1, 0.15, 0.2, 0.25, 0.3],
        }

        # Direct call
        direct = transient_gating.score_transient_likelihood(
            drift_score=params["drift_score"],
            drift_trend=self._compute_trend(params["drift_history"]),
            shock_activity=params["shock_activity"],
            system_phase=params["system_phase"],
            state=params["state"],
            drift_history=params["drift_history"],
        )

        # Pipeline (first call - miss)
        pipeline_miss, _ = self.pipeline.phase_transient_detection(**params)

        # Pipeline (second call - hit)
        pipeline_hit, _ = self.pipeline.phase_transient_detection(**params)

        # All should be equal
        self.assertAlmostEqual(direct, pipeline_miss, places=6)
        self.assertAlmostEqual(direct, pipeline_hit, places=6)

    def test_cache_hit_rate(self):
        """Verify cache hit rate tracking works correctly."""
        params = {
            "state": "DEGRADING",
            "drift_score": 0.65,
            "relational_instability": 0.42,
            "system_phase": "degrading",
        }

        # Make several calls
        for _ in range(5):
            self.pipeline.phase_severity_classification(**params)

        stats = self.pipeline.cache_stats()
        metrics = stats["phases"]

        # Verify metrics are tracking calls
        self.assertGreater(len(metrics), 0)

    def test_force_recompute_ignores_cache(self):
        """Verify force_recompute bypasses cache."""
        params = {
            "state": "DEGRADING",
            "drift_score": 0.65,
            "relational_instability": 0.42,
            "system_phase": "degrading",
        }

        # First call (miss)
        result1 = self.pipeline.phase_severity_classification(**params)

        # Second call with force_recompute (should not use cache)
        result2 = self.pipeline.phase_severity_classification(
            **params, force_recompute=True
        )

        # Results should still match
        self.assertEqual(result1, result2)

    def test_different_params_different_cache_keys(self):
        """Verify different parameters use different cache entries."""
        params1 = {
            "drift_score": 0.5,
            "signal_count": 10,
            "data_quality_issues": 0,
            "state": "STABLE",
            "relational_instability": 0.2,
        }

        params2 = {
            "drift_score": 0.9,  # Different
            "signal_count": 10,
            "data_quality_issues": 0,
            "state": "STABLE",
            "relational_instability": 0.2,
        }

        conf1 = self.pipeline.phase_finding_confidence(**params1)
        conf2 = self.pipeline.phase_finding_confidence(**params2)

        # Different inputs should produce different outputs
        self.assertNotEqual(conf1, conf2)

    @staticmethod
    def _compute_trend(history: list[float]) -> float:
        """Compute drift trend for testing."""
        if len(history) < 2:
            return 0.0
        try:
            recent = [float(v) for v in history[-5:]]
            return (recent[-1] - recent[0]) / max(len(recent) - 1, 1)
        except (ValueError, TypeError):
            return 0.0


if __name__ == "__main__":
    unittest.main()
