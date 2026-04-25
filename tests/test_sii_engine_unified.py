"""
Tests for System Instability Intelligence (SII) Unified Engine.

Validates:
1. Baseline fitting and covariance computation
2. Structural drift calculation
3. Drift velocity computation
4. Transition pressure dynamics
5. Unified instability score composition
6. Regime classification
7. Urgency mapping
8. End-to-end pipeline
9. State persistence/restoration
10. Edge cases (missing data, singular covariance, warmup)
"""

from __future__ import annotations

import numpy as np
import pytest

from neraium_core.sii_engine_unified import (
    SIIEngine,
    SIIEngineOutput,
    BaselineProfile,
    STABLE_THRESHOLD,
    TRANSITION_THRESHOLD,
    UNSTABLE_THRESHOLD,
    LOCK_IN_THRESHOLD,
)


class TestBaselineProfile:
    """Test baseline profile data structure."""

    def test_baseline_profile_validity(self):
        """Empty baseline is invalid."""
        profile = BaselineProfile()
        assert not profile.is_valid()

    def test_baseline_profile_with_data(self):
        """Baseline with mean and covariance is valid."""
        profile = BaselineProfile()
        profile.mean = np.array([0.0, 0.0])
        profile.cov = np.eye(2)
        profile.cov_inv = np.eye(2)
        profile.sample_count = 50
        assert profile.is_valid()


class TestBaselineFitting:
    """Test baseline computation from initial window."""

    def test_fit_baseline_from_data(self):
        """Baseline fit computes mean and covariance correctly."""
        engine = SIIEngine(baseline_window=30)

        # Create synthetic data: 2D Gaussian
        np.random.seed(42)
        data = np.random.randn(50, 2) + np.array([5.0, 10.0])

        engine.fit_baseline(data)

        assert engine.baseline.is_valid()
        assert engine.baseline.mean is not None
        assert engine.baseline.cov is not None
        assert engine.baseline.cov_inv is not None
        assert engine.baseline.sample_count == 30

    def test_fit_baseline_insufficient_data(self):
        """Fitting with too few samples raises error."""
        engine = SIIEngine(baseline_window=50)
        data = np.random.randn(30, 3)

        with pytest.raises(ValueError):
            engine.fit_baseline(data)

    def test_fit_baseline_with_missing_values(self):
        """Baseline fitting handles NaN values via forward-fill."""
        engine = SIIEngine(baseline_window=20)

        data = np.random.randn(30, 2)
        # Introduce NaN values
        data[5:8, 0] = np.nan
        data[10, 1] = np.nan

        engine.fit_baseline(data)
        assert engine.baseline.is_valid()


class TestStructuralDrift:
    """Test structural drift computation."""

    def test_drift_at_baseline_is_zero(self):
        """Drift should be near zero when data matches baseline."""
        engine = SIIEngine()

        # Create and fit baseline
        baseline_data = np.random.randn(50, 3)
        engine.fit_baseline(baseline_data)

        # Compute drift from same data
        cov_t = engine.compute_covariance(baseline_data)
        drift = engine.compute_structural_drift(cov_t)

        assert drift < 0.05

    def test_drift_increases_with_covariance_change(self):
        """Drift should increase as covariance deviates from baseline."""
        engine = SIIEngine()

        # Baseline data: unit variance
        baseline_data = np.random.randn(50, 2)
        engine.fit_baseline(baseline_data)

        # Modified data: higher variance
        modified_data = np.random.randn(50, 2) * 3.0
        cov_modified = engine.compute_covariance(modified_data)
        drift_modified = engine.compute_structural_drift(cov_modified)

        # Drift at baseline
        cov_baseline = engine.compute_covariance(baseline_data)
        drift_baseline = engine.compute_structural_drift(cov_baseline)

        assert drift_modified > drift_baseline


class TestDriftVelocity:
    """Test drift velocity (rate of change)."""

    def test_velocity_accumulates_with_changes(self):
        """Velocity should track rate of drift change."""
        engine = SIIEngine()

        baseline_data = np.random.randn(50, 2)
        engine.fit_baseline(baseline_data)

        velocities = []
        for _ in range(5):
            frame = np.random.randn(1, 2) * 2.0
            cov_t = engine.compute_covariance(
                np.vstack([np.random.randn(20, 2), frame])
            )
            drift = engine.compute_structural_drift(cov_t)
            velocity = engine.compute_velocity(drift, float(_))
            velocities.append(velocity)

        # First velocity is zero (no previous drift)
        assert abs(velocities[0]) < 1e-6

    def test_velocity_sign_indicates_direction(self):
        """Velocity sign should indicate drift direction."""
        engine = SIIEngine()

        baseline_data = np.ones((50, 2))
        engine.fit_baseline(baseline_data)

        # Increasing drift
        v1 = engine.compute_velocity(0.1, 0.0)
        v2 = engine.compute_velocity(0.2, 1.0)
        v3 = engine.compute_velocity(0.3, 2.0)

        # Velocities should be positive (increasing drift)
        assert v2 > 0
        assert v3 > 0


class TestTransitionPressure:
    """Test transition pressure computation."""

    def test_pressure_zero_at_stable_low_velocity(self):
        """Pressure should be near zero when drift and velocity are low."""
        engine = SIIEngine()

        pressure = engine.compute_transition_pressure(S_t=0.05, V_t=0.01)
        assert pressure < 0.1

    def test_pressure_increases_with_drift(self):
        """Pressure should increase as drift magnitude increases."""
        engine = SIIEngine()

        p1 = engine.compute_transition_pressure(S_t=0.1, V_t=0.0)
        p2 = engine.compute_transition_pressure(S_t=0.5, V_t=0.0)
        p3 = engine.compute_transition_pressure(S_t=0.9, V_t=0.0)

        assert p1 < p2 < p3

    def test_pressure_increases_with_velocity(self):
        """Pressure should increase as velocity magnitude increases."""
        engine = SIIEngine()

        p1 = engine.compute_transition_pressure(S_t=0.5, V_t=0.01)
        p2 = engine.compute_transition_pressure(S_t=0.5, V_t=0.1)
        p3 = engine.compute_transition_pressure(S_t=0.5, V_t=0.5)

        assert p1 < p2 < p3


class TestInstabilityScore:
    """Test unified instability score computation."""

    def test_score_is_normalized(self):
        """Instability score should be in [0, 1]."""
        engine = SIIEngine()

        for _ in range(10):
            s_t = np.random.rand()
            v_t = np.random.randn() * 2.0
            p_t = np.random.rand()

            score = engine.compute_instability_score(s_t, v_t, p_t)
            assert 0.0 <= score <= 1.0

    def test_score_zero_when_all_components_zero(self):
        """Score should be zero when all components are zero."""
        engine = SIIEngine()
        score = engine.compute_instability_score(0.0, 0.0, 0.0)
        assert score == pytest.approx(0.0, abs=1e-6)

    def test_score_increases_with_components(self):
        """Score should increase as components increase."""
        engine = SIIEngine()

        s1 = engine.compute_instability_score(0.2, 0.01, 0.1)
        s2 = engine.compute_instability_score(0.5, 0.01, 0.1)
        s3 = engine.compute_instability_score(0.8, 0.01, 0.1)

        assert s1 < s2 < s3


class TestRegimeClassification:
    """Test regime classification based on instability score."""

    def test_stable_regime(self):
        """I_t <= 0.30 -> STABLE"""
        engine = SIIEngine()

        assert engine.classify_regime(0.0) == "STABLE"
        assert engine.classify_regime(0.15) == "STABLE"
        assert engine.classify_regime(STABLE_THRESHOLD) == "STABLE"

    def test_transition_regime(self):
        """0.30 < I_t <= 0.65 -> TRANSITION"""
        engine = SIIEngine()

        assert engine.classify_regime(0.40) == "TRANSITION"
        assert engine.classify_regime(0.50) == "TRANSITION"
        assert engine.classify_regime(TRANSITION_THRESHOLD) == "TRANSITION"

    def test_unstable_regime(self):
        """0.65 < I_t <= 0.85 -> UNSTABLE"""
        engine = SIIEngine()

        assert engine.classify_regime(0.70) == "UNSTABLE"
        assert engine.classify_regime(0.75) == "UNSTABLE"
        assert engine.classify_regime(UNSTABLE_THRESHOLD) == "UNSTABLE"

    def test_lock_in_regime(self):
        """I_t > 0.85 -> LOCK_IN"""
        engine = SIIEngine()

        assert engine.classify_regime(0.90) == "LOCK_IN"
        assert engine.classify_regime(0.95) == "LOCK_IN"
        assert engine.classify_regime(1.0) == "LOCK_IN"


class TestUrgencyMapping:
    """Test urgency mapping from regime and velocity."""

    def test_urgency_nominal_in_stable(self):
        """STABLE regime with low velocity -> NOMINAL"""
        engine = SIIEngine()
        urgency = engine.compute_urgency("STABLE", 0.01)
        assert urgency == "NOMINAL"

    def test_urgency_watch_in_transition(self):
        """TRANSITION regime -> WATCH"""
        engine = SIIEngine()
        urgency = engine.compute_urgency("TRANSITION", 0.01)
        assert urgency == "WATCH"

    def test_urgency_alert_in_unstable(self):
        """UNSTABLE regime -> ALERT"""
        engine = SIIEngine()
        urgency = engine.compute_urgency("UNSTABLE", 0.01)
        assert urgency == "ALERT"

    def test_urgency_critical_in_lock_in(self):
        """LOCK_IN regime -> CRITICAL"""
        engine = SIIEngine()
        urgency = engine.compute_urgency("LOCK_IN", 0.01)
        assert urgency == "CRITICAL"

    def test_urgency_escalates_with_velocity(self):
        """High velocity escalates urgency."""
        engine = SIIEngine()

        u1 = engine.compute_urgency("STABLE", 0.001)
        u2 = engine.compute_urgency("STABLE", 0.1)

        # u1 should be NOMINAL, u2 should be WATCH
        assert u1 == "NOMINAL"
        assert u2 == "WATCH"


class TestEndToEndPipeline:
    """Test complete pipeline from input to output."""

    def test_update_during_warmup(self):
        """Updates during warmup return safe defaults."""
        engine = SIIEngine(baseline_window=20)

        # Process frames during warmup
        np.random.seed(42)
        for i in range(15):
            frame = np.random.randn(3)
            output = engine.update(frame, float(i))

            assert output.regime == "WARMUP"
            assert output.instability_score == 0.0
            assert not engine.baseline_ready

    def test_update_after_warmup(self):
        """Updates after warmup produce non-zero scores."""
        engine = SIIEngine(baseline_window=20)

        # Warmup phase
        np.random.seed(42)
        baseline_data = np.random.randn(20, 3)
        for i, frame in enumerate(baseline_data):
            engine.update(frame, float(i))

        assert engine.baseline_ready

        # Post-warmup: introduce perturbation
        perturbed = np.random.randn(3) * 3.0
        output = engine.update(perturbed, 20.0)

        assert output.regime != "WARMUP"
        assert output.instability_score > 0.0

    def test_output_structure(self):
        """Output object has all required fields."""
        engine = SIIEngine()

        # Fit baseline
        baseline_data = np.random.randn(50, 2)
        engine.fit_baseline(baseline_data)

        # Process frame
        frame = np.random.randn(2)
        output = engine.update(frame, 50.0)

        # Check all fields
        assert isinstance(output, SIIEngineOutput)
        assert output.timestamp == 50.0
        assert hasattr(output, "instability_score")
        assert hasattr(output, "structural_drift")
        assert hasattr(output, "drift_velocity")
        assert hasattr(output, "transition_pressure")
        assert hasattr(output, "regime")
        assert hasattr(output, "urgency")
        assert hasattr(output, "confidence")
        assert hasattr(output, "gradient_norm")
        assert hasattr(output, "recovery_alignment")

    def test_output_to_dict(self):
        """Output can be serialized to dictionary."""
        engine = SIIEngine()
        baseline_data = np.random.randn(50, 2)
        engine.fit_baseline(baseline_data)

        frame = np.random.randn(2)
        output = engine.update(frame, 50.0)

        d = output.to_dict()
        assert isinstance(d, dict)
        assert "instability_score" in d
        assert "regime" in d
        assert "urgency" in d


class TestStatePersistence:
    """Test state save/restore for long-running deployments."""

    def test_state_snapshot_and_restore(self):
        """State can be saved and restored."""
        engine1 = SIIEngine(baseline_window=20)

        # Fit and process some frames
        baseline_data = np.random.randn(20, 2)
        engine1.fit_baseline(baseline_data)

        for i in range(10):
            frame = np.random.randn(2)
            engine1.update(frame, float(i))

        # Snapshot
        state = engine1.get_state()

        # Create new engine and restore
        engine2 = SIIEngine(baseline_window=20)
        engine2.restore_state(state)

        # Verify state matches
        assert engine2.baseline_ready == engine1.baseline_ready
        assert engine2.frame_count == engine1.frame_count
        assert np.allclose(engine2.baseline.mean, engine1.baseline.mean)


class TestEdgeCases:
    """Test edge cases and robustness."""

    def test_all_nan_frame(self):
        """All-NaN frame is handled gracefully."""
        engine = SIIEngine()
        baseline_data = np.random.randn(50, 2)
        engine.fit_baseline(baseline_data)

        # All NaN
        frame = np.full(2, np.nan)
        output = engine.update(frame, 50.0)

        # Should not crash
        assert output is not None
        assert isinstance(output, SIIEngineOutput)

    def test_single_feature(self):
        """Engine handles single-feature data."""
        engine = SIIEngine()

        baseline_data = np.random.randn(50, 1)
        engine.fit_baseline(baseline_data)

        frame = np.random.randn(1)
        output = engine.update(frame, 50.0)

        assert output.instability_score >= 0.0

    def test_high_dimensional(self):
        """Engine handles high-dimensional data."""
        engine = SIIEngine()

        baseline_data = np.random.randn(50, 100)
        engine.fit_baseline(baseline_data)

        frame = np.random.randn(100)
        output = engine.update(frame, 50.0)

        assert output.instability_score >= 0.0

    def test_singular_covariance(self):
        """Singular covariance is handled via regularization."""
        engine = SIIEngine()

        # Create singular covariance (rank-deficient)
        baseline_data = np.zeros((50, 3))
        baseline_data[:, 0] = np.random.randn(50)
        baseline_data[:, 1] = baseline_data[:, 0] * 2.0  # Linear dependency
        baseline_data[:, 2] = np.random.randn(50)

        # Should not crash
        engine.fit_baseline(baseline_data)
        assert engine.baseline.is_valid()


class TestConsistency:
    """Test internal consistency of the pipeline."""

    def test_score_dominated_by_largest_component(self):
        """Score is dominated by largest component."""
        engine = SIIEngine()

        # Low drift, high velocity
        score1 = engine.compute_instability_score(0.1, 0.5, 0.1)
        # High drift, low velocity
        score2 = engine.compute_instability_score(0.9, 0.05, 0.1)

        # Score2 should be higher due to higher drift weight
        assert score2 > score1

    def test_regime_transitions_are_monotonic(self):
        """As score increases, regime should not decrease in severity."""
        engine = SIIEngine()

        regimes_by_score = []
        for i in range(101):
            score = i / 100.0
            regime = engine.classify_regime(score)
            regimes_by_score.append(regime)

        # Map regimes to severity
        severity = {
            "STABLE": 0,
            "TRANSITION": 1,
            "UNSTABLE": 2,
            "LOCK_IN": 3,
        }

        # Check monotonicity
        for i in range(len(regimes_by_score) - 1):
            s1 = severity[regimes_by_score[i]]
            s2 = severity[regimes_by_score[i + 1]]
            assert s2 >= s1

    def test_urgency_consistent_with_regime(self):
        """Urgency should be consistent with regime severity."""
        engine = SIIEngine()

        urgency_rank = {
            "NOMINAL": 0,
            "WATCH": 1,
            "ALERT": 2,
            "CRITICAL": 3,
        }

        regimes = ["STABLE", "TRANSITION", "UNSTABLE", "LOCK_IN"]

        for regime in regimes:
            urgency = engine.compute_urgency(regime, 0.01)
            # Regime order should match urgency severity
            regime_rank = {r: i for i, r in enumerate(regimes)}[regime]
            assert urgency_rank[urgency] >= regime_rank


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
