"""
Unit tests for Stability Energy + Gradient Field Layer.

Tests cover:
1. Stable state near baseline has low energy
2. State far from baseline has higher energy
3. Motion away from baseline creates positive escape_alignment
4. Motion toward baseline creates positive recovery_alignment
5. Warmup / insufficient history does not crash
"""

from __future__ import annotations

import numpy as np
import pytest

from neraium_core.stability_energy import (
    compute_alignment,
    compute_energy,
    compute_gradient,
    compute_stability_energy,
    insufficient_history_payload,
    label_stability_direction,
)


class TestEnergyComputation:
    """Test energy landscape computation."""

    def test_energy_at_baseline_is_zero(self):
        """State at baseline mean should have zero energy."""
        dim = 5
        z = np.zeros(dim)
        mu_B = np.zeros(dim)
        sigma_B_inv = np.eye(dim)

        energy = compute_energy(z, mu_B, sigma_B_inv)
        assert energy == pytest.approx(0.0, abs=1e-9)

    def test_energy_increases_with_distance(self):
        """Energy should increase as state moves away from baseline."""
        dim = 3
        mu_B = np.array([0.0, 0.0, 0.0])
        sigma_B_inv = np.eye(dim)

        z_near = np.array([0.1, 0.0, 0.0])
        z_far = np.array([1.0, 0.0, 0.0])

        energy_near = compute_energy(z_near, mu_B, sigma_B_inv)
        energy_far = compute_energy(z_far, mu_B, sigma_B_inv)

        assert energy_far > energy_near
        assert energy_near > 0.0

    def test_energy_is_positive(self):
        """Energy should always be non-negative."""
        dim = 4
        mu_B = np.array([1.0, 2.0, 3.0, 4.0])
        sigma_B_inv = np.eye(dim)

        z = np.array([2.5, 2.5, 2.5, 2.5])
        energy = compute_energy(z, mu_B, sigma_B_inv)

        assert energy >= 0.0


class TestGradientComputation:
    """Test gradient and force computations."""

    def test_gradient_at_baseline_is_zero(self):
        """Gradient at baseline should be zero."""
        dim = 3
        z = np.zeros(dim)
        mu_B = np.zeros(dim)
        sigma_B_inv = np.eye(dim)

        grad = compute_gradient(z, mu_B, sigma_B_inv)

        np.testing.assert_array_almost_equal(grad, np.zeros(dim))

    def test_gradient_points_away_from_baseline(self):
        """Gradient should point away from baseline (direction of increasing energy)."""
        dim = 2
        mu_B = np.array([0.0, 0.0])
        sigma_B_inv = np.eye(dim)
        z = np.array([1.0, 0.0])

        grad = compute_gradient(z, mu_B, sigma_B_inv)

        # Gradient should be in same direction as (z - mu_B)
        assert grad[0] > 0.0  # Positive x component
        assert grad[1] == pytest.approx(0.0, abs=1e-9)  # Zero y component

    def test_recovery_force_points_toward_baseline(self):
        """Recovery force (negative gradient) should point toward baseline."""
        dim = 2
        mu_B = np.array([0.0, 0.0])
        sigma_B_inv = np.eye(dim)
        z = np.array([1.0, 1.0])

        grad = compute_gradient(z, mu_B, sigma_B_inv)
        recovery_force = -grad

        # Recovery force should point toward baseline (negative direction)
        assert recovery_force[0] < 0.0
        assert recovery_force[1] < 0.0


class TestAlignmentComputation:
    """Test alignment between velocity and directions."""

    def test_alignment_same_direction(self):
        """Vectors in same direction should have alignment close to 1.0."""
        v = np.array([1.0, 0.0, 0.0])
        d = np.array([1.0, 0.0, 0.0])

        alignment = compute_alignment(v, d)
        assert alignment == pytest.approx(1.0, abs=1e-6)

    def test_alignment_opposite_direction(self):
        """Vectors in opposite direction should have alignment close to -1.0."""
        v = np.array([1.0, 0.0, 0.0])
        d = np.array([-1.0, 0.0, 0.0])

        alignment = compute_alignment(v, d)
        assert alignment == pytest.approx(-1.0, abs=1e-6)

    def test_alignment_orthogonal(self):
        """Orthogonal vectors should have alignment close to 0.0."""
        v = np.array([1.0, 0.0, 0.0])
        d = np.array([0.0, 1.0, 0.0])

        alignment = compute_alignment(v, d)
        assert alignment == pytest.approx(0.0, abs=1e-6)

    def test_alignment_is_bounded(self):
        """Alignment should always be in [-1, 1]."""
        v = np.random.randn(5)
        d = np.random.randn(5)

        alignment = compute_alignment(v, d)
        assert -1.0 <= alignment <= 1.0


class TestStabilityDirectionLabeling:
    """Test direction labeling logic."""

    def test_escaping_stability_label(self):
        """High escape alignment + positive energy_delta -> ESCAPING_STABILITY."""
        label, interp = label_stability_direction(
            escape_alignment=0.6,
            recovery_alignment=0.1,
            energy_delta=0.5,
        )
        assert label == "ESCAPING_STABILITY"
        assert "escape" in interp.lower() or "increasing" in interp.lower()

    def test_recovering_label(self):
        """High recovery alignment + negative energy_delta -> RECOVERING."""
        label, interp = label_stability_direction(
            escape_alignment=0.1,
            recovery_alignment=0.6,
            energy_delta=-0.5,
        )
        assert label == "RECOVERING"
        assert "recover" in interp.lower()

    def test_sideways_transition_label(self):
        """Low escape alignment -> SIDEWAYS_TRANSITION."""
        label, interp = label_stability_direction(
            escape_alignment=0.1,
            recovery_alignment=0.2,
            energy_delta=0.0,
        )
        assert label == "SIDEWAYS_TRANSITION"
        assert "sideways" in interp.lower() or "across" in interp.lower()

    def test_degrading_label(self):
        """Neither escape nor recovery dominant -> DEGRADING."""
        label, interp = label_stability_direction(
            escape_alignment=0.3,
            recovery_alignment=0.3,
            energy_delta=0.5,
        )
        assert label == "DEGRADING"
        assert "degrad" in interp.lower() or "elevated" in interp.lower()


class TestStabilityEnergyComputation:
    """Test comprehensive stability energy computation."""

    def test_warmup_insufficient_history(self):
        """Insufficient history returns safe defaults."""
        z_current = np.array([1.0, 2.0, 3.0])
        mu_baseline = np.array([0.0, 0.0, 0.0])
        sigma_baseline = np.eye(3)

        # First frame with no history
        result = compute_stability_energy(
            z_current=z_current,
            z_previous=None,
            mu_baseline=mu_baseline,
            sigma_baseline=sigma_baseline,
            prev_energy=None,
            prev_energy_velocity=None,
        )

        assert result["direction_label"] == "INSUFFICIENT_HISTORY"
        assert result["energy"] == pytest.approx(14.0, abs=1e-6)  # sum of squares
        assert "interpretation" in result

    def test_stable_state_near_baseline(self):
        """Stable state close to baseline should have low energy."""
        dim = 5
        mu_baseline = np.zeros(dim)
        sigma_baseline = np.eye(dim)

        z_stable = np.random.randn(dim) * 0.05  # Small perturbation
        result = compute_stability_energy(
            z_current=z_stable,
            z_previous=None,
            mu_baseline=mu_baseline,
            sigma_baseline=sigma_baseline,
        )

        assert result["energy"] < 0.1
        assert result["instability_gradient_norm"] < 0.5

    def test_state_far_from_baseline_high_energy(self):
        """State far from baseline should have higher energy."""
        dim = 5
        mu_baseline = np.zeros(dim)
        sigma_baseline = np.eye(dim)

        z_far = np.ones(dim) * 2.0  # Far from baseline
        result = compute_stability_energy(
            z_current=z_far,
            z_previous=None,
            mu_baseline=mu_baseline,
            sigma_baseline=sigma_baseline,
        )

        assert result["energy"] > 10.0

    def test_escape_trajectory(self):
        """Motion away from baseline creates positive escape_alignment."""
        dim = 3
        mu_baseline = np.array([0.0, 0.0, 0.0])
        sigma_baseline = np.eye(dim)

        z_prev = np.array([0.5, 0.0, 0.0])
        z_curr = np.array([1.0, 0.0, 0.0])  # Moving away

        result = compute_stability_energy(
            z_current=z_curr,
            z_previous=z_prev,
            mu_baseline=mu_baseline,
            sigma_baseline=sigma_baseline,
            prev_energy=compute_energy(z_prev, mu_baseline, sigma_baseline),
        )

        assert result["escape_alignment"] > 0.5
        assert result["recovery_alignment"] < 0.3

    def test_recovery_trajectory(self):
        """Motion toward baseline creates positive recovery_alignment."""
        dim = 3
        mu_baseline = np.array([0.0, 0.0, 0.0])
        sigma_baseline = np.eye(dim)

        z_prev = np.array([1.0, 0.0, 0.0])
        z_curr = np.array([0.5, 0.0, 0.0])  # Moving toward baseline

        result = compute_stability_energy(
            z_current=z_curr,
            z_previous=z_prev,
            mu_baseline=mu_baseline,
            sigma_baseline=sigma_baseline,
            prev_energy=compute_energy(z_prev, mu_baseline, sigma_baseline),
        )

        assert result["recovery_alignment"] > 0.5
        assert result["escape_alignment"] < 0.3

    def test_energy_dynamics_smooth(self):
        """Energy velocity should be smooth due to EMA."""
        dim = 3
        mu_baseline = np.zeros(dim)
        sigma_baseline = np.eye(dim)

        z_prev = np.array([0.5, 0.0, 0.0])
        z_curr = np.array([1.0, 0.0, 0.0])

        prev_energy = compute_energy(z_prev, mu_baseline, sigma_baseline)

        result1 = compute_stability_energy(
            z_current=z_curr,
            z_previous=z_prev,
            mu_baseline=mu_baseline,
            sigma_baseline=sigma_baseline,
            prev_energy=prev_energy,
            prev_energy_velocity=None,
            energy_velocity_alpha=0.8,
        )

        # First frame energy velocity
        ev1 = result1["energy_velocity"]

        # Second frame with same velocity
        z_next = np.array([1.5, 0.0, 0.0])
        result2 = compute_stability_energy(
            z_current=z_next,
            z_previous=z_curr,
            mu_baseline=mu_baseline,
            sigma_baseline=sigma_baseline,
            prev_energy=compute_energy(z_curr, mu_baseline, sigma_baseline),
            prev_energy_velocity=ev1,
            energy_velocity_alpha=0.8,
        )

        # Energy velocity should be smoothed, not equal to energy_delta
        assert result2["energy_velocity"] != pytest.approx(result2["energy_delta"], abs=0.01)

    def test_singular_covariance_handling(self):
        """Should handle singular covariance gracefully via regularization."""
        dim = 3
        mu_baseline = np.zeros(dim)
        # Singular covariance (rank-deficient)
        sigma_baseline = np.zeros((dim, dim))
        sigma_baseline[0, 0] = 1.0

        z_current = np.array([0.5, 0.5, 0.5])

        result = compute_stability_energy(
            z_current=z_current,
            z_previous=None,
            mu_baseline=mu_baseline,
            sigma_baseline=sigma_baseline,
        )

        # Should not crash, should return reasonable values
        assert isinstance(result["energy"], float)
        assert result["energy"] >= 0.0

    def test_insufficient_history_payload(self):
        """Insufficient history payload should be safe."""
        payload = insufficient_history_payload()

        assert payload["energy"] == 0.0
        assert payload["energy_delta"] == 0.0
        assert payload["direction_label"] == "INSUFFICIENT_HISTORY"
        assert "interpretation" in payload


class TestStreamingCompatibility:
    """Test streaming-safe behavior."""

    def test_no_full_history_required(self):
        """Should not require loading full history at once."""
        dim = 10
        mu_baseline = np.zeros(dim)
        sigma_baseline = np.eye(dim)

        # Simulate streaming: one frame at a time
        z_prev = np.random.randn(dim)
        z_curr = np.random.randn(dim)

        result = compute_stability_energy(
            z_current=z_curr,
            z_previous=z_prev,
            mu_baseline=mu_baseline,
            sigma_baseline=sigma_baseline,
            prev_energy=compute_energy(z_prev, mu_baseline, sigma_baseline),
        )

        assert "energy" in result
        assert "direction_label" in result

    def test_state_carryover_between_frames(self):
        """State should be carryable between frame calls."""
        dim = 3
        mu_baseline = np.zeros(dim)
        sigma_baseline = np.eye(dim)

        # Frame 1
        z1 = np.array([0.5, 0.0, 0.0])
        result1 = compute_stability_energy(
            z_current=z1,
            z_previous=None,
            mu_baseline=mu_baseline,
            sigma_baseline=sigma_baseline,
        )
        state1 = result1.get("history", {})

        # Frame 2 using saved state
        z2 = np.array([1.0, 0.0, 0.0])
        result2 = compute_stability_energy(
            z_current=z2,
            z_previous=z1,
            mu_baseline=mu_baseline,
            sigma_baseline=sigma_baseline,
            prev_energy=state1.get("prev_energy"),
            prev_energy_velocity=state1.get("prev_energy_velocity"),
        )

        assert result2["energy"] > result1["energy"]
        assert result2["direction_label"] != "INSUFFICIENT_HISTORY"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
