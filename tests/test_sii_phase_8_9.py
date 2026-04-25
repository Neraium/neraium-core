"""
Tests for PHASE 8+9: Comprehensive validation and white paper output.

Tests:
- Phase 8: Mathematical, architectural, and debug visibility consistency
- Phase 9: White paper output generation
"""

import pytest
import numpy as np
from neraium_core.sii_comprehensive_validation import (
    Phase8ConsistencyValidator,
    Phase9WhitePaperGenerator,
    ComprehensiveValidationRunner,
)
from neraium_core.sii_fd004_validation import DetectionMetrics
from neraium_core.sii_weight_sensitivity import WeightVariation


@pytest.fixture
def synthetic_units():
    """Create synthetic unit data for testing."""
    units = {}
    num_units = 3
    num_cycles = 200
    num_sensors = 14

    for unit_id in range(num_units):
        # Synthetic sensor data with drift in last 50 cycles
        sensor_data = np.random.randn(num_cycles, num_sensors)

        # Add progressive drift in last 50 cycles to simulate failure
        for cycle in range(num_cycles - 50, num_cycles):
            progress = (cycle - (num_cycles - 50)) / 50.0
            sensor_data[cycle, :] += progress * 2.0

        timestamps = np.arange(num_cycles, dtype=float)
        failure_cycle = num_cycles - 1

        units[f"unit_{unit_id}"] = (sensor_data, timestamps, failure_cycle)

    return units


class TestPhase8ConsistencyValidator:
    """Tests for PHASE 8 consistency validation."""

    def test_validate_mathematical_consistency(self, synthetic_units):
        """Test that mathematical consistency check runs without error."""
        is_consistent, violations = Phase8ConsistencyValidator.validate_mathematical_consistency(
            synthetic_units, baseline_window=50
        )

        # Should pass with synthetic data
        assert isinstance(is_consistent, bool)
        assert isinstance(violations, list)

    def test_validate_architectural_consistency(self):
        """Test that architectural consistency check runs without error."""
        is_consistent, violations = Phase8ConsistencyValidator.validate_architectural_consistency()

        assert isinstance(is_consistent, bool)
        assert isinstance(violations, list)

    def test_validate_debug_visibility(self):
        """Test that debug visibility check runs without error."""
        is_visible, violations = Phase8ConsistencyValidator.validate_debug_visibility()

        assert isinstance(is_visible, bool)
        assert isinstance(violations, list)


class TestPhase9WhitePaperGenerator:
    """Tests for PHASE 9 white paper generation."""

    def test_generate_key_claims(self):
        """Test key claims generation."""
        # Create mock metrics
        baseline_metrics = DetectionMetrics(
            method_name="SII",
            detection_cycles=[50, 75, 100],
            lead_times=[125, 100, 150],
            detected_count=3,
            total_count=3,
        )

        threshold_metrics = DetectionMetrics(
            method_name="Threshold",
            detection_cycles=[80, 95, 120],
            lead_times=[95, 80, 130],
            detected_count=3,
            total_count=3,
        )

        claims = Phase9WhitePaperGenerator.generate_key_claims(
            baseline_metrics, threshold_metrics
        )

        assert "key_claim" in claims
        assert "positioning" in claims
        assert "lead_time_improvement_cycles" in claims
        assert "lead_time_improvement_percent" in claims
        assert float(claims["lead_time_improvement_cycles"]) > 0

    def test_generate_robustness_claim(self):
        """Test weight sensitivity robustness claim generation."""
        baseline = WeightVariation(
            drift_weight=0.40,
            velocity_weight=0.35,
            pressure_weight=0.25,
            detection_rate=95.2,
            mean_lead_time=156.0,
            median_lead_time=143.0,
            std_lead_time=67.0,
            label="baseline",
        )

        variations = [
            baseline,
            WeightVariation(
                drift_weight=0.48,
                velocity_weight=0.30,
                pressure_weight=0.22,
                detection_rate=94.0,
                mean_lead_time=158.0,
                median_lead_time=145.0,
                std_lead_time=68.0,
                label="drift_weight_+20%",
            ),
        ]

        claim = Phase9WhitePaperGenerator.generate_robustness_claim(baseline, variations)

        assert "claim" in claim
        assert "detection_rate_robustness" in claim
        assert "lead_time_robustness" in claim

    def test_generate_method_summary(self):
        """Test method summary generation."""
        summary = Phase9WhitePaperGenerator.generate_method_summary()

        assert "title" in summary
        assert "core_equation" in summary
        assert "I_t" in summary["core_equation"]
        assert "regime_definition" in summary
        assert "urgency_definition" in summary
        assert "abstract" in summary


class TestComprehensiveValidationRunner:
    """Tests for comprehensive validation orchestration."""

    def test_run_full_validation(self, synthetic_units):
        """Test full validation run."""
        result = ComprehensiveValidationRunner.run_full_validation(
            synthetic_units, baseline_window=50
        )

        assert result is not None
        assert isinstance(result.mathematical_consistency_verified, bool)
        assert isinstance(result.architectural_consistency_verified, bool)
        assert isinstance(result.debug_visibility_enabled, bool)
        assert isinstance(result.constraint_violations, list)
        assert isinstance(result.deployment_readiness, bool)

    def test_validation_result_has_whitepaper_claims(self, synthetic_units):
        """Test that validation result includes white paper claims."""
        result = ComprehensiveValidationRunner.run_full_validation(synthetic_units)

        assert "whitepaper_claims" in result.__dict__
        assert "method" in result.whitepaper_claims
        assert "key_claims" in result.whitepaper_claims


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
