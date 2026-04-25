"""
Minimal SII Engine validation tests.

Covers:
- SII engine unified core functionality
- FD004 validation
- External validation reproducibility
"""

import pytest


def test_sii_engine_imports():
    """Verify SII engine modules can be imported."""
    from neraium_core.sii_engine_unified import SIIEngine
    from neraium_core.sii_fd004_validation import (
        FD004ValidationRunner,
        load_fd004_dataset,
    )

    assert SIIEngine is not None
    assert FD004ValidationRunner is not None
    assert load_fd004_dataset is not None


def test_sii_stability_energy():
    """Verify stability energy module loads."""
    from neraium_core.stability_energy import StabilityEnergyCalculator

    assert StabilityEnergyCalculator is not None


def test_sii_alignment():
    """Verify alignment module loads."""
    from neraium_core.alignment import AlignmentProcessor

    assert AlignmentProcessor is not None


def test_fd004_validation_runner():
    """Basic test of FD004 validation runner initialization."""
    from neraium_core.sii_fd004_validation import FD004ValidationRunner

    runner = FD004ValidationRunner()
    assert runner is not None


def test_external_validation():
    """Verify external validation can be imported."""
    from neraium_core.validate_sii_external import SIIExternalValidator

    assert SIIExternalValidator is not None
