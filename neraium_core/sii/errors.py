from __future__ import annotations


class SIIError(Exception):
    """Base error for Systemic Infrastructure Intelligence."""


class SIIValidationError(SIIError):
    """Raised when telemetry input cannot be normalized safely."""


class SIIConfigurationError(SIIError):
    """Raised when SII configuration is invalid."""

