"""
early_warning.py — Critical Slowing Down (CSD) indicators.

Theory:
    Near a bifurcation, a dynamical system loses its ability to recover from
    perturbations.  This manifests as:

        1. Increasing variance:          Var(ε(t)) → ∞  as stability margin → 0
        2. Increasing autocorrelation:   lag-1 ACF(ε(t)) → 1 as ρ(A) → 1

    These are the two canonical CSD precursors (Scheffer et al. 2009,
    Dakos et al. 2012).

Important implementation note:
    CSD analysis should be applied to the **innovation residuals**
        ε(t) = X(t) - A · X(t-1)
    rather than to raw observations X(t).

    Applying CSD to X(t) conflates three effects:
        (a) genuine structural change (rising ρ(A))
        (b) level drift (rising mean — e.g. temperature creep)
        (c) measurement noise

    Residuals ε(t) isolate effect (a).  As ρ(A_t) approaches 1, the system
    becomes a near-random-walk in ε-space; variance and autocorrelation of ε
    both increase.

    The function `early_warning_metrics` accepts either raw observations
    (backward-compatible) or pre-computed residuals.  When a StructuralOperator
    is available, always prefer the residual path.

Public API:
    early_warning_metrics(observations) — backward-compatible; operates on raw X
    csd_on_residuals(residuals)         — primary path; operates on ε(t)
    early_warning_from_operator(observations, operator) — full pipeline
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from neraium_core.operator.structural_operator import StructuralOperator

ArrayLike = Any


def _lag1_acf_per_signal(data: np.ndarray) -> np.ndarray:
    """Per-signal lag-1 autocorrelation. Returns array of length N."""
    T, N = data.shape
    result = np.zeros(N, dtype=float)
    if T < 2:
        return result
    with np.errstate(invalid="ignore", divide="ignore"):
        for idx in range(N):
            result[idx] = np.corrcoef(data[:-1, idx], data[1:, idx])[0, 1]
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


def csd_on_residuals(residuals: ArrayLike) -> dict[str, float]:
    """
    Compute CSD indicators on innovation residuals ε(t).

    This is the **primary** early-warning path when a StructuralOperator is
    available.  The residuals ε(t) = X(t) - A·X(t-1) have mean zero and
    covariance Σ under the null model; deviations in their variance and
    autocorrelation directly indicate instability.

    Args:
        residuals: (T-1, N) array of innovation residuals from
                   StructuralOperator.innovation_residuals().

    Returns:
        Dict with:
            variance:            Mean per-signal residual variance (should ≈ mean(diag(Σ)))
            lag1_autocorrelation: Mean per-signal lag-1 ACF of residuals
            max_variance:        Max per-signal variance (identifies loudest residual)
            max_lag1_acf:        Max per-signal lag-1 ACF (identifies most persistent residual)
            csd_pressure:        Composite CSD indicator ∈ [0, 1]; rising = approaching bifurcation
            source:              "residuals"
    """
    data = np.asarray(residuals, dtype=float)
    if data.ndim != 2:
        raise ValueError(f"residuals must be 2D (T, N); got shape {data.shape}")
    safe = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    variances = np.var(safe, axis=0)
    lag1 = _lag1_acf_per_signal(safe)

    mean_var = float(np.mean(variances)) if variances.size else 0.0
    mean_acf = float(np.mean(lag1)) if lag1.size else 0.0
    max_var = float(np.max(variances)) if variances.size else 0.0
    max_acf = float(np.max(lag1)) if lag1.size else 0.0

    # CSD pressure: lag-1 ACF ∈ [0, 1] is the most direct stability indicator.
    # Variance normalized by max observed does not generalize across assets;
    # ACF is unit-free and directly comparable.
    csd_pressure = float(np.clip(0.7 * max(mean_acf, 0.0) + 0.3 * min(1.0, mean_var), 0.0, 1.0))

    return {
        "variance": round(mean_var, 6),
        "lag1_autocorrelation": round(mean_acf, 6),
        "max_variance": round(max_var, 6),
        "max_lag1_acf": round(max_acf, 6),
        "csd_pressure": round(csd_pressure, 6),
        "source": "residuals",
    }


def early_warning_metrics(observations: ArrayLike) -> dict[str, float]:
    """
    Compute CSD indicators on raw observations X(t).

    This is the **backward-compatible** path.  It produces the same fields as
    the original implementation.  When a StructuralOperator is available,
    prefer `csd_on_residuals(operator.innovation_residuals(observations))` for
    more precise CSD estimation.

    Args:
        observations: (T, N) array.

    Returns:
        Dict with:
            variance:            Mean per-signal variance
            lag1_autocorrelation: Mean per-signal lag-1 ACF
    """
    data = np.asarray(observations, dtype=float)
    if data.ndim != 2:
        raise ValueError("Expected 2D observations")
    safe = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    variances = np.var(safe, axis=0)
    lag1 = _lag1_acf_per_signal(safe)

    return {
        "variance": float(np.mean(variances) if variances.size else 0.0),
        "lag1_autocorrelation": float(np.mean(lag1) if lag1.size else 0.0),
    }


def early_warning_from_operator(
    observations: ArrayLike,
    operator: "StructuralOperator",
) -> dict[str, float]:
    """
    Full CSD pipeline: compute innovation residuals from the structural operator
    and apply CSD analysis to them.

    This is the preferred entry point when an operator is available.

    Args:
        observations: (T, N) array used to compute residuals.
        operator:     A fitted StructuralOperator.

    Returns:
        CSD metrics dict from csd_on_residuals(), plus stability properties
        from the operator for context.
    """
    residuals = operator.innovation_residuals(observations)
    if residuals.shape[0] < 2:
        # Not enough data for residual CSD; fall back to raw observations
        raw = early_warning_metrics(observations)
        raw["source"] = "raw_fallback"
        return raw

    csd = csd_on_residuals(residuals)
    # Add operator stability context so the caller has a complete picture
    csd["spectral_radius"] = round(operator.spectral_radius, 6)
    csd["stability_margin"] = round(operator.stability_margin, 6)
    return csd
