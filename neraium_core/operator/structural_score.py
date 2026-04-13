"""
structural_score.py — Alert score derived from the structural operator.

This module lives in the operator package because it is explicitly downstream
of the operator model.  The data flow is:

    StructuralOperator  →  structural_metrics()  →  alert_score_from_operator()
         (Layer 0)               (Layer 1)                 (Layer 6 interface)

The scalar composite score produced here is a convenience for display and
alerting systems.  It is not the internal model.  The internal model is
the structural operator 𝒮_t = (A_t, Σ_t).

Design:
    Three structural observables are sufficient to derive a principled alert:

    1. operator_drift:  ||A_t - A_0||_F normalized by N
       → measures how much the propagation law has changed

    2. stability_margin:  1 - ρ(A_t)
       → measures distance to the unit circle (negative = unstable)
       → converted to instability score: max(0, 1 - stability_margin)

    3. innovation_growth:  ||Σ_t - Σ_0||_F normalized by N
       → measures dissolution of constraint structure

    These three quantities are grounded in the operator and have clear
    interpretations.  They are NOT ad-hoc components of a pre-specified
    weighted average; they are derived properties of the central object.

    The composite is a weighted combination with weights that reflect
    theoretical precedence:
        - stability_margin has the highest weight because ρ(A_t) has a
          direct stability interpretation (unit circle = bifurcation)
        - operator_drift has medium weight (structural change)
        - innovation_growth has lower weight (secondary signal)

    All three are normalized to [0, 1] before combining.

Relationship to legacy scoring.py:
    scoring.py (top level) computes a weighted average of 8 ad-hoc components
    with weights {1.0, 0.8, 0.95, ...} chosen by intuition.  That file is
    preserved as a compatibility shim for existing callers.  This module is
    the correct replacement when a StructuralOperator is available.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


from .structural_operator import StructuralOperator


@dataclass(frozen=True)
class OperatorMetrics:
    """
    Derived structural metrics from a StructuralOperator pair (current, baseline).

    These are the correct structural observables.  All three are in [0, ∞);
    normalisation to [0, 1] is deferred to the alert layer so raw values
    remain available for comparison across time.

    Attributes:
        operator_drift:      ||A_t - A_0||_F — propagation law change
        stability_instability: max(0, ρ(A_t) - threshold) — proximity to unit circle
                               threshold defaults to 0 (any deviation from full margin)
        innovation_growth:   ||Σ_t - Σ_0||_F — constraint dissolution
        spectral_radius:     ρ(A_t) — raw stability indicator
        stability_margin:    1 - ρ(A_t) — positive = stable, negative = unstable
        coupling_asymmetry:  fraction of directed (asymmetric) coupling in A_t
        n_signals:           dimensionality N of the operator
    """
    operator_drift: float
    stability_instability: float
    innovation_growth: float
    spectral_radius: float
    stability_margin: float
    coupling_asymmetry: float
    n_signals: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "operator_drift": round(self.operator_drift, 6),
            "stability_instability": round(self.stability_instability, 6),
            "innovation_growth": round(self.innovation_growth, 6),
            "spectral_radius": round(self.spectral_radius, 6),
            "stability_margin": round(self.stability_margin, 6),
            "coupling_asymmetry": round(self.coupling_asymmetry, 6),
            "n_signals": self.n_signals,
        }


def structural_metrics(
    current: StructuralOperator,
    baseline: StructuralOperator,
    *,
    stability_threshold: float = 0.8,
) -> OperatorMetrics:
    """
    Extract the three primary structural observables from the operator pair.

    Args:
        current:             Current structural operator 𝒮_t.
        baseline:            Baseline structural operator 𝒮_0.
        stability_threshold: ρ value above which to report instability.
                             Default 0.8: stable systems with ρ < 0.8 contribute
                             zero instability; values above this threshold begin
                             accumulating instability signal.
    """
    N = current.n_signals
    drift_dict = current.drift_from(baseline)

    # Normalise drifts by N so they are comparable across different system sizes.
    # (Frobenius norm of an N×N matrix scales as sqrt(N) for fixed per-element change,
    #  so dividing by N gives per-element average change.)
    scale = max(float(N), 1.0)
    operator_drift = drift_dict["operator_drift"] / scale
    innovation_growth = drift_dict["innovation_drift"] / scale

    rho = current.spectral_radius
    margin = current.stability_margin
    # Instability: how far above the stability threshold is ρ?
    stability_instability = float(max(0.0, rho - float(stability_threshold)))

    return OperatorMetrics(
        operator_drift=float(operator_drift),
        stability_instability=float(stability_instability),
        innovation_growth=float(innovation_growth),
        spectral_radius=float(rho),
        stability_margin=float(margin),
        coupling_asymmetry=float(current.coupling_asymmetry),
        n_signals=int(N),
    )


# Weights reflecting theoretical precedence:
#   stability_instability is the most interpretable (unit circle = bifurcation)
#   operator_drift is the primary structural change signal
#   innovation_growth is secondary (constraint dissolution, harder to normalise)
_DEFAULT_OPERATOR_WEIGHTS: dict[str, float] = {
    "stability_instability": 0.45,
    "operator_drift": 0.40,
    "innovation_growth": 0.15,
}

# Normalization caps: values beyond these are clipped to 1.0 before weighting.
# These are approximate operational calibration constants; they should be
# re-estimated from system-specific baseline data when available.
_DEFAULT_CAPS: dict[str, float] = {
    "stability_instability": 0.2,   # ρ = threshold + 0.2 → full instability signal
    "operator_drift": 2.0,          # per-element avg drift of 2.0 sigma → full drift signal
    "innovation_growth": 1.0,       # per-element avg Σ change of 1.0 → full growth signal
}


def alert_score_from_operator(
    current: StructuralOperator,
    baseline: StructuralOperator,
    *,
    weights: dict[str, float] | None = None,
    caps: dict[str, float] | None = None,
    stability_threshold: float = 0.8,
) -> dict[str, Any]:
    """
    Compute a scalar alert score from the structural operator.

    This is the **interface convenience** function.  It converts operator-space
    structural metrics into a single scalar for alerting and display.  The
    score ∈ [0, 1].

    It is explicitly downstream of the operator.  Do not use this score as
    an internal model state — use OperatorMetrics directly for that.

    Args:
        current:             Current structural operator.
        baseline:            Baseline structural operator.
        weights:             Override default component weights.
        caps:                Override normalization caps.
        stability_threshold: ρ threshold below which ρ contributes zero instability.

    Returns:
        Dict with:
            alert_score:          Scalar ∈ [0, 1]
            components:           Normalized per-component values used in scoring
            raw_metrics:          Un-normalized OperatorMetrics as dict
            dominant_component:   Which component is highest (for explanation)
    """
    w = dict(_DEFAULT_OPERATOR_WEIGHTS)
    if weights:
        w.update({k: float(v) for k, v in weights.items() if k in w})

    c = dict(_DEFAULT_CAPS)
    if caps:
        c.update({k: float(v) for k, v in caps.items() if k in c})

    metrics = structural_metrics(current, baseline, stability_threshold=stability_threshold)

    # Normalise each component to [0, 1] using caps
    normed: dict[str, float] = {
        k: float(min(1.0, max(0.0, getattr(metrics, k) / max(c[k], 1e-9))))
        for k in w
    }

    weight_sum = sum(w.values())
    score = float(sum(normed[k] * w[k] for k in w) / max(weight_sum, 1e-12))
    score = float(min(1.0, max(0.0, score)))

    dominant = max(normed, key=lambda k: normed[k] * w[k])

    return {
        "alert_score": round(score, 6),
        "components": {k: round(normed[k], 6) for k in w},
        "weights": {k: round(w[k], 4) for k in w},
        "raw_metrics": metrics.as_dict(),
        "dominant_component": dominant,
        "source": "structural_operator",
    }
