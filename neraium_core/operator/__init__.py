"""
Structural operator package — the central mathematical object of Neraium.

A complex system under observation is modelled as a VAR(1) process:

    X(t) = A_t · X(t-1) + ε(t),   ε(t) ~ N(0, Σ_t)

where:
    A_t ∈ ℝ^{N×N}  is the propagation operator   (directed dynamic coupling)
    Σ_t ∈ ℝ^{N×N}  is the innovation covariance   (unexplained noise geometry)

The pair 𝒮_t = (A_t, Σ_t) is the *structural operator* at time t.

Structural health means 𝒮_t is statistically consistent with a learned baseline 𝒮_0.
Structural drift means ||A_t - A_0||_F is growing.
Structural instability means ρ(A_t) is approaching or exceeding 1.
Constraint dissolution means rank(Σ_t) is growing (less predictive structure remains).

All other observables in the system — correlation drift, spectral properties, regime
distance, entropy — are derived from or approximated by (A_t, Σ_t).

Public API:
    from neraium_core.operator.structural_operator import (
        StructuralOperator,
        fit_structural_operator,
        operator_drift,
        structural_stability,
    )
"""
from __future__ import annotations

from .structural_operator import (
    StructuralOperator,
    fit_structural_operator,
    operator_drift,
    structural_stability,
)
from .operator_attribution import (
    attribute_operator_drift,
    summarise_structural_change,
)
from .structural_score import (
    OperatorMetrics,
    structural_metrics,
    alert_score_from_operator,
)

__all__ = [
    # Core object
    "StructuralOperator",
    "fit_structural_operator",
    "operator_drift",
    "structural_stability",
    # Attribution (operator-space)
    "attribute_operator_drift",
    "summarise_structural_change",
    # Alert interface (downstream of operator — not the model itself)
    "OperatorMetrics",
    "structural_metrics",
    "alert_score_from_operator",
]
