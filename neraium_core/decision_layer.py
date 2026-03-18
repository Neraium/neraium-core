from __future__ import annotations

from typing import Dict, Any


def _risk_level(score: float) -> str:
    if score > 2.5:
        return "HIGH"
    if score > 1.5:
        return "ELEVATED"
    if score > 0.7:
        return "MODERATE"
    return "LOW"


def _confidence(components: Dict[str, float]) -> float:
    # Confidence increases when multiple signals agree
    active = [v for v in components.values() if abs(v) > 1e-6]
    return round(min(1.0, len(active) / 6.0), 3)


def _interpret_state(
    relational_drift: float,
    regime_drift: float,
    directional: float,
    spectral: float,
) -> str:
    """
    Core logic separating:
    - regime change vs instability
    """

    # --- Case 1: Regime shift (structure changed but matches known pattern)
    if relational_drift > 1.2 and regime_drift < 0.8:
        return "REGIME_SHIFT_OBSERVED"

    # --- Case 2: True instability (structure deviates from known regime)
    if relational_drift > 1.2 and regime_drift >= 0.8:
        return "STRUCTURAL_INSTABILITY_OBSERVED"

    # --- Case 3: directional/coupling instability
    if directional > 1.0 or spectral > 1.2:
        return "COUPLING_INSTABILITY_OBSERVED"

    return "NOMINAL_STRUCTURE"


def _operator_message(
    state: str,
    risk: str,
    drift: float,
    regime_drift: float,
) -> str:
    """
    IMPORTANT:
    - No control language
    - No prescriptions
    - Observational only (legally safe)
    """

    if state == "STRUCTURAL_INSTABILITY_OBSERVED":
        return (
            "Observed structural relationships are diverging from previously seen system patterns. "
            "Current configuration exhibits elevated instability characteristics."
        )

    if state == "REGIME_SHIFT_OBSERVED":
        return (
            "Observed system relationships indicate a transition into a different structural regime. "
            "Current behavior differs from prior baseline but remains internally consistent."
        )

    if state == "COUPLING_INSTABILITY_OBSERVED":
        return (
            "Observed coupling and directional interactions between signals show elevated variability. "
            "System coordination patterns appear less stable than baseline."
        )

    return (
        "System relationships remain consistent with previously observed baseline behavior. "
        "No significant structural deviation detected."
    )


def decision_output(
    composite_score: float,
    components: Dict[str, float],
    forecast: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Final decision layer.

    Produces:
    - risk level
    - interpreted state
    - operator-safe message
    """

    relational_drift = float(components.get("relational_drift", 0.0))
    regime_drift = float(components.get("regime_drift", 0.0))
    directional = float(components.get("directional_divergence", 0.0))
    spectral = float(components.get("spectral", 0.0))

    state = _interpret_state(
        relational_drift=relational_drift,
        regime_drift=regime_drift,
        directional=directional,
        spectral=spectral,
    )

    risk = _risk_level(composite_score)

    confidence = _confidence(components)

    message = _operator_message(
        state=state,
        risk=risk,
        drift=relational_drift,
        regime_drift=regime_drift,
    )

    return {
        "status": {
            "state": state,
            "risk_level": risk,
            "confidence": confidence,
            "operator_message": message,
        },
        "scores": {
            "composite_instability": round(float(composite_score), 4),
            "relational_drift": round(relational_drift, 4),
            "regime_drift": round(regime_drift, 4),
        },
        "forecast": forecast,
    }