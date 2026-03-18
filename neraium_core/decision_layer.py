from __future__ import annotations

from typing import Dict, Any


def _risk_level(score: float) -> str:
    if score >= 2.5:
        return "HIGH"
    if score >= 1.5:
        return "MEDIUM"
    return "LOW"


def _signal_strength(score: float, trend: float) -> str:
    if score > 2.5 and trend > 0:
        return "high"
    if score > 1.5:
        return "medium"
    return "low"


def _confidence(score: float, components: dict[str, float]) -> str:
    active = [v for v in components.values() if v > 0]

    if score > 2.0 and len(active) >= 4:
        return "high"
    if score > 1.0:
        return "medium"
    return "low"


def _phase(score: float, trend: float) -> str:
    if score < 0.5:
        return "stable"
    if trend > 0.05:
        return "degrading"
    if trend < -0.05:
        return "recovering"
    return "transitional"


def _operator_message(
    score: float,
    trend: float,
    tti: float | None,
) -> str:
    """
    MUST remain observational.
    No control, no directives, no prescriptions.
    """

    if score < 0.5:
        return (
            "Observed structural relationships remain consistent with a stable operating regime "
            "under current analysis."
        )

    if score < 1.5:
        return (
            "Observed structural patterns indicate mild deviation from baseline relational geometry, "
            "without clear evidence of escalating instability at this time."
        )

    if score < 2.5:
        if trend > 0:
            return (
                "Observed structural patterns indicate increasing deviation from baseline relational "
                "geometry, with emerging instability characteristics under current analysis."
            )
        return (
            "Observed structural patterns indicate sustained deviation from baseline relational geometry, "
            "with mixed indications of progression under current analysis."
        )

    # HIGH
    if tti is not None:
        return (
            "Observed structural patterns indicate significant deviation from baseline relational geometry, "
            "with characteristics consistent with escalating instability under current analysis. "
            f"Current trend projection suggests continued progression over approximately {round(tti, 1)} time units."
        )

    return (
        "Observed structural patterns indicate significant deviation from baseline relational geometry, "
        "with characteristics consistent with escalating instability under current analysis."
    )


def decision_output(
    composite_score: float,
    components: Dict[str, float],
    forecast: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Convert structural analytics into operator-safe decision output.

    This layer:
    - does NOT control anything
    - does NOT prescribe actions
    - only describes observed structural conditions
    """

    trend = float(forecast.get("trend", 0.0))
    tti = forecast.get("time_to_instability")

    risk = _risk_level(composite_score)
    strength = _signal_strength(composite_score, trend)
    confidence = _confidence(composite_score, components)
    phase = _phase(composite_score, trend)

    signal_emitted = composite_score > 1.5

    return {
        "phase": phase,
        "risk_level": risk,
        "signal_emitted": signal_emitted,
        "signal_strength": strength,
        "confidence": confidence,
        "operator_message": _operator_message(
            composite_score,
            trend,
            tti,
        ),
    }