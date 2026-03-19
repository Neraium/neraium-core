from __future__ import annotations

from typing import Any


def _risk_level(score: float) -> str:
    if score >= 2.5:
        return "HIGH"
    if score >= 1.5:
        return "ELEVATED"
    if score >= 0.7:
        return "MODERATE"
    return "LOW"


def _signal_strength(score: float, trend: float) -> str:
    if score >= 2.5 and trend > 0:
        return "high"
    if score >= 1.5:
        return "medium"
    return "low"


def _confidence(components: dict[str, float]) -> str:
    active = [v for v in components.values() if abs(v) > 1e-6]

    if len(active) >= 5:
        return "high"
    if len(active) >= 3:
        return "medium"
    return "low"


def _phase(score: float, trend: float) -> str:
    if score < 0.7:
        return "stable"
    if trend > 0.05:
        return "degrading"
    if trend < -0.05:
        return "recovering"
    return "transitional"


def _interpret_state(
    relational_drift: float,
    regime_drift: float,
    directional: float,
    spectral: float,
) -> str:
    # Structure changed, but current structure still fits known regime
    if relational_drift > 1.2 and regime_drift < 0.8:
        return "REGIME_SHIFT_OBSERVED"

    # Structure changed and also departs from assigned regime
    if relational_drift > 1.2 and regime_drift >= 0.8:
        return "STRUCTURAL_INSTABILITY_OBSERVED"

    # Elevated interaction instability even without strong global drift
    if directional > 1.0 or spectral > 1.2:
        return "COUPLING_INSTABILITY_OBSERVED"

    return "NOMINAL_STRUCTURE"


def _operator_message(
    state: str,
    trend: float,
    time_to_instability: float | None,
) -> str:
    """
    Strictly observational language.
    No control, no directives, no operational commands.
    """

    if state == "STRUCTURAL_INSTABILITY_OBSERVED":
        if time_to_instability is not None:
            return (
                "Observed structural relationships are diverging from previously seen "
                "system patterns. Current configuration exhibits elevated instability "
                "characteristics under current analysis, with continued progression "
                f"projected over approximately {round(time_to_instability, 1)} time units."
            )
        return (
            "Observed structural relationships are diverging from previously seen "
            "system patterns. Current configuration exhibits elevated instability "
            "characteristics under current analysis."
        )

    if state == "REGIME_SHIFT_OBSERVED":
        return (
            "Observed system relationships indicate a transition into a different "
            "structural regime. Current behavior differs from prior baseline but "
            "remains internally consistent under current analysis."
        )

    if state == "COUPLING_INSTABILITY_OBSERVED":
        return (
            "Observed coupling and directional interactions between signals show "
            "elevated variability. System coordination patterns appear less stable "
            "than baseline under current analysis."
        )

    if trend > 0.0:
        return (
            "Observed structural patterns remain broadly consistent with previously "
            "seen behavior, with limited upward movement in current instability signals."
        )

    return (
        "Observed structural patterns are consistent with previously seen baseline "
        "behavior under current analysis. No significant structural deviation detected."
    )


def decision_output(
    composite_score: float,
    components: dict[str, float],
    forecast: dict[str, Any],
) -> dict[str, Any]:
    """
    Convert structural analytics into operator-safe decision output.

    This layer is observational only.
    It does not prescribe action or imply control authority.
    """
    relational_drift = float(components.get("relational_drift", 0.0))
    regime_drift = float(components.get("regime_drift", 0.0))
    directional = float(components.get("directional_divergence", 0.0))
    spectral = float(components.get("spectral", 0.0))

    trend = float(forecast.get("trend", 0.0))
    time_to_instability = forecast.get("ar1_time_to_instability")
    if time_to_instability is None:
        time_to_instability = forecast.get("time_to_instability")

    state = _interpret_state(
        relational_drift=relational_drift,
        regime_drift=regime_drift,
        directional=directional,
        spectral=spectral,
    )

    risk_level = _risk_level(composite_score)
    signal_strength = _signal_strength(composite_score, trend)
    confidence = _confidence(components)
    phase = _phase(composite_score, trend)

    signal_emitted = composite_score >= 1.5 or state in {
        "REGIME_SHIFT_OBSERVED",
        "STRUCTURAL_INSTABILITY_OBSERVED",
        "COUPLING_INSTABILITY_OBSERVED",
    }

    operator_message = _operator_message(
        state=state,
        trend=trend,
        time_to_instability=time_to_instability,
    )

    return {
        "phase": phase,
        "risk_level": risk_level,
        "signal_emitted": signal_emitted,
        "signal_strength": signal_strength,
        "confidence": confidence,
        "operator_message": operator_message,
        "interpreted_state": state,
    }


def evaluate_signal(timeseries: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    """
    Backward-compatible decision helper used by the test suite.

    This function evaluates a short time-series summary and returns an
    operator-safe decision-like payload. It is observational only and does
    not prescribe control actions.
    """
    if not timeseries:
        return {
            "signal_emitted": False,
            "signal_strength": "low",
            "confidence": "low",
            "operator_message": "No material structural instability detected.",
            "reason": [],
        }

    peak_instability = float(config.get("peak_instability", 1.5))

    phases = [str(row.get("phase", "") or "").lower().strip() for row in timeseries]
    all_stable = bool(phases) and all(p == "stable" for p in phases)

    values = []
    for row in timeseries:
        try:
            values.append(float(row.get("composite_instability", 0.0)))
        except (TypeError, ValueError):
            values.append(0.0)

    latest_instability = float(values[-1])
    max_instability = max(values) if values else 0.0

    # Consistency rule: require a small run of elevated instability at the end
    # of the series, rather than a single noisy spike.
    required_cycles = min(3, len(values))
    high_cut = peak_instability * 0.85

    consecutive_high = 0
    for v in reversed(values):
        if v >= high_cut:
            consecutive_high += 1
        else:
            break

    consistency_ok = consecutive_high >= required_cycles

    # If everything is explicitly stable, keep the message minimal.
    if all_stable:
        return {
            "signal_emitted": False,
            "signal_strength": "low",
            "confidence": "low",
            "operator_message": "No material structural instability detected.",
            "reason": [],
        }

    if max_instability < peak_instability:
        return {
            "signal_emitted": False,
            "signal_strength": "low",
            "confidence": "low",
            "operator_message": "No material structural instability detected.",
            "reason": [],
        }

    if consistency_ok:
        # Strength tier based on how close we are to the configured peak.
        if latest_instability >= peak_instability:
            signal_strength = "high"
        else:
            signal_strength = "medium"

        confidence = "high" if consecutive_high >= required_cycles else "medium"
        return {
            "signal_emitted": True,
            "signal_strength": signal_strength,
            "confidence": confidence,
            "operator_message": (
                "Elevated structural instability characteristics observed; "
                "human review for confirmation is appropriate."
            ),
            "reason": [],
        }

    # Suppress signal when the configured peak was hit but the evidence was not consistent.
    return {
        "signal_emitted": False,
        "signal_strength": "low",
        "confidence": "low",
        "reason": [
            "Signal suppressed because it did not satisfy consistency requirements.",
        ],
        "operator_message": (
            "Observed instability did not satisfy consistency requirements for emission."
        ),
    }