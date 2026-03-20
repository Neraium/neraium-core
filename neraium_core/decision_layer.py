from __future__ import annotations

import os
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


def _confidence_categorical_from_score(score: float) -> str:
    """Map a [0, 1] confidence score to categorical for backward compatibility."""
    if score >= 0.7:
        return "high"
    if score >= 0.4:
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


def _response_recommendations_enabled() -> bool:
    # Default to disabled to preserve existing "observational only" behavior.
    return os.environ.get("NERAIUM_AUTONOMOUS_RESPONSE", "0").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
        "",
    )


def _response_recommendations(
    *,
    state: str,
    risk_level: str,
    time_to_instability: float | None,
    scenario_projections: Any,
) -> list[dict[str, Any]]:
    """
    Operator-facing recommendations only (no control authority / no actuation).
    """
    actions: list[dict[str, Any]] = []

    horizon_urgent = time_to_instability is not None and time_to_instability <= 12.0

    # Core, structurally grounded suggestions mapped to typical control-system themes.
    if state == "STRUCTURAL_INSTABILITY_OBSERVED":
        actions.append(
            {
                "action_type": "maintenance_scheduling",
                "integration_trigger": "SCHEDULE_MAINTENANCE",
                "rationale": "Observed multi-indicator structural instability suggests risk of near-term transition.",
            }
        )
        actions.append(
            {
                "action_type": "failover_routing_planning",
                "integration_trigger": "FAILOVER_ROUTING_PREP",
                "rationale": "Prepare routing safeguards for coordination loss across infrastructure signals.",
            }
        )
        actions.append(
            {
                "action_type": "configuration_sanity_check",
                "integration_trigger": "VERIFY_CONTROL_SETPOINTS",
                "rationale": "Structural regime divergence can be amplified by recent configuration changes.",
            }
        )
    elif state == "COUPLING_INSTABILITY_OBSERVED":
        actions.append(
            {
                "action_type": "throttling_consideration",
                "integration_trigger": "THROTTLING_PREP",
                "rationale": "Coupling/directional breakdown implies higher coordination volatility; reduce stress until stable.",
            }
        )
        actions.append(
            {
                "action_type": "load_redistribution_planning",
                "integration_trigger": "LOAD_REDISTRIBUTION_PREP",
                "rationale": "Propagation-aware causal proxy suggests some signals can dominate system motion.",
            }
        )
    elif state == "REGIME_SHIFT_OBSERVED":
        actions.append(
            {
                "action_type": "sensor_calibration_verify",
                "integration_trigger": "VERIFY_SENSOR_CALIBRATION",
                "rationale": "A regime shift may reflect operational reconfiguration or instrumentation change; verify both.",
            }
        )
        actions.append(
            {
                "action_type": "increase_monitoring_cadence",
                "integration_trigger": "ALERTING_CADENCE_UP",
                "rationale": "Regime transitions benefit from faster operator review windows.",
            }
        )
    else:
        actions.append(
            {
                "action_type": "continue_observation",
                "integration_trigger": "NO_ACTUATION",
                "rationale": "System state is consistent with baseline under current analysis.",
            }
        )

    # Horizon-based readiness (still human-in-the-loop).
    if horizon_urgent and risk_level in {"HIGH", "ELEVATED"}:
        actions.append(
            {
                "action_type": "urgent_readiness_check",
                "integration_trigger": "HUMAN_APPROVAL_REQUIRED",
                "rationale": f"Projected time-to-threshold is short (~{round(time_to_instability, 1)} time units). Escalate readiness.",
            }
        )

    # Keep recommendation language grounded: do not directly encode control commands.
    actions.append(
        {
            "action_type": "human_approval_required",
            "integration_trigger": "OPERATOR_REVIEW_ONLY",
            "rationale": "Neraium emits recommendations as decision-support; no automated actuation is executed from this layer.",
        }
    )

    return actions


def _interpret_state(
    relational_drift: float,
    regime_drift: float,
    directional: float,
    spectral: float,
    early_warning: float,
    entropy: float,
    trend: float,
    persistence: dict[str, float] | None,
) -> str:
    """
    Interpret structural state with clear separation of conditions.

    - REGIME_SHIFT_OBSERVED: relational geometry has moved into a different regime
      without strong directional/coupling breakdown or sustained instability.
    - COUPLING_INSTABILITY_OBSERVED: directional/interaction (and spectral) breakdown
      dominates; depends more strongly on directional + spectral evidence.
    - STRUCTURAL_INSTABILITY_OBSERVED: relational drift + entropy + regime drift
      with sustained, multi-indicator confirmation.
    """
    persistence = persistence or {}
    history_len = float(persistence.get("history_len", 0.0))
    consecutive_elevated = float(persistence.get("consecutive_elevated", 0.0))
    consecutive_high = float(persistence.get("consecutive_high", 0.0))
    rolling_mean = float(persistence.get("rolling_mean", 0.0))

    # Warmup/hysteresis: avoid jumping straight into strong instability classes.
    if history_len < 8:
        if relational_drift > 1.2:
            return "REGIME_SHIFT_OBSERVED"
        return "NOMINAL_STRUCTURE"

    motion = relational_drift > 1.2
    regime_departure = regime_drift >= 0.8
    # Coupling instability: depends more strongly on directional and spectral interaction breakdown.
    directional_breakdown = directional > 1.0
    spectral_breakdown = spectral > 1.2
    coupling_instability = directional_breakdown or spectral_breakdown

    bounded_persistence = consecutive_high < 2 and consecutive_elevated < 4 and rolling_mean < 2.0
    no_degradation_trend = abs(trend) <= 0.06
    sustained = consecutive_high >= 2 or consecutive_elevated >= 5 or rolling_mean >= 2.2
    # Slightly lower bar for coupling: directional/spectral breakdown can persist with less elevation.
    sustained_coupling = consecutive_high >= 1 or consecutive_elevated >= 3 or rolling_mean >= 1.7

    # Coupling instability first: sustained directional/spectral breakdown (interaction-focused).
    if coupling_instability and sustained_coupling:
        return "COUPLING_INSTABILITY_OBSERVED"

    # Regime shift only: structure moved, no strong coupling/directional breakdown, bounded.
    if motion and not coupling_instability and bounded_persistence and no_degradation_trend:
        return "REGIME_SHIFT_OBSERVED"

    # Constrained coherence: motion with correction-like activity but no sustained breakdown.
    correction_present = early_warning > 0.9 or coupling_instability
    if motion and correction_present and bounded_persistence and no_degradation_trend:
        return "COHERENCE_UNDER_CONSTRAINT"

    # Structural instability: depends more strongly on relational drift + entropy + regime drift.
    entropy_elevated = entropy > 0.8
    structural_evidence = (motion and regime_departure) or (motion and entropy_elevated)
    multi_indicator = structural_evidence or (regime_departure and entropy_elevated) or (
        motion and coupling_instability and early_warning > 1.1
    )
    degrading = trend > 0.06

    if sustained and multi_indicator and (degrading or regime_departure):
        return "STRUCTURAL_INSTABILITY_OBSERVED"

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

    if state == "COHERENCE_UNDER_CONSTRAINT":
        return (
            "Observed structure is moving under apparent correction activity. "
            "Current relationships remain bounded and internally coherent under current analysis, "
            "without clear evidence of coordinated structural breakdown."
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
    components: dict[str, Any],
    forecast: dict[str, Any],
    *,
    confidence_score: float | None = None,
    classification_stability: float | None = None,
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
    early_warning = float(components.get("early_warning", 0.0))
    entropy = float(components.get("entropy", 0.0))

    trend = float(forecast.get("trend", 0.0))
    persistence = forecast.get("persistence") if isinstance(forecast.get("persistence"), dict) else None
    time_to_instability = forecast.get("ar1_time_to_instability")
    if time_to_instability is None:
        time_to_instability = forecast.get("time_to_instability")

    state = _interpret_state(
        relational_drift=relational_drift,
        regime_drift=regime_drift,
        directional=directional,
        spectral=spectral,
        early_warning=early_warning,
        entropy=entropy,
        trend=trend,
        persistence=persistence,
    )

    risk_level = _risk_level(composite_score)
    signal_strength = _signal_strength(composite_score, trend)
    conf_val = confidence_score if confidence_score is not None else components.get("_confidence_score")
    if conf_val is not None:
        try:
            confidence = _confidence_categorical_from_score(float(conf_val))
        except (TypeError, ValueError):
            confidence = _confidence({k: v for k, v in components.items() if not k.startswith("_")})
    else:
        confidence = _confidence({k: v for k, v in components.items() if not k.startswith("_")})
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

    out: dict[str, Any] = {
        "phase": phase,
        "risk_level": risk_level,
        "signal_emitted": signal_emitted,
        "signal_strength": signal_strength,
        "confidence": confidence,
        "operator_message": operator_message,
        "interpreted_state": state,
    }

    if _response_recommendations_enabled():
        scenario_projections = forecast.get("scenario_projections") if isinstance(forecast, dict) else None
        out["response_recommendations"] = _response_recommendations(
            state=state,
            risk_level=risk_level,
            time_to_instability=time_to_instability,
            scenario_projections=scenario_projections,
        )
        out["autonomous_response_enabled"] = True
    else:
        out["autonomous_response_enabled"] = False

    if classification_stability is not None:
        out["classification_stability"] = round(classification_stability, 4)
    return out


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