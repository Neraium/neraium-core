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
    def risk_score(level: str) -> float:
        # Map risk_level categories to a conservative [0,1] "escalation urgency" score.
        mapping = {"HIGH": 0.95, "ELEVATED": 0.75, "MODERATE": 0.45, "LOW": 0.15}
        return float(mapping.get(level, 0.3))

    def action_cost_tier(action_type: str) -> float:
        # Non-monetary cost tier in [0,1]: higher means more operational overhead.
        cost_map = {
            "maintenance_scheduling": 0.55,
            "failover_routing_planning": 0.45,
            "load_redistribution_planning": 0.40,
            "configuration_sanity_check": 0.20,
            "sensor_calibration_verify": 0.25,
            "throttling_consideration": 0.15,
            "increase_monitoring_cadence": 0.08,
            "urgent_readiness_check": 0.05,
            "human_approval_required": 0.00,
            "continue_observation": 0.00,
        }
        return float(cost_map.get(action_type, 0.20))

    def time_impact_tier(action_type: str) -> float:
        # Time impact tier in [0,1], higher means longer lead-time.
        t_map = {
            "maintenance_scheduling": 0.70,
            "failover_routing_planning": 0.55,
            "load_redistribution_planning": 0.50,
            "configuration_sanity_check": 0.15,
            "sensor_calibration_verify": 0.35,
            "throttling_consideration": 0.20,
            "increase_monitoring_cadence": 0.10,
            "urgent_readiness_check": 0.05,
            "human_approval_required": 0.00,
            "continue_observation": 0.00,
        }
        return float(t_map.get(action_type, 0.25))

    actions: list[dict[str, Any]] = []

    horizon_urgent = time_to_instability is not None and time_to_instability <= 12.0

    # Core, structurally grounded suggestions mapped to typical control-system themes.
    base_risk = risk_score(risk_level)
    if state == "STRUCTURAL_INSTABILITY_OBSERVED":
        actions.append(
            {
                "action_type": "maintenance_scheduling",
                "integration_trigger": "SCHEDULE_MAINTENANCE",
                "rationale": "Observed multi-indicator structural instability suggests risk of near-term transition.",
                "rank_hint": 1,
                "risk": base_risk,
                "cost_tier": action_cost_tier("maintenance_scheduling"),
                "time_impact_tier": time_impact_tier("maintenance_scheduling"),
            }
        )
        actions.append(
            {
                "action_type": "failover_routing_planning",
                "integration_trigger": "FAILOVER_ROUTING_PREP",
                "rationale": "Prepare routing safeguards for coordination loss across infrastructure signals.",
                "rank_hint": 2,
                "risk": base_risk,
                "cost_tier": action_cost_tier("failover_routing_planning"),
                "time_impact_tier": time_impact_tier("failover_routing_planning"),
            }
        )
        actions.append(
            {
                "action_type": "configuration_sanity_check",
                "integration_trigger": "VERIFY_CONTROL_SETPOINTS",
                "rationale": "Structural regime divergence can be amplified by recent configuration changes.",
                "rank_hint": 3,
                "risk": base_risk * 0.8,
                "cost_tier": action_cost_tier("configuration_sanity_check"),
                "time_impact_tier": time_impact_tier("configuration_sanity_check"),
            }
        )
    elif state == "COUPLING_INSTABILITY_OBSERVED":
        actions.append(
            {
                "action_type": "throttling_consideration",
                "integration_trigger": "THROTTLING_PREP",
                "rationale": "Coupling/directional breakdown implies higher coordination volatility; reduce stress until stable.",
                "rank_hint": 1,
                "risk": base_risk * 0.9,
                "cost_tier": action_cost_tier("throttling_consideration"),
                "time_impact_tier": time_impact_tier("throttling_consideration"),
            }
        )
        actions.append(
            {
                "action_type": "load_redistribution_planning",
                "integration_trigger": "LOAD_REDISTRIBUTION_PREP",
                "rationale": "Propagation-aware causal proxy suggests some signals can dominate system motion.",
                "rank_hint": 2,
                "risk": base_risk * 0.85,
                "cost_tier": action_cost_tier("load_redistribution_planning"),
                "time_impact_tier": time_impact_tier("load_redistribution_planning"),
            }
        )
    elif state == "REGIME_SHIFT_OBSERVED":
        actions.append(
            {
                "action_type": "sensor_calibration_verify",
                "integration_trigger": "VERIFY_SENSOR_CALIBRATION",
                "rationale": "A regime shift may reflect operational reconfiguration or instrumentation change; verify both.",
                "rank_hint": 1,
                "risk": base_risk * 0.7,
                "cost_tier": action_cost_tier("sensor_calibration_verify"),
                "time_impact_tier": time_impact_tier("sensor_calibration_verify"),
            }
        )
        actions.append(
            {
                "action_type": "increase_monitoring_cadence",
                "integration_trigger": "ALERTING_CADENCE_UP",
                "rationale": "Regime transitions benefit from faster operator review windows.",
                "rank_hint": 2,
                "risk": base_risk * 0.55,
                "cost_tier": action_cost_tier("increase_monitoring_cadence"),
                "time_impact_tier": time_impact_tier("increase_monitoring_cadence"),
            }
        )
    else:
        actions.append(
            {
                "action_type": "continue_observation",
                "integration_trigger": "NO_ACTUATION",
                "rationale": "System state is consistent with baseline under current analysis.",
                "rank_hint": 1,
                "risk": base_risk * 0.2,
                "cost_tier": action_cost_tier("continue_observation"),
                "time_impact_tier": time_impact_tier("continue_observation"),
            }
        )

    # Horizon-based readiness (still human-in-the-loop).
    if horizon_urgent and risk_level in {"HIGH", "ELEVATED"}:
        actions.append(
            {
                "action_type": "urgent_readiness_check",
                "integration_trigger": "HUMAN_APPROVAL_REQUIRED",
                "rationale": f"Projected time-to-threshold is short (~{round(time_to_instability, 1)} time units). Escalate readiness.",
                "rank_hint": 0,
                "risk": min(1.0, base_risk * 1.05),
                "cost_tier": action_cost_tier("urgent_readiness_check"),
                "time_impact_tier": time_impact_tier("urgent_readiness_check"),
            }
        )

    # Keep recommendation language grounded: do not directly encode control commands.
    actions.append(
        {
            "action_type": "human_approval_required",
            "integration_trigger": "OPERATOR_REVIEW_ONLY",
            "rationale": "Neraium emits recommendations as decision-support; no automated actuation is executed from this layer.",
            "rank_hint": 999,
            "risk": 0.0,
            "cost_tier": action_cost_tier("human_approval_required"),
            "time_impact_tier": time_impact_tier("human_approval_required"),
        }
    )

    # Rank by rank_hint if present; stable tie-break by original order.
    for i, a in enumerate(actions):
        if "rank_hint" not in a:
            a["rank_hint"] = 100 + i
    actions.sort(key=lambda a: (float(a.get("rank_hint", 100.0)),))
    for rank, a in enumerate(actions, start=1):
        a["rank"] = rank
        a.pop("rank_hint", None)

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
    watch_threshold: float | None = None,
    alert_threshold: float | None = None,
    min_history_for_alerts: int = 8,
    require_persistence: bool = True,
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

    effective_watch_threshold = float(watch_threshold) if watch_threshold is not None else 1.5
    effective_alert_threshold = float(alert_threshold) if alert_threshold is not None else 1.5

    persistence = persistence or {}
    history_len = int(float(persistence.get("history_len", 0.0)))
    consecutive_elevated = int(float(persistence.get("consecutive_elevated", 0.0)))
    consecutive_high = int(float(persistence.get("consecutive_high", 0.0)))

    warmup_ok = history_len >= int(min_history_for_alerts)
    sustained = (consecutive_high >= 1) or (consecutive_elevated >= 2)
    if not require_persistence:
        sustained = True

    strong_states = {
        "REGIME_SHIFT_OBSERVED",
        "STRUCTURAL_INSTABILITY_OBSERVED",
        "COUPLING_INSTABILITY_OBSERVED",
    }

    signal_emitted = (state in strong_states) or (
        warmup_ok and sustained and composite_score > effective_alert_threshold
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

    # Optional verbose debugging: keep off by default.
    debug_enabled = os.environ.get("NERAIUM_DEBUG_SII", "0").strip().lower() not in {"0", "false", "no", "off", ""}
    if debug_enabled:
        print(
            "[NERAIUM_DEBUG_SII]"
            f" composite={composite_score:.4f}"
            f" watch_thr={effective_watch_threshold:.4f}"
            f" alert_thr={effective_alert_threshold:.4f}"
            f" history_len={history_len}"
            f" consec_elev={consecutive_elevated}"
            f" consec_high={consecutive_high}"
            f" warmup_ok={warmup_ok}"
            f" sustained={sustained}"
            f" interpreted_state={state}"
            f" signal_emitted={signal_emitted}"
        )

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