"""Operator-facing decision layer derived from engine results (no control paths)."""

from __future__ import annotations

from typing import Any

from neraium_core.product._util import clamp01, safe_float, safe_get


def _severity(state: str, drift: float, pressure: float, pre_inst: float) -> str:
    st = (state or "STABLE").upper()
    if st == "ALERT":
        return "critical" if drift >= 3.2 or pressure >= 0.82 else "high"
    if st == "WATCH":
        return "high" if pressure >= 0.62 or pre_inst >= 0.62 else "medium"
    if drift >= 2.0 or pressure >= 0.55:
        return "medium"
    return "low"


def _actionability(
    transition_ready: bool,
    structural_ready: bool,
    readiness_reason: str,
) -> str:
    if not structural_ready:
        return "blocked"
    if not transition_ready:
        return "limited"
    if readiness_reason in {"partial_subsystems_pending", "insufficient_transition_pressure_history"}:
        return "limited"
    return "full"


def _next_action(state: str, horizon: str, severity: str) -> str:
    st = (state or "STABLE").upper()
    h = (horizon or "LONGER_HORIZON").upper()
    if st == "ALERT":
        return "Schedule focused inspection and review structural divergence within the current operating window."
    if st == "WATCH":
        if h in {"IMMINENT", "NEAR_TERM"}:
            return "Increase monitoring frequency and prepare maintenance options."
        return "Continue observation and validate trend against baseline."
    if severity in {"medium", "high"}:
        return "Maintain routine monitoring and log any operational changes."
    return "Continue normal operations with periodic structural health review."


def _urgency(state: str, horizon: str) -> str:
    st = (state or "STABLE").upper()
    h = (horizon or "").upper()
    if st == "ALERT" or h == "IMMINENT":
        return "critical"
    if st == "WATCH" and h in {"NEAR_TERM", "IMMINENT"}:
        return "high"
    if st == "WATCH":
        return "medium"
    if h in {"NEAR_TERM", "MID_TERM"}:
        return "low"
    return "low"


def _inspection_priority(state: str, pressure: float, lock_in: float) -> int:
    st = (state or "STABLE").upper()
    if st == "ALERT":
        return 5
    if st == "WATCH":
        if pressure >= 0.65 or lock_in >= 0.65:
            return 4
        return 3
    if pressure >= 0.5 or lock_in >= 0.55:
        return 2
    return 1


def _why_now(
    state: str,
    transition_state: str,
    ew_state: str,
    dominant_path: str,
    pressure: float,
) -> str:
    parts = [
        f"Structural state is {state}.",
        f"Transition label is {transition_state}.",
        f"Early warning mode is {ew_state}.",
        f"Trajectory path is {dominant_path}.",
        f"Transition pressure is {pressure:.3f}.",
    ]
    return " ".join(parts)


def _evidence_summary(
    drift: float,
    pressure: float,
    pre_inst: float,
    lock_in: float,
    dominant_path: str,
    transition_state: str,
) -> list[str]:
    return [
        f"Structural drift score {drift:.3f}",
        f"Transition pressure {pressure:.3f}",
        f"Pre-instability signal {pre_inst:.3f}",
        f"Lock-in proxy {lock_in:.3f}",
        f"Dominant trajectory path {dominant_path}",
        f"Transition classification {transition_state}",
    ]


def build_decision_layer(result: dict[str, Any]) -> dict[str, Any]:
    experimental = safe_get(result, "experimental_analytics") or {}
    early_warning = safe_get(result, "early_warning") or {}
    readiness = safe_get(result, "readiness") or {}

    state = str(result.get("state", "STABLE"))
    drift = safe_float(result.get("structural_drift_score"))
    pressure = safe_float(result.get("transition_pressure"))
    transition_state = str(result.get("transition_state", "NONE"))
    conf = clamp01(safe_float(result.get("confidence_score", 0.5)))

    traj = experimental.get("trajectory_analysis") or {}
    if isinstance(traj, dict) and traj.get("available") is False:
        dominant_path = "UNAVAILABLE"
        path_conf = 0.0
    else:
        dominant_path = str((traj or {}).get("dominant_path", "METASTABLE"))
        path_conf = clamp01(safe_float((traj or {}).get("path_confidence", 0.0)))

    horizon = experimental.get("horizon_analysis") or {}
    time_horizon = str((horizon or {}).get("risk_horizon", "LONGER_HORIZON"))
    horizon_score = clamp01(safe_float((horizon or {}).get("horizon_score", 0.0)))

    constraint = experimental.get("constraint_analysis") or {}
    lock_in = clamp01(safe_float((constraint or {}).get("lock_in_score", 0.0)))

    pre_inst = clamp01(safe_float(early_warning.get("pre_instability_score", 0.0)))
    ew_state = str(early_warning.get("early_warning_state", "stable"))

    sev = _severity(state, drift, pressure, pre_inst)
    transition_ready = bool(result.get("transition_outputs_actionable") or result.get("engine_transition_detectable"))
    structural_ready = bool(readiness.get("structural_ready", False))
    readiness_reason = str(readiness.get("reason", ""))

    blended_conf = clamp01(0.55 * conf + 0.35 * path_conf + 0.10 * horizon_score)

    trust_summary = (
        f"Confidence {blended_conf:.2f} with actionability {_actionability(transition_ready, structural_ready, readiness_reason)} "
        f"for current structural assessment."
    )

    return {
        "current_state": state,
        "severity": sev,
        "trajectory": dominant_path,
        "time_horizon": time_horizon,
        "confidence": round(blended_conf, 4),
        "actionability": _actionability(transition_ready, structural_ready, readiness_reason),
        "recommended_next_action": _next_action(state, time_horizon, sev),
        "recommended_urgency": _urgency(state, time_horizon),
        "inspection_priority": _inspection_priority(state, pressure, lock_in),
        "why_this_now": _why_now(state, transition_state, ew_state, dominant_path, pressure),
        "evidence_summary": _evidence_summary(drift, pressure, pre_inst, lock_in, dominant_path, transition_state),
        "trust_summary": trust_summary,
    }
