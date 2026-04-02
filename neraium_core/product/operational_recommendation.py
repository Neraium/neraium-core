"""Operational recommendation text (advisory only, no actuation)."""

from __future__ import annotations

from typing import Any

from neraium_core.product._util import clamp01, safe_float, safe_get


def _status(state: str) -> str:
    st = (state or "STABLE").upper()
    if st == "ALERT":
        return "escalate"
    if st == "WATCH":
        return "observe"
    return "ok"


def _action(state: str, horizon: str) -> str:
    st = (state or "STABLE").upper()
    h = (horizon or "").upper()
    if st == "ALERT":
        return "Prioritize engineering review and inspection planning for this asset."
    if st == "WATCH":
        if h in {"IMMINENT", "NEAR_TERM"}:
            return "Tighten review cadence and validate sensors against baseline."
        return "Continue scheduled monitoring and document any process changes."
    return "Retain standard operating review rhythm."


def _target_scope(origin_scope: str) -> str:
    s = (origin_scope or "LOCAL").upper()
    if s == "SYSTEMIC":
        return "fleet_and_asset"
    if s == "MULTI_SUBSYSTEM":
        return "asset_multi_subsystem"
    return "asset_local"


def build_operational_recommendation(result: dict[str, Any]) -> dict[str, Any]:
    experimental = safe_get(result, "experimental_analytics") or {}
    early_warning = safe_get(result, "early_warning") or {}
    readiness = safe_get(result, "readiness") or {}

    state = str(result.get("state", "STABLE"))
    horizon_block = experimental.get("horizon_analysis") or {}
    horizon = str((horizon_block or {}).get("risk_horizon", "LONGER_HORIZON"))

    hierarchy = experimental.get("hierarchy_analysis") or {}
    origin_scope = str((hierarchy or {}).get("origin_scope", "LOCAL"))

    decision = result.get("decision_layer") or {}
    urgency = str(decision.get("recommended_urgency", "low")) if isinstance(decision, dict) else "low"
    priority = int(decision.get("inspection_priority", 1)) if isinstance(decision, dict) else 1

    drift = safe_float(result.get("structural_drift_score"))
    pressure = safe_float(result.get("transition_pressure"))
    pre_inst = clamp01(safe_float(early_warning.get("pre_instability_score", 0.0)))

    transition_ready = bool(result.get("transition_outputs_actionable", False))
    rec_conf = clamp01(
        0.4 * safe_float(result.get("confidence_score", 0.5))
        + 0.35 * (1.0 if transition_ready else 0.5)
        + 0.25 * (1.0 if readiness.get("structural_ready") else 0.4)
    )

    rationale = (
        f"State {state}, horizon {horizon}, drift {drift:.3f}, pressure {pressure:.3f}, "
        f"pre-instability {pre_inst:.3f}. Scope {origin_scope}."
    )

    supporting = [
        f"interpreted_state={result.get('interpreted_state')}",
        f"transition_state={result.get('transition_state')}",
        f"early_warning_state={early_warning.get('early_warning_state')}",
        f"readiness_reason={readiness.get('reason')}",
    ]

    return {
        "status": _status(state),
        "recommended_action": _action(state, horizon),
        "recommended_target": _target_scope(origin_scope),
        "priority": priority,
        "urgency": urgency,
        "recommendation_confidence": round(rec_conf, 4),
        "rationale": rationale,
        "supporting_evidence": supporting,
        "operator_note": "Advisory only. No automated control. Confirm with site procedures before any change.",
    }
