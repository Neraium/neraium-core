from __future__ import annotations

from typing import Any


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _extract_action_parts(candidate: Any) -> tuple[str | None, str | None, int | None]:
    if isinstance(candidate, str):
        text = candidate.strip()
        return (text or None, None, None)
    if not isinstance(candidate, dict):
        return (None, None, None)
    action = str(candidate.get("action", "")).strip() or None
    target = str(candidate.get("target", "")).strip() or None
    priority_raw = candidate.get("priority")
    try:
        priority = int(priority_raw) if priority_raw is not None else None
    except (TypeError, ValueError):
        priority = None
    return (action, target, priority)


def compute_decision_confidence(
    *,
    causal_analysis: dict[str, Any] | None,
    risk_assessment: dict[str, Any] | None,
    attribution: dict[str, Any] | None,
    source: dict[str, bool],
) -> float:
    causal = causal_analysis or {}
    risk = risk_assessment or {}
    attr = attribution or {}

    top_hypotheses = causal.get("top_hypotheses") if isinstance(causal.get("top_hypotheses"), list) else []
    top_h = top_hypotheses[0] if top_hypotheses and isinstance(top_hypotheses[0], dict) else {}
    hypothesis_conf = _clamp01(_safe_float(top_h.get("confidence"), 0.0))
    robustness = _clamp01(_safe_float(top_h.get("robustness"), 0.0))
    counterfactual = _clamp01(_safe_float(top_h.get("counterfactual_strength"), 0.0))

    trend = str(risk.get("projected_near_term_trend", "uncertain")).lower()
    risk_score = _clamp01(_safe_float(risk.get("risk_score"), 0.0))
    projected_score = _clamp01(_safe_float(risk.get("projected_score"), risk_score))
    trend_clarity = 1.0 if trend in {"increasing", "decreasing"} else 0.4
    risk_strength = _clamp01(0.55 * max(risk_score, projected_score) + 0.45 * trend_clarity)

    top_sensors = attr.get("top_sensors") if isinstance(attr.get("top_sensors"), list) else []
    contribution_scores = attr.get("contribution_scores") if isinstance(attr.get("contribution_scores"), dict) else {}
    top_sensor_score = 0.0
    if top_sensors and isinstance(top_sensors[0], dict):
        top_sensor_score = _clamp01(_safe_float(top_sensors[0].get("score"), 0.0))
    concentration = _clamp01(min(1.0, len(top_sensors) / 3.0))
    localization = _clamp01(0.55 * top_sensor_score + 0.45 * concentration)
    if contribution_scores:
        strongest = max((_safe_float(v) for v in contribution_scores.values()), default=0.0)
        localization = _clamp01(max(localization, strongest))

    converging_count = int(sum(1 for value in source.values() if value))
    converging_evidence = _clamp01(converging_count / 4.0)

    confidence = _clamp01(
        0.30 * hypothesis_conf
        + 0.20 * robustness
        + 0.10 * counterfactual
        + 0.20 * risk_strength
        + 0.12 * localization
        + 0.08 * converging_evidence
    )
    return round(float(confidence), 4)


def summarize_decision_reason(
    *,
    action: str | None,
    target: str | None,
    confidence: float,
    evidence: list[dict[str, Any]],
    source: dict[str, bool],
    risk_assessment: dict[str, Any] | None,
    available: bool,
    unavailable_reason: str,
) -> str:
    if not available:
        return unavailable_reason

    trend = "uncertain"
    if isinstance(risk_assessment, dict):
        trend = str(risk_assessment.get("projected_near_term_trend", "uncertain"))
    support = []
    if source.get("from_causal_analysis"):
        support.append("causal ranking")
    if source.get("from_operator_guidance"):
        support.append("operator guidance")
    if source.get("from_risk_assessment"):
        support.append(f"risk trend ({trend})")
    if source.get("from_attribution"):
        support.append("attribution localization")

    support_text = ", ".join(support[:3]) if support else "available structural evidence"
    target_text = f" on {target}" if target else ""
    return f"Prioritized '{action}'{target_text} from {support_text}; confidence={confidence:.2f}."


def build_decision_fallback(
    *,
    reason: str,
    evidence: list[dict[str, Any]] | None = None,
    source: dict[str, bool] | None = None,
    confidence: float = 0.0,
) -> dict[str, Any]:
    source_flags = source or {
        "from_causal_analysis": False,
        "from_operator_guidance": False,
        "from_risk_assessment": False,
        "from_attribution": False,
    }
    return {
        "action": None,
        "target": None,
        "priority": None,
        "confidence": round(_clamp01(confidence), 4),
        "reason": reason,
        "evidence": list(evidence or []),
        "source": source_flags,
        "status": {
            "available": False,
            "reason": reason,
        },
    }


def resolve_best_action(
    *,
    attribution: dict[str, Any] | None,
    regime_memory: dict[str, Any] | None,
    risk_assessment: dict[str, Any] | None,
    operator_guidance: dict[str, Any] | None,
    causal_analysis: dict[str, Any] | None,
    readiness: dict[str, Any] | None = None,
) -> dict[str, Any]:
    attr = attribution or {}
    risk = risk_assessment or {}
    guidance = operator_guidance or {}
    causal = causal_analysis or {}

    if str(causal.get("status", "")).lower() in {"warming_up", "insufficient_history"}:
        return build_decision_fallback(
            reason="Warmup window is incomplete; continue monitoring until sufficient evidence accumulates.",
            evidence=[{"signal": "causal_analysis.status", "value": causal.get("status", "warming_up")}],
        )

    if isinstance(readiness, dict) and readiness.get("ready") is False:
        reason = str(readiness.get("reason", "Engine not ready for a single prioritized action."))
        return build_decision_fallback(
            reason=reason,
            evidence=[{"signal": "readiness.ready", "value": False}],
        )

    source = {
        "from_causal_analysis": False,
        "from_operator_guidance": False,
        "from_risk_assessment": isinstance(risk, dict) and bool(risk),
        "from_attribution": isinstance(attr, dict) and bool(attr),
    }
    evidence: list[dict[str, Any]] = []
    action: str | None = None
    target: str | None = None
    priority: int | None = None

    best_next = causal.get("best_next_action")
    top_h = {}
    top_hypotheses = causal.get("top_hypotheses") if isinstance(causal.get("top_hypotheses"), list) else []
    if top_hypotheses and isinstance(top_hypotheses[0], dict):
        top_h = top_hypotheses[0]

    best_conf = _safe_float((best_next or {}).get("confidence") if isinstance(best_next, dict) else None, 0.0)
    if best_next is not None and best_conf >= 0.25:
        action, target, priority = _extract_action_parts(best_next)
        source["from_causal_analysis"] = bool(action)
        if action:
            evidence.append(
                {
                    "signal": "causal_analysis.best_next_action",
                    "value": action,
                    "confidence": round(_clamp01(best_conf), 4),
                }
            )

    if action is None:
        seq = causal.get("recommended_sequence") if isinstance(causal.get("recommended_sequence"), list) else []
        if seq:
            action, target, priority = _extract_action_parts(seq[0])
            if action:
                source["from_causal_analysis"] = True
                evidence.append({"signal": "causal_analysis.recommended_sequence[0]", "value": action})

    if action is None:
        first_target = str(guidance.get("first_inspection_target", "")).strip()
        rec_actions = guidance.get("recommended_actions") if isinstance(guidance.get("recommended_actions"), list) else []
        if rec_actions:
            action = str(rec_actions[0]).strip() or None
            target = None if first_target in {"", "none", "unknown"} else first_target
            priority = 1
            source["from_operator_guidance"] = bool(action)
            if action:
                evidence.append({"signal": "operator_guidance.recommended_actions[0]", "value": action})

    if action is None:
        top_sensors = attr.get("top_sensors") if isinstance(attr.get("top_sensors"), list) else []
        if top_sensors and isinstance(top_sensors[0], dict):
            sensor = str(top_sensors[0].get("sensor", "")).strip()
            sensor_score = _safe_float(top_sensors[0].get("score"), 0.0)
            if sensor:
                action = "Inspect localized subsystem"
                target = sensor
                priority = 2
                source["from_attribution"] = True
                evidence.append(
                    {
                        "signal": "attribution.top_sensors[0]",
                        "value": sensor,
                        "score": round(_clamp01(sensor_score), 4),
                    }
                )

    if action is None:
        fallback_evidence = []
        if isinstance(causal, dict) and causal:
            fallback_evidence.append({"signal": "causal_analysis.status", "value": causal.get("status", "unknown")})
        if isinstance(risk, dict) and risk:
            fallback_evidence.append({"signal": "risk_assessment.projected_near_term_trend", "value": risk.get("projected_near_term_trend", "uncertain")})
        return build_decision_fallback(
            reason="Insufficient converging evidence for a single prioritized action.",
            evidence=fallback_evidence,
            source=source,
            confidence=0.05,
        )

    if top_h:
        evidence.append(
            {
                "signal": "causal_analysis.top_hypotheses[0]",
                "value": top_h.get("hypothesis", "unknown"),
                "confidence": round(_clamp01(_safe_float(top_h.get("confidence"), 0.0)), 4),
            }
        )
    if isinstance(risk, dict) and risk:
        evidence.append(
            {
                "signal": "risk_assessment.projected_near_term_trend",
                "value": risk.get("projected_near_term_trend", "uncertain"),
                "risk_score": round(_clamp01(_safe_float(risk.get("risk_score"), 0.0)), 4),
            }
        )
    if isinstance(attr, dict) and attr:
        top_sensors = attr.get("top_sensors") if isinstance(attr.get("top_sensors"), list) else []
        if top_sensors and isinstance(top_sensors[0], dict):
            evidence.append(
                {
                    "signal": "attribution.top_sensors[0]",
                    "value": top_sensors[0].get("sensor"),
                }
            )

    confidence = compute_decision_confidence(
        causal_analysis=causal,
        risk_assessment=risk,
        attribution=attr,
        source=source,
    )
    reason = summarize_decision_reason(
        action=action,
        target=target,
        confidence=confidence,
        evidence=evidence,
        source=source,
        risk_assessment=risk,
        available=True,
        unavailable_reason="",
    )
    return {
        "action": action,
        "target": target,
        "priority": priority,
        "confidence": confidence,
        "reason": reason,
        "evidence": evidence,
        "source": source,
        "status": {
            "available": True,
            "reason": "decision_available",
        },
    }


__all__ = [
    "resolve_best_action",
    "compute_decision_confidence",
    "summarize_decision_reason",
    "build_decision_fallback",
]
