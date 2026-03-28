from __future__ import annotations

from typing import Any, Mapping

CANONICAL_SCHEMA_VERSION = "2026-03-28"

OPERATOR_BOUNDARY_NOTE = (
    "Recommendations are advisory outputs intended to support, not replace, "
    "qualified operator judgment and site-specific procedures."
)

REQUIRED_FIELDS = {
    "schema_version",
    "timestamp",
    "cycle",
    "attribution",
    "regime_memory",
    "risk_assessment",
    "causal_analysis",
    "operational_recommendation",
    "confidence",
    "explanation_text",
    "events",
}

OPTIONAL_FIELDS = {
    "session",
    "aliases",
    "history_id",
    "persisted_at",
    "customer_id",
    "run_id",
}

_RISK_RANK = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "UNKNOWN": -1}


def _normalize_top_drivers(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, dict):
            driver = str(item.get("driver") or item.get("name") or "unknown")
            score = item.get("score", item.get("contribution", 0.0))
        elif isinstance(item, (tuple, list)) and item:
            driver = str(item[0])
            score = item[1] if len(item) > 1 else 0.0
        else:
            driver = str(item)
            score = 0.0
        try:
            numeric_score = float(score)
        except (TypeError, ValueError):
            numeric_score = 0.0
        out.append({"driver": driver, "score": round(numeric_score, 6)})
    return out[:5]


def _coerce_recommendation(raw_result: Mapping[str, Any]) -> dict[str, Any]:
    raw_decision = raw_result.get("decision") if isinstance(raw_result.get("decision"), dict) else {}
    raw_recommendation = (
        raw_result.get("operational_recommendation")
        if isinstance(raw_result.get("operational_recommendation"), dict)
        else {}
    )

    state = str(
        raw_decision.get("state")
        or raw_result.get("state")
        or raw_result.get("action_state")
        or "UNKNOWN"
    ).upper()
    action = raw_recommendation.get("recommended_action") or raw_decision.get("action") or raw_decision.get("resolved_action")
    target = raw_recommendation.get("recommended_target") or raw_decision.get("target")

    status_block = raw_recommendation.get("status") if isinstance(raw_recommendation.get("status"), dict) else {}
    legacy_status = raw_decision.get("status") if isinstance(raw_decision.get("status"), dict) else {}
    available = bool(status_block.get("available", legacy_status.get("available", state != "UNKNOWN")))

    confidence = raw_recommendation.get("recommendation_confidence")
    if confidence is None:
        confidence = raw_decision.get("confidence", raw_result.get("confidence", raw_result.get("confidence_score", 0.0)))
    try:
        recommendation_confidence = round(max(0.0, min(float(confidence), 1.0)), 6)
    except (TypeError, ValueError):
        recommendation_confidence = 0.0

    supporting_evidence = raw_recommendation.get("supporting_evidence")
    if not isinstance(supporting_evidence, list):
        attribution = raw_result.get("attribution") if isinstance(raw_result.get("attribution"), dict) else {}
        supporting_evidence = _normalize_top_drivers(attribution.get("top_drivers"))

    rationale = str(
        raw_recommendation.get("rationale")
        or raw_decision.get("reason")
        or "Recommendation available from converging structural evidence."
    )

    return {
        "status": {
            "available": available,
            "advisory": True,
            "reason": str(status_block.get("reason") or legacy_status.get("reason") or ("recommendation_available" if available else "recommendation_unavailable")),
        },
        "recommended_action": str(action) if action not in (None, "") else None,
        "recommended_target": str(target) if target not in (None, "") else None,
        "priority": raw_recommendation.get("priority"),
        "recommendation_confidence": recommendation_confidence,
        "urgency": raw_recommendation.get("urgency") if raw_recommendation.get("urgency") is not None else raw_decision.get("urgency"),
        "rationale": rationale,
        "supporting_evidence": supporting_evidence,
        "operator_note": str(raw_recommendation.get("operator_note") or OPERATOR_BOUNDARY_NOTE),
    }


def _build_legacy_decision_alias(recommendation: Mapping[str, Any]) -> dict[str, Any]:
    status = recommendation.get("status") if isinstance(recommendation.get("status"), dict) else {}
    return {
        "state": "ALERT" if bool(status.get("available")) else "UNKNOWN",
        "action": recommendation.get("recommended_action") or "none",
        "reason": recommendation.get("rationale") or status.get("reason") or "legacy_alias",
        "confidence": recommendation.get("recommendation_confidence", 0.0),
        "status": {
            "available": bool(status.get("available")),
            "reason": status.get("reason", "recommendation_available"),
        },
        "deprecated": True,
    }


def _normalize_risk(raw_result: Mapping[str, Any]) -> dict[str, Any]:
    risk = raw_result.get("risk_assessment") if isinstance(raw_result.get("risk_assessment"), dict) else {}
    risk_level = str(risk.get("risk_level") or raw_result.get("risk_level") or "UNKNOWN").upper()
    trend = str(risk.get("trend") or risk.get("projected_near_term_trend") or raw_result.get("trend") or "UNKNOWN").upper()
    latest_instability = risk.get("latest_instability", raw_result.get("latest_instability", 0.0))
    try:
        latest_instability = float(latest_instability)
    except (TypeError, ValueError):
        latest_instability = 0.0
    return {
        "risk_level": risk_level,
        "trend": trend,
        "latest_instability": round(latest_instability, 6),
    }


def _normalize_confidence(raw_result: Mapping[str, Any], recommendation: Mapping[str, Any]) -> float:
    recommendation_confidence = recommendation.get("recommendation_confidence")
    try:
        if recommendation_confidence is not None:
            return round(max(0.0, min(float(recommendation_confidence), 1.0)), 6)
    except (TypeError, ValueError):
        pass

    for key in ("confidence", "confidence_score"):
        if key in raw_result:
            try:
                return round(max(0.0, min(float(raw_result[key]), 1.0)), 6)
            except (TypeError, ValueError):
                continue
    return 0.0


def derive_product_events(current: Mapping[str, Any], previous: Mapping[str, Any] | None = None) -> list[str]:
    events: list[str] = []
    risk = current.get("risk_assessment") if isinstance(current.get("risk_assessment"), dict) else {}
    recommendation = (
        current.get("operational_recommendation")
        if isinstance(current.get("operational_recommendation"), dict)
        else {}
    )
    status = recommendation.get("status") if isinstance(recommendation.get("status"), dict) else {}

    level = str(risk.get("risk_level", "UNKNOWN")).upper()
    latest_instability = float(risk.get("latest_instability", 0.0) or 0.0)
    trend = str(risk.get("trend", "UNKNOWN")).upper()

    if latest_instability >= 0.9 and level in {"LOW", "MEDIUM"}:
        events.append("early_instability_detected")

    prev_level = "UNKNOWN"
    prev_instability = 0.0
    if isinstance(previous, Mapping):
        prev_risk = previous.get("risk_assessment") if isinstance(previous.get("risk_assessment"), dict) else {}
        prev_level = str(prev_risk.get("risk_level", "UNKNOWN")).upper()
        prev_instability = float(prev_risk.get("latest_instability", 0.0) or 0.0)

    if _RISK_RANK.get(level, -1) > _RISK_RANK.get(prev_level, -1):
        events.append("risk_escalated")

    if bool(status.get("available")):
        events.append("recommendation_available")

    next_action = str(recommendation.get("recommended_action", "")).lower()
    if level == "HIGH" or "inspect" in next_action or "diagn" in next_action:
        events.append("inspection_recommended")

    if trend in {"RISING", "UP"} or latest_instability > prev_instability + 0.1:
        events.append("deterioration_detected")

    return sorted(set(events))


def build_canonical_output(
    raw_result: Mapping[str, Any],
    *,
    cycle: int,
    run_id: str | None = None,
    customer_id: str | None = None,
    previous: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    attribution = raw_result.get("attribution") if isinstance(raw_result.get("attribution"), dict) else {}
    causal_analysis = raw_result.get("causal_analysis") if isinstance(raw_result.get("causal_analysis"), dict) else {}
    regime_memory = raw_result.get("regime_memory") if isinstance(raw_result.get("regime_memory"), dict) else {}

    recommendation = _coerce_recommendation(raw_result)
    canonical = {
        "schema_version": CANONICAL_SCHEMA_VERSION,
        "timestamp": str(raw_result.get("timestamp", "")),
        "cycle": int(cycle),
        "session": {"run_id": run_id, "customer_id": customer_id},
        "attribution": {
            "top_drivers": _normalize_top_drivers(attribution.get("top_drivers")),
            "group_contributions": attribution.get("group_contributions", {}),
        },
        "regime_memory": regime_memory,
        "risk_assessment": _normalize_risk(raw_result),
        "causal_analysis": causal_analysis,
        "operational_recommendation": recommendation,
        "confidence": _normalize_confidence(raw_result, recommendation),
        "explanation_text": str(raw_result.get("explanation_text") or raw_result.get("explanation") or ""),
    }
    canonical["events"] = derive_product_events(canonical, previous=previous)

    aliases: dict[str, Any] = {
        "legacy_decision": _build_legacy_decision_alias(recommendation),
    }
    if "decision" in raw_result and isinstance(raw_result.get("decision"), dict):
        aliases["decision"] = raw_result.get("decision")
    if "regime_memory_state" in raw_result:
        aliases["regime_memory_state"] = raw_result.get("regime_memory_state")
    if "explanation" in raw_result:
        aliases["explanation"] = raw_result.get("explanation")
    if aliases:
        canonical["aliases"] = aliases

    missing = REQUIRED_FIELDS - set(canonical.keys())
    if missing:
        raise ValueError(f"Canonical output missing required fields: {sorted(missing)}")

    return canonical


def is_canonical_output(payload: Mapping[str, Any]) -> bool:
    keys = set(payload.keys())
    return REQUIRED_FIELDS.issubset(keys) and keys.issubset(REQUIRED_FIELDS | OPTIONAL_FIELDS)


__all__ = [
    "CANONICAL_SCHEMA_VERSION",
    "OPERATOR_BOUNDARY_NOTE",
    "REQUIRED_FIELDS",
    "OPTIONAL_FIELDS",
    "build_canonical_output",
    "derive_product_events",
    "is_canonical_output",
]
