from __future__ import annotations

from typing import Any, Mapping

CANONICAL_SCHEMA_VERSION = "2026-03-01"

REQUIRED_FIELDS = {
    "schema_version",
    "timestamp",
    "cycle",
    "attribution",
    "regime_memory",
    "risk_assessment",
    "causal_analysis",
    "decision",
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


def _normalize_decision(raw_result: Mapping[str, Any]) -> dict[str, Any]:
    raw_decision = raw_result.get("decision") if isinstance(raw_result.get("decision"), dict) else {}
    state = str(
        raw_decision.get("state")
        or raw_result.get("state")
        or raw_result.get("action_state")
        or "UNKNOWN"
    ).upper()
    action = str(
        raw_decision.get("action")
        or raw_decision.get("resolved_action")
        or raw_decision.get("recommended_action")
        or "none"
    )
    return {
        "state": state,
        "action": action,
        "reason": str(raw_decision.get("reason", "decision_available")),
        "source": raw_decision.get("source", {}),
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


def _normalize_confidence(raw_result: Mapping[str, Any]) -> float:
    for key in ("confidence", "confidence_score"):
        if key in raw_result:
            try:
                return round(max(0.0, min(float(raw_result[key]), 1.0)), 6)
            except (TypeError, ValueError):
                continue
    decision = raw_result.get("decision") if isinstance(raw_result.get("decision"), dict) else {}
    try:
        return round(max(0.0, min(float(decision.get("confidence", 0.0)), 1.0)), 6)
    except (TypeError, ValueError):
        return 0.0


def derive_product_events(current: Mapping[str, Any], previous: Mapping[str, Any] | None = None) -> list[str]:
    events: list[str] = []
    risk = current.get("risk_assessment") if isinstance(current.get("risk_assessment"), dict) else {}
    decision = current.get("decision") if isinstance(current.get("decision"), dict) else {}

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

    if decision and str(decision.get("state", "UNKNOWN")).upper() != "UNKNOWN":
        events.append("decision_available")

    next_action = str(decision.get("action", "")).lower()
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
        "decision": _normalize_decision(raw_result),
        "confidence": _normalize_confidence(raw_result),
        "explanation_text": str(raw_result.get("explanation_text") or raw_result.get("explanation") or ""),
    }
    canonical["events"] = derive_product_events(canonical, previous=previous)

    aliases: dict[str, Any] = {}
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
    "REQUIRED_FIELDS",
    "OPTIONAL_FIELDS",
    "build_canonical_output",
    "derive_product_events",
    "is_canonical_output",
]
