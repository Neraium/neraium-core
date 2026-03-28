from __future__ import annotations

from typing import Any, Mapping

from neraium_core.memory_recall import determine_pattern_family

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
    "memory_recall",
    "operational_recommendation",
    "recommendation_available",
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


def _normalize_memory_recall(raw: Any) -> dict[str, Any]:
    base = {
        "status": {"enabled": False, "memory_records_considered": 0, "scope": "none"},
        "novelty": {"is_novel": True, "score": 1.0, "reason": "memory_not_available"},
        "nearest_match": {
            "found": False,
            "similarity": 0.0,
            "memory_id": None,
            "asset_id": None,
            "run_id": None,
            "cycle": None,
            "summary": None,
            "scope": None,
        },
        "top_matches": [],
        "pattern_family": {"label": None, "confidence": None},
    }
    if not isinstance(raw, Mapping):
        return base

    out = dict(base)
    status = raw.get("status")
    if isinstance(status, Mapping):
        out["status"] = {
            "enabled": bool(status.get("enabled", True)),
            "memory_records_considered": int(status.get("memory_records_considered", 0) or 0),
            "scope": str(status.get("scope", "customer")),
        }
    novelty = raw.get("novelty")
    if isinstance(novelty, Mapping):
        out["novelty"] = {
            "is_novel": bool(novelty.get("is_novel", True)),
            "score": round(max(0.0, min(float(novelty.get("score", 1.0) or 1.0), 1.0)), 6),
            "reason": str(novelty.get("reason", "unspecified")),
        }
    nearest = raw.get("nearest_match")
    if isinstance(nearest, Mapping):
        out["nearest_match"] = {
            "found": bool(nearest.get("found", False)),
            "similarity": round(max(0.0, min(float(nearest.get("similarity", 0.0) or 0.0), 1.0)), 6),
            "memory_id": nearest.get("memory_id"),
            "asset_id": nearest.get("asset_id"),
            "run_id": nearest.get("run_id"),
            "cycle": nearest.get("cycle"),
            "summary": nearest.get("summary"),
            "scope": nearest.get("scope"),
        }
    raw_matches = raw.get("top_matches")
    matches = []
    if isinstance(raw_matches, list):
        for item in raw_matches[:5]:
            if not isinstance(item, Mapping):
                continue
            matches.append(
                {
                    "memory_id": item.get("memory_id"),
                    "similarity": round(max(0.0, min(float(item.get("similarity", 0.0) or 0.0), 1.0)), 6),
                    "asset_id": item.get("asset_id"),
                    "run_id": item.get("run_id"),
                    "cycle": item.get("cycle"),
                    "summary": item.get("summary"),
                    "scope": item.get("scope"),
                }
            )
    out["top_matches"] = matches

    family = raw.get("pattern_family")
    if isinstance(family, Mapping):
        label = family.get("label")
        confidence = family.get("confidence")
        out["pattern_family"] = {
            "label": str(label) if label is not None else None,
            "confidence": None if confidence is None else round(max(0.0, min(float(confidence), 1.0)), 6),
        }
    elif out["nearest_match"]["found"]:
        out["pattern_family"] = determine_pattern_family(raw.get("signature", {}), similarity=out["nearest_match"]["similarity"])

    return out




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

    memory = current.get("memory_recall") if isinstance(current.get("memory_recall"), Mapping) else {}
    novelty = memory.get("novelty") if isinstance(memory.get("novelty"), Mapping) else {}
    nearest = memory.get("nearest_match") if isinstance(memory.get("nearest_match"), Mapping) else {}

    if bool(nearest.get("found")):
        events.append("known_pattern_recalled")
        if nearest.get("asset_id") and str(nearest.get("asset_id")) != str(current.get("session", {}).get("asset_id", "")):
            events.append("cross_asset_pattern_match")
        if float(nearest.get("similarity", 0.0) or 0.0) >= 0.85 and level in {"MEDIUM", "HIGH"}:
            events.append("recurring_degradation_pattern")

    if bool(novelty.get("is_novel", False)):
        events.append("novel_pattern_detected")

    return sorted(set(events))


def build_canonical_output(
    raw_result: Mapping[str, Any],
    *,
    cycle: int,
    run_id: str | None = None,
    customer_id: str | None = None,
    previous: Mapping[str, Any] | None = None,
    memory_recall: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    attribution = raw_result.get("attribution") if isinstance(raw_result.get("attribution"), dict) else {}
    causal_analysis = raw_result.get("causal_analysis") if isinstance(raw_result.get("causal_analysis"), dict) else {}
    regime_memory = raw_result.get("regime_memory") if isinstance(raw_result.get("regime_memory"), dict) else {}

    recommendations = raw_result.get("response_recommendations")
    if not isinstance(recommendations, list):
        recommendations = []

    canonical = {
        "schema_version": CANONICAL_SCHEMA_VERSION,
        "timestamp": str(raw_result.get("timestamp", "")),
        "cycle": int(cycle),
        "session": {
            "run_id": run_id,
            "customer_id": customer_id,
            "site_id": raw_result.get("site_id"),
            "asset_id": raw_result.get("asset_id"),
        },
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
        "memory_recall": _normalize_memory_recall(memory_recall if memory_recall is not None else raw_result.get("memory_recall")),
        "operational_recommendation": raw_result.get("operational_recommendation") or (recommendations[0] if recommendations else None),
        "recommendation_available": bool(raw_result.get("recommendation_available", bool(recommendations))),
    }
    canonical["events"] = derive_product_events(canonical, previous=previous)

    aliases: dict[str, Any] = {}
    if "regime_memory_state" in raw_result:
        aliases["regime_memory_state"] = raw_result.get("regime_memory_state")
    if "explanation" in raw_result:
        aliases["explanation"] = raw_result.get("explanation")
    if "response_recommendations" in raw_result:
        aliases["response_recommendations"] = raw_result.get("response_recommendations")
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
