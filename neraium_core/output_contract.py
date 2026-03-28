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
    "memory_recall",
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

    recommendation = _coerce_recommendation(raw_result)
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
        "operational_recommendation": recommendation,
        "confidence": _normalize_confidence(raw_result, recommendation),
        "explanation_text": str(raw_result.get("explanation_text") or raw_result.get("explanation") or ""),
        "memory_recall": _normalize_memory_recall(memory_recall if memory_recall is not None else raw_result.get("memory_recall")),
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
