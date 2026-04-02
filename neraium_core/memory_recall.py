from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


NOVELTY_SIMILARITY_THRESHOLD = 0.72
NOVELTY_OVERLAP_THRESHOLD = 0.45
DEDUPLICATE_SIMILARITY_THRESHOLD = 0.97


@dataclass(frozen=True)
class SimilarityBreakdown:
    total: float
    categorical: float
    numeric: float
    drivers_overlap: float


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalized_top_drivers(raw: Any, limit: int = 3) -> list[str]:
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for item in raw:
        if isinstance(item, Mapping):
            label = item.get("driver") or item.get("name")
        elif isinstance(item, (list, tuple)) and item:
            label = item[0]
        else:
            label = item
        text = str(label or "").strip()
        if text:
            out.append(text)
    return out[:limit]


def _driver_concentration(raw: Any) -> float:
    if not isinstance(raw, list):
        return 0.0
    vals: list[float] = []
    for item in raw:
        if isinstance(item, Mapping):
            score = item.get("score", item.get("contribution", 0.0))
        elif isinstance(item, (list, tuple)) and len(item) > 1:
            score = item[1]
        else:
            score = 0.0
        vals.append(abs(_safe_float(score, 0.0)))
    total = sum(vals)
    if total <= 1e-9:
        return 0.0
    return _clamp01(max(vals) / total)


def _regime_key(regime_memory: Mapping[str, Any] | None) -> str:
    if not isinstance(regime_memory, Mapping):
        return "unknown"
    for key in (
        "state",
        "regime_state",
        "label",
        "regime_label",
        "current_regime",
        "nearest_regime",
    ):
        if key in regime_memory and str(regime_memory.get(key)).strip():
            return str(regime_memory.get(key)).strip().lower()
    return "unknown"


def _dominant_hypothesis(causal_analysis: Mapping[str, Any] | None) -> str:
    if not isinstance(causal_analysis, Mapping):
        return "none"
    for key in ("dominant_hypothesis", "primary_hypothesis", "hypothesis"):
        if key in causal_analysis and str(causal_analysis.get(key)).strip():
            return str(causal_analysis.get(key)).strip().lower()
    dominant_drivers = causal_analysis.get("dominant_drivers")
    if isinstance(dominant_drivers, list) and dominant_drivers:
        return str(dominant_drivers[0]).strip().lower()
    return "none"


def _decision_target(action: str) -> str:
    text = str(action or "none").strip().lower()
    if not text or text == "none":
        return "none"
    tokens = [t for t in text.replace("-", "_").split("_") if t]
    return tokens[-1] if tokens else text


def build_structural_memory_signature(canonical_output: Mapping[str, Any]) -> dict[str, Any]:
    attribution = canonical_output.get("attribution") if isinstance(canonical_output.get("attribution"), Mapping) else {}
    risk = canonical_output.get("risk_assessment") if isinstance(canonical_output.get("risk_assessment"), Mapping) else {}
    decision = canonical_output.get("decision") if isinstance(canonical_output.get("decision"), Mapping) else {}
    causal = canonical_output.get("causal_analysis") if isinstance(canonical_output.get("causal_analysis"), Mapping) else {}
    regime_memory = canonical_output.get("regime_memory") if isinstance(canonical_output.get("regime_memory"), Mapping) else {}

    top_driver_items = attribution.get("top_drivers")
    top_drivers = _normalized_top_drivers(top_driver_items, limit=3)
    concentration = _driver_concentration(top_driver_items)
    latest_instability = _clamp01(_safe_float(risk.get("latest_instability"), 0.0) / 2.0)
    confidence = _clamp01(_safe_float(canonical_output.get("confidence"), 0.0))

    signature = {
        "top_drivers": top_drivers,
        "driver_concentration": round(concentration, 4),
        "regime_key": _regime_key(regime_memory),
        "risk_level": str(risk.get("risk_level", "UNKNOWN")).upper(),
        "risk_trend": str(risk.get("trend", "UNKNOWN")).upper(),
        "instability_bucket": round(latest_instability, 4),
        "decision_state": str(decision.get("state", "UNKNOWN")).upper(),
        "decision_action": str(decision.get("action", "none")).strip().lower() or "none",
        "decision_target": _decision_target(str(decision.get("action", "none"))),
        "dominant_hypothesis": _dominant_hypothesis(causal),
        "confidence_bucket": round(confidence, 4),
    }
    return signature


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _categorical_similarity(sig_a: Mapping[str, Any], sig_b: Mapping[str, Any]) -> float:
    weights = {
        "regime_key": 0.22,
        "risk_level": 0.14,
        "risk_trend": 0.08,
        "decision_state": 0.14,
        "decision_action": 0.14,
        "decision_target": 0.06,
        "dominant_hypothesis": 0.10,
    }
    score = 0.0
    for key, w in weights.items():
        a = str(sig_a.get(key, "")).strip().lower()
        b = str(sig_b.get(key, "")).strip().lower()
        score += w if (a and a == b) else 0.0
    return _clamp01(score)


def _numeric_similarity(sig_a: Mapping[str, Any], sig_b: Mapping[str, Any]) -> float:
    def close(key: str) -> float:
        a = _safe_float(sig_a.get(key), 0.0)
        b = _safe_float(sig_b.get(key), 0.0)
        return _clamp01(1.0 - abs(a - b))

    return round(
        0.4 * close("driver_concentration")
        + 0.4 * close("instability_bucket")
        + 0.2 * close("confidence_bucket"),
        6,
    )


def signature_similarity(sig_a: Mapping[str, Any], sig_b: Mapping[str, Any]) -> SimilarityBreakdown:
    drivers_a = {str(x).strip().lower() for x in sig_a.get("top_drivers", []) if str(x).strip()}
    drivers_b = {str(x).strip().lower() for x in sig_b.get("top_drivers", []) if str(x).strip()}
    drivers_overlap = _jaccard(drivers_a, drivers_b)
    categorical = _categorical_similarity(sig_a, sig_b)
    numeric = _numeric_similarity(sig_a, sig_b)
    total = _clamp01(0.5 * categorical + 0.3 * drivers_overlap + 0.2 * numeric)
    return SimilarityBreakdown(
        total=round(total, 6),
        categorical=round(categorical, 6),
        numeric=round(numeric, 6),
        drivers_overlap=round(drivers_overlap, 6),
    )


def classify_novelty(*, best_similarity: float, best_overlap: float, memory_count: int) -> dict[str, Any]:
    if memory_count <= 0:
        return {
            "is_novel": True,
            "score": 1.0,
            "reason": "no_prior_structural_memory",
        }

    novelty_score = _clamp01(1.0 - max(0.0, best_similarity))
    is_novel = best_similarity < NOVELTY_SIMILARITY_THRESHOLD or best_overlap < NOVELTY_OVERLAP_THRESHOLD
    if not is_novel:
        reason = "resembles_known_structural_pattern"
    elif best_similarity >= NOVELTY_SIMILARITY_THRESHOLD:
        reason = "insufficient_feature_overlap"
    else:
        reason = "similarity_below_threshold"

    return {
        "is_novel": bool(is_novel),
        "score": round(novelty_score, 6),
        "reason": reason,
    }


def determine_pattern_family(signature: Mapping[str, Any], *, similarity: float | None = None) -> dict[str, Any]:
    risk = str(signature.get("risk_level", "UNKNOWN")).upper()
    regime = str(signature.get("regime_key", "unknown")).lower()
    decision_state = str(signature.get("decision_state", "UNKNOWN")).upper()

    if risk == "HIGH" and decision_state in {"ALERT", "WATCH"}:
        label = "degradation_path"
        confidence = 0.74
    elif regime not in {"", "unknown"}:
        label = f"regime:{regime}"
        confidence = 0.66
    elif risk == "LOW":
        label = "nominal_path"
        confidence = 0.58
    else:
        return {"label": None, "confidence": None}

    if similarity is not None:
        confidence = max(0.45, min(0.95, (confidence + float(similarity)) / 2.0))

    return {"label": label, "confidence": round(confidence, 4)}


def short_memory_summary(signature: Mapping[str, Any]) -> str:
    drivers = signature.get("top_drivers") or []
    head = ", ".join([str(d) for d in drivers[:2]]) if isinstance(drivers, list) else ""
    risk = str(signature.get("risk_level", "UNKNOWN")).upper()
    state = str(signature.get("decision_state", "UNKNOWN")).upper()
    regime = str(signature.get("regime_key", "unknown"))
    if head:
        return f"{risk}/{state} regime={regime} drivers={head}"
    return f"{risk}/{state} regime={regime}"


__all__ = [
    "DEDUPLICATE_SIMILARITY_THRESHOLD",
    "NOVELTY_OVERLAP_THRESHOLD",
    "NOVELTY_SIMILARITY_THRESHOLD",
    "SimilarityBreakdown",
    "build_structural_memory_signature",
    "classify_novelty",
    "determine_pattern_family",
    "short_memory_summary",
    "signature_similarity",
]
