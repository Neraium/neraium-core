from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from neraium_core.doctrine.defaults import DEFAULT_DOCTRINE_V1
from neraium_core.doctrine.schema import Doctrine


@dataclass(frozen=True)
class DoctrineEvaluation:
    allowed: bool
    refused: bool
    refusal_reason: str | None
    normalized_statement: str


def _normalize(text: str | None) -> str:
    return str(text or "").strip().lower()


def is_recommendation_crossing_boundary(
    statement: str | None,
    doctrine: Doctrine = DEFAULT_DOCTRINE_V1,
) -> bool:
    normalized = _normalize(statement)
    if not normalized:
        return False
    return any(token in normalized for token in doctrine.forbidden_outputs)


def is_assertion_allowed(
    assertion: str | None,
    doctrine: Doctrine = DEFAULT_DOCTRINE_V1,
) -> bool:
    normalized = _normalize(assertion)
    if not normalized:
        return False
    if is_recommendation_crossing_boundary(normalized, doctrine):
        return False
    return any(phrase in normalized for phrase in doctrine.allowed_assertions)


def refusal_reason_for_statement(
    statement: str | None,
    *,
    confidence: float | int | None = None,
    corroborating_signals: int | None = None,
    doctrine: Doctrine = DEFAULT_DOCTRINE_V1,
) -> str | None:
    normalized = _normalize(statement)
    if not normalized:
        return "insufficient_evidence"

    if is_recommendation_crossing_boundary(normalized, doctrine):
        return "prescriptive_or_actuation_language"

    try:
        conf = float(confidence) if confidence is not None else None
    except (TypeError, ValueError):
        conf = None
    if (
        doctrine.uncertainty_override.refuse_on_low_confidence
        and conf is not None
        and conf < doctrine.uncertainty_override.min_confidence
    ):
        return "low_confidence"

    if conf is not None and conf < doctrine.corroboration_thresholds.min_confidence:
        return "low_confidence"

    corroboration = corroborating_signals if corroborating_signals is not None else 0
    if corroboration < doctrine.corroboration_thresholds.min_confirming_signals:
        return "insufficient_corroboration"

    if not is_assertion_allowed(normalized, doctrine):
        return "insufficient_evidence"

    return None


def should_refuse_statement(
    statement: str | None,
    *,
    confidence: float | int | None = None,
    corroborating_signals: int | None = None,
    doctrine: Doctrine = DEFAULT_DOCTRINE_V1,
) -> bool:
    return refusal_reason_for_statement(
        statement,
        confidence=confidence,
        corroborating_signals=corroborating_signals,
        doctrine=doctrine,
    ) is not None


def evaluate_statement(
    statement: str | None,
    *,
    confidence: float | int | None = None,
    corroborating_signals: int | None = None,
    doctrine: Doctrine = DEFAULT_DOCTRINE_V1,
) -> DoctrineEvaluation:
    normalized = _normalize(statement)
    reason = refusal_reason_for_statement(
        normalized,
        confidence=confidence,
        corroborating_signals=corroborating_signals,
        doctrine=doctrine,
    )
    refused = reason is not None
    return DoctrineEvaluation(
        allowed=not refused,
        refused=refused,
        refusal_reason=reason,
        normalized_statement=normalized,
    )


def count_confirming_signals(values: Iterable[bool | int | float]) -> int:
    count = 0
    for value in values:
        if isinstance(value, bool):
            if value:
                count += 1
            continue
        if float(value) > 0:
            count += 1
    return count
