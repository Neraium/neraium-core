from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

DOCTRINE_VERSION = "2026.04-conservative-observational"


@dataclass(frozen=True)
class DoctrineAssessment:
    allowed: bool
    classification: Literal["observational", "prescriptive", "ambiguous", "missing"]
    reason: str | None
    sanitized_assertion: str | None
    doctrine_version: str = DOCTRINE_VERSION


_PRESCRIPTIVE_TOKENS = {
    "must",
    "should",
    "need to",
    "required",
    "immediately",
    "replace",
    "shutdown",
    "shut down",
    "do this",
    "action now",
}


def assess_assertion(candidate_assertion: str | None) -> DoctrineAssessment:
    """Assess whether candidate text stays within observational doctrine boundaries."""
    if candidate_assertion is None:
        return DoctrineAssessment(
            allowed=True,
            classification="missing",
            reason="No candidate assertion supplied; doctrinally neutral.",
            sanitized_assertion=None,
        )

    text = str(candidate_assertion).strip()
    if not text:
        return DoctrineAssessment(
            allowed=True,
            classification="missing",
            reason="Candidate assertion is empty; doctrinally neutral.",
            sanitized_assertion=None,
        )

    lowered = text.lower()
    for token in _PRESCRIPTIVE_TOKENS:
        if token in lowered:
            return DoctrineAssessment(
                allowed=False,
                classification="prescriptive",
                reason=(
                    "Candidate assertion includes prescriptive language and cannot be "
                    "institutionally surfaced as an observation."
                ),
                sanitized_assertion=None,
            )

    if "?" in text:
        return DoctrineAssessment(
            allowed=False,
            classification="ambiguous",
            reason="Candidate assertion is interrogative/ambiguous and not defensible as an observation.",
            sanitized_assertion=None,
        )

    return DoctrineAssessment(
        allowed=True,
        classification="observational",
        reason="Candidate assertion remains observational and doctrine-conformant.",
        sanitized_assertion=text,
    )


__all__ = ["DOCTRINE_VERSION", "DoctrineAssessment", "assess_assertion"]
