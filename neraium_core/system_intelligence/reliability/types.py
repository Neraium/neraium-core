from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

OutputFamily = Literal[
    "trajectory_escalation_forecast",
    "law_candidate_confidence",
    "intervention_recommendation_confidence",
    "risk_escalation_advisory",
]

Horizon = Literal["immediate", "short_term", "medium_term"]


@dataclass
class ReliabilityContext:
    trajectory_family: str = "unknown"
    transition_path: str = "unknown"
    regime: str = "unknown"
    novelty_level: float = 0.5
    support_count: int = 0
    mechanism_family: str = "unknown"
    evidence_mix: str = "local_dominant"


@dataclass
class ReliabilityIssueRecord:
    record_id: str
    family: OutputFamily
    context: ReliabilityContext
    raw_confidence: float
    local_evidence_weight: float
    transferred_evidence_weight: float
    calibration_version: str
    issued_at_step: int
    issued_timestamp: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReliabilityOutcomeRecord:
    record_id: str
    family: OutputFamily
    horizon: Horizon
    realized_outcome: float
    finalized_at_step: int


@dataclass
class ReliabilityTrace:
    raw_confidence: float
    calibrated_confidence: float
    calibration_delta: float
    support_adjusted_confidence: float
    reliability_band: str
    reliability_bucket: str
    calibration_bucket: str
    bucket_support: int
    local_bucket_support: int
    global_bucket_support: int
    fallback_bucket_used: bool
    support_count: int
    novelty_penalty: float
    uncertainty_penalty: float
    transfer_penalty: float
    mismatch_penalty: float
    stale_evidence_penalty: float
    local_evidence_weight: float
    transferred_evidence_weight: float
    calibration_source: str
    fallback_used: bool
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_confidence": round(self.raw_confidence, 6),
            "calibrated_confidence": round(self.calibrated_confidence, 6),
            "calibration_delta": round(self.calibration_delta, 6),
            "support_adjusted_confidence": round(self.support_adjusted_confidence, 6),
            "reliability_band": self.reliability_band,
            "reliability_bucket": self.reliability_bucket,
            "calibration_bucket": self.calibration_bucket,
            "bucket_support": int(self.bucket_support),
            "local_bucket_support": int(self.local_bucket_support),
            "global_bucket_support": int(self.global_bucket_support),
            "fallback_bucket_used": bool(self.fallback_bucket_used),
            "support_count": int(self.support_count),
            "novelty_penalty": round(self.novelty_penalty, 6),
            "uncertainty_penalty": round(self.uncertainty_penalty, 6),
            "transfer_penalty": round(self.transfer_penalty, 6),
            "mismatch_penalty": round(self.mismatch_penalty, 6),
            "stale_evidence_penalty": round(self.stale_evidence_penalty, 6),
            "local_evidence_weight": round(self.local_evidence_weight, 6),
            "transferred_evidence_weight": round(self.transferred_evidence_weight, 6),
            "calibration_source": self.calibration_source,
            "fallback_used": bool(self.fallback_used),
            "warnings": list(self.warnings),
        }
