from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .memory import ReliabilityRecordStore
from .types import Horizon, OutputFamily, ReliabilityContext, ReliabilityTrace


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _band(x: float) -> str:
    if x >= 0.75:
        return "high"
    if x >= 0.5:
        return "moderate"
    return "low"


@dataclass
class CalibrationInputs:
    family: OutputFamily
    raw_confidence: float
    context: ReliabilityContext
    local_evidence_weight: float
    transferred_evidence_weight: float
    mismatch_penalty: float = 0.0
    uncertainty: float = 0.5
    stale_evidence_penalty: float = 0.0
    horizon: Horizon = "short_term"
    global_bucket_support: int = 0


class StructuralCalibrationEngine:
    """Transparent, bucketed empirical calibrator for structural-intelligence outputs."""

    def __init__(self, store: ReliabilityRecordStore, *, calibration_source: str = "beta_shrinkage_bucketed.v1") -> None:
        self.store = store
        self.calibration_source = calibration_source

    def buckets_for(self, *, family: OutputFamily, context: ReliabilityContext) -> list[str]:
        support_band = self.store.support_band(support_count=context.support_count)
        novelty_band = self.store.novelty_band(novelty=context.novelty_level)
        mix = context.evidence_mix
        fine = (
            f"{family}|tf={context.trajectory_family}|path={context.transition_path}|reg={context.regime}"
            f"|nov={novelty_band}|sup={support_band}|mix={mix}|mech={context.mechanism_family}"
        )
        mid = f"{family}|tf={context.trajectory_family}|reg={context.regime}|nov={novelty_band}|sup={support_band}"
        coarse = f"{family}|reg={context.regime}|sup={support_band}"
        global_bucket = f"{family}|global"
        return [fine, mid, coarse, global_bucket]

    def calibrate(self, inp: CalibrationInputs) -> ReliabilityTrace:
        raw = _clip01(inp.raw_confidence)
        buckets = self.buckets_for(family=inp.family, context=inp.context)
        empirical = self.store.empirical_reliability(family=inp.family, buckets=buckets, horizon=inp.horizon)

        support = int(empirical["support"])
        hit_rate = float(empirical["hit_rate"])
        local_bucket_support = support
        global_bucket_support = int(inp.global_bucket_support)

        support_strength = _clip01(support / 16.0)
        empirical_weight = min(0.82, 0.15 + 0.67 * support_strength)
        support_adjusted = (1.0 - empirical_weight) * raw + empirical_weight * hit_rate

        novelty_penalty = 0.28 * _clip01(inp.context.novelty_level)
        uncertainty_penalty = 0.22 * _clip01(inp.uncertainty)
        transfer_penalty = 0.35 * _clip01(inp.transferred_evidence_weight) * _clip01(inp.mismatch_penalty)
        stale_penalty = 0.22 * _clip01(inp.stale_evidence_penalty)

        weak_support_penalty = 0.0
        if support < 4:
            weak_support_penalty = 0.12
        elif support < 8:
            weak_support_penalty = 0.06
        failure_signal = self.store.recent_high_confidence_failure_signal(family=inp.family)
        repeated_failure_penalty = 0.0
        if float(failure_signal.get("support", 0.0)) >= 3.0:
            repeated_failure_penalty = 0.28 * _clip01(float(failure_signal.get("high_confidence_fail_rate", 0.0)))

        total_penalty = min(
            0.72,
            novelty_penalty + uncertainty_penalty + transfer_penalty + stale_penalty + weak_support_penalty + repeated_failure_penalty,
        )
        calibrated = _clip01(support_adjusted * (1.0 - total_penalty))

        warnings: list[str] = []
        if support < 4:
            warnings.append("Sparse calibration support; confidence is conservative.")
        if inp.context.novelty_level > 0.7:
            warnings.append("Context is novel/out-of-family; reliability discounted.")
        if inp.transferred_evidence_weight > 0.45 and inp.mismatch_penalty > 0.35:
            warnings.append("Transferred evidence mismatch detected; local-first penalties applied.")
        if repeated_failure_penalty >= 0.1:
            warnings.append("Recent high-confidence misses detected; confidence penalized until reliability recovers.")

        return ReliabilityTrace(
            raw_confidence=raw,
            calibrated_confidence=calibrated,
            calibration_delta=calibrated - raw,
            support_adjusted_confidence=support_adjusted,
            reliability_band=_band(calibrated),
            reliability_bucket=str(empirical["bucket"]),
            calibration_bucket=str(empirical["bucket"]),
            bucket_support=support,
            local_bucket_support=local_bucket_support,
            global_bucket_support=global_bucket_support,
            fallback_bucket_used=bool(empirical["fallback"]),
            support_count=inp.context.support_count,
            novelty_penalty=novelty_penalty,
            uncertainty_penalty=uncertainty_penalty,
            transfer_penalty=transfer_penalty,
            mismatch_penalty=inp.mismatch_penalty,
            stale_evidence_penalty=stale_penalty,
            local_evidence_weight=_clip01(inp.local_evidence_weight),
            transferred_evidence_weight=_clip01(inp.transferred_evidence_weight),
            calibration_source=self.calibration_source,
            fallback_used=bool(empirical["fallback"]),
            warnings=warnings,
        )
