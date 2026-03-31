from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

from .types import Horizon, OutputFamily, ReliabilityContext, ReliabilityIssueRecord, ReliabilityOutcomeRecord


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _band(value: float, cuts: tuple[float, float] = (0.34, 0.67)) -> str:
    if value < cuts[0]:
        return "low"
    if value < cuts[1]:
        return "mid"
    return "high"


class ReliabilityRecordStore:
    """Compact, inspectable reliability memory with delayed finalization support."""

    def __init__(self, *, calibration_version: str = "reliability.v1", max_finalized: int = 4096) -> None:
        self.calibration_version = calibration_version
        self.max_finalized = int(max_finalized)
        self._issued_by_id: dict[str, ReliabilityIssueRecord] = {}
        self._pending_by_asset: dict[str, list[str]] = defaultdict(list)
        self._finalized: list[tuple[ReliabilityIssueRecord, ReliabilityOutcomeRecord]] = []
        self._counter = 0

    def issue(
        self,
        *,
        asset_id: str,
        family: OutputFamily,
        context: ReliabilityContext,
        raw_confidence: float,
        local_evidence_weight: float,
        transferred_evidence_weight: float,
        step: int,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        self._counter += 1
        rec_id = f"rel::{step}::{self._counter}"
        issued = ReliabilityIssueRecord(
            record_id=rec_id,
            family=family,
            context=context,
            raw_confidence=_clip01(raw_confidence),
            local_evidence_weight=_clip01(local_evidence_weight),
            transferred_evidence_weight=_clip01(transferred_evidence_weight),
            calibration_version=self.calibration_version,
            issued_at_step=int(step),
            issued_timestamp=datetime.now(tz=timezone.utc).isoformat(),
            metadata=dict(metadata or {}),
        )
        self._issued_by_id[rec_id] = issued
        self._pending_by_asset[str(asset_id)].append(rec_id)
        return rec_id

    def finalize(
        self,
        *,
        record_id: str,
        realized_outcome: float,
        horizon: Horizon,
        step: int,
    ) -> bool:
        issued = self._issued_by_id.pop(record_id, None)
        if issued is None:
            return False
        outcome = ReliabilityOutcomeRecord(
            record_id=record_id,
            family=issued.family,
            horizon=horizon,
            realized_outcome=_clip01(realized_outcome),
            finalized_at_step=int(step),
        )
        self._finalized.append((issued, outcome))
        if len(self._finalized) > self.max_finalized:
            self._finalized = self._finalized[-self.max_finalized :]
        return True

    def finalize_due_for_asset(
        self,
        *,
        asset_id: str,
        current_step: int,
        horizon_outcomes: dict[Horizon, float],
    ) -> int:
        pending = list(self._pending_by_asset.get(str(asset_id), []))
        kept: list[str] = []
        closed = 0
        for rec_id in pending:
            issued = self._issued_by_id.get(rec_id)
            if issued is None:
                continue
            age = int(current_step) - int(issued.issued_at_step)
            target_horizon: Horizon = "medium_term" if age >= 3 else "short_term"
            if age < 1:
                kept.append(rec_id)
                continue
            outcome = float(horizon_outcomes.get(target_horizon, horizon_outcomes.get("immediate", 0.0)))
            if self.finalize(record_id=rec_id, realized_outcome=outcome, horizon=target_horizon, step=current_step):
                closed += 1
        self._pending_by_asset[str(asset_id)] = kept
        return closed

    def summary_by_bucket(self, *, family: OutputFamily, bucket: str, horizon: Horizon = "short_term") -> dict[str, float]:
        rows = [
            (issued, outcome)
            for issued, outcome in self._finalized
            if issued.family == family
            and issued.metadata.get("calibration_bucket") == bucket
            and outcome.horizon == horizon
        ]
        n = len(rows)
        if n == 0:
            return {"support": 0, "hit_rate": 0.5, "issued_mean": 0.5, "brier": 0.25}
        hit_rate = sum(float(outcome.realized_outcome) for _, outcome in rows) / n
        issued_mean = sum(float(issued.raw_confidence) for issued, _ in rows) / n
        brier = sum((float(issued.raw_confidence) - float(outcome.realized_outcome)) ** 2 for issued, outcome in rows) / n
        return {
            "support": float(n),
            "hit_rate": float(hit_rate),
            "issued_mean": float(issued_mean),
            "brier": float(brier),
        }

    def empirical_reliability(
        self,
        *,
        family: OutputFamily,
        buckets: list[str],
        horizon: Horizon = "short_term",
    ) -> dict[str, Any]:
        for idx, bucket in enumerate(buckets):
            summary = self.summary_by_bucket(family=family, bucket=bucket, horizon=horizon)
            if int(summary["support"]) >= (6 if idx == 0 else 3):
                return {
                    "bucket": bucket,
                    "support": int(summary["support"]),
                    "hit_rate": float(summary["hit_rate"]),
                    "issued_mean": float(summary["issued_mean"]),
                    "brier": float(summary["brier"]),
                    "fallback": idx > 0,
                }
        return {
            "bucket": buckets[-1],
            "support": 0,
            "hit_rate": 0.5,
            "issued_mean": 0.5,
            "brier": 0.25,
            "fallback": True,
        }

    def support_band(self, *, support_count: int) -> str:
        return _band(min(1.0, max(0.0, float(support_count) / 12.0)), cuts=(0.25, 0.65))

    def novelty_band(self, *, novelty: float) -> str:
        return _band(_clip01(novelty), cuts=(0.33, 0.7))

    def inspect(self, *, limit: int = 64) -> dict[str, Any]:
        finalized_tail = self._finalized[-int(limit) :]
        return {
            "calibration_version": self.calibration_version,
            "issued_pending_count": len(self._issued_by_id),
            "finalized_count": len(self._finalized),
            "finalized_tail": [
                {
                    "issued": asdict(issued),
                    "outcome": asdict(outcome),
                }
                for issued, outcome in finalized_tail
            ],
        }

    def recent_high_confidence_failure_signal(self, *, family: OutputFamily, limit: int = 96) -> dict[str, float]:
        rows = [
            (issued, outcome)
            for issued, outcome in self._finalized[-int(limit) :]
            if issued.family == family
        ]
        if not rows:
            return {"support": 0.0, "high_confidence_fail_rate": 0.0}
        high_conf_rows = [(issued, outcome) for issued, outcome in rows if float(issued.raw_confidence) >= 0.72]
        if not high_conf_rows:
            return {"support": float(len(rows)), "high_confidence_fail_rate": 0.0}
        failures = sum(1.0 for _, outcome in high_conf_rows if float(outcome.realized_outcome) <= 0.25)
        return {
            "support": float(len(high_conf_rows)),
            "high_confidence_fail_rate": float(failures / max(1.0, float(len(high_conf_rows)))),
        }
