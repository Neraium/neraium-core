from __future__ import annotations

from typing import Any

from .calibration import CalibrationInputs, StructuralCalibrationEngine
from .memory import ReliabilityRecordStore
from .types import ReliabilityContext


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


class StructuralReliabilityLayer:
    def __init__(self) -> None:
        self.store = ReliabilityRecordStore()
        self.calibrator = StructuralCalibrationEngine(self.store)

    def _build_context(
        self,
        *,
        trajectory_family: str,
        transition_path: str,
        regime: str,
        novelty: float,
        support_count: int,
        mechanism_family: str,
        local_weight: float,
    ) -> ReliabilityContext:
        return ReliabilityContext(
            trajectory_family=trajectory_family,
            transition_path=transition_path,
            regime=regime,
            novelty_level=_clip01(novelty),
            support_count=max(0, int(support_count)),
            mechanism_family=mechanism_family,
            evidence_mix="local_dominant" if local_weight >= 0.6 else "transferred_assisted",
        )

    def finalize_due(self, *, asset_id: str, step: int, transition: dict[str, Any]) -> int:
        escalation = _clip01(float(transition.get("escalation_probability", 0.0)))
        horizon_outcomes = {
            "immediate": escalation,
            "short_term": _clip01(escalation * 0.92 + (0.08 if str(transition.get("transition_path", "")) == "escalating" else 0.0)),
            "medium_term": _clip01(escalation * 0.85 + (0.12 if str(transition.get("regime", "")) == "critical" else 0.0)),
        }
        return self.store.finalize_due_for_asset(asset_id=asset_id, current_step=step, horizon_outcomes=horizon_outcomes)

    def calibrate_all(
        self,
        *,
        asset_id: str,
        step: int,
        transition: dict[str, Any],
        trajectory_forecast: dict[str, Any],
        laws: dict[str, Any],
        intervention: dict[str, Any],
        cross: dict[str, Any],
    ) -> dict[str, Any]:
        weights = dict((cross.get("evidence_source_weights") or {}))
        local_w = _clip01(float(weights.get("local_evidence_weight", 0.8)))
        transfer_w = _clip01(float(weights.get("global_evidence_weight", weights.get("transferred_evidence_weight", 0.0))))
        mismatch_penalty = _clip01(float(weights.get("mismatch_penalty", 0.0)))
        stale_penalty = _clip01(float(cross.get("stale_evidence_penalty", 0.0)))
        trajectory_family = str(trajectory_forecast.get("likely_next_path_family", transition.get("transition_path", "unknown")))
        transition_path = str(transition.get("transition_path", "unknown"))
        regime = str(transition.get("regime", "unknown"))
        novelty = float(trajectory_forecast.get("novelty_aware_confidence_penalty", 0.0))
        support = int((trajectory_forecast.get("matched_historical_progressions") or [{}])[0].get("frequency", 0) if trajectory_forecast.get("matched_historical_progressions") else 0)

        context = self._build_context(
            trajectory_family=trajectory_family,
            transition_path=transition_path,
            regime=regime,
            novelty=novelty,
            support_count=support,
            mechanism_family="trajectory",
            local_weight=local_w,
        )
        traj_raw = _clip01(float(trajectory_forecast.get("trajectory_conditioned_escalation_probability", transition.get("escalation_probability", 0.0))))
        traj_trace_short = self.calibrator.calibrate(
            CalibrationInputs(
                family="trajectory_escalation_forecast",
                raw_confidence=traj_raw,
                context=context,
                local_evidence_weight=local_w,
                transferred_evidence_weight=transfer_w,
                mismatch_penalty=mismatch_penalty,
                uncertainty=float(trajectory_forecast.get("uncertainty", transition.get("uncertainty", 0.5))),
                stale_evidence_penalty=stale_penalty,
                horizon="short_term",
                global_bucket_support=int((cross.get("shared_trajectory_knowledge") or {}).get("global_support", 0)),
            )
        )
        traj_trace_med = self.calibrator.calibrate(
            CalibrationInputs(
                family="trajectory_escalation_forecast",
                raw_confidence=traj_raw,
                context=context,
                local_evidence_weight=local_w,
                transferred_evidence_weight=transfer_w,
                mismatch_penalty=min(1.0, mismatch_penalty + 0.08),
                uncertainty=min(1.0, float(trajectory_forecast.get("uncertainty", 0.5)) + 0.08),
                stale_evidence_penalty=min(1.0, stale_penalty + 0.05),
                horizon="medium_term",
                global_bucket_support=int((cross.get("shared_trajectory_knowledge") or {}).get("global_support", 0)),
            )
        )

        self.store.issue(
            asset_id=asset_id,
            family="trajectory_escalation_forecast",
            context=context,
            raw_confidence=traj_raw,
            local_evidence_weight=local_w,
            transferred_evidence_weight=transfer_w,
            step=step,
            metadata={"calibration_bucket": traj_trace_short.calibration_bucket},
        )

        law_traces = []
        for law in list(laws.get("law_candidates") or []):
            law_support = int(law.get("support", 0))
            law_context = self._build_context(
                trajectory_family=str((law.get("applicability") or {}).get("trajectory_family", trajectory_family)),
                transition_path=transition_path,
                regime=regime,
                novelty=novelty,
                support_count=law_support,
                mechanism_family=str(law.get("mechanism", "unknown")),
                local_weight=local_w,
            )
            shared_law = next((row for row in list(cross.get("shared_law_intelligence") or []) if row.get("law_id") == law.get("law_id")), None)
            conflict = 1.0 if (shared_law and str(shared_law.get("status", "")).endswith("conflicting")) else 0.0
            law_trace = self.calibrator.calibrate(
                CalibrationInputs(
                    family="law_candidate_confidence",
                    raw_confidence=float(law.get("confidence", 0.0)),
                    context=law_context,
                    local_evidence_weight=local_w,
                    transferred_evidence_weight=transfer_w,
                    mismatch_penalty=min(1.0, mismatch_penalty + 0.25 * conflict),
                    uncertainty=max(0.0, 1.0 - float(law.get("consistency_across_assets_runs", 0.5))),
                    stale_evidence_penalty=stale_penalty,
                    horizon="short_term",
                    global_bucket_support=int((shared_law or {}).get("cross_system_support", 0)),
                )
            )
            law_traces.append({
                "law_id": law.get("law_id"),
                "local_law_confidence": round(_clip01(float(law.get("confidence", 0.0))), 6),
                "cross_system_generalization_confidence": round(_clip01(float((shared_law or {}).get("generalization_confidence", 0.0))), 6),
                "calibrated_applicability_confidence": round(law_trace.calibrated_confidence, 6),
                "classification": (
                    "strong_and_calibrated"
                    if law_trace.calibrated_confidence >= 0.7 and law_trace.bucket_support >= 8
                    else "globally_recurrent_locally_uncertain"
                    if float((shared_law or {}).get("generalization_confidence", 0.0)) >= 0.6
                    else "plausible_but_weakly_calibrated"
                    if law_trace.calibrated_confidence >= 0.45
                    else "novel_or_unsupported"
                ),
                "reliability_trace": law_trace.to_dict(),
            })
            self.store.issue(
                asset_id=asset_id,
                family="law_candidate_confidence",
                context=law_context,
                raw_confidence=float(law.get("confidence", 0.0)),
                local_evidence_weight=local_w,
                transferred_evidence_weight=transfer_w,
                step=step,
                metadata={"calibration_bucket": law_trace.calibration_bucket, "law_id": law.get("law_id")},
            )

        rec = dict((intervention.get("recommendation") or {}).get("best_intervention") or {})
        rec_raw = float(rec.get("confidence", (intervention.get("uncertainty_summary") or {}).get("confidence", 0.0)) or 0.0)
        rec_support = int(((intervention.get("historical_evidence") or {}).get("support_summary") or {}).get("support_count", 0) or 0)
        rec_context = self._build_context(
            trajectory_family=str((intervention.get("context") or {}).get("trajectory_family", trajectory_family)),
            transition_path=transition_path,
            regime=str((intervention.get("context") or {}).get("regime", regime)),
            novelty=float((intervention.get("context") or {}).get("novelty_score", novelty)),
            support_count=rec_support,
            mechanism_family="intervention",
            local_weight=local_w,
        )
        rec_trace = self.calibrator.calibrate(
            CalibrationInputs(
                family="intervention_recommendation_confidence",
                raw_confidence=rec_raw,
                context=rec_context,
                local_evidence_weight=local_w,
                transferred_evidence_weight=transfer_w,
                mismatch_penalty=mismatch_penalty,
                uncertainty=max(0.0, 1.0 - float((intervention.get("uncertainty_summary") or {}).get("confidence", rec_raw))),
                stale_evidence_penalty=stale_penalty,
                horizon="short_term",
                global_bucket_support=int(sum(int(row.get("evidence_source_weights", {}).get("global", 0.0) > 0.0) for row in list(cross.get("shared_intervention_evidence") or []))),
            )
        )
        self.store.issue(
            asset_id=asset_id,
            family="intervention_recommendation_confidence",
            context=rec_context,
            raw_confidence=rec_raw,
            local_evidence_weight=local_w,
            transferred_evidence_weight=transfer_w,
            step=step,
            metadata={"calibration_bucket": rec_trace.calibration_bucket, "intervention_name": rec.get("name")},
        )

        risk_raw = _clip01(float(transition.get("escalation_probability", 0.0)))
        risk_context = self._build_context(
            trajectory_family=trajectory_family,
            transition_path=transition_path,
            regime=regime,
            novelty=novelty,
            support_count=max(support, rec_support),
            mechanism_family="risk",
            local_weight=local_w,
        )
        risk_trace = self.calibrator.calibrate(
            CalibrationInputs(
                family="risk_escalation_advisory",
                raw_confidence=risk_raw,
                context=risk_context,
                local_evidence_weight=local_w,
                transferred_evidence_weight=transfer_w,
                mismatch_penalty=mismatch_penalty,
                uncertainty=float(transition.get("uncertainty", 0.5)),
                stale_evidence_penalty=stale_penalty,
                horizon="immediate",
            )
        )
        self.store.issue(
            asset_id=asset_id,
            family="risk_escalation_advisory",
            context=risk_context,
            raw_confidence=risk_raw,
            local_evidence_weight=local_w,
            transferred_evidence_weight=transfer_w,
            step=step,
            metadata={"calibration_bucket": risk_trace.calibration_bucket},
        )

        horizon_support = {
            "short_term": int(traj_trace_short.bucket_support),
            "medium_term": int(traj_trace_med.bucket_support),
        }
        horizon_notes = []
        if traj_trace_short.calibrated_confidence > traj_trace_med.calibrated_confidence + 0.08:
            horizon_notes.append("Short-horizon calibration stronger than medium-horizon calibration.")
        elif traj_trace_med.bucket_support < 4:
            horizon_notes.append("Medium-horizon support is sparse; treat as exploratory.")

        return {
            "trajectory_forecast": {
                "raw_confidence": round(traj_raw, 6),
                "calibrated_short_horizon_confidence": round(traj_trace_short.calibrated_confidence, 6),
                "calibrated_medium_horizon_confidence": round(traj_trace_med.calibrated_confidence, 6),
                "horizon_support": horizon_support,
                "horizon_reliability_notes": horizon_notes,
                "reliability_trace": traj_trace_short.to_dict(),
            },
            "intervention_recommendation": {
                "recommendation_raw_confidence": round(_clip01(rec_raw), 6),
                "recommendation_calibrated_confidence": round(rec_trace.calibrated_confidence, 6),
                "recommendation_reliability": rec_trace.reliability_band,
                "support_band": self.store.support_band(support_count=rec_support),
                "novelty_penalty": round(rec_trace.novelty_penalty, 6),
                "transfer_penalty": round(rec_trace.transfer_penalty, 6),
                "reliability_trace": rec_trace.to_dict(),
            },
            "law_candidates": law_traces,
            "risk_advisory": {
                "raw_confidence": round(risk_raw, 6),
                "calibrated_confidence": round(risk_trace.calibrated_confidence, 6),
                "reliability_trace": risk_trace.to_dict(),
            },
            "record_store": self.store.inspect(limit=24),
        }

    def ingest_feedback_records(self, *, asset_id: str, step: int, feedback_records: list[dict[str, Any]]) -> int:
        finalized = 0
        for row in feedback_records:
            confidence = float(row.get("confidence", 0.5))
            label = str(row.get("outcome_label", "neutral"))
            realized = 1.0 if label in {"helpful", "recovery-associated"} else 0.0 if label == "harmful" else 0.5
            context = self._build_context(
                trajectory_family=str(row.get("trajectory_family", "unknown")),
                transition_path=str(row.get("transition_path", "unknown")),
                regime=str(row.get("regime", "unknown")),
                novelty=float(row.get("novelty", 0.5)),
                support_count=int(row.get("support_count", 1)),
                mechanism_family="feedback",
                local_weight=1.0,
            )
            rec_id = self.store.issue(
                asset_id=asset_id,
                family="intervention_recommendation_confidence",
                context=context,
                raw_confidence=confidence,
                local_evidence_weight=1.0,
                transferred_evidence_weight=0.0,
                step=step,
                metadata={"calibration_bucket": f"feedback|{label}"},
            )
            if self.store.finalize(record_id=rec_id, realized_outcome=realized, horizon="short_term", step=step):
                finalized += 1
        return finalized
