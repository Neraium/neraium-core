from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable


_INTERVENTION_EVENT_MAP: dict[str, tuple[str, str]] = {
    "remove_top_driver_contribution": ("remove_or_suppress_top_driver", "top_driver"),
    "restore_relationship_cluster_to_baseline": ("restore_relationship_cluster", "relationship_cluster"),
    "suppress_subsystem_instability": ("damp_subsystem_instability", "subsystem"),
}


@dataclass
class ReplayStepResult:
    timestep: int
    timestamp: float
    asset_id: str
    domain: str
    scenario_family: str
    scenario_id: str
    system_type: str
    novelty: float
    support_count: int
    drift_warning: bool
    predicted_state: dict[str, Any]
    recommended_intervention: str | None
    confidence: float
    advisory_mode: str
    fallback_triggered: bool
    fallback_reasons: list[str]
    intervention_memory_contribution: dict[str, Any] | None
    law_usage: list[str]
    actual_intervention: str | None
    actual_outcome: str | None


class HistoricalReplayEngine:
    def __init__(self, decision_fn: Callable[[dict[str, Any]], dict[str, Any]]) -> None:
        self.decision_fn = decision_fn

    @staticmethod
    def _float_or_default(value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _int_or_default(value: Any, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def replay(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        step_logs: list[ReplayStepResult] = []
        trajectory_logs: dict[str, list[dict[str, Any]]] = {}

        for idx, row in enumerate(records):
            row = dict(row or {})
            observation = dict(row.get("observation") or {})
            observation["asset_id"] = row.get("asset_id", "unknown")
            actual_intervention = row.get("actual_intervention")
            if actual_intervention and "intervention_event" not in observation and "applied_intervention" not in observation:
                mapped_type, mapped_target = _INTERVENTION_EVENT_MAP.get(
                    str(actual_intervention),
                    ("other", "system"),
                )
                observation["intervention_event"] = {
                    "name": str(actual_intervention),
                    "type": mapped_type,
                    "target": mapped_target,
                    "outcome_label": row.get("outcome_label"),
                    "confidence": 0.7,
                }
            output = self.decision_fn(observation)

            transition = dict(output.get("transition_dynamics") or {})
            trajectory = dict(output.get("trajectory_archetypes") or output.get("trajectory_intelligence") or {})
            intervention = dict(output.get("intervention_intelligence") or {})
            recommendation = dict(intervention.get("recommendation") or {})
            best = dict(intervention.get("best_intervention") or {})
            if not best:
                best = dict(recommendation.get("best_intervention") or {})
            recommended = best.get("name")
            if bool(recommendation.get("no_intervention_recommended", False)) and not bool(recommendation.get("fallback_triggered", False)):
                recommended = "insufficient_support_monitor"
            confidence = self._float_or_default(
                best.get("confidence", (output.get("reliability_intelligence") or {}).get("risk_advisory", {}).get("calibrated_confidence", 0.0)),
                0.0,
            )
            confidence = max(0.0, min(1.0, confidence))
            recommendation_trace = dict(recommendation.get("intervention_memory_contribution") or {})
            reliability = dict(output.get("reliability_intelligence") or {})
            novelty = float(trajectory.get("novelty_score", 0.0))
            support_count = int(
                (intervention.get("context") or {}).get("support_count", 0)
                or ((intervention.get("historical_evidence") or {}).get("support_summary") or {}).get("support_count", 0)
                or trajectory.get("support_count", 0)
            )
            drift_warning = bool(
                trajectory.get("drift_warning", False)
                or (reliability.get("risk_advisory") or {}).get("drift_warning", False)
            )
            fallback_triggered = bool(recommendation.get("fallback_triggered", False))
            fallback_reasons = list(recommendation.get("fallback_reasons", []))
            advisory_mode = str(recommendation.get("recommended_posture", "standard_advisory"))
            if recommended is None or confidence == 0.0:
                fallback = self._fallback_policy(
                    row=row,
                    output=output,
                    recommended_intervention=str(recommended or "monitor"),
                    confidence=confidence,
                )
                if recommended is None:
                    recommended = fallback["recommended_intervention"]
                if confidence == 0.0:
                    confidence = fallback["confidence"]
                if not fallback_triggered:
                    fallback_triggered = bool(fallback["fallback_triggered"])
                if not fallback_reasons:
                    fallback_reasons = list(fallback["fallback_reasons"])
                if advisory_mode == "standard_advisory":
                    advisory_mode = str(fallback["advisory_mode"])
            elif not fallback_triggered:
                fallback = self._fallback_policy(
                    row=row,
                    output=output,
                    recommended_intervention=str(recommended or "monitor"),
                    confidence=confidence,
                )
                if bool(fallback["fallback_triggered"]):
                    recommended = fallback["recommended_intervention"]
                    confidence = fallback["confidence"]
                    fallback_triggered = True
                    fallback_reasons = list(fallback["fallback_reasons"])
                    advisory_mode = str(fallback["advisory_mode"])
            if recommended not in {None, "monitor", "insufficient_support_monitor"} and confidence < 0.42 and not fallback_triggered:
                recommended = "insufficient_support_monitor"
            if fallback_triggered and str(recommended) == "monitor":
                confidence = min(confidence, 0.15)
            governance = dict((output.get("structural_law_intelligence") or {}).get("structural_law_governance") or {})
            law_usage = [
                str(l.get("law_id"))
                for l in list(governance.get("laws") or [])
                if l.get("current_stage") != "rejected_or_falsified"
            ]

            step = ReplayStepResult(
                timestep=idx,
                timestamp=self._float_or_default(row.get("timestamp", idx), float(idx)),
                asset_id=str(row.get("asset_id", "unknown")),
                domain=str(row.get("domain") or row.get("vertical") or "unknown"),
                scenario_family=str(row.get("scenario_family") or row.get("scenario") or "unknown"),
                scenario_id=str(row.get("scenario_id") or f"{row.get('asset_id', 'unknown')}::{idx}"),
                system_type=str(row.get("system_type") or row.get("asset_type") or "unknown"),
                novelty=max(0.0, min(1.0, novelty)),
                support_count=max(0, support_count),
                drift_warning=drift_warning,
                predicted_state=transition,
                recommended_intervention=str(recommended or "monitor"),
                confidence=confidence,
                advisory_mode=advisory_mode,
                fallback_triggered=fallback_triggered,
                fallback_reasons=fallback_reasons,
                intervention_memory_contribution=recommendation_trace or None,
                law_usage=law_usage,
                actual_intervention=row.get("actual_intervention"),
                actual_outcome=row.get("outcome_label"),
            )
            step_logs.append(step)
            trajectory_logs.setdefault(step.asset_id, []).append(asdict(step))

        return {
            "step_logs": [asdict(s) for s in step_logs],
            "trajectory_logs": trajectory_logs,
        }

    def _fallback_policy(
        self,
        *,
        row: dict[str, Any],
        output: dict[str, Any],
        recommended_intervention: str | None,
        confidence: float,
    ) -> dict[str, Any]:
        novelty = max(0.0, min(1.0, self._float_or_default(row.get("novelty"), 0.0)))
        support_count = max(0, self._int_or_default(row.get("support_count"), 0))
        drift_warning = bool(row.get("drift_warning", False))
        reliability = dict(output.get("reliability_intelligence") or {})
        recommendation = dict(reliability.get("intervention_recommendation") or {})
        trace = dict(recommendation.get("reliability_trace") or {})
        warnings = [str(w) for w in list(trace.get("warnings") or [])]
        transfer = dict(output.get("transfer_adaptation") or {})
        transfer_mismatch = float((transfer.get("transfer_metrics") or {}).get("mismatch_penalty", 0.0) or 0.0)
        calibrated_conf = self._float_or_default(recommendation.get("recommendation_calibrated_confidence"), confidence)
        structural_uncertainty = dict((output.get("intervention_intelligence") or {}).get("structural_uncertainty_mode") or {})
        family_similarity = self._float_or_default(
            ((output.get("trajectory_archetypes") or {}).get("family_similarity")),
            1.0,
        )

        reasons: list[str] = []
        high_risk = drift_warning or novelty >= 0.75 or transfer_mismatch >= 0.65
        if drift_warning:
            reasons.append("drift_warning")
        if novelty >= 0.75:
            reasons.append("high_novelty")
        if support_count <= 2 and high_risk:
            reasons.append("sparse_support")
        if transfer_mismatch >= 0.65:
            reasons.append("transfer_mismatch")
        if calibrated_conf < 0.55 and high_risk:
            reasons.append("weak_calibration_support")
        if warnings and high_risk:
            reasons.append("reliability_warning")

        weak_support = support_count <= 2 or calibrated_conf < 0.55 or bool(warnings)
        if high_risk and weak_support:
            return {
                "recommended_intervention": "monitor",
                "confidence": min(confidence, 0.35),
                "fallback_triggered": True,
                "fallback_reasons": sorted(set(reasons)),
                "advisory_mode": "conservative_monitoring",
            }
        if reasons:
            return {
                "recommended_intervention": recommended_intervention,
                "confidence": min(confidence, 0.55),
                "fallback_triggered": True,
                "fallback_reasons": sorted(set(reasons)),
                "advisory_mode": "bounded_advisory",
            }
        return {
            "recommended_intervention": str(recommended_intervention or "monitor"),
            "confidence": confidence,
            "fallback_triggered": False,
            "fallback_reasons": [],
            "advisory_mode": "standard_advisory",
        }
