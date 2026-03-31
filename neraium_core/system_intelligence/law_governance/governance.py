from __future__ import annotations

from typing import Any


STAGES = [
    "observed_pattern",
    "repeated_pattern",
    "context_conditioned_pattern",
    "intervention_sensitive_pattern",
    "decision_grade_heuristic",
]


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _support_band(support_count: int) -> str:
    if support_count >= 12:
        return "strong"
    if support_count >= 7:
        return "moderate"
    if support_count >= 4:
        return "emerging"
    return "thin"


def _reliability_band(conf: float) -> str:
    if conf >= 0.78:
        return "high"
    if conf >= 0.6:
        return "medium"
    if conf >= 0.42:
        return "guarded"
    return "low"


class StructuralLawGovernance:
    """Conservative governance layer for structural-law maturity and decision eligibility."""

    def __init__(self, *, require_human_review_for_decision_grade: bool = False) -> None:
        self.require_human_review_for_decision_grade = bool(require_human_review_for_decision_grade)
        self._registry: dict[str, dict[str, Any]] = {}
        self._real_world_evidence: dict[str, dict[str, float]] = {}
        self._tick: int = 0

    def evaluate(
        self,
        *,
        law_candidates: list[dict[str, Any]],
        trajectory_info: dict[str, Any],
        reliability: dict[str, Any] | None = None,
        falsification: dict[str, Any] | None = None,
        intervention_info: dict[str, Any] | None = None,
        allow_auto_promotion: bool = True,
    ) -> dict[str, Any]:
        self._tick += 1
        reliability_by_law = self._reliability_map(reliability or {})
        contradiction_by_law = self._contradictions_map(falsification or {})

        laws_out: list[dict[str, Any]] = []
        promotion_candidates: list[dict[str, Any]] = []
        decision_grade_laws: list[dict[str, Any]] = []
        blocked_promotions: list[dict[str, Any]] = []
        demotions: list[dict[str, Any]] = []

        for law in law_candidates:
            law_id = str(law.get("law_id") or "law::unknown")
            previous = self._registry.get(law_id)
            previous_stage = str((previous or {}).get("current_stage") or "observed_pattern")

            support = max(0, int(_f(law.get("support"), 0)))
            rw = self._real_world_evidence.get(law_id, {})
            real_support = max(0.0, _f(rw.get("support_count"), 0.0))
            real_contradiction = max(0.0, _f(rw.get("contradiction_count"), 0.0))
            real_validation_score = _clamp01(_f(rw.get("validation_score"), 0.0))
            intervention_sensitive_support = max(0.0, _f(rw.get("intervention_sensitive_support"), 0.0))
            intervention_sensitive_contradiction = max(0.0, _f(rw.get("intervention_sensitive_contradiction"), 0.0))
            weighted_support = support + int(round(real_support * 1.8))
            contradiction_count = max(0, int(contradiction_by_law.get(law_id, round(max(0.0, support * (1.0 - _clamp01(_f(law.get("robustness"), 0.0))))))))
            contradiction_count += int(round(real_contradiction * 1.8))
            contradiction_rate = _clamp01(contradiction_count / max(1.0, float(support + contradiction_count)))
            intervention_contradiction_rate = _clamp01(
                intervention_sensitive_contradiction / max(1.0, intervention_sensitive_support + intervention_sensitive_contradiction)
            )
            cross_domain_consistency = _clamp01(_f(law.get("consistency_across_assets_runs"), 0.0))
            domain_coverage = _clamp01(len(set((law.get("applicability") or {}).get("archetypes") or [])) / 3.0)
            intervention_effect_correlation = self._intervention_correlation(law=law, intervention=intervention_info or {})
            recency_weight = self._recency_weight(previous=previous)

            rel = reliability_by_law.get(law_id, {})
            novelty_penalty = _clamp01(_f(rel.get("novelty_penalty"), 0.45 * _clamp01(_f(trajectory_info.get("novelty_score"), 1.0))))
            transfer_penalty = _clamp01(_f(rel.get("transfer_penalty"), 0.0))
            calibrated_law_confidence = _clamp01(
                _f(rel.get("calibrated_applicability_confidence"), _f(law.get("confidence"), 0.0) * (1.0 - novelty_penalty * 0.4))
            )

            evidence_strength = _clamp01(
                0.35 * min(1.0, weighted_support / 14.0)
                + 0.20 * (1.0 - contradiction_rate)
                + 0.20 * cross_domain_consistency
                + 0.15 * calibrated_law_confidence
                + 0.10 * recency_weight
            )
            stage_confidence = _clamp01(evidence_strength * (1.0 - 0.5 * novelty_penalty) * (1.0 - 0.4 * transfer_penalty))

            target_stage, blockers, risk_flags = self._target_stage(
                support=weighted_support,
                contradiction_rate=contradiction_rate,
                real_validation_score=real_validation_score,
                domain_coverage=domain_coverage,
                cross_domain_consistency=cross_domain_consistency,
                intervention_effect_correlation=intervention_effect_correlation,
                intervention_sensitive_support=intervention_sensitive_support,
                intervention_contradiction_rate=intervention_contradiction_rate,
                calibrated_law_confidence=calibrated_law_confidence,
                novelty_penalty=novelty_penalty,
                transfer_penalty=transfer_penalty,
            )

            applied_stage = target_stage
            promotion_eligibility = target_stage != previous_stage
            promotion_blockers = list(blockers)
            if target_stage == "decision_grade_heuristic" and self.require_human_review_for_decision_grade and allow_auto_promotion:
                promotion_blockers.append("human_review_required_for_decision_grade")
                applied_stage = previous_stage

            if not allow_auto_promotion and target_stage != previous_stage:
                promotion_blockers.append("auto_promotion_disabled")
                applied_stage = previous_stage

            if target_stage == "rejected_or_falsified":
                applied_stage = target_stage

            history = list((previous or {}).get("promotion_history") or [])
            if applied_stage != previous_stage:
                event_type = "promotion" if self._stage_index(applied_stage) > self._stage_index(previous_stage) else "demotion"
                history.append(
                    {
                        "at_step": self._tick,
                        "from_stage": previous_stage,
                        "to_stage": applied_stage,
                        "event": event_type,
                        "reason": "threshold_transition",
                    }
                )
                if event_type == "demotion":
                    demotions.append(
                        {
                            "law_id": law_id,
                            "from_stage": previous_stage,
                            "to_stage": applied_stage,
                            "trigger": "evidence_regression",
                        }
                    )

            registry_entry = {
                "law_id": law_id,
                "description": str(law.get("condition") or law.get("outcome") or ""),
                "current_stage": applied_stage,
                "stage_confidence": round(stage_confidence, 4),
                "promotion_history": history,
                "evidence_metrics": {
                    "support_count": support,
                    "real_world_support_count": int(real_support),
                    "contradiction_count": contradiction_count,
                    "real_world_contradiction_count": int(real_contradiction),
                    "contradiction_rate": round(contradiction_rate, 4),
                    "domain_coverage": round(domain_coverage, 4),
                    "cross_domain_consistency": round(cross_domain_consistency, 4),
                    "intervention_effect_correlation": round(intervention_effect_correlation, 4),
                    "recency_weighting": round(recency_weight, 4),
                    "real_world_validation_score": round(real_validation_score, 4),
                    "intervention_sensitive_support": int(intervention_sensitive_support),
                    "intervention_sensitive_contradiction_rate": round(intervention_contradiction_rate, 4),
                },
                "reliability": {
                    "calibrated_law_confidence": round(calibrated_law_confidence, 4),
                    "reliability_band": str(rel.get("reliability_band") or _reliability_band(calibrated_law_confidence)),
                    "support_band": str(rel.get("support_band") or _support_band(support)),
                    "novelty_penalty": round(novelty_penalty, 4),
                    "transfer_penalty": round(transfer_penalty, 4),
                },
                "domains": list((law.get("applicability") or {}).get("archetypes") or []),
                "mechanisms": [str(law.get("mechanism") or "unknown")],
                "last_updated": self._tick,
                "validation_status": "validated" if applied_stage == "decision_grade_heuristic" else "advisory_only",
                "promotion_eligibility": promotion_eligibility,
                "promotion_blockers": promotion_blockers,
                "risk_flags": risk_flags,
                "real_world_gate": {
                    "real_world_validation_score": round(real_validation_score, 4),
                    "intervention_sensitive_support": int(intervention_sensitive_support),
                    "passes_decision_grade_real_world_gate": bool(
                        real_validation_score >= 0.62 and intervention_sensitive_support >= 3 and intervention_contradiction_rate <= 0.2
                    ),
                },
            }
            self._registry[law_id] = registry_entry
            laws_out.append(registry_entry)

            if promotion_eligibility:
                candidate = {
                    "law_id": law_id,
                    "from_stage": previous_stage,
                    "to_stage": target_stage,
                    "promotion_eligibility": not promotion_blockers,
                    "promotion_blockers": promotion_blockers,
                    "risk_flags": risk_flags,
                }
                promotion_candidates.append(candidate)
                if promotion_blockers:
                    blocked_promotions.append(candidate)

            if applied_stage == "decision_grade_heuristic":
                decision_grade_laws.append(registry_entry)

        return {
            "laws": laws_out,
            "promotion_candidates": promotion_candidates,
            "decision_grade_laws": decision_grade_laws,
            "blocked_promotions": blocked_promotions,
            "demotions": demotions,
        }

    def ingest_real_world_validation(
        self,
        *,
        law_id: str,
        helpful_count: int = 0,
        harmful_count: int = 0,
        neutral_count: int = 0,
        intervention_sensitive_helpful_count: int = 0,
        intervention_sensitive_harmful_count: int = 0,
    ) -> dict[str, Any]:
        total = max(1, helpful_count + harmful_count + neutral_count)
        consistency = helpful_count / total
        contradiction_rate = harmful_count / total
        validation_score = _clamp01(0.7 * consistency + 0.3 * (1.0 - contradiction_rate))
        row = self._real_world_evidence.setdefault(
            law_id,
            {
                "support_count": 0.0,
                "contradiction_count": 0.0,
                "validation_score": 0.0,
                "intervention_sensitive_support": 0.0,
                "intervention_sensitive_contradiction": 0.0,
            },
        )
        row["support_count"] += float(helpful_count + neutral_count)
        row["contradiction_count"] += float(harmful_count)
        row["intervention_sensitive_support"] += float(intervention_sensitive_helpful_count)
        row["intervention_sensitive_contradiction"] += float(intervention_sensitive_harmful_count)
        row["validation_score"] = float(validation_score)
        return dict(row)

    def _target_stage(
        self,
        *,
        support: int,
        contradiction_rate: float,
        real_validation_score: float,
        domain_coverage: float,
        cross_domain_consistency: float,
        intervention_effect_correlation: float,
        intervention_sensitive_support: float,
        intervention_contradiction_rate: float,
        calibrated_law_confidence: float,
        novelty_penalty: float,
        transfer_penalty: float,
    ) -> tuple[str, list[str], list[str]]:
        blockers: list[str] = []
        risk_flags: list[str] = []

        if contradiction_rate >= 0.45:
            return "rejected_or_falsified", ["contradiction_rate_critical"], ["falsification_pressure"]

        stage = "observed_pattern"
        if support >= 4:
            stage = "repeated_pattern"
        if support >= 6 and contradiction_rate <= 0.25 and domain_coverage >= 0.33:
            stage = "context_conditioned_pattern"
        else:
            if support < 6:
                blockers.append("insufficient_repeated_support")
            if contradiction_rate > 0.25:
                blockers.append("contradiction_rate_too_high")
            if domain_coverage < 0.33:
                blockers.append("insufficient_context_coverage")

        if stage == "context_conditioned_pattern" and intervention_effect_correlation >= 0.45:
            stage = "intervention_sensitive_pattern"
        elif stage == "context_conditioned_pattern":
            blockers.append("intervention_signal_too_weak")

        if (
            stage == "intervention_sensitive_pattern"
            and support >= 10
            and contradiction_rate <= 0.16
            and real_validation_score >= 0.62
            and intervention_sensitive_support >= 3
            and intervention_contradiction_rate <= 0.2
            and calibrated_law_confidence >= 0.72
            and (cross_domain_consistency >= 0.62 or domain_coverage <= 0.34)
            and novelty_penalty <= 0.38
            and transfer_penalty <= 0.4
        ):
            stage = "decision_grade_heuristic"
        elif stage == "intervention_sensitive_pattern":
            if support < 10:
                blockers.append("support_below_decision_grade_threshold")
            if contradiction_rate > 0.16:
                blockers.append("contradiction_rate_above_decision_grade_limit")
            if real_validation_score < 0.62:
                blockers.append("insufficient_real_world_validation_score")
            if intervention_sensitive_support < 3:
                blockers.append("intervention_sensitive_evidence_insufficient")
            if intervention_contradiction_rate > 0.2:
                blockers.append("intervention_sensitive_contradiction_too_high")
            if calibrated_law_confidence < 0.72:
                blockers.append("calibrated_reliability_below_decision_grade_threshold")
            if cross_domain_consistency < 0.62 and domain_coverage > 0.34:
                blockers.append("inconsistent_cross_domain_transfer")
            if novelty_penalty > 0.38:
                blockers.append("novelty_penalty_too_high")
            if transfer_penalty > 0.4:
                blockers.append("transfer_penalty_too_high")

        if novelty_penalty > 0.5:
            risk_flags.append("high_novelty_context")
        if transfer_penalty > 0.45:
            risk_flags.append("transfer_risk")
        if contradiction_rate > 0.25:
            risk_flags.append("evidence_drift")

        return stage, sorted(set(blockers)), sorted(set(risk_flags))

    def _stage_index(self, stage: str) -> int:
        if stage == "rejected_or_falsified":
            return -1
        try:
            return STAGES.index(stage)
        except ValueError:
            return -1

    def _recency_weight(self, *, previous: dict[str, Any] | None) -> float:
        if not previous:
            return 1.0
        age = max(0, self._tick - int(previous.get("last_updated", self._tick)))
        return _clamp01(1.0 / (1.0 + 0.35 * age))

    def _intervention_correlation(self, *, law: dict[str, Any], intervention: dict[str, Any]) -> float:
        base = _clamp01(_f(law.get("counterfactual_consistency"), 0.0))
        scenario_rankings = list(intervention.get("scenario_rankings") or [])
        if scenario_rankings:
            aligned = [_clamp01(abs(_f(item.get("risk_delta"), 0.0))) for item in scenario_rankings[:3]]
            if aligned:
                base = _clamp01(0.65 * base + 0.35 * (sum(aligned) / len(aligned)))
        return base

    def _reliability_map(self, reliability: dict[str, Any]) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for row in list(reliability.get("law_applicability") or []):
            law_id = str(row.get("law_id") or "")
            if not law_id:
                continue
            trace = dict(row.get("reliability_trace") or {})
            out[law_id] = {
                "calibrated_applicability_confidence": row.get("calibrated_applicability_confidence", row.get("local_law_confidence", 0.0)),
                "reliability_band": trace.get("reliability_band"),
                "support_band": trace.get("support_band"),
                "novelty_penalty": trace.get("novelty_penalty"),
                "transfer_penalty": trace.get("transfer_penalty"),
            }
        return out

    def _contradictions_map(self, falsification: dict[str, Any]) -> dict[str, int]:
        out: dict[str, int] = {}
        for row in list((falsification.get("falsification_summary") or {}).get("law_evaluations") or []):
            law_id = str(row.get("law_id") or "")
            if not law_id:
                continue
            counters = dict((row.get("counterexample_summary") or {}))
            out[law_id] = max(0, int(_f(counters.get("contradiction_count"), 0)))
        return out
