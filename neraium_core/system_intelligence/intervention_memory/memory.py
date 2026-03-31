from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import math


@dataclass
class InterventionEvidenceRecord:
    asset_id: str
    pre_intervention_state: dict[str, float]
    post_intervention_state: dict[str, float]
    trajectory_family: str
    transition_path: str
    regime: str
    mechanism_context: list[str]
    law_context: list[str]
    intervention_type: str
    intervention_target: str
    outcome_delta: dict[str, float]
    evidence_metadata: dict[str, Any] = field(default_factory=dict)


class InterventionMemoryStore:
    """Compact, inspectable store for intervention evidence.

    This layer tracks empirical structural outcomes and does not claim causal certainty.
    """

    def __init__(self) -> None:
        self.records: list[InterventionEvidenceRecord] = []
        self._pending: dict[str, dict[str, Any]] = {}
        self.feedback_evidence: dict[str, dict[str, float]] = {}

    def register_intervention_start(
        self,
        *,
        asset_id: str,
        intervention_type: str,
        intervention_target: str,
        context: dict[str, Any],
        pre_state: dict[str, float],
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._pending[asset_id] = {
            "intervention_type": intervention_type,
            "intervention_target": intervention_target,
            "context": context,
            "pre_state": pre_state,
            "metadata": metadata or {},
        }
        return {"status": "registered", "asset_id": asset_id}

    def finalize_if_ready(self, *, asset_id: str, post_state: dict[str, float]) -> dict[str, Any] | None:
        pending = self._pending.pop(asset_id, None)
        if pending is None:
            return None

        pre_state = dict(pending["pre_state"])
        context = dict(pending["context"])
        outcome = {
            "escalation_reduction": max(0.0, pre_state["escalation_probability"] - post_state["escalation_probability"]),
            "reversibility_improvement": max(0.0, post_state["reversibility_score"] - pre_state["reversibility_score"]),
            "distance_from_critical_improvement": max(0.0, post_state["distance_to_critical_region"] - pre_state["distance_to_critical_region"]),
            "escalation_worsening": max(0.0, post_state["escalation_probability"] - pre_state["escalation_probability"]),
        }

        record = InterventionEvidenceRecord(
            asset_id=asset_id,
            pre_intervention_state=pre_state,
            post_intervention_state=post_state,
            trajectory_family=str(context.get("trajectory_family", "unknown")),
            transition_path=str(context.get("transition_path", "unknown")),
            regime=str(context.get("regime", "unknown")),
            mechanism_context=list(context.get("mechanism_candidates") or []),
            law_context=list(context.get("law_candidates") or []),
            intervention_type=str(pending["intervention_type"]),
            intervention_target=str(pending["intervention_target"]),
            outcome_delta=outcome,
            evidence_metadata=dict(pending.get("metadata") or {}),
        )
        self.records.append(record)
        return {
            "status": "recorded",
            "record_count": len(self.records),
            "record": self._record_to_dict(record),
        }

    def _context_match(self, record: InterventionEvidenceRecord, context: dict[str, Any]) -> float:
        score = 0.0
        total = 0.0

        def _add(condition: bool, weight: float) -> None:
            nonlocal score, total
            total += weight
            if condition:
                score += weight

        _add(record.trajectory_family == context.get("trajectory_family"), 0.25)
        _add(record.transition_path == context.get("transition_path"), 0.15)
        _add(record.regime == context.get("regime"), 0.15)

        current_mech = set(context.get("mechanism_candidates") or [])
        current_laws = set(context.get("law_candidates") or [])
        record_mech = set(record.mechanism_context)
        record_law = set(record.law_context)

        mech_overlap = len(current_mech & record_mech) / max(1, len(current_mech | record_mech))
        law_overlap = len(current_laws & record_law) / max(1, len(current_laws | record_law))
        score += mech_overlap * 0.25
        score += law_overlap * 0.20
        total += 0.45
        latent_now = [float(x) for x in list(context.get("latent_state") or [])]
        latent_then = [float(x) for x in list((record.evidence_metadata or {}).get("latent_state") or [])]
        if latent_now and latent_then and len(latent_now) == len(latent_then):
            mse = sum((a - b) ** 2 for a, b in zip(latent_now, latent_then)) / max(1.0, float(len(latent_now)))
            latent_similarity = 1.0 / (1.0 + math.sqrt(max(0.0, mse)))
            score += latent_similarity * 0.20
            total += 0.20
        return 0.0 if total <= 0 else score / total

    def query(
        self,
        *,
        context: dict[str, Any],
        intervention_type: str,
        intervention_target: str | None = None,
        min_match: float = 0.35,
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for rec in self.records:
            if rec.intervention_type != intervention_type:
                continue
            if intervention_target and rec.intervention_target != intervention_target:
                continue
            match = self._context_match(rec, context)
            if match < min_match:
                continue
            out.append({"match_quality": round(match, 4), "record": self._record_to_dict(rec)})
        return sorted(out, key=lambda x: x["match_quality"], reverse=True)

    def support_summary(self) -> dict[str, Any]:
        helpful: list[dict[str, Any]] = []
        neutral: list[dict[str, Any]] = []
        harmful: list[dict[str, Any]] = []
        recovery: list[dict[str, Any]] = []

        for record in self.records:
            payload = {
                "intervention_type": record.intervention_type,
                "intervention_target": record.intervention_target,
                "trajectory_family": record.trajectory_family,
                "outcome_delta": record.outcome_delta,
            }
            improvement = (
                record.outcome_delta.get("escalation_reduction", 0.0)
                + record.outcome_delta.get("reversibility_improvement", 0.0)
                + record.outcome_delta.get("distance_from_critical_improvement", 0.0)
            )
            worsening = record.outcome_delta.get("escalation_worsening", 0.0)

            if record.outcome_delta.get("reversibility_improvement", 0.0) > 0.03:
                recovery.append(payload)
            if worsening > 0.04:
                harmful.append(payload)
            elif improvement > 0.08:
                helpful.append(payload)
            else:
                neutral.append(payload)

        return {
            "helpful_interventions": helpful,
            "neutral_interventions": neutral,
            "harmful_interventions": harmful,
            "recovery_associated_interventions": recovery,
            "support_count": len(self.records),
            "feedback_evidence": self.feedback_evidence,
        }

    def ingest_feedback_event(
        self,
        *,
        intervention_type: str,
        action: str,
        outcome_label: str,
    ) -> None:
        key = str(intervention_type or "other")
        bucket = self.feedback_evidence.setdefault(
            key,
            {
                "accepted_success": 0.0,
                "accepted_harmful": 0.0,
                "overridden": 0.0,
                "ignored": 0.0,
            },
        )
        label = str(outcome_label or "neutral")
        if action == "accepted" and label in {"helpful", "recovery-associated"}:
            bucket["accepted_success"] += 1.0
        elif action == "accepted" and label == "harmful":
            bucket["accepted_harmful"] += 1.0
        elif action == "overridden":
            bucket["overridden"] += 1.0
        else:
            bucket["ignored"] += 1.0

    @staticmethod
    def _record_to_dict(record: InterventionEvidenceRecord) -> dict[str, Any]:
        return {
            "asset_id": record.asset_id,
            "pre_intervention_state": record.pre_intervention_state,
            "post_intervention_state": record.post_intervention_state,
            "trajectory_family": record.trajectory_family,
            "transition_path": record.transition_path,
            "regime": record.regime,
            "mechanism_context": record.mechanism_context,
            "law_context": record.law_context,
            "intervention_type": record.intervention_type,
            "intervention_target": record.intervention_target,
            "outcome_delta": record.outcome_delta,
            "evidence_metadata": record.evidence_metadata,
        }
