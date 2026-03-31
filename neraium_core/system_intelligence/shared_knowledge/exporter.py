from __future__ import annotations

from typing import Any

from ..federation.types import (
    CalibrationSummary,
    InterventionEvidenceSummary,
    LawEvidenceSummary,
    PacketMetadata,
    SharedKnowledgePacket,
    TrajectoryFamilySummary,
)


def _bucket(value: float, cuts: tuple[float, float] = (0.33, 0.66)) -> str:
    if value < cuts[0]:
        return "low"
    if value < cuts[1]:
        return "mid"
    return "high"


def export_local_structural_knowledge(
    *,
    instance_id: str,
    step: int,
    trajectory_info: dict[str, Any],
    law_info: dict[str, Any],
    intervention_info: dict[str, Any],
) -> SharedKnowledgePacket:
    trajectory_rows: list[TrajectoryFamilySummary] = []
    if trajectory_info.get("status") == "ready":
        current = dict((trajectory_info.get("trajectory_families") or {}).get("current") or {})
        trajectory_rows.append(
            TrajectoryFamilySummary(
                family_id=str(trajectory_info.get("current_trajectory_family", "unknown")),
                path_family=str(trajectory_info.get("current_trajectory_path_family", "unknown")),
                support=int(trajectory_info.get("support", current.get("support", 0)) or 0),
                escalation_frequency=float(trajectory_info.get("historical_escalation_frequency", current.get("escalation_frequency", 0.0)) or 0.0),
                phase_shift_frequency=float(trajectory_info.get("historical_phase_shift_frequency", current.get("phase_shift_frequency", 0.0)) or 0.0),
                novelty_rate=float(trajectory_info.get("novelty_score", 0.5)),
                representative_segment=list(trajectory_info.get("representative_segment") or current.get("representative_segment") or []),
                reliability=max(0.05, min(1.0, 1.0 - float(trajectory_info.get("novelty_score", 0.5)))),
            )
        )

    law_rows = [
        LawEvidenceSummary(
            law_key=str(row.get("law_id", "unknown")),
            condition_family=str((row.get("applicability") or {}).get("trajectory_family", "unknown")),
            mechanism=str(row.get("mechanism", "unknown")),
            outcome_family="escalation",
            support=int(row.get("support", 0)),
            effect_estimate=float(row.get("estimated_effect", 0.0)),
            uncertainty=max(0.0, min(1.0, 1.0 - float(row.get("confidence", 0.0)))),
            consistency=float(row.get("consistency_across_assets_runs", 0.0)),
            source_diversity=1,
        )
        for row in list(law_info.get("law_candidates") or [])
    ]

    intervention_rows: list[InterventionEvidenceSummary] = []
    for ranked in list((intervention_info.get("recommendation") or {}).get("ranked_interventions") or []):
        h = dict((ranked.get("evidence_sources") or {}).get("historical_evidence") or {})
        context = dict(intervention_info.get("context") or {})
        intervention_rows.append(
            InterventionEvidenceSummary(
                context_bucket=f"{context.get('trajectory_family', 'unknown')}|{context.get('regime', 'unknown')}",
                trajectory_family=str(context.get("trajectory_family", "unknown")),
                intervention_type=str(ranked.get("intervention_type", "other")),
                intervention_target=str(ranked.get("intervention_target", "system")),
                support=int(h.get("support", 0)),
                effectiveness=float(h.get("intervention_effectiveness", 0.0)),
                worsening_rate=float(h.get("worsening_signal", 0.0)),
                reliability=max(0.05, min(1.0, 1.0 - float(h.get("uncertainty", 0.7)))),
            )
        )

    confidence = float((intervention_info.get("uncertainty_summary") or {}).get("confidence", 0.0))
    novelty = float((intervention_info.get("context") or {}).get("novelty_score", 0.5))
    calibration = [
        CalibrationSummary(
            bucket=str((intervention_info.get("context") or {}).get("trajectory_family", "unknown")),
            support_band=_bucket(float((intervention_info.get("historical_evidence") or {}).get("support_summary", {}).get("support_count", 0)) / 10.0),
            novelty_band=_bucket(novelty),
            support=int((intervention_info.get("historical_evidence") or {}).get("support_summary", {}).get("support_count", 0)),
            confidence_mean=confidence,
            realized_success_rate=float(max(0.0, confidence - 0.08)),
        )
    ]

    return SharedKnowledgePacket(
        metadata=PacketMetadata(source_instance_id=instance_id, generated_at_step=int(step), recency_score=1.0),
        trajectory_families=trajectory_rows,
        law_evidence=law_rows,
        intervention_evidence=intervention_rows,
        calibration=calibration,
        mechanism_family_associations=list((trajectory_info.get("trajectory_families") or {}).get("current", {}).get("co_observed_mechanisms") or []),
    )
