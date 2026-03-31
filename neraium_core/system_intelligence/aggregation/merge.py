from __future__ import annotations

from collections import defaultdict
from statistics import median
from typing import Any

from ..federation.types import (
    CalibrationSummary,
    InterventionEvidenceSummary,
    LawEvidenceSummary,
    PacketMetadata,
    SharedKnowledgePacket,
    TrajectoryFamilySummary,
)


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def merge_shared_packets(packets: list[SharedKnowledgePacket], *, current_step: int) -> dict[str, Any]:
    if not packets:
        return {
            "status": "empty",
            "global_packet": SharedKnowledgePacket(metadata=PacketMetadata(source_instance_id="global", generated_at_step=current_step)),
            "conflict_flags": [],
        }

    traj_groups: dict[str, list[TrajectoryFamilySummary]] = defaultdict(list)
    law_groups: dict[str, list[LawEvidenceSummary]] = defaultdict(list)
    int_groups: dict[str, list[InterventionEvidenceSummary]] = defaultdict(list)
    cal_groups: dict[str, list[CalibrationSummary]] = defaultdict(list)

    for packet in packets:
        age = max(0, current_step - int(packet.metadata.generated_at_step))
        recency = _clip01(float(packet.metadata.recency_score) * (0.92 ** age))
        for row in packet.trajectory_families:
            c = TrajectoryFamilySummary(**{**row.__dict__, "reliability": _clip01(row.reliability * recency)})
            traj_groups[f"{row.family_id}|{row.path_family}"].append(c)
        for row in packet.law_evidence:
            c = LawEvidenceSummary(**{**row.__dict__, "uncertainty": _clip01(row.uncertainty + (1.0 - recency) * 0.25)})
            law_groups[row.law_key].append(c)
        for row in packet.intervention_evidence:
            c = InterventionEvidenceSummary(**{**row.__dict__, "reliability": _clip01(row.reliability * recency)})
            int_groups[f"{row.context_bucket}|{row.intervention_type}|{row.intervention_target}"].append(c)
        for row in packet.calibration:
            cal_groups[f"{row.bucket}|{row.support_band}|{row.novelty_band}"].append(row)

    conflict_flags: list[dict[str, Any]] = []

    merged_trajectory: list[TrajectoryFamilySummary] = []
    for key, rows in traj_groups.items():
        support = sum(r.support for r in rows)
        merged_trajectory.append(
            TrajectoryFamilySummary(
                family_id=rows[0].family_id,
                path_family=rows[0].path_family,
                support=support,
                escalation_frequency=sum(r.escalation_frequency * r.support for r in rows) / max(1, support),
                phase_shift_frequency=sum(r.phase_shift_frequency * r.support for r in rows) / max(1, support),
                novelty_rate=sum(r.novelty_rate * r.support for r in rows) / max(1, support),
                representative_segment=rows[0].representative_segment,
                reliability=sum(r.reliability * r.support for r in rows) / max(1, support),
            )
        )

    merged_laws: list[LawEvidenceSummary] = []
    law_status: dict[str, str] = {}
    for law_key, rows in law_groups.items():
        support = sum(r.support for r in rows)
        weighted_effect = sum(r.effect_estimate * r.support for r in rows) / max(1, support)
        signs = {1 if r.effect_estimate >= 0 else -1 for r in rows if abs(r.effect_estimate) > 0.03}
        conflicting = len(signs) > 1
        if conflicting:
            conflict_flags.append({"type": "law_conflict", "law_key": law_key, "detail": "Opposing effect signs across systems"})
        consistency = _clip01(sum(r.consistency for r in rows) / max(1, len(rows)) - (0.3 if conflicting else 0.0))
        merged_laws.append(
            LawEvidenceSummary(
                law_key=law_key,
                condition_family=rows[0].condition_family,
                mechanism=rows[0].mechanism,
                outcome_family=rows[0].outcome_family,
                support=support,
                effect_estimate=weighted_effect,
                uncertainty=min(1.0, median([r.uncertainty for r in rows]) + (0.2 if conflicting else 0.0)),
                consistency=consistency,
                source_diversity=len(rows),
            )
        )
        law_status[law_key] = (
            "globally_conflicting"
            if conflicting
            else "globally_consistent"
            if support >= 10 and consistency >= 0.6
            else "multi_system_recurrent"
            if len(rows) >= 2
            else "local_only"
        )

    merged_interventions: list[InterventionEvidenceSummary] = []
    for _, rows in int_groups.items():
        support = sum(r.support for r in rows)
        harmful = sum(1 for r in rows if r.worsening_rate > max(0.05, r.effectiveness))
        if harmful > 0 and len(rows) > 1:
            conflict_flags.append({"type": "intervention_conflict", "context_bucket": rows[0].context_bucket, "intervention": rows[0].intervention_type})
        merged_interventions.append(
            InterventionEvidenceSummary(
                context_bucket=rows[0].context_bucket,
                trajectory_family=rows[0].trajectory_family,
                intervention_type=rows[0].intervention_type,
                intervention_target=rows[0].intervention_target,
                support=support,
                effectiveness=sum(r.effectiveness * r.support for r in rows) / max(1, support),
                worsening_rate=sum(r.worsening_rate * r.support for r in rows) / max(1, support),
                reliability=_clip01(sum(r.reliability * r.support for r in rows) / max(1, support) - 0.2 * harmful),
            )
        )

    merged_calibration = [
        CalibrationSummary(
            bucket=rows[0].bucket,
            support_band=rows[0].support_band,
            novelty_band=rows[0].novelty_band,
            support=sum(r.support for r in rows),
            confidence_mean=sum(r.confidence_mean * r.support for r in rows) / max(1, sum(r.support for r in rows)),
            realized_success_rate=sum(r.realized_success_rate * r.support for r in rows) / max(1, sum(r.support for r in rows)),
        )
        for rows in cal_groups.values()
    ]

    global_packet = SharedKnowledgePacket(
        metadata=PacketMetadata(source_instance_id="global", generated_at_step=current_step, recency_score=1.0),
        trajectory_families=merged_trajectory,
        law_evidence=merged_laws,
        intervention_evidence=merged_interventions,
        calibration=merged_calibration,
    )

    return {
        "status": "ready",
        "global_packet": global_packet,
        "conflict_flags": conflict_flags,
        "law_status": law_status,
    }
