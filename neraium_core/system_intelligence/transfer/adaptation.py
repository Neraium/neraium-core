from __future__ import annotations

from typing import Any

from ..federation.types import SharedKnowledgePacket


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _support_weight(support: int, strong: int = 8) -> float:
    return _clip01(float(support) / max(1.0, float(strong)))


def adapt_global_to_local(
    *,
    local_packet: SharedKnowledgePacket,
    global_packet: SharedKnowledgePacket,
    trajectory_info: dict[str, Any],
    local_laws: dict[str, Any],
    local_intervention: dict[str, Any],
) -> dict[str, Any]:
    local_traj = (local_packet.trajectory_families or [None])[0]
    local_support = int(getattr(local_traj, "support", 0) or 0)
    novelty = float(trajectory_info.get("novelty_score", 0.5))

    global_traj_match = None
    if local_traj is not None:
        same = [row for row in global_packet.trajectory_families if row.path_family == local_traj.path_family]
        if same:
            global_traj_match = max(same, key=lambda row: row.support)

    mismatch_penalty = _clip01(0.55 * novelty + (0.25 if global_traj_match is None else 0.0))
    local_evidence_weight = _clip01(0.75 * _support_weight(local_support, strong=6) + 0.25)
    global_raw = 1.0 - local_evidence_weight
    global_evidence_weight = _clip01(global_raw * (1.0 - mismatch_penalty))
    if local_support >= 10 and mismatch_penalty > 0.2:
        global_evidence_weight *= 0.25

    transfer_confidence = _clip01(global_evidence_weight * (1.0 - mismatch_penalty))

    law_lookup = {row.law_key: row for row in global_packet.law_evidence}
    shared_law_intel = []
    for law in list(local_laws.get("law_candidates") or []):
        law_id = str(law.get("law_id", ""))
        g = law_lookup.get(law_id)
        if g is None:
            shared_law_intel.append({
                "law_id": law_id,
                "cross_system_support": 0,
                "cross_system_consistency": 0.0,
                "generalization_confidence": 0.0,
                "status": "local_only",
                "applicability_notes": "No shared evidence yet.",
            })
            continue
        consistency = _clip01(g.consistency * (1.0 - g.uncertainty))
        gen_conf = _clip01(_support_weight(g.support, strong=12) * consistency * (1.0 - mismatch_penalty))
        status = "globally_consistent" if g.support >= 12 and consistency >= 0.6 else "multi_system_recurrent"
        if g.uncertainty > 0.65:
            status = "globally_conflicting"
        shared_law_intel.append({
            "law_id": law_id,
            "cross_system_support": g.support,
            "cross_system_consistency": round(consistency, 4),
            "generalization_confidence": round(gen_conf, 4),
            "status": status,
            "applicability_notes": "Global support is advisory; local contradiction blocks promotion.",
        })

    ranked = list((local_intervention.get("recommendation") or {}).get("ranked_interventions") or [])
    global_int_lookup = {f"{row.context_bucket}|{row.intervention_type}|{row.intervention_target}": row for row in global_packet.intervention_evidence}
    context = dict(local_intervention.get("context") or {})
    bucket = f"{context.get('trajectory_family', 'unknown')}|{context.get('regime', 'unknown')}"
    shared_intervention = []
    for row in ranked:
        key = f"{bucket}|{row.get('intervention_type', 'other')}|{row.get('intervention_target', 'system')}"
        g = global_int_lookup.get(key)
        global_effect = float(g.effectiveness) if g is not None else 0.0
        context_match = 1.0 if g is not None else 0.0
        borrowed_weight = global_evidence_weight * context_match
        local_effect = float(((row.get("evidence_sources") or {}).get("historical_evidence") or {}).get("intervention_effectiveness", 0.0))
        blended = _clip01(local_evidence_weight * local_effect + borrowed_weight * global_effect)
        if local_effect > 0.4 and global_effect < 0.1:
            blended = max(blended, local_effect * 0.9)
        shared_intervention.append({
            "name": row.get("name"),
            "local_effectiveness": round(local_effect, 4),
            "global_effectiveness": round(global_effect, 4),
            "blended_effectiveness": round(blended, 4),
            "context_family_match_quality": round(context_match, 4),
            "evidence_source_weights": {
                "local": round(local_evidence_weight, 4),
                "global": round(borrowed_weight, 4),
            },
            "global_penalized_for_mismatch": mismatch_penalty > 0.25,
        })

    traj_novelty = "locally_known"
    if global_traj_match is None or int(getattr(global_traj_match, "support", 0) or 0) <= local_support + 1:
        traj_novelty = "globally_unknown"
    elif local_traj is not None and global_traj_match is not None and local_traj.support <= 2:
        traj_novelty = "globally_known_but_locally_novel"

    return {
        "transfer_confidence": round(transfer_confidence, 4),
        "evidence_source_weights": {
            "local_evidence_weight": round(local_evidence_weight, 4),
            "global_evidence_weight": round(global_evidence_weight, 4),
            "mismatch_penalty": round(mismatch_penalty, 4),
        },
        "shared_trajectory_knowledge": {
            "status": traj_novelty,
            "local_support": local_support,
            "global_support": int(getattr(global_traj_match, "support", 0) or 0),
            "novelty_flags": [traj_novelty],
        },
        "shared_law_intelligence": shared_law_intel,
        "shared_intervention_evidence": shared_intervention,
        "shared_calibration": {
            "borrowed": bool(global_packet.calibration) and local_support < 4,
            "local_sparse": local_support < 4,
            "calibration_buckets": [row.bucket for row in global_packet.calibration[:5]],
        },
        "generalization_warnings": [
            "Global knowledge is assistive only and cannot override strong contradictory local evidence."
        ],
    }
