from __future__ import annotations

from neraium_core.system_intelligence.aggregation.merge import merge_shared_packets
from neraium_core.system_intelligence.federation.layer import CrossSystemStructuralIntelligenceLayer
from neraium_core.system_intelligence.federation.schema import packet_from_dict, packet_has_raw_telemetry, packet_to_dict
from neraium_core.system_intelligence.shared_knowledge.exporter import export_local_structural_knowledge
from neraium_core.system_intelligence.transfer.adaptation import adapt_global_to_local


def _trajectory(path_family: str, *, support: int, novelty: float = 0.2) -> dict:
    return {
        "status": "ready",
        "current_trajectory_family": f"family::{path_family}",
        "current_trajectory_archetype": f"family::{path_family}",
        "current_trajectory_path_family": path_family,
        "historical_escalation_frequency": 0.75 if path_family == "escalating" else 0.2,
        "historical_phase_shift_frequency": 0.35,
        "support": support,
        "novelty_score": novelty,
        "representative_segment": [[0.1, 0.2, 0.3], [0.2, 0.25, 0.4]],
        "trajectory_families": {"current": {"co_observed_mechanisms": [{"mechanism": "A->B", "support": 4}]}}
    }


def _laws(effect: float, support: int, confidence: float = 0.7) -> dict:
    return {
        "status": "ready",
        "law_candidates": [
            {
                "law_id": "law::family::escalating|mech:A->B",
                "mechanism": "mech:A->B",
                "support": support,
                "estimated_effect": effect,
                "confidence": confidence,
                "consistency_across_assets_runs": 0.8,
                "applicability": {"trajectory_family": "escalating"},
            }
        ],
    }


def _intervention(support: int, effect: float, uncertainty: float, novelty: float = 0.2) -> dict:
    return {
        "context": {"trajectory_family": "escalating", "regime": "critical", "novelty_score": novelty},
        "historical_evidence": {"support_summary": {"support_count": support}},
        "uncertainty_summary": {"confidence": max(0.0, effect - 0.1)},
        "recommendation": {
            "ranked_interventions": [
                {
                    "name": "remove_top_driver_contribution",
                    "intervention_type": "remove_or_suppress_top_driver",
                    "intervention_target": "top_driver",
                    "evidence_sources": {
                        "historical_evidence": {
                            "support": support,
                            "intervention_effectiveness": effect,
                            "uncertainty": uncertainty,
                            "worsening_signal": 0.0,
                        }
                    },
                }
            ]
        },
    }


def test_export_import_packet_without_raw_telemetry() -> None:
    packet = export_local_structural_knowledge(
        instance_id="system-a",
        step=4,
        trajectory_info=_trajectory("escalating", support=3),
        law_info=_laws(0.34, support=4),
        intervention_info=_intervention(2, effect=0.32, uncertainty=0.3),
    )
    payload = packet_to_dict(packet)
    assert payload["metadata"]["source_instance_id"] == "system-a"
    assert not packet_has_raw_telemetry(payload)
    restored = packet_from_dict(payload)
    assert restored.trajectory_families[0].family_id.startswith("family::")


def test_shared_trajectory_knowledge_boosts_sparse_local_matching() -> None:
    local = export_local_structural_knowledge(instance_id="sparse", step=3, trajectory_info=_trajectory("escalating", support=1, novelty=0.3), law_info=_laws(0.22, 2), intervention_info=_intervention(1, 0.2, 0.6))
    remote1 = export_local_structural_knowledge(instance_id="remote-1", step=3, trajectory_info=_trajectory("escalating", support=9, novelty=0.1), law_info=_laws(0.28, 8), intervention_info=_intervention(6, 0.41, 0.2))
    remote2 = export_local_structural_knowledge(instance_id="remote-2", step=3, trajectory_info=_trajectory("escalating", support=7, novelty=0.15), law_info=_laws(0.3, 7), intervention_info=_intervention(5, 0.37, 0.25))
    global_snapshot = merge_shared_packets([local, remote1, remote2], current_step=3)
    adapted = adapt_global_to_local(local_packet=local, global_packet=global_snapshot["global_packet"], trajectory_info=_trajectory("escalating", support=1, novelty=0.3), local_laws=_laws(0.22, 2), local_intervention=_intervention(1, 0.2, 0.6))
    assert adapted["shared_trajectory_knowledge"]["status"] == "globally_known_but_locally_novel"
    assert adapted["shared_trajectory_knowledge"]["global_support"] > adapted["shared_trajectory_knowledge"]["local_support"]


def test_law_confidence_requires_cross_system_consistency() -> None:
    local = export_local_structural_knowledge(instance_id="a", step=5, trajectory_info=_trajectory("escalating", support=2), law_info=_laws(0.35, 3, confidence=0.7), intervention_info=_intervention(1, 0.2, 0.5))
    same = export_local_structural_knowledge(instance_id="b", step=5, trajectory_info=_trajectory("escalating", support=5), law_info=_laws(0.33, 7, confidence=0.75), intervention_info=_intervention(2, 0.22, 0.5))
    conflict = export_local_structural_knowledge(instance_id="c", step=5, trajectory_info=_trajectory("escalating", support=5), law_info=_laws(-0.3, 7, confidence=0.7), intervention_info=_intervention(2, 0.22, 0.5))

    consistent = merge_shared_packets([local, same], current_step=5)
    conflicting = merge_shared_packets([local, conflict], current_step=5)

    a1 = adapt_global_to_local(local_packet=local, global_packet=consistent["global_packet"], trajectory_info=_trajectory("escalating", support=2), local_laws=_laws(0.35, 3, confidence=0.7), local_intervention=_intervention(1, 0.2, 0.5))
    a2 = adapt_global_to_local(local_packet=local, global_packet=conflicting["global_packet"], trajectory_info=_trajectory("escalating", support=2), local_laws=_laws(0.35, 3, confidence=0.7), local_intervention=_intervention(1, 0.2, 0.5))

    assert a1["shared_law_intelligence"][0]["generalization_confidence"] > a2["shared_law_intelligence"][0]["generalization_confidence"]
    assert conflicting["conflict_flags"]


def test_local_evidence_dominates_when_stronger_than_transfer() -> None:
    local = export_local_structural_knowledge(instance_id="local", step=10, trajectory_info=_trajectory("escalating", support=12, novelty=0.1), law_info=_laws(0.31, 12), intervention_info=_intervention(8, 0.55, 0.1, novelty=0.1))
    remote = export_local_structural_knowledge(instance_id="remote", step=10, trajectory_info=_trajectory("drift", support=3, novelty=0.2), law_info=_laws(0.05, 2, confidence=0.3), intervention_info=_intervention(1, 0.1, 0.8, novelty=0.2))
    merged = merge_shared_packets([local, remote], current_step=10)
    adapted = adapt_global_to_local(local_packet=local, global_packet=merged["global_packet"], trajectory_info=_trajectory("escalating", support=12, novelty=0.1), local_laws=_laws(0.31, 12), local_intervention=_intervention(8, 0.55, 0.1, novelty=0.1))
    assert adapted["evidence_source_weights"]["local_evidence_weight"] > adapted["evidence_source_weights"]["global_evidence_weight"]


def test_intervention_transfer_helps_sparse_local_with_context_match() -> None:
    local = export_local_structural_knowledge(instance_id="sparse", step=7, trajectory_info=_trajectory("escalating", support=1, novelty=0.2), law_info=_laws(0.25, 2), intervention_info=_intervention(1, 0.08, 0.8))
    remote = export_local_structural_knowledge(instance_id="dense", step=7, trajectory_info=_trajectory("escalating", support=9, novelty=0.1), law_info=_laws(0.3, 8), intervention_info=_intervention(9, 0.62, 0.2))
    merged = merge_shared_packets([local, remote], current_step=7)
    adapted = adapt_global_to_local(local_packet=local, global_packet=merged["global_packet"], trajectory_info=_trajectory("escalating", support=1, novelty=0.2), local_laws=_laws(0.25, 2), local_intervention=_intervention(1, 0.08, 0.8))
    shared = adapted["shared_intervention_evidence"][0]
    assert shared["blended_effectiveness"] > shared["local_effectiveness"]


def test_novelty_penalty_applies_for_out_of_distribution_system() -> None:
    local = export_local_structural_knowledge(instance_id="novel", step=8, trajectory_info=_trajectory("spiral-collapse", support=1, novelty=0.95), law_info=_laws(0.15, 1), intervention_info=_intervention(1, 0.2, 0.8, novelty=0.95))
    remote = export_local_structural_knowledge(instance_id="global", step=8, trajectory_info=_trajectory("escalating", support=8, novelty=0.1), law_info=_laws(0.32, 9), intervention_info=_intervention(8, 0.5, 0.25))
    merged = merge_shared_packets([local, remote], current_step=8)
    adapted = adapt_global_to_local(local_packet=local, global_packet=merged["global_packet"], trajectory_info=_trajectory("spiral-collapse", support=1, novelty=0.95), local_laws=_laws(0.15, 1), local_intervention=_intervention(1, 0.2, 0.8, novelty=0.95))
    assert adapted["evidence_source_weights"]["mismatch_penalty"] >= 0.5
    assert adapted["shared_trajectory_knowledge"]["status"] == "globally_unknown"


def test_layer_integration_exposes_cross_system_output_sections() -> None:
    layer = CrossSystemStructuralIntelligenceLayer(instance_id="site-a")
    remote = export_local_structural_knowledge(instance_id="site-b", step=1, trajectory_info=_trajectory("escalating", support=8), law_info=_laws(0.3, 8), intervention_info=_intervention(7, 0.5, 0.2))
    layer.ingest_remote_packet(remote)

    out = layer.update(
        trajectory_info=_trajectory("escalating", support=1, novelty=0.2),
        law_info=_laws(0.22, 2),
        intervention_info=_intervention(1, 0.1, 0.75),
    )
    cross = out["cross_system_structural_intelligence"]
    assert "shared_trajectory_knowledge" in cross
    assert "shared_law_intelligence" in cross
    assert "shared_intervention_evidence" in cross
    assert "transfer_confidence" in cross
