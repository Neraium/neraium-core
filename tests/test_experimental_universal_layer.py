from __future__ import annotations

from neraium_core.system_intelligence.universal.experimental_layer import ExperimentalUniversalStructuralLayer


def _trajectory(style: str) -> list[list[float]]:
    if style == "drift":
        return [[i * 0.03, i * 0.015, i * 0.01] for i in range(18)]
    if style == "oscillatory":
        return [[i * 0.01, (1 if i % 2 else -1) * 0.08, i * 0.005] for i in range(18)]
    if style == "reversible":
        up = [[i * 0.02, i * 0.01, 0.0] for i in range(9)]
        dn = [[(17 - i) * 0.02, (17 - i) * 0.01, 0.0] for i in range(9, 18)]
        return up + dn
    return [[0.0, 0.0, i * 0.01] for i in range(18)]


def _intervention(effect: float) -> dict:
    return {
        "recommendation": {
            "ranked_interventions": [
                {
                    "intervention_type": "remove_or_suppress_top_driver",
                    "evidence_sources": {
                        "historical_evidence": {
                            "intervention_effectiveness": effect,
                            "uncertainty": 0.2,
                        }
                    },
                }
            ]
        }
    }


def test_cross_domain_archetype_and_similarity_outputs_present() -> None:
    layer = ExperimentalUniversalStructuralLayer()
    out1 = layer.update(
        system_id="asset-a",
        system_type="compressor",
        domain="energy",
        latent_trajectory=_trajectory("drift"),
        escalating=True,
        mechanism_name="pressure->temperature",
        intervention_info=_intervention(0.4),
    )
    exp = out1["experimental_universal_structural_intelligence"]
    assert exp["status"] == "experimental"
    assert exp["safety_boundary"]["can_drive_operator_recommendations"] is False
    assert exp["cross_domain_archetype"]["cross_domain_archetype_id"].startswith("cross::")

    out2 = layer.update(
        system_id="asset-b",
        system_type="pump",
        domain="water",
        latent_trajectory=_trajectory("drift"),
        escalating=True,
        mechanism_name="pressure->temperature",
        intervention_info=_intervention(0.45),
    )
    sim = out2["experimental_universal_structural_intelligence"]["domain_similarity_map"]
    assert sim["nearest_systems"]
    assert sim["transfer_confidence"] >= 0.0


def test_universal_law_candidate_requires_cross_domain_consistency() -> None:
    layer = ExperimentalUniversalStructuralLayer()
    for i in range(8):
        layer.update(
            system_id=f"e-{i}",
            system_type="compressor",
            domain="energy",
            latent_trajectory=_trajectory("drift"),
            escalating=True,
            mechanism_name="coupling::A",
            intervention_info=_intervention(0.5),
        )
    for i in range(8):
        layer.update(
            system_id=f"w-{i}",
            system_type="pump",
            domain="water",
            latent_trajectory=_trajectory("drift"),
            escalating=True,
            mechanism_name="coupling::A",
            intervention_info=_intervention(0.5),
        )
    out = layer.update(
        system_id="m-1",
        system_type="reactor",
        domain="manufacturing",
        latent_trajectory=_trajectory("drift"),
        escalating=True,
        mechanism_name="coupling::A",
        intervention_info=_intervention(0.45),
    )
    law = out["experimental_universal_structural_intelligence"]["universal_structural_law_candidate"]
    assert law["cross_domain_support"] >= 17
    assert law["classification"] in {"multi_system_pattern", "candidate_universal_structural_law"}

    out_fail = layer.update(
        system_id="x-1",
        system_type="reactor",
        domain="manufacturing",
        latent_trajectory=_trajectory("oscillatory"),
        escalating=False,
        mechanism_name="coupling::A",
        intervention_info=_intervention(-0.1),
    )
    law_fail = out_fail["experimental_universal_structural_intelligence"]["universal_structural_law_candidate"]
    assert law_fail["epistemic_note"].startswith("Experimental")
