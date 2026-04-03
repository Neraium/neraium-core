from __future__ import annotations

from typing import Any

from ..alignment.cross_system_alignment import CrossSystemTrajectoryAlignment
from ..cross_domain.archetypes import CrossDomainArchetypeClusterer
from ..cross_domain.domain_similarity import DomainSimilarityMapper
from ..cross_domain.intervention_transfer import CrossDomainInterventionTransfer
from ..cross_domain_law.universal_law import UniversalStructuralLawEngine
from ..invariant_representation.canonical import CanonicalStructuralRepresentation, canonical_to_dict


class ExperimentalUniversalStructuralLayer:
    """Experimental layer for cross-domain structural invariant discovery.

    Safety boundary: outputs are inspect-only and must never directly control operator recommendations.
    """

    def __init__(self) -> None:
        self.representation = CanonicalStructuralRepresentation()
        self.alignment = CrossSystemTrajectoryAlignment()
        self.archetypes = CrossDomainArchetypeClusterer()
        self.domain_similarity = DomainSimilarityMapper()
        self.universal_law = UniversalStructuralLawEngine()
        self.intervention_transfer = CrossDomainInterventionTransfer()
        self._recent: list[dict[str, Any]] = []

    def update(
        self,
        *,
        system_id: str,
        system_type: str,
        domain: str,
        latent_trajectory: list[list[float]],
        escalating: bool,
        mechanism_name: str,
        intervention_info: dict[str, Any],
    ) -> dict[str, Any]:
        canonical = self.representation.encode(
            system_type=system_type,
            domain=domain,
            latent_trajectory=latent_trajectory,
        )
        canonical_dict = canonical_to_dict(canonical)

        alignments = []
        for prev in self._recent[-10:]:
            comp = self.alignment.compare(
                path_a=canonical.normalized_path,
                path_b=list(prev["normalized_path"]),
                descriptor_a=canonical.descriptor_map,
                descriptor_b=dict(prev["descriptor_map"]),
            )
            alignments.append(
                {
                    "against_system_id": prev["system_id"],
                    "against_system_type": prev["system_type"],
                    "against_domain": prev["domain"],
                    **comp,
                }
            )
        alignments.sort(key=lambda row: row["similarity_score"], reverse=True)

        cross_archetype = self.archetypes.update(canonical=canonical_dict, system_id=system_id)
        similarity = self.domain_similarity.update(system_type=system_type, descriptor_vector=canonical.descriptor_vector)
        law = self.universal_law.update(
            domain=domain,
            trajectory_archetype_id=str(cross_archetype.get("cross_domain_archetype_id", "cross::unknown")),
            mechanism=mechanism_name,
            escalating=escalating,
        )
        intervention = self.intervention_transfer.update(
            domain=domain,
            intervention_info=intervention_info,
            context_similarity=float(similarity.get("transfer_confidence", 0.0)),
        )

        self._recent.append(
            {
                "system_id": system_id,
                "system_type": system_type,
                "domain": domain,
                "normalized_path": canonical.normalized_path,
                "descriptor_map": canonical.descriptor_map,
            }
        )
        self._recent = self._recent[-40:]

        confidence = float(law.get("cross_domain_effect_consistency", 0.0)) * float(similarity.get("transfer_confidence", 0.0))
        warnings = []
        if law.get("classification") != "candidate_universal_structural_law":
            warnings.append("insufficient cross-domain evidence for universality")
        if float(similarity.get("transfer_confidence", 0.0)) < 0.45:
            warnings.append("domain mismatch risk: low structural similarity")

        return {
            "experimental_universal_structural_intelligence": {
                "status": "experimental",
                "safety_boundary": {
                    "can_drive_operator_recommendations": False,
                    "minimum_confidence_for_any_future_influence": 0.86,
                    "warnings": warnings,
                },
                "canonical_representation": canonical_dict,
                "cross_system_alignment": alignments[:5],
                "cross_domain_archetype": cross_archetype,
                "domain_similarity_map": similarity,
                "universal_structural_law_candidate": law,
                "intervention_transfer": intervention,
                "uncertainty": round(float(1.0 - max(0.0, min(1.0, confidence))), 6),
                "low_confidence_flag": confidence < 0.6,
            }
        }
