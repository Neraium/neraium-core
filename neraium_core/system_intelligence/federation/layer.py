from __future__ import annotations

from typing import Any

from ..aggregation.merge import merge_shared_packets
from ..shared_knowledge.exporter import export_local_structural_knowledge
from ..transfer.adaptation import adapt_global_to_local
from .schema import packet_to_dict
from .types import SharedKnowledgePacket


class CrossSystemStructuralIntelligenceLayer:
    """Local-first decentralized structural intelligence exchange layer."""

    def __init__(self, *, instance_id: str = "local-instance", max_remote_packets: int = 24) -> None:
        self.instance_id = instance_id
        self.max_remote_packets = int(max_remote_packets)
        self._remote_packets: list[SharedKnowledgePacket] = []
        self._step = 0

    def ingest_remote_packet(self, packet: SharedKnowledgePacket) -> None:
        self._remote_packets.append(packet)
        if len(self._remote_packets) > self.max_remote_packets:
            self._remote_packets = self._remote_packets[-self.max_remote_packets :]

    def update(
        self,
        *,
        trajectory_info: dict[str, Any],
        law_info: dict[str, Any],
        intervention_info: dict[str, Any],
    ) -> dict[str, Any]:
        self._step += 1
        local_packet = export_local_structural_knowledge(
            instance_id=self.instance_id,
            step=self._step,
            trajectory_info=trajectory_info,
            law_info=law_info,
            intervention_info=intervention_info,
        )
        merged = merge_shared_packets([local_packet, *self._remote_packets], current_step=self._step)
        global_packet = merged["global_packet"]
        adapted = adapt_global_to_local(
            local_packet=local_packet,
            global_packet=global_packet,
            trajectory_info=trajectory_info,
            local_laws=law_info,
            local_intervention=intervention_info,
        )
        stale_penalty = 0.0
        if self._remote_packets:
            ages = [max(0, self._step - int(p.metadata.generated_at_step)) for p in self._remote_packets]
            stale_penalty = min(1.0, max(ages) / 25.0)

        return {
            "cross_system_structural_intelligence": {
                "status": "active",
                "shared_schema_version": local_packet.metadata.schema_version,
                "local_packet": packet_to_dict(local_packet),
                "global_packet": packet_to_dict(global_packet),
                "conflict_flags": merged.get("conflict_flags") or [],
                "stale_evidence_penalty": round(float(stale_penalty), 4),
                **adapted,
            }
        }
