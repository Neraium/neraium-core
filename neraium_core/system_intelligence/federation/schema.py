from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .types import (
    CalibrationSummary,
    InterventionEvidenceSummary,
    LawEvidenceSummary,
    PacketMetadata,
    SharedKnowledgePacket,
    TrajectoryFamilySummary,
)


def packet_to_dict(packet: SharedKnowledgePacket) -> dict[str, Any]:
    return asdict(packet)


def packet_from_dict(payload: dict[str, Any]) -> SharedKnowledgePacket:
    metadata = PacketMetadata(**dict(payload.get("metadata") or {}))
    return SharedKnowledgePacket(
        metadata=metadata,
        trajectory_families=[TrajectoryFamilySummary(**row) for row in list(payload.get("trajectory_families") or [])],
        law_evidence=[LawEvidenceSummary(**row) for row in list(payload.get("law_evidence") or [])],
        intervention_evidence=[InterventionEvidenceSummary(**row) for row in list(payload.get("intervention_evidence") or [])],
        calibration=[CalibrationSummary(**row) for row in list(payload.get("calibration") or [])],
        mechanism_family_associations=list(payload.get("mechanism_family_associations") or []),
    )


def packet_has_raw_telemetry(payload: dict[str, Any]) -> bool:
    blocked = {"raw_signal", "timeseries", "sensor_trace", "telemetry", "samples"}
    serialized = str(payload).lower()
    return any(token in serialized for token in blocked)
