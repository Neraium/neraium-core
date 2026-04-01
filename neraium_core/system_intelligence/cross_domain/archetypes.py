from __future__ import annotations

from collections import defaultdict
from typing import Any


def _clamp(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


class CrossDomainArchetypeClusterer:
    """Experimental cross-domain trajectory archetype clustering."""

    def __init__(self) -> None:
        self._clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)

    def update(self, *, canonical: dict[str, Any], system_id: str) -> dict[str, Any]:
        d = dict(canonical.get("descriptor_map") or {})
        archetype_id = self._assign(d)
        member = {
            "system_id": str(system_id),
            "system_type": str(canonical.get("system_type", "unknown")),
            "domain": str(canonical.get("domain", "unknown")),
            "trajectory_example": list(canonical.get("normalized_path") or []),
            "descriptor": d,
        }
        self._clusters[archetype_id].append(member)
        members = self._clusters[archetype_id]
        support_by_domain: dict[str, int] = defaultdict(int)
        system_types = set()
        for m in members:
            support_by_domain[m["domain"]] += 1
            system_types.add(m["system_type"])
        consistency = _clamp(len(support_by_domain) / max(1.0, len(system_types) + 0.5))
        return {
            "cross_domain_archetype_id": archetype_id,
            "member_system_types": sorted(system_types),
            "trajectory_examples": [m["trajectory_example"] for m in members[-3:]],
            "support_by_domain": dict(support_by_domain),
            "consistency_score": round(float(consistency), 6),
            "support_count": len(members),
        }

    def _assign(self, d: dict[str, float]) -> str:
        drift = float(d.get("drift_signed", 0.0))
        accel = float(d.get("acceleration_mean", 0.0))
        osc = float(d.get("oscillation_ratio", 0.0))
        rev = float(d.get("reversibility", 0.0))
        if osc > 0.42:
            return "cross::oscillatory_instability"
        if drift > 0.2 and accel > 0.02:
            return "cross::slow_drift_to_instability"
        if drift > 0.3 and accel > 0.06:
            return "cross::sudden_collapse"
        if rev > 0.65 and drift < 0.1:
            return "cross::reversible_degradation"
        return "cross::mixed_transition"
