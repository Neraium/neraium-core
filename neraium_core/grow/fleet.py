from __future__ import annotations

from collections import Counter
from typing import Any


def _rank(items: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    return sorted(items, key=lambda item: float(item.get(key, 0.0)), reverse=True)


def build_fleet_summary(
    plants: list[dict[str, Any]],
    zones: list[dict[str, Any]],
    rooms: list[dict[str, Any]],
    subsystems: list[dict[str, Any]],
    facilities: list[dict[str, Any]],
) -> dict[str, Any]:
    all_entities = [*plants, *zones, *rooms, *subsystems, *facilities]
    drivers = Counter(entity.get("cross_system_driver", "Unknown") for entity in all_entities if entity.get("cross_system_driver"))
    ineff_sorted = sorted(
        all_entities,
        key=lambda entity: (
            float(entity.get("energy_inefficiency", 0.0))
            + float(entity.get("climate_inefficiency", 0.0))
            + float(entity.get("process_inefficiency", 0.0))
            + float(entity.get("biological_inefficiency", 0.0))
        ),
        reverse=True,
    )
    action_sorted = sorted(
        [entity for entity in all_entities if entity.get("estimated_time_to_impact_hours") is not None],
        key=lambda entity: float(entity.get("estimated_time_to_impact_hours", 9999.0)),
    )
    return {
        "ranked_plants_by_local_risk": _rank(plants, "composite_instability"),
        "ranked_zones_by_risk": _rank(zones, "composite_instability"),
        "ranked_rooms_by_risk": _rank(rooms, "composite_instability"),
        "ranked_subsystems_by_drift_contribution": _rank(subsystems, "structural_drift_score"),
        "ranked_facilities_by_instability": _rank(facilities, "composite_instability"),
        "highest_inefficiency_areas": ineff_sorted[:10],
        "fastest_deteriorating_entities": _rank(all_entities, "trend_score")[:10],
        "entities_requiring_action_soonest": action_sorted[:10],
        "top_cross_system_drivers": [{"driver": name, "count": count} for name, count in drivers.most_common(6)],
    }
