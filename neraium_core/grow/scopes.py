from __future__ import annotations

from typing import Any

from neraium_core.grow.schemas import ScopeType


SCOPE_ORDER: list[ScopeType] = ["plant", "zone", "room", "subsystem", "facility", "portfolio"]


def portfolio_scope_id() -> str:
    return "portfolio-global"


def scope_parent(scope_type: ScopeType, row: dict[str, Any]) -> str | None:
    if scope_type == "plant":
        return row.get("zone_id") or row.get("room_id")
    if scope_type == "zone":
        return row.get("room_id")
    if scope_type == "room":
        return row.get("facility_id")
    if scope_type == "subsystem":
        return row.get("room_id") or row.get("facility_id")
    if scope_type == "facility":
        return portfolio_scope_id()
    return None


def scope_id_for_record(scope_type: ScopeType, row: dict[str, Any]) -> str | None:
    if scope_type == "plant":
        return row.get("plant_id")
    if scope_type == "zone":
        return row.get("zone_id")
    if scope_type == "room":
        return row.get("room_id")
    if scope_type == "subsystem":
        return row.get("subsystem_id") or row.get("asset_id")
    if scope_type == "facility":
        return row.get("facility_id")
    if scope_type == "portfolio":
        return portfolio_scope_id()
    return None
