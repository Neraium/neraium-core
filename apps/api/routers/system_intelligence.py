"""System Intelligence endpoint serving unified state data for the frontend.

Provides complete system state including tetrahedron geometry, subsystem analysis,
trajectory projection, and operator intelligence for the main system interface.
"""

from __future__ import annotations

from typing import Any
from fastapi import APIRouter

from ui.app import load_builtin_demo_rows
from ui.config import UIConfig
from ui.core_integration import build_system_state
from ui.unified_shell_data import (
    build_facility_rooms_data,
    build_subsystems_data,
    build_timeline_states,
    build_intelligence_insights,
    get_critical_alerts,
    _compute_no_action_projection,
)

router = APIRouter(prefix="/api/system", tags=["system-intelligence"])


def _build_unified_state(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Build complete unified system state from records."""
    if not records:
        records = load_builtin_demo_rows(use_synthetic=True)

    latest_record = records[-1] if records else {}
    system_state = build_system_state(records, config=UIConfig())

    # Extract metrics from latest record
    drift = system_state.drift_intensity if system_state else 0.2
    coherence = float(latest_record.get("coherence_score", 0.75))
    stability = float(latest_record.get("relational_stability_score", 0.8))

    # Determine system state
    sys_state = "stable"
    if drift > 0.6:
        sys_state = "critical"
    elif drift > 0.4:
        sys_state = "instability"
    elif drift > 0.2:
        sys_state = "drift"

    # Build component data
    rooms = build_facility_rooms_data(records=records)
    subsystems = build_subsystems_data(records=records, drift=drift, coherence=coherence)
    states_timeline = build_timeline_states(records=records)
    no_action_projection = _compute_no_action_projection(records, steps=3)
    insights = build_intelligence_insights(
        records=records,
        drift=drift,
        coherence=coherence,
        stability=stability,
        no_action_projection=no_action_projection,
    )
    critical_alerts = get_critical_alerts(records=records)

    return {
        "timestamp": latest_record.get("timestamp", ""),
        "state": sys_state,
        "drift": float(drift),
        "coherence": float(coherence),
        "stability": float(stability),
        "rooms": rooms,
        "subsystems": subsystems,
        "timeline_states": states_timeline,
        "no_action_projection": no_action_projection,
        "critical_alerts": critical_alerts,
        "insights": insights,
        "records": records,
    }


@router.get("/unified-state")
async def get_unified_state(use_synthetic: bool = True) -> dict[str, Any]:
    """Get complete unified system intelligence state.

    Returns all data needed for the system intelligence interface:
    - System metrics (drift, coherence, stability)
    - Tetrahedron geometry info
    - Subsystem analysis
    - State trajectory and no-action projection
    - Operator intelligence insights

    Args:
        use_synthetic: Use synthetic demo data if True

    Returns:
        Complete unified state dictionary
    """
    records = load_builtin_demo_rows(use_synthetic=use_synthetic)
    return _build_unified_state(records)


@router.get("/frame/{frame_index}")
async def get_frame(frame_index: int, use_synthetic: bool = True) -> dict[str, Any]:
    """Get a specific frame of system state.

    Args:
        frame_index: Index of the frame (0-based)
        use_synthetic: Use synthetic demo data if True

    Returns:
        System state at that frame
    """
    records = load_builtin_demo_rows(use_synthetic=use_synthetic)
    if 0 <= frame_index < len(records):
        records = records[: frame_index + 1]
    return _build_unified_state(records)
