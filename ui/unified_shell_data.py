"""Helper to generate data for the unified app shell from system state.

Converts SystemState and related data into the format expected by
unified shell components (facility command strip, subsystems, timeline, intelligence).
"""

from __future__ import annotations

from typing import Any
from ui.core_integration import SystemState


def build_facility_rooms_data(
    system_state: SystemState | None,
    records: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build room status data for facility command strip.

    Args:
        system_state: Current SystemState
        records: Historical records list

    Returns:
        List of room dicts with keys: room_id, room_name, phase, state, confidence, changed_minutes_ago
    """
    rooms = []

    if not records or len(records) == 0:
        return [
            {
                "room_id": "room_0",
                "room_name": "Climate Chamber",
                "phase": "veg",
                "state": "nominal",
                "confidence": 0.85,
                "changed_minutes_ago": 12,
            },
            {
                "room_id": "room_1",
                "room_name": "Airflow Zone",
                "phase": "veg",
                "state": "nominal",
                "confidence": 0.78,
                "changed_minutes_ago": 5,
            },
            {
                "room_id": "room_2",
                "room_name": "Irrigation Grid",
                "phase": "veg",
                "state": "watch",
                "confidence": 0.65,
                "changed_minutes_ago": 0,
            },
        ]

    latest = records[-1] if records else {}

    phase = str(latest.get("regime_name", "unknown")).lower().replace("_", " ")
    health = str(latest.get("system_health", "nominal")).lower()
    confidence = float(latest.get("confidence_score", 0.7))

    state_map = {"nominal": "nominal", "watch": "watch", "degraded": "degraded", "critical": "critical"}
    state = state_map.get(health, "nominal")

    rooms.append(
        {
            "room_id": "climate",
            "room_name": "Climate",
            "phase": phase,
            "state": state,
            "confidence": confidence,
            "changed_minutes_ago": 3 if len(records) > 1 else 0,
        }
    )
    rooms.append(
        {
            "room_id": "airflow",
            "room_name": "Airflow",
            "phase": phase,
            "state": state,
            "confidence": confidence - 0.05,
            "changed_minutes_ago": 7,
        }
    )
    rooms.append(
        {
            "room_id": "irrigation",
            "room_name": "Irrigation",
            "phase": phase,
            "state": "watch" if state == "nominal" else state,
            "confidence": confidence - 0.1,
            "changed_minutes_ago": 0,
        }
    )

    return rooms


def build_subsystems_data(
    system_state: SystemState | None,
    records: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build subsystem influence data.

    Args:
        system_state: Current SystemState
        records: Historical records list

    Returns:
        List of subsystem dicts
    """
    subsystems = []

    drift = system_state.drift_intensity if system_state else 0.2
    stability = system_state.stability if system_state else 0.8
    coherence = system_state.coherence if system_state else 0.75

    subsystems.append(
        {
            "subsystem_id": "climate",
            "subsystem_name": "Climate",
            "condition": f"{22 + drift * 5:.1f}°C",
            "behavioral_state": "Stabilizing" if stability > 0.6 else "Drifting",
            "drift_contribution_pct": drift * 30,
            "confidence_pct": (1.0 - drift) * 100,
            "explanation": "Temperature regulation within normal parameters.",
            "micro_activity": drift * 0.5,
        }
    )

    subsystems.append(
        {
            "subsystem_id": "airflow",
            "subsystem_name": "Airflow",
            "condition": f"{1.2 + drift * 0.4:.1f} m/s",
            "behavioral_state": "Recovering" if drift > 0 else "Optimal",
            "drift_contribution_pct": drift * 25,
            "confidence_pct": stability * 100,
            "explanation": "Airflow velocity trending upward in recovery window.",
            "micro_activity": drift * 0.3,
        }
    )

    subsystems.append(
        {
            "subsystem_id": "irrigation",
            "subsystem_name": "Irrigation",
            "condition": f"{65 - drift * 20:.0f}% capacity",
            "behavioral_state": "Post-cycle" if drift > 0.3 else "Nominal",
            "drift_contribution_pct": drift * 35,
            "confidence_pct": coherence * 100,
            "explanation": "Irrigation cycle completed. System in recovery phase.",
            "micro_activity": (1.0 - stability) * 0.6,
        }
    )

    subsystems.append(
        {
            "subsystem_id": "plant_response",
            "subsystem_name": "Plant Response",
            "condition": "Robust",
            "behavioral_state": "Growing" if coherence > 0.7 else "Slowing",
            "drift_contribution_pct": max(0, drift * 40 - 10),
            "confidence_pct": coherence * 100,
            "explanation": "Plant layer showing healthy growth trajectory.",
            "micro_activity": 0.2,
        }
    )

    return subsystems


def build_timeline_states(
    records: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build state timeline data.

    Args:
        records: Historical records list

    Returns:
        List of state dicts
    """
    if not records or len(records) == 0:
        return [
            {
                "timestamp": "—",
                "state_label": "baseline",
                "coherence": 0.85,
                "drift": 0.1,
                "stability": 0.9,
                "is_admitted": False,
                "emphasis": "low",
            }
        ]

    states = []
    for i, record in enumerate(records[-24:]):
        drift = float(record.get("structural_drift_score", 0.2))
        stability = float(record.get("relational_stability_score", 0.8))
        coherence = float(record.get("coherence_score", 0.75))

        if drift > 0.6:
            state_label = "critical"
        elif drift > 0.4:
            state_label = "persistent"
        elif drift > 0.2:
            state_label = "emerging"
        elif drift > 0.05:
            state_label = "drift"
        else:
            state_label = "baseline"

        emphasis = "high" if bool(record.get("event_admitted")) else "medium" if drift > 0.15 else "low"

        timestamp = str(record.get("timestamp", "")).split("T")[-1] if record.get("timestamp") else "—"

        states.append(
            {
                "timestamp": timestamp,
                "state_label": state_label,
                "coherence": coherence,
                "drift": drift,
                "stability": stability,
                "is_admitted": bool(record.get("event_admitted")),
                "emphasis": emphasis,
            }
        )

    return states


def build_intelligence_insights(
    system_state: SystemState | None,
    records: list[dict[str, Any]] | None = None,
    gate_decision: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Build operator intelligence insights.

    Args:
        system_state: Current SystemState
        records: Historical records list
        gate_decision: Gate decision dict

    Returns:
        Dictionary with insight keys
    """
    drift = system_state.drift_intensity if system_state else 0.2
    stability = system_state.stability if system_state else 0.8
    coherence = system_state.coherence if system_state else 0.75

    gate_decision = gate_decision or {}
    decision = str(gate_decision.get("decision", "SUPPRESS")).upper()

    if drift > 0.6:
        current_state = "Critical structural instability detected across multiple domains"
    elif drift > 0.4:
        current_state = "Persistent instability forming; recovery path uncertain"
    elif drift > 0.2:
        current_state = "Mild drift emerging in climate and irrigation coupling"
    else:
        current_state = "System operating nominally with stable coherence"

    if drift > 0.3:
        onset = "Drift began ~45 minutes ago; has persisted through one full cycle"
    elif drift > 0.1:
        onset = "Current state established 12 minutes ago during post-irrigation phase"
    else:
        onset = "Stable baseline maintained for 3+ cycles"

    if coherence > 0.8:
        coherence_text = "Strong integration across all subsystems; coupled dynamics intact"
    elif coherence > 0.6:
        coherence_text = "Coherence weakening between climate and plant-response layers"
    else:
        coherence_text = "Subsystem decoupling detected; feedback pathways degraded"

    if drift > 0.4:
        driver = "Irrigation timing misalignment driving plant-response feedback loop"
    elif drift > 0.2:
        driver = "Climate volatility in final stage of veg cycle triggering minor instability"
    else:
        driver = "Stable environmental equilibrium maintained by active feedback"

    if decision == "ADMIT":
        focus = "⚠️ CRITICAL: Structural transition admitted. Execute intervention protocol immediately."
    elif drift > 0.3:
        focus = "Monitor recovery trajectory. If drift continues to rise, structural intervention required."
    elif drift > 0.1:
        focus = "Observe next irrigation cycle for coherence recovery. Adjust timing if drift persists."
    else:
        focus = "Maintain current operational parameters. System shows stable recovery profile."

    if len(records or []) > 6:
        if records[-1].get("structural_drift_score", 0) < records[-6].get("structural_drift_score", 1):
            outlook = "Recovery trajectory confirmed. System moving toward baseline within next cycle window."
        else:
            outlook = "Drift acceleration detected. Current path may become critical if coherence continues degrading."
    else:
        outlook = "Insufficient history to project trajectory. Wait for additional cycle data."

    return {
        "current_state": current_state,
        "onset": onset,
        "coherence": coherence_text,
        "primary_driver": driver,
        "operator_focus": focus,
        "path_outlook": outlook,
    }


def get_critical_alerts(
    system_state: SystemState | None,
    gate_decision: dict[str, Any] | None = None,
) -> list[str]:
    """Build list of critical alerts.

    Args:
        system_state: Current SystemState
        gate_decision: Gate decision dict

    Returns:
        List of critical alert strings
    """
    alerts = []

    gate_decision = gate_decision or {}
    if str(gate_decision.get("decision", "SUPPRESS")).upper() == "ADMIT":
        alerts.append("Structural transition ADMITTED. Execute intervention protocol.")

    drift = system_state.drift_intensity if system_state else 0.2
    if drift > 0.5:
        alerts.append("High structural drift. Recovery path becoming constrained.")

    stability = system_state.stability if system_state else 0.8
    if stability < 0.3:
        alerts.append("Relational stability critically low. Subsystem coupling degraded.")

    coherence = system_state.coherence if system_state else 0.75
    if coherence < 0.4:
        alerts.append("System coherence below operational threshold. Imminent structural failure risk.")

    return alerts
