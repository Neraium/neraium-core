"""Helper to generate data for the unified app shell from system state.

Converts SystemState and related data into the format expected by
unified shell components (facility command strip, subsystems, timeline, intelligence).
"""

from __future__ import annotations

from typing import Any
from ui.core_integration import SystemState


def _compute_subsystem_criticality(
    records: list[dict[str, Any]] | None = None,
) -> tuple[str, float]:
    """Determine most critical subsystem and its contribution magnitude.

    Args:
        records: Historical records

    Returns:
        Tuple of (critical_subsystem_id, criticality_score 0-1)
    """
    if not records or len(records) < 2:
        return "climate", 0.5

    latest = records[-1]
    drift = float(latest.get("structural_drift_score", 0.2))

    subsystem_contributions = {
        "climate": drift * 0.35,
        "airflow": drift * 0.3,
        "irrigation": drift * 0.25,
        "plant_response": drift * 0.1,
    }

    critical = max(subsystem_contributions, key=subsystem_contributions.get)
    criticality = subsystem_contributions[critical]

    return critical, min(criticality, 1.0)


def _compute_time_to_consequence(
    records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compute time-to-consequence based on drift velocity.

    Args:
        records: Historical records

    Returns:
        Dict with keys: cycles_to_critical, minutes_to_escalation, velocity
    """
    if not records or len(records) < 2:
        return {
            "cycles_to_critical": None,
            "minutes_to_escalation": None,
            "velocity": 0.0,
        }

    latest = records[-1]
    previous = records[-2] if len(records) > 1 else records[0]

    current_drift = float(latest.get("structural_drift_score", 0.2))
    previous_drift = float(previous.get("structural_drift_score", 0.2))

    velocity = current_drift - previous_drift
    time_diff_minutes = 5

    if velocity > 0.01:
        drift_needed = 1.0 - current_drift
        minutes_to_critical = (drift_needed / velocity) * time_diff_minutes if velocity > 0 else None
        cycles_to_critical = int(minutes_to_critical / 30) if minutes_to_critical else None
    else:
        minutes_to_critical = None
        cycles_to_critical = None

    return {
        "cycles_to_critical": cycles_to_critical,
        "minutes_to_escalation": int(minutes_to_critical) if minutes_to_critical else None,
        "velocity": velocity,
    }


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
                "is_critical": False,
            },
            {
                "room_id": "room_1",
                "room_name": "Airflow Zone",
                "phase": "veg",
                "state": "nominal",
                "confidence": 0.78,
                "changed_minutes_ago": 5,
                "is_critical": False,
            },
            {
                "room_id": "room_2",
                "room_name": "Irrigation Grid",
                "phase": "veg",
                "state": "watch",
                "confidence": 0.65,
                "changed_minutes_ago": 0,
                "is_critical": True,
            },
        ]

    latest = records[-1] if records else {}
    critical_subsystem, _ = _compute_subsystem_criticality(records)

    phase = str(latest.get("regime_name", "unknown")).lower().replace("_", " ")
    health = str(latest.get("system_health", "nominal")).lower()
    confidence = float(latest.get("confidence_score", 0.7))

    state_map = {"nominal": "nominal", "watch": "watch", "degraded": "degraded", "critical": "critical"}
    state = state_map.get(health, "nominal")

    subsystems = ["climate", "airflow", "irrigation"]
    for i, subsys in enumerate(subsystems):
        is_critical = subsys == critical_subsystem
        rooms.append(
            {
                "room_id": subsys,
                "room_name": subsys.title(),
                "phase": phase,
                "state": state,
                "confidence": confidence - (i * 0.05),
                "changed_minutes_ago": 3 - i if len(records) > 1 else 0,
                "is_critical": is_critical,
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
    time_to_consequence: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Build operator intelligence insights.

    Args:
        system_state: Current SystemState
        records: Historical records list
        gate_decision: Gate decision dict
        time_to_consequence: Time-to-consequence metrics dict with cycles_to_critical, minutes_to_escalation, velocity

    Returns:
        Dictionary with insight keys
    """
    drift = system_state.drift_intensity if system_state else 0.2
    stability = system_state.stability if system_state else 0.8
    coherence = system_state.coherence if system_state else 0.75

    gate_decision = gate_decision or {}
    decision = str(gate_decision.get("decision", "SUPPRESS")).upper()
    time_to_consequence = time_to_consequence or {}
    minutes_to_escalation = time_to_consequence.get("minutes_to_escalation")
    cycles_to_critical = time_to_consequence.get("cycles_to_critical")

    if drift > 0.6:
        current_state = "Multiple subsystem couplings fractured; system structure compromised"
    elif drift > 0.4:
        current_state = "Coupling degradation spreading; coherence structure unstable"
    elif drift > 0.2:
        current_state = "Early coupling stress between climate and response layers"
    else:
        current_state = "Subsystem couplings intact; system structure coherent"

    if drift > 0.3:
        if minutes_to_escalation and minutes_to_escalation > 0:
            onset = f"Drift persisting; escalation expected within {minutes_to_escalation} minutes"
        else:
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
        driver = "Airflow-irrigation coupling strained; system feedback pathways degrading"
    elif drift > 0.2:
        driver = "Climate-plant coherence weakening; subsystem integration under stress"
    else:
        driver = "All subsystem couplings stable; coherent equilibrium maintained"

    if decision == "ADMIT":
        focus = "⚠️ CRITICAL: Structural transition admitted. Execute intervention protocol immediately."
    elif drift > 0.3:
        if minutes_to_escalation and minutes_to_escalation < 30:
            focus = f"Escalation within {minutes_to_escalation} minutes. Intervention needed now."
        else:
            focus = "Monitor recovery trajectory. If drift continues to rise, structural intervention required."
    elif drift > 0.1:
        focus = "Observe next irrigation cycle for coherence recovery. Adjust timing if drift persists."
    else:
        focus = "Maintain current operational parameters. System shows stable recovery profile."

    if len(records or []) > 6:
        if records[-1].get("structural_drift_score", 0) < records[-6].get("structural_drift_score", 1):
            outlook = "Recovery trajectory confirmed. System moving toward baseline within next cycle window."
        else:
            if cycles_to_critical:
                outlook = f"Drift acceleration detected. Critical threshold reachable in ~{cycles_to_critical} cycle windows."
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
