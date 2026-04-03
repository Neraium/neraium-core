from __future__ import annotations

from typing import Any

from neraium_core.grow.forecast import humanize_impact_window
from neraium_core.grow.recommendations import recommended_action_for_driver


def dominant_relationship_shift(subsystem_scores: dict[str, float], signals: dict[str, Any]) -> tuple[str, str]:
    ranked = sorted(subsystem_scores.items(), key=lambda item: item[1], reverse=True)
    top_driver = ranked[0][0] if ranked else "cross_system_stability"

    if top_driver == "hvac_loop_inefficiency":
        return (
            "Supply air temperature and discharge pressure are diverging from baseline behavior",
            "Room instability is primarily driven by HVAC loop inefficiency",
        )
    if top_driver == "dehumidification_lag":
        return (
            "Compressor load no longer tracks room RH as expected",
            "Dehumidification lag is increasing compressor burden across the room",
        )
    if top_driver == "airflow_restriction":
        return (
            "Humidity to temperature coupling is weakening",
            "Room pressure and airflow behavior suggest an upstream restriction",
        )
    if top_driver == "irrigation_fertigation_drift":
        return (
            "Drain EC is diverging from nutrient EC and irrigation timing",
            "Nutrient delivery variability is now affecting room-level biological stability",
        )
    if top_driver == "lighting_thermal_mismatch":
        return (
            "Lighting load and room thermal recovery are no longer aligned",
            "Lighting heat load is no longer matched by cooling recovery",
        )
    if top_driver == "co2_response_instability":
        return (
            "CO2 injection is no longer producing expected thermal response",
            "Environmental drift appears upstream of plant-response risk",
        )
    if top_driver == "energy_recovery_mismatch":
        return (
            "Lighting load and room thermal recovery are no longer aligned",
            "Rising site load is not being matched by cooling and recovery performance",
        )
    if top_driver == "biological_stress_escalation":
        return (
            "Transpiration proxy is decoupling from VPD response",
            "Biological response is no longer matching adjacent baseline peers",
        )
    return (
        "Humidity to temperature coupling is weakening",
        "Environmental drift appears upstream of plant-response risk",
    )


def biological_risk_label(
    *,
    state: str,
    signals: dict[str, Any],
    threshold_breach_detected: bool,
) -> str:
    stress = float(signals.get("stress_index") or 0.0)
    image_health = float(signals.get("image_health_score") or 0.8)
    transpiration = float(signals.get("transpiration_proxy") or 0.0)
    if state == "DEGRADING" or threshold_breach_detected or stress >= 0.75:
        return "QUALITY_RISK_EMERGING"
    if state == "AT_RISK" or image_health < 0.55:
        return "YIELD_RISK_EMERGING"
    if state == "DRIFTING" or transpiration > 0.75:
        return "ELEVATED_BIOLOGICAL_RISK"
    return "LOW_BIOLOGICAL_RISK"


def urgency_for_state(state: str, impact_hours: float | None) -> str:
    if impact_hours == 0.0 or state == "DEGRADING":
        return "IMMEDIATE"
    if state == "AT_RISK":
        return "HIGH"
    if state == "DRIFTING":
        return "MEDIUM"
    return "LOW"


def operator_message(
    *,
    state: str,
    relationship_shift: str,
    cross_system_driver: str,
    impact_hours: float | None,
    threshold_text: str,
    driver_key: str,
) -> str:
    action = recommended_action_for_driver(driver_key)
    return (
        f"{state.title().replace('_', ' ')}: {relationship_shift}. "
        f"{cross_system_driver}. {threshold_text} {humanize_impact_window(impact_hours)}. "
        f"Recommended action: {action}."
    )
