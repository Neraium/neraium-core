from __future__ import annotations


DRIVER_TO_ACTION = {
    "hvac_loop_inefficiency": "Inspect refrigerant loop for early leak or charge imbalance",
    "dehumidification_lag": "Inspect dehumidification stage lag",
    "airflow_restriction": "Verify fan performance and static pressure restrictions",
    "irrigation_fertigation_drift": "Inspect dosing and reservoir mixing because drain chemistry is drifting from baseline",
    "lighting_thermal_mismatch": "Review lighting schedule transition effects on thermal recovery",
    "co2_response_instability": "Review CO2 delivery timing and verify expected thermal response at the room level",
    "energy_recovery_mismatch": "Check compressor, fan, and lighting recovery mismatch because energy burden is rising faster than recovery",
    "biological_stress_escalation": "Check room airflow balancing and canopy microclimate distribution",
}


def recommended_action_for_driver(driver: str) -> str:
    return DRIVER_TO_ACTION.get(driver, "Inspect cross-system behavior to restore baseline operating stability")
