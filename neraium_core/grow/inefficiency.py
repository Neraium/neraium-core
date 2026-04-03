from __future__ import annotations

from typing import Any


def clamp_score(value: float) -> float:
    return round(max(0.0, min(1.0, value)), 3)


def score_inefficiencies(signals: dict[str, Any], subsystem_scores: dict[str, float]) -> dict[str, float | str]:
    room_power = float(signals.get("room_power_kw") or signals.get("power_kw") or 0.0)
    fixture_power = float(signals.get("fixture_power_kw") or 0.0)
    load_percent = float(signals.get("load_percent") or 0.0)
    temp = float(signals.get("zone_temperature") or 0.0)
    rh = float(signals.get("zone_relative_humidity") or 0.0)
    vpd = float(signals.get("vapor_pressure_deficit") or 0.0)
    co2 = float(signals.get("co2_ppm") or 0.0)
    drain_ec = float(signals.get("drain_ec") or 0.0)
    nutrient_ec = float(signals.get("nutrient_ec") or 0.0)
    irrigation_pressure = float(signals.get("irrigation_pressure") or 0.0)
    transpiration = float(signals.get("transpiration_proxy") or 0.0)
    stress = float(signals.get("stress_index") or 0.0)
    image = float(signals.get("image_health_score") or 0.8)

    energy = clamp_score(max(room_power / max(fixture_power + 25.0, 1.0), load_percent / 100.0) * 0.6)
    climate = clamp_score(abs(temp - 76.0) / 12.0 + abs(rh - 58.0) / 30.0 + abs(vpd - 1.1) / 1.4 + max(0.0, co2 - 1100.0) / 700.0)
    process = clamp_score(abs(drain_ec - nutrient_ec) / 1.4 + max(0.0, 22.0 - irrigation_pressure) / 25.0)
    biological = clamp_score(max(transpiration - vpd, 0.0) * 0.5 + stress * 0.8 + max(0.0, 0.75 - image))

    scores = {
        "energy_inefficiency": energy,
        "climate_inefficiency": climate,
        "process_inefficiency": process,
        "biological_inefficiency": biological,
    }
    dominant = max(scores, key=scores.get)
    summary_map = {
        "energy_inefficiency": "Operating worse than expected due to rising energy burden and weak recovery",
        "climate_inefficiency": "Climate response is operating worse than expected relative to baseline and peers",
        "process_inefficiency": "Process response is decoupling from expected irrigation, fertigation, or transition behavior",
        "biological_inefficiency": "Biological response is weaker than expected relative to baseline peers",
    }
    if subsystem_scores:
        dominant_subsystem = max(subsystem_scores, key=subsystem_scores.get)
        summary = f"{summary_map[dominant]} driven primarily by {dominant_subsystem.replace('_', ' ')}"
    else:
        summary = summary_map[dominant]
    scores["inefficiency_summary"] = summary
    return scores
