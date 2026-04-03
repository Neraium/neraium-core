from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any

from neraium_core.grow.schemas import GrowTelemetryRecord


SUPPORTED_SIGNALS: set[str] = {
    "zone_temperature",
    "zone_relative_humidity",
    "vapor_pressure_deficit",
    "co2_ppm",
    "airflow_rate",
    "static_pressure",
    "ambient_temperature",
    "dew_point",
    "leaf_temperature",
    "canopy_temperature",
    "suction_temperature",
    "discharge_temperature",
    "refrigerant_temperature",
    "return_air_temperature",
    "supply_air_temperature",
    "oil_temperature",
    "motor_winding_temperature",
    "suction_pressure",
    "discharge_pressure",
    "oil_pressure",
    "refrigerant_pressure",
    "superheat",
    "subcooling",
    "condensate_drain_status",
    "filter_differential_pressure",
    "condenser_fan_speed",
    "evaporator_fan_speed",
    "compressor_speed_rpm",
    "load_percent",
    "vibration",
    "run_status",
    "cycle_count",
    "irrigation_flow_rate",
    "irrigation_pressure",
    "irrigation_duration",
    "irrigation_event_count",
    "nutrient_ec",
    "nutrient_ph",
    "reservoir_level",
    "reservoir_temperature",
    "drain_ec",
    "drain_ph",
    "runoff_volume",
    "water_temperature",
    "dissolved_oxygen",
    "dosing_pump_status",
    "valve_status",
    "light_status",
    "photoperiod_state",
    "ppfd",
    "dimming_level",
    "fixture_power_kw",
    "lighting_zone_status",
    "motor_current",
    "power_kw",
    "voltage",
    "power_factor",
    "breaker_status",
    "panel_load_kw",
    "room_power_kw",
    "canopy_stage",
    "growth_rate_proxy",
    "biomass_proxy",
    "transpiration_proxy",
    "stress_index",
    "image_health_score",
    "leaf_temp_delta",
    "door_open_event",
    "occupancy_event",
    "sanitation_event",
    "maintenance_event",
    "alarm_event_count",
}

IDENTITY_FIELDS = {
    "timestamp",
    "facility_id",
    "site_id",
    "building_id",
    "room_id",
    "zone_id",
    "plant_id",
    "asset_id",
    "asset_type",
    "subsystem_id",
    "controller_id",
    "crop_type",
    "strain_or_crop_type",
    "growth_stage",
    "batch_id",
    "telemetry_source",
}

DEFAULT_THRESHOLDS: dict[str, tuple[float, float]] = {
    "zone_temperature": (68.0, 84.0),
    "zone_relative_humidity": (45.0, 72.0),
    "vapor_pressure_deficit": (0.7, 1.6),
    "co2_ppm": (350.0, 1500.0),
    "filter_differential_pressure": (0.0, 2.2),
    "irrigation_pressure": (10.0, 45.0),
    "nutrient_ec": (1.2, 3.2),
    "drain_ec": (1.0, 3.5),
    "stress_index": (0.0, 0.8),
    "power_factor": (0.88, 1.05),
}

PLANT_MEASURED_SIGNALS = {
    "growth_rate_proxy",
    "biomass_proxy",
    "image_health_score",
    "leaf_temp_delta",
    "leaf_temperature",
    "canopy_temperature",
}


def _coerce_value(value: Any) -> float | int | bool | str | None:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float, bool)):
        return value
    text = str(value).strip()
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    try:
        if "." in text:
            return float(text)
        return int(text)
    except ValueError:
        return text


def normalize_record(record: dict[str, Any]) -> GrowTelemetryRecord:
    base: dict[str, Any] = {}
    signals: dict[str, Any] = {}
    for key, value in record.items():
        if key in IDENTITY_FIELDS:
            base[key] = value
        elif key == "signals" and isinstance(value, dict):
            signals.update({name: _coerce_value(raw) for name, raw in value.items() if name in SUPPORTED_SIGNALS})
        elif key in SUPPORTED_SIGNALS:
            signals[key] = _coerce_value(value)
    base["signals"] = signals
    return GrowTelemetryRecord(**base)


def parse_json_payload(payload: Any) -> list[GrowTelemetryRecord]:
    if isinstance(payload, dict):
        items = payload.get("records", [payload])
    elif isinstance(payload, list):
        items = payload
    else:
        raise ValueError("json payload must be a record or list of records")
    return [normalize_record(dict(item)) for item in items]


def parse_json_text(text: str) -> list[GrowTelemetryRecord]:
    return parse_json_payload(json.loads(text))


def parse_csv_text(text: str) -> list[GrowTelemetryRecord]:
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        raise ValueError("csv payload must include a header row")
    return [normalize_record(dict(row)) for row in reader]


def load_records_from_path(path: str | Path) -> list[GrowTelemetryRecord]:
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8")
    if file_path.suffix.lower() == ".json":
        return parse_json_text(text)
    if file_path.suffix.lower() == ".csv":
        return parse_csv_text(text)
    raise ValueError(f"unsupported file format: {file_path.suffix}")
