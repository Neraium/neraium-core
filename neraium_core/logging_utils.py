from __future__ import annotations

import json
import logging
import os
import time
from typing import Any


def pilot_debug_enabled() -> bool:
    """
    Enable extra pilot observability without changing runtime behavior.

    Controlled by `NERAIUM_DEBUG_PILOT`.
    """

    raw = os.getenv("NERAIUM_DEBUG_PILOT", "0").strip().lower()
    return raw not in {"0", "false", "no", "off", ""}


def _json_default(o: Any) -> str:
    try:
        if hasattr(o, "isoformat"):
            return str(o.isoformat())
    except Exception:
        pass
    return str(o)


def _to_json(fields: dict[str, Any]) -> str:
    return json.dumps(fields, default=_json_default, sort_keys=True, separators=(",", ":"))


def log_structured(logger: logging.Logger, *, event: str, fields: dict[str, Any], level: int = logging.INFO) -> None:
    """
    Emit a single-line structured log.

    The implementation uses an `event=... {json}` format to avoid reliance on
    external log processors while keeping logs parseable.
    """

    msg = f"event={event} fields={_to_json(fields)}"
    logger.log(level, msg)


def summarize_payload_for_logs(payload: dict[str, Any]) -> dict[str, Any]:
    sensor_values = payload.get("sensor_values")
    sensor_count: int = 0
    if isinstance(sensor_values, dict):
        sensor_count = len(sensor_values)

    timestamp = payload.get("timestamp")
    return {
        "site_id": payload.get("site_id"),
        "asset_id": payload.get("asset_id"),
        "has_timestamp": bool(timestamp not in (None, "")),
        "sensor_count": sensor_count,
        "sensor_values_type": type(sensor_values).__name__,
    }


def summarize_result_for_logs(result: dict[str, Any]) -> dict[str, Any]:
    status = result.get("action_state") or result.get("state")
    score = result.get("latest_instability")
    aligned = result.get("aligned")
    anomaly = result.get("anomaly")

    dq = result.get("data_quality_summary")
    gate_passed = None
    statuses: list[str] = []
    if isinstance(dq, dict):
        gate_passed = dq.get("gate_passed")
        raw_statuses = dq.get("statuses")
        if isinstance(raw_statuses, list):
            statuses = [str(s) for s in raw_statuses[:5]]

    return {
        "status": status,
        "score": score,
        "aligned": aligned,
        "anomaly": anomaly,
        "gate_passed": gate_passed,
        "dq_statuses_top": statuses,
    }


def summarize_exception_for_logs(exc: Exception) -> dict[str, Any]:
    msg = str(exc)

    # Redact raw sensor values from strict-pilot validation messages.
    # Examples (from `pipeline.py`):
    # - "Invalid signal value for 's1': 'abc'"
    # - "Invalid signal type for 's1': dict"
    if msg.startswith("Invalid signal value for"):
        # Keep: "Invalid signal value for '<sensor>'"
        try:
            rest = msg.split("Invalid signal value for", 1)[1].strip()
            sensor = rest.split(":", 1)[0].strip()
            msg = f"Invalid signal value for {sensor}"
        except Exception:
            msg = "Invalid signal value (redacted)"
    elif msg.startswith("Invalid signal type for"):
        try:
            rest = msg.split("Invalid signal type for", 1)[1].strip()
            sensor = rest.split(":", 1)[0].strip()
            msg = f"Invalid signal type for {sensor}"
        except Exception:
            msg = "Invalid signal type (redacted)"

    return {
        "error_type": type(exc).__name__,
        "error_message": msg,
    }


class Timer:
    """Small timer helper for ingest step latency logging."""

    def __init__(self) -> None:
        self._t0 = time.perf_counter()

    def ms(self) -> float:
        return (time.perf_counter() - self._t0) * 1000.0

