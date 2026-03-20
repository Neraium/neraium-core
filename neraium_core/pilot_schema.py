from __future__ import annotations

import math
from typing import Any, TypedDict


class PilotOutput(TypedDict):
    """
    Pilot hardening output schema.

    Required keys (always present when pilot mode is enabled):
    - `timestamp`: ISO-8601 UTC string
    - `signals`: dict of `{signal_name: float | None}`
    - `score`: float (structural instability score)
    - `status`: `STABLE` / `WATCH` / `ALERT`
    - `aligned`: boolean gate flag (data quality alignment)
    - `anomaly`: boolean derived from status
    """

    timestamp: str
    signals: dict[str, float | None]
    score: float
    status: str
    aligned: bool
    anomaly: bool


def build_pilot_output(*, frame: dict[str, Any], result: dict[str, Any]) -> PilotOutput:
    """
    Build pilot schema output from normalized input frame + decorated engine result.

    This function is strict about presence of the six required keys, but it does not
    validate or mutate raw payload content (validation happens earlier in `pipeline.py`).
    """

    timestamp = str(result.get("timestamp") or frame.get("timestamp") or "")

    raw_signals = frame.get("sensor_values")
    signals_out: dict[str, float | None] = {}
    if isinstance(raw_signals, dict):
        for k, v in raw_signals.items():
            key = str(k)
            if v is None:
                signals_out[key] = None
                continue

            try:
                f = float(v)
            except (TypeError, ValueError):
                signals_out[key] = None
                continue

            if math.isnan(f) or math.isinf(f):
                signals_out[key] = None
            else:
                signals_out[key] = f

    score = float(result.get("latest_instability", 0.0))

    status = str(result.get("action_state") or result.get("state") or "STABLE").upper()
    aligned = bool((result.get("data_quality_summary") or {}).get("gate_passed", False))
    anomaly = status in {"WATCH", "ALERT"}

    return PilotOutput(
        timestamp=timestamp,
        signals=signals_out,
        score=score,
        status=status,
        aligned=aligned,
        anomaly=anomaly,
    )

