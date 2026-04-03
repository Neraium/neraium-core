from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PilotConfig:
    """
    Pilot hardening configuration with safe defaults.

    This config is intentionally small: it replaces hard-coded thresholds used by
    operator-facing interpretation with externally tunable values.
    """

    drift_high_threshold: float = 3.0
    drift_watch_threshold: float = 1.5


def _parse_float(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def load_pilot_config() -> PilotConfig:
    """
    Load pilot config from:
    - `NERAIUM_PILOT_CONFIG_PATH` (optional JSON file)
    - environment overrides (`NERAIUM_PILOT_DRIFT_HIGH_THRESHOLD`, `NERAIUM_PILOT_DRIFT_WATCH_THRESHOLD`)

    Falls back to safe defaults when inputs are missing/invalid.
    """

    raw_path = os.getenv("NERAIUM_PILOT_CONFIG_PATH")
    file_cfg: dict[str, object] = {}
    if raw_path:
        try:
            path = Path(raw_path)
            if path.exists():
                file_cfg = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            file_cfg = {}

    drift_high = _parse_float(
        os.getenv(
            "NERAIUM_PILOT_DRIFT_HIGH_THRESHOLD",
            file_cfg.get("drift_high_threshold", PilotConfig.drift_high_threshold),
        )
    )
    drift_watch = _parse_float(
        os.getenv(
            "NERAIUM_PILOT_DRIFT_WATCH_THRESHOLD",
            file_cfg.get("drift_watch_threshold", PilotConfig.drift_watch_threshold),
        )
    )

    return PilotConfig(
        drift_high_threshold=drift_high if drift_high is not None else PilotConfig.drift_high_threshold,
        drift_watch_threshold=drift_watch if drift_watch is not None else PilotConfig.drift_watch_threshold,
    )

