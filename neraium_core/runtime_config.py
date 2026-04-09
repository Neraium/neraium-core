from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any


_RUNTIME_MODES = {"development", "staging", "production"}


def _parse_int_env(name: str, default: int, *, minimum: int) -> int:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return int(default)
    try:
        parsed = int(str(raw).strip())
    except ValueError as exc:
        raise ValueError(f"Invalid {name}: expected integer, got {raw!r}") from exc
    if parsed < minimum:
        raise ValueError(f"Invalid {name}: must be >= {minimum}, got {parsed}")
    return parsed


def _parse_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class RuntimeConfig:
    baseline_window: int
    recent_window: int
    log_level: str
    runtime_mode: str
    input_path: str | None
    data_path: str | None
    require_input_path: bool
    require_data_path: bool

    def validate(self) -> None:
        if self.runtime_mode not in _RUNTIME_MODES:
            allowed = ", ".join(sorted(_RUNTIME_MODES))
            raise ValueError(f"Invalid NERAIUM_RUNTIME_MODE: {self.runtime_mode!r}. Allowed: {allowed}")
        if self.baseline_window < 4:
            raise ValueError("Invalid NERAIUM_BASELINE_WINDOW: must be >= 4")
        if self.recent_window < 3:
            raise ValueError("Invalid NERAIUM_RECENT_WINDOW: must be >= 3")
        if self.baseline_window < self.recent_window:
            raise ValueError(
                "Invalid runtime window config: NERAIUM_BASELINE_WINDOW must be >= NERAIUM_RECENT_WINDOW"
            )
        if not self.log_level:
            raise ValueError("Invalid NERAIUM_LOG_LEVEL: cannot be empty")

        if self.require_input_path and not self.input_path:
            raise ValueError(
                "Missing required runtime configuration: NERAIUM_INPUT_PATH must be set when "
                "NERAIUM_REQUIRE_INPUT_PATH=1"
            )
        if self.input_path:
            path = Path(self.input_path)
            if not path.exists():
                raise ValueError(f"Invalid NERAIUM_INPUT_PATH: path does not exist: {self.input_path}")

        if self.require_data_path and not self.data_path:
            raise ValueError(
                "Missing required runtime configuration: NERAIUM_DATA_PATH must be set when "
                "NERAIUM_REQUIRE_DATA_PATH=1"
            )
        if self.data_path:
            path = Path(self.data_path)
            if not path.exists():
                raise ValueError(f"Invalid NERAIUM_DATA_PATH: path does not exist: {self.data_path}")

    def to_diagnostics(self) -> dict[str, Any]:
        return {
            "runtime_mode": self.runtime_mode,
            "baseline_window": self.baseline_window,
            "recent_window": self.recent_window,
            "log_level": self.log_level,
            "input_path_configured": bool(self.input_path),
            "data_path_configured": bool(self.data_path),
            "require_input_path": bool(self.require_input_path),
            "require_data_path": bool(self.require_data_path),
        }

    @staticmethod
    def from_env() -> "RuntimeConfig":
        cfg = RuntimeConfig(
            baseline_window=_parse_int_env("NERAIUM_BASELINE_WINDOW", 24, minimum=4),
            recent_window=_parse_int_env("NERAIUM_RECENT_WINDOW", 8, minimum=3),
            log_level=str(os.getenv("NERAIUM_LOG_LEVEL", "INFO")).strip().upper(),
            runtime_mode=str(os.getenv("NERAIUM_RUNTIME_MODE", "development")).strip().lower(),
            input_path=(str(os.getenv("NERAIUM_INPUT_PATH", "")).strip() or None),
            data_path=(str(os.getenv("NERAIUM_DATA_PATH", "")).strip() or None),
            require_input_path=_parse_bool_env("NERAIUM_REQUIRE_INPUT_PATH", False),
            require_data_path=_parse_bool_env("NERAIUM_REQUIRE_DATA_PATH", False),
        )
        cfg.validate()
        return cfg
