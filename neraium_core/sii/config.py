from __future__ import annotations

from dataclasses import dataclass
import logging
import os

from .errors import SIIConfigurationError


_LOG = logging.getLogger("neraium.sii.config")


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    try:
        return float(raw.strip())
    except ValueError:
        _LOG.warning("invalid_float_env", extra={"key": name, "value": raw, "default": default})
        return float(default)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return int(default)
    try:
        return int(raw.strip())
    except ValueError:
        _LOG.warning("invalid_int_env", extra={"key": name, "value": raw, "default": default})
        return int(default)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


@dataclass(frozen=True)
class SIIConfig:
    baseline_window: int = 50
    recent_window: int = 12
    max_history: int = 500
    relation_threshold: float = 0.6
    graph_edge_threshold: float = 0.6
    regime_distance_threshold: float = 2.0
    baseline_adaptation_alpha: float = 0.92
    freeze_baseline_frames: int = 12
    regime_min_persistence: int = 3
    regime_max_prototypes: int = 5
    watch_threshold: float = 1.5
    alert_threshold: float = 2.5
    min_samples_for_alerts: int = 28
    allow_context_provider: bool = True
    regime_store_path: str = "sii_regimes.json"
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        if self.baseline_window < 4:
            raise SIIConfigurationError("baseline_window must be >= 4")
        if self.recent_window < 3:
            raise SIIConfigurationError("recent_window must be >= 3")
        if self.max_history < max(self.baseline_window, self.recent_window):
            raise SIIConfigurationError("max_history must be >= max(baseline_window, recent_window)")
        if not (0.0 < self.relation_threshold <= 1.0):
            raise SIIConfigurationError("relation_threshold must be in (0, 1]")
        if not (0.0 < self.graph_edge_threshold <= 1.0):
            raise SIIConfigurationError("graph_edge_threshold must be in (0, 1]")
        if self.regime_distance_threshold <= 0.0:
            raise SIIConfigurationError("regime_distance_threshold must be > 0")
        if not (0.0 <= self.baseline_adaptation_alpha < 1.0):
            raise SIIConfigurationError("baseline_adaptation_alpha must be in [0, 1)")
        if self.freeze_baseline_frames < 0:
            raise SIIConfigurationError("freeze_baseline_frames must be >= 0")
        if self.regime_min_persistence < 1:
            raise SIIConfigurationError("regime_min_persistence must be >= 1")
        if self.regime_max_prototypes < 1:
            raise SIIConfigurationError("regime_max_prototypes must be >= 1")
        if self.watch_threshold < 0.0 or self.alert_threshold < 0.0:
            raise SIIConfigurationError("watch_threshold and alert_threshold must be >= 0")
        if self.alert_threshold < self.watch_threshold:
            raise SIIConfigurationError("alert_threshold must be >= watch_threshold")
        if self.min_samples_for_alerts < 1:
            raise SIIConfigurationError("min_samples_for_alerts must be >= 1")
        if not str(self.regime_store_path).strip():
            raise SIIConfigurationError("regime_store_path cannot be empty")
        if not str(self.log_level).strip():
            raise SIIConfigurationError("log_level cannot be empty")

    @staticmethod
    def from_env() -> "SIIConfig":
        return SIIConfig(
            baseline_window=_env_int("SII_BASELINE_WINDOW", 50),
            recent_window=_env_int("SII_RECENT_WINDOW", 12),
            max_history=_env_int("SII_MAX_HISTORY", 500),
            relation_threshold=_env_float("SII_RELATION_THRESHOLD", 0.6),
            graph_edge_threshold=_env_float("SII_GRAPH_EDGE_THRESHOLD", 0.6),
            regime_distance_threshold=_env_float("SII_REGIME_DISTANCE_THRESHOLD", 2.0),
            baseline_adaptation_alpha=_env_float("SII_BASELINE_ADAPTATION_ALPHA", 0.92),
            freeze_baseline_frames=_env_int("SII_FREEZE_BASELINE_FRAMES", 12),
            regime_min_persistence=_env_int("SII_REGIME_MIN_PERSISTENCE", 3),
            regime_max_prototypes=_env_int("SII_REGIME_MAX_PROTOTYPES", 5),
            watch_threshold=_env_float("SII_WATCH_THRESHOLD", 1.5),
            alert_threshold=_env_float("SII_ALERT_THRESHOLD", 2.5),
            min_samples_for_alerts=_env_int("SII_MIN_SAMPLES_FOR_ALERTS", 28),
            allow_context_provider=_env_bool("SII_ALLOW_CONTEXT_PROVIDER", True),
            regime_store_path=os.getenv("SII_REGIME_STORE_PATH", "sii_regimes.json"),
            log_level=os.getenv("SII_LOG_LEVEL", "INFO").upper(),
        )


def load_sii_config() -> SIIConfig:
    return SIIConfig.from_env()

