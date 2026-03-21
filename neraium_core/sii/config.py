from __future__ import annotations

from dataclasses import dataclass
import os


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    try:
        return float(raw.strip())
    except ValueError:
        return float(default)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return int(default)
    try:
        return int(raw.strip())
    except ValueError:
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
        )


def load_sii_config() -> SIIConfig:
    return SIIConfig.from_env()

