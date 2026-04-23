"""Day 12–13: deterministic configuration profiles, validation, and source selection."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import config as project_config  # noqa: E402

VALID_PROFILES = frozenset({"dev", "test", "prod_local"})

DEFAULT_SOURCE_QUALITY_THRESHOLDS: dict[str, float] = {
    "max_null_timestamp_ratio": 0.05,
    "max_null_close_ratio": 0.05,
    "max_duplicate_timestamp_ratio": 0.01,
    "min_rows": 5.0,
}

STRICT_SOURCE_QUALITY_THRESHOLDS: dict[str, float] = {
    "max_null_timestamp_ratio": 0.01,
    "max_null_close_ratio": 0.01,
    "max_duplicate_timestamp_ratio": 0.0,
    "min_rows": 10.0,
}


@dataclass(frozen=True)
class Settings:
    """Resolved settings for a Neraium Markets run (read-only advisory pipeline)."""

    profile: str
    project_root: Path
    data_dir: Path
    output_dir: Path
    latest_dir: Path
    runs_dir: Path
    manifests_dir: Path
    alerts_dir: Path
    evidence_dir: Path
    reviews_dir: Path
    health_dir: Path
    default_mode: str
    polling_interval_seconds: int
    max_cycles: int
    alert_suppression_window_seconds: float
    freshness_stale_hours: float
    save_outputs: bool
    active_timeframes: tuple[str, ...]
    active_assets: tuple[str, ...] | None
    # Day 13
    source_type: str
    mock_api_fixture_dir: Path | None
    mock_api_simulated_latency_ms: float
    source_quality_thresholds: dict[str, float]
    reject_invalid_assets: bool
    allowed_timeframes: tuple[str, ...]
    # Day 14
    state_dir: Path
    snapshot_dir: Path
    cache_dir: Path
    enable_cache: bool
    enable_incremental: bool
    incremental_lookback_buffer: int
    persist_input_snapshots: bool
    persist_output_snapshots: bool


def _profile_template(profile: str) -> dict[str, Any]:
    """Per-profile defaults (deterministic; paths are relative names under project_root)."""
    root = _PROJECT_ROOT
    common = {
        "data_rel": Path(project_config.DATA_DIR),
        "output_rel": Path("output"),
        "default_mode": "batch",
        "polling_interval_seconds": 60,
        "max_cycles": 1,
        "alert_suppression_window_seconds": 300.0,
        "freshness_stale_hours": 24.0,
        "active_timeframes": ("daily", "1h", "15m"),
        "active_assets": None,
        "allowed_timeframes": ("daily", "1h", "15m"),
    }
    if profile == "dev":
        return {
            **common,
            "save_outputs": False,
            "max_cycles": 1,
            "polling_interval_seconds": 60,
            "source_type": "csv",
            "mock_api_fixture_rel": None,
            "mock_api_simulated_latency_ms": 2.0,
            "source_quality_thresholds": dict(DEFAULT_SOURCE_QUALITY_THRESHOLDS),
            "reject_invalid_assets": False,
            "enable_cache": True,
            "enable_incremental": False,
            "incremental_lookback_buffer": 100,
            "persist_input_snapshots": False,
            "persist_output_snapshots": False,
        }
    if profile == "test":
        return {
            **common,
            "save_outputs": False,
            "max_cycles": 2,
            "polling_interval_seconds": 1,
            "freshness_stale_hours": 1.0,
            "source_type": "mock_api",
            "mock_api_fixture_rel": Path("tests/fixtures/mock_api"),
            "mock_api_simulated_latency_ms": 0.0,
            "source_quality_thresholds": dict(DEFAULT_SOURCE_QUALITY_THRESHOLDS),
            "reject_invalid_assets": False,
            "enable_cache": False,
            "enable_incremental": False,
            "incremental_lookback_buffer": 50,
            "persist_input_snapshots": True,
            "persist_output_snapshots": True,
        }
    if profile == "prod_local":
        return {
            **common,
            "save_outputs": True,
            "max_cycles": 10,
            "polling_interval_seconds": 30,
            "freshness_stale_hours": 48.0,
            "source_type": "csv",
            "mock_api_fixture_rel": None,
            "mock_api_simulated_latency_ms": 2.0,
            "source_quality_thresholds": dict(STRICT_SOURCE_QUALITY_THRESHOLDS),
            "reject_invalid_assets": True,
            "enable_cache": True,
            "enable_incremental": True,
            "incremental_lookback_buffer": 120,
            "persist_input_snapshots": True,
            "persist_output_snapshots": True,
        }
    raise ValueError(f"Unknown profile: {profile!r}")


def _env_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def load_settings(profile: str = "dev") -> Settings:
    """
    Load settings for ``profile`` (``dev`` | ``test`` | ``prod_local``).

    Environment overrides (optional):

    - ``NERAIUM_PROFILE``: profile name
    - ``NERAIUM_DATA_DIR``: path to CSV directory (absolute or relative to project root)
    - ``NERAIUM_OUTPUT_DIR``: base output directory (absolute or relative to project root)
    - ``NERAIUM_SAVE_OUTPUTS``: ``true`` / ``false`` — default artifact saving flag
    - ``NERAIUM_SOURCE_TYPE``: ``csv`` | ``mock_api``
    - ``NERAIUM_MOCK_API_FIXTURE_DIR``: override mock API fixture directory
    """
    prof = os.environ.get("NERAIUM_PROFILE", profile).strip() or "dev"
    if prof not in VALID_PROFILES:
        raise ValueError(f"Unknown profile {prof!r}; expected one of {sorted(VALID_PROFILES)}")

    tmpl = _profile_template(prof)
    root = _PROJECT_ROOT

    data_override = os.environ.get("NERAIUM_DATA_DIR")
    out_override = os.environ.get("NERAIUM_OUTPUT_DIR")

    data_rel = Path(data_override) if data_override else tmpl["data_rel"]
    out_rel = Path(out_override) if out_override else tmpl["output_rel"]

    data_dir = data_rel if data_rel.is_absolute() else (root / data_rel).resolve()
    output_dir = out_rel if out_rel.is_absolute() else (root / out_rel).resolve()

    save_outputs = _env_bool("NERAIUM_SAVE_OUTPUTS", bool(tmpl["save_outputs"]))

    latest_dir = output_dir / "latest"
    runs_dir = output_dir / "runs"
    manifests_dir = output_dir / "manifests"
    alerts_dir = output_dir / "alerts"
    evidence_dir = output_dir / "evidence"
    reviews_dir = output_dir / "reviews"
    health_dir = output_dir / "health"
    state_dir = output_dir / "state"
    snapshot_dir = output_dir / "snapshots"
    cache_dir = output_dir / "cache"

    assets = tmpl["active_assets"]
    if isinstance(assets, tuple):
        pass
    elif assets is None:
        assets = None
    else:
        assets = tuple(assets)

    source_type = os.environ.get("NERAIUM_SOURCE_TYPE", tmpl["source_type"]).strip().lower()
    mock_fix_rel = tmpl.get("mock_api_fixture_rel")
    mock_override = os.environ.get("NERAIUM_MOCK_API_FIXTURE_DIR")
    if mock_override:
        mpath = Path(mock_override)
        mock_api_fixture_dir = mpath if mpath.is_absolute() else (root / mpath).resolve()
    elif mock_fix_rel is not None:
        mock_api_fixture_dir = (root / mock_fix_rel).resolve()
    else:
        mock_api_fixture_dir = None

    sq = dict(tmpl["source_quality_thresholds"])

    return Settings(
        profile=prof,
        project_root=root,
        data_dir=data_dir,
        output_dir=output_dir,
        latest_dir=latest_dir,
        runs_dir=runs_dir,
        manifests_dir=manifests_dir,
        alerts_dir=alerts_dir,
        evidence_dir=evidence_dir,
        reviews_dir=reviews_dir,
        health_dir=health_dir,
        default_mode=str(tmpl["default_mode"]),
        polling_interval_seconds=int(tmpl["polling_interval_seconds"]),
        max_cycles=int(tmpl["max_cycles"]),
        alert_suppression_window_seconds=float(tmpl["alert_suppression_window_seconds"]),
        freshness_stale_hours=float(tmpl["freshness_stale_hours"]),
        save_outputs=save_outputs,
        active_timeframes=tuple(tmpl["active_timeframes"]),
        active_assets=assets,
        source_type=source_type,
        mock_api_fixture_dir=mock_api_fixture_dir,
        mock_api_simulated_latency_ms=float(tmpl["mock_api_simulated_latency_ms"]),
        source_quality_thresholds=sq,
        reject_invalid_assets=bool(tmpl["reject_invalid_assets"]),
        allowed_timeframes=tuple(tmpl["allowed_timeframes"]),
        state_dir=state_dir,
        snapshot_dir=snapshot_dir,
        cache_dir=cache_dir,
        enable_cache=bool(tmpl["enable_cache"]),
        enable_incremental=bool(tmpl["enable_incremental"]),
        incremental_lookback_buffer=int(tmpl["incremental_lookback_buffer"]),
        persist_input_snapshots=bool(tmpl["persist_input_snapshots"]),
        persist_output_snapshots=bool(tmpl["persist_output_snapshots"]),
    )


def with_overrides(base: Settings, **kwargs: Any) -> Settings:
    """Return a copy of ``base`` with dataclass fields replaced (only known keys)."""
    allowed = {f.name for f in Settings.__dataclass_fields__.values()}
    bad = set(kwargs) - allowed
    if bad:
        raise ValueError(f"Unknown settings keys: {sorted(bad)}")
    return replace(base, **kwargs)


def validate_settings(s: Settings) -> None:
    """Raise ``ValueError`` if settings are inconsistent or paths are unusable."""
    if s.profile not in VALID_PROFILES:
        raise ValueError(f"Invalid profile: {s.profile!r}")

    if s.default_mode not in {"batch", "realtime"}:
        raise ValueError(f"default_mode must be batch or realtime, got {s.default_mode!r}")

    if s.polling_interval_seconds < 1:
        raise ValueError("polling_interval_seconds must be >= 1")

    if s.max_cycles < 1:
        raise ValueError("max_cycles must be >= 1")

    if s.alert_suppression_window_seconds < 0:
        raise ValueError("alert_suppression_window_seconds must be >= 0")

    if s.freshness_stale_hours <= 0:
        raise ValueError("freshness_stale_hours must be > 0")

    if s.source_type not in {"csv", "mock_api"}:
        raise ValueError(f"source_type must be csv or mock_api, got {s.source_type!r}")

    if s.source_type == "csv" and not s.data_dir.is_dir():
        raise ValueError(f"data_dir does not exist or is not a directory: {s.data_dir}")

    if s.source_type == "mock_api":
        fx = s.mock_api_fixture_dir
        if fx is None or not fx.is_dir():
            raise ValueError(f"mock_api requires mock_api_fixture_dir to exist: {fx}")

    for tf in s.active_timeframes:
        if tf not in {"daily", "1h", "15m"}:
            raise ValueError(f"Unsupported timeframe in active_timeframes: {tf!r}")

    for tf in s.allowed_timeframes:
        if tf not in {"daily", "1h", "15m"}:
            raise ValueError(f"Unsupported timeframe in allowed_timeframes: {tf!r}")

    if s.active_assets is not None and len(s.active_assets) == 0:
        raise ValueError("active_assets, if set, must be non-empty")

    if s.incremental_lookback_buffer < 1:
        raise ValueError("incremental_lookback_buffer must be >= 1")

    try:
        s.output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ValueError(f"Cannot create output_dir {s.output_dir}: {exc}") from exc
