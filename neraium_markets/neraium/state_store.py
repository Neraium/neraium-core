"""Day 14: persistent pipeline state (JSON, local-first)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config as project_config  # noqa: E402
from neraium.settings import Settings  # noqa: E402


def save_pipeline_state(state: dict[str, Any], path: str | Path) -> None:
    """Persist pipeline state as UTF-8 JSON (atomic write via temp + replace)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True, default=str), encoding="utf-8")
    tmp.replace(p)


def load_pipeline_state(path: str | Path) -> dict[str, Any] | None:
    """Load pipeline state JSON, or ``None`` if missing/unreadable."""
    p = Path(path)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def build_default_state(settings: Settings) -> dict[str, Any]:
    """Initial state when no prior run exists."""
    assets = list(settings.active_assets) if settings.active_assets else list(project_config.ASSETS)
    return {
        "version": 1,
        "last_successful_run_id": None,
        "latest_timestamp_by_asset": {},
        "latest_timestamp_by_timeframe": {},
        "source_type": settings.source_type,
        "active_assets": assets,
        "latest_health_status": None,
        "latest_manifest_path": None,
        "last_good_output_paths": {},
        "cache_key_meta": {},
        "updated_at": None,
    }
