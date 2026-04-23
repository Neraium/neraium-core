"""Day 12: run manifests for reproducibility and inspection."""

from __future__ import annotations

import json
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config as project_config  # noqa: E402

from neraium.settings import Settings


def _git_commit_short(project_root: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if out.returncode != 0 or not out.stdout.strip():
            return None
        return out.stdout.strip()[:40]
    except (OSError, subprocess.SubprocessError):
        return None


def build_run_manifest(
    settings: Settings,
    mode: str,
    output_paths: dict[str, str | Path],
    *,
    run_id: str | None = None,
    success: bool = True,
    error_message: str | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable manifest for one run."""
    rid = run_id or str(uuid.uuid4())
    ts = datetime.now(timezone.utc).isoformat()
    assets = list(settings.active_assets) if settings.active_assets else list(project_config.ASSETS)
    snap = {
        "profile": settings.profile,
        "data_dir": str(settings.data_dir),
        "output_dir": str(settings.output_dir),
        "default_mode": settings.default_mode,
        "polling_interval_seconds": settings.polling_interval_seconds,
        "max_cycles": settings.max_cycles,
        "alert_suppression_window_seconds": settings.alert_suppression_window_seconds,
        "freshness_stale_hours": settings.freshness_stale_hours,
        "save_outputs": settings.save_outputs,
        "active_timeframes": list(settings.active_timeframes),
        "active_assets": assets,
        "source_type": settings.source_type,
        "reject_invalid_assets": settings.reject_invalid_assets,
        "allowed_timeframes": list(settings.allowed_timeframes),
    }
    out_files: dict[str, str] = {}
    for k, v in output_paths.items():
        out_files[str(k)] = str(Path(v))

    manifest: dict[str, Any] = {
        "run_id": rid,
        "timestamp": ts,
        "profile": settings.profile,
        "mode": mode,
        "timeframes_used": list(settings.active_timeframes),
        "assets_used": assets,
        "output_files": out_files,
        "git_commit": _git_commit_short(settings.project_root),
        "settings_snapshot": snap,
        "success": bool(success),
        "error_message": error_message,
    }
    if extra_metadata:
        manifest["extra"] = extra_metadata
    return manifest


def save_run_manifest(manifest: dict[str, Any], output_path: str | Path) -> Path:
    """Write ``manifest`` to ``output_path`` (UTF-8 JSON)."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(manifest, indent=2, sort_keys=True, default=str)
    p.write_text(text, encoding="utf-8")
    return p
