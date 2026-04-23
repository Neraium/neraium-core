"""Day 12: safe pipeline execution, output layout, atomic writes, fallbacks."""

from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TypeVar

from neraium.settings import Settings

T = TypeVar("T")


def ensure_output_layout(settings: Settings) -> None:
    """Create standard subdirectories under ``settings.output_dir``."""
    for d in (
        settings.latest_dir,
        settings.runs_dir,
        settings.manifests_dir,
        settings.alerts_dir,
        settings.evidence_dir,
        settings.reviews_dir,
        settings.health_dir,
        settings.state_dir,
        settings.snapshot_dir,
        settings.cache_dir,
    ):
        d.mkdir(parents=True, exist_ok=True)


def evidence_log_path(settings: Settings) -> Path:
    return settings.evidence_dir / "evidence_log.jsonl"


def alert_log_path(settings: Settings) -> Path:
    return settings.alerts_dir / "alert_log.jsonl"


def health_log_path(settings: Settings) -> Path:
    return settings.health_dir / "health_log.jsonl"


def operator_inbox_path(settings: Settings) -> Path:
    return settings.reviews_dir / "operator_inbox.csv"


def realtime_snapshot_path(settings: Settings) -> Path:
    return settings.latest_dir / "realtime_snapshot.json"


def run_status_path(settings: Settings) -> Path:
    return settings.latest_dir / "run_status.json"


def pipeline_state_path(settings: Settings) -> Path:
    """Latest pipeline state JSON (Day 14)."""
    return settings.latest_dir / "pipeline_state.json"


def run_resume_summary_path(settings: Settings) -> Path:
    """Resume / incremental operator summary (Day 14)."""
    return settings.latest_dir / "run_resume_summary.json"


def atomic_write_json(path: Path, obj: dict[str, Any]) -> None:
    """Write JSON atomically (write temp + replace) to avoid partial files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    data = json.dumps(obj, indent=2, sort_keys=True, default=str)
    tmp.write_text(data, encoding="utf-8")
    tmp.replace(path)


def safe_run_pipeline(
    fn: Callable[[], T],
    *,
    stage: str = "pipeline",
    output_dir: Path | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """
    Run ``fn`` and return a structured status dict (never raises from ``fn``).

    Fields: ``success``, ``error_message``, ``stage_failed``, ``latest_available_output``,
    ``health_status``, ``result`` (on success).
    """
    latest_out: str | None = None
    if output_dir is None and settings is not None:
        output_dir = settings.output_dir
    if output_dir is not None:
        fb = fallback_to_last_good_output(output_dir)
        latest_out = fb.get("best_path")

    try:
        result = fn()
        return {
            "success": True,
            "error_message": None,
            "stage_failed": None,
            "latest_available_output": latest_out,
            "health_status": "ok",
            "result": result,
        }
    except Exception as exc:  # noqa: BLE001 — intentional broad catch for operator safety
        return {
            "success": False,
            "error_message": str(exc),
            "stage_failed": stage,
            "latest_available_output": latest_out,
            "health_status": "degraded",
            "result": None,
        }


def fallback_to_last_good_output(output_dir: Path | str) -> dict[str, Any]:
    """
    Best-effort pointer to the latest usable artifacts under ``output_dir``.

    Prefers ``latest/run_status.json`` (manifest_path, outputs), then newest file in
    ``manifests/``, then ``latest/realtime_snapshot.json``.
    """
    base = Path(output_dir)
    latest_status = base / "latest" / "run_status.json"
    best_path: str | None = None
    manifest_path: str | None = None

    if latest_status.is_file():
        try:
            data = json.loads(latest_status.read_text(encoding="utf-8"))
            manifest_path = data.get("manifest_path")
            outs = data.get("outputs_written") or {}
            if isinstance(outs, dict) and outs:
                # pick first path value
                best_path = str(next(iter(outs.values())))
            elif manifest_path:
                best_path = manifest_path
        except (OSError, json.JSONDecodeError, StopIteration):
            pass

    manifests_dir = base / "manifests"
    if best_path is None and manifests_dir.is_dir():
        files = sorted(
            manifests_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if files:
            manifest_path = str(files[0])
            best_path = manifest_path

    snap = base / "latest" / "realtime_snapshot.json"
    if best_path is None and snap.is_file():
        best_path = str(snap)

    return {
        "best_path": best_path,
        "manifest_path": manifest_path,
        "run_status_path": str(latest_status) if latest_status.is_file() else None,
    }


def new_run_id() -> str:
    return str(uuid.uuid4())


def build_run_status_payload(
    *,
    run_id: str,
    mode: str,
    success: bool,
    settings: Settings,
    manifest_path: str | None,
    outputs_written: dict[str, str],
    error_message: str | None,
    health_status: str | None,
    latest_timestamp_processed: str | None = None,
    alerts_fired: int = 0,
    day14: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "success": success,
        "profile": settings.profile,
        "latest_timestamp_processed": latest_timestamp_processed,
        "alerts_fired": alerts_fired,
        "outputs_written": outputs_written,
        "health_status": health_status,
        "manifest_path": manifest_path,
        "error_message": error_message,
    }
    if day14:
        out["day14"] = day14
    return out


def save_run_status(settings: Settings, payload: dict[str, Any]) -> Path:
    """Write ``output/latest/run_status.json`` atomically."""
    p = run_status_path(settings)
    atomic_write_json(p, payload)
    return p


def copy_tree_to_run_folder(settings: Settings, run_id: str, paths: list[Path]) -> Path | None:
    """Optional copy of key artifacts into ``runs/{run_id}/`` for retention."""
    dest = settings.runs_dir / run_id
    if not paths:
        return None
    dest.mkdir(parents=True, exist_ok=True)
    for src in paths:
        if src.is_file():
            shutil.copy2(src, dest / src.name)
    return dest
