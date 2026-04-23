from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Callable

from .config import dir_is_writable


def validate_runtime_or_raise(runtime_status: dict[str, Any]) -> None:
    runtime_fallback_active = bool(runtime_status.get("using_fallback", False))
    runtime_env = str(
        os.getenv("NERAIUM_RUNTIME_MODE")
        or os.getenv("NERAIUM_RUNTIME_ENV")
        or os.getenv("NERAIUM_ENV")
        or os.getenv("APP_ENV")
        or ""
    ).strip().lower()
    if runtime_env in {"prod"}:
        runtime_env = "production"
    strict_runtime_flag = str(os.getenv("NERAIUM_REQUIRE_FULL_CORE_RUNTIME", "")).strip().lower()
    require_full_runtime = strict_runtime_flag in {"1", "true", "yes", "on"}
    if strict_runtime_flag == "":
        require_full_runtime = runtime_env in {"production"}
    if require_full_runtime and runtime_fallback_active:
        notes = ", ".join(str(x) for x in runtime_status.get("notes", []))
        raise RuntimeError(
            "Core runtime fallback is active while strict runtime mode is enabled. "
            f"runtime_mode={runtime_env or 'unknown'} notes={notes or 'n/a'}"
        )


def validate_startup_assumptions_or_raise(
    *,
    runtime_mode: str,
    persistence_available: bool,
    runtime_state_diagnostics: dict[str, Any],
) -> None:
    normalized_mode = str(runtime_mode or "").strip().lower()
    if normalized_mode == "prod":
        normalized_mode = "production"

    temp_dir_writable = bool(runtime_state_diagnostics.get("temp_dir_writable", False))
    db_path_writable = bool(runtime_state_diagnostics.get("db_path_writable", False))
    upload_temp_dir_writable = bool(runtime_state_diagnostics.get("upload_temp_dir_writable", False))

    if normalized_mode == "production":
        if not persistence_available:
            raise RuntimeError(
                "Startup validation failed: persistence is unavailable in production mode. "
                "Set NERAIUM_DB_PATH to a writable persistent path."
            )
        if not db_path_writable:
            raise RuntimeError(
                "Startup validation failed: database path directory is not writable in production mode."
            )
        if not temp_dir_writable or not upload_temp_dir_writable:
            raise RuntimeError(
                "Startup validation failed: temporary upload directory is not writable in production mode."
            )


def resolve_runtime_mode_from_env() -> str:
    runtime_mode = str(
        os.getenv("NERAIUM_RUNTIME_MODE")
        or os.getenv("NERAIUM_RUNTIME_ENV")
        or os.getenv("NERAIUM_ENV")
        or os.getenv("APP_ENV")
        or ""
    ).strip().lower()
    if runtime_mode in {"", "dev"}:
        return "development"
    if runtime_mode in {"prod"}:
        return "production"
    return runtime_mode


def build_runtime_state_diagnostics(
    *,
    request_body_limit: int,
    db_path: str,
    writable_checker: Callable[[str], bool] = dir_is_writable,
) -> dict[str, Any]:
    temp_dir = tempfile.gettempdir()
    return {
        "persisted_state_enabled": False,
        "persisted_state_store": "none",
        "request_body_limit_bytes": int(request_body_limit),
        "db_path": db_path,
        "db_path_writable": os.access(str(Path(db_path).parent), os.W_OK),
        "temp_dir": temp_dir,
        "temp_dir_writable": writable_checker(temp_dir),
        "upload_temp_dir": temp_dir,
        "upload_temp_dir_writable": writable_checker(temp_dir),
        "memory_only_state": [
            "pull_integration_worker_threads",
            "service_engine_runtime_memory",
        ],
    }
