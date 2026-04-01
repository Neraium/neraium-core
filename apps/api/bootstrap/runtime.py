from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Callable

from .config import dir_is_writable


def validate_runtime_or_raise(runtime_status: dict[str, Any]) -> None:
    runtime_fallback_active = bool(runtime_status.get("using_fallback", False))
    runtime_env = str(
        os.getenv("NERAIUM_RUNTIME_ENV")
        or os.getenv("NERAIUM_ENV")
        or os.getenv("APP_ENV")
        or ""
    ).strip().lower()
    strict_runtime_flag = str(os.getenv("NERAIUM_REQUIRE_FULL_CORE_RUNTIME", "")).strip().lower()
    require_full_runtime = strict_runtime_flag in {"1", "true", "yes", "on"}
    if strict_runtime_flag == "":
        require_full_runtime = runtime_env in {"prod", "production"}
    if require_full_runtime and runtime_fallback_active:
        notes = ", ".join(str(x) for x in runtime_status.get("notes", []))
        raise RuntimeError(
            "Core runtime fallback is active while strict runtime mode is enabled. "
            f"runtime_env={runtime_env or 'unknown'} notes={notes or 'n/a'}"
        )


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
            "demo_jobs",
            "cmapss_fd004_cache",
            "pull_integration_worker_threads",
            "service_engine_runtime_memory",
        ],
    }
