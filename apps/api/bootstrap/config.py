from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_MAX_REQUEST_BODY_BYTES = 50 * 1024 * 1024
# Keep parser allowance above app-level request cap so oversize requests
# are handled by middleware with a clean 413 response instead of reset.
DEFAULT_UVICORN_H11_MAX_INCOMPLETE_EVENT_SIZE = 64 * 1024 * 1024
DEFAULT_CORS_ALLOW_ORIGINS: tuple[str, ...] = ()
DEFAULT_CORS_ALLOW_HEADERS = (
    "Content-Type",
    "Authorization",
    "X-API-Key",
    "Accept",
    # Browser tracing stacks (Sentry/OpenTelemetry) can attach these automatically.
    "baggage",
    "sentry-trace",
    "traceparent",
    "tracestate",
)


def request_body_limit_bytes() -> int:
    raw = os.getenv("NERAIUM_MAX_REQUEST_BODY_BYTES")
    if not raw:
        return DEFAULT_MAX_REQUEST_BODY_BYTES
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "Invalid NERAIUM_MAX_REQUEST_BODY_BYTES=%r; using default=%s",
            raw,
            DEFAULT_MAX_REQUEST_BODY_BYTES,
        )
        return DEFAULT_MAX_REQUEST_BODY_BYTES
    return max(value, DEFAULT_MAX_REQUEST_BODY_BYTES)


def uvicorn_h11_max_incomplete_event_size() -> int:
    raw = os.getenv("NERAIUM_UVICORN_H11_MAX_INCOMPLETE_EVENT_SIZE")
    if not raw:
        return DEFAULT_UVICORN_H11_MAX_INCOMPLETE_EVENT_SIZE
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "Invalid NERAIUM_UVICORN_H11_MAX_INCOMPLETE_EVENT_SIZE=%r; using default=%s",
            raw,
            DEFAULT_UVICORN_H11_MAX_INCOMPLETE_EVENT_SIZE,
        )
        return DEFAULT_UVICORN_H11_MAX_INCOMPLETE_EVENT_SIZE
    return max(value, DEFAULT_UVICORN_H11_MAX_INCOMPLETE_EVENT_SIZE)


def cors_allow_origins() -> list[str]:
    raw = str(os.getenv("NERAIUM_CORS_ALLOW_ORIGINS") or "").strip()
    configured = [x.strip() for x in raw.split(",") if x.strip()] if raw else []
    merged: list[str] = []
    for origin in [*DEFAULT_CORS_ALLOW_ORIGINS, *configured]:
        if origin and origin not in merged:
            merged.append(origin)
    return merged


def cors_allow_headers() -> list[str]:
    raw = str(os.getenv("NERAIUM_CORS_ALLOW_HEADERS") or "").strip()
    configured = [x.strip() for x in raw.split(",") if x.strip()] if raw else []
    merged: list[str] = []
    seen: set[str] = set()
    for header in [*DEFAULT_CORS_ALLOW_HEADERS, *configured]:
        normalized = header.lower() if header else ""
        if header and normalized not in seen:
            merged.append(header)
            seen.add(normalized)
    return merged


def cors_allow_origin_regex() -> str | None:
    raw = str(os.getenv("NERAIUM_CORS_ALLOW_ORIGIN_REGEX") or "").strip()
    return raw or None


def persistence_available(db_path: str) -> bool:
    try:
        db_file = Path(db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)
        with db_file.open("a", encoding="utf-8"):
            pass
        return True
    except OSError:
        return False


def dir_is_writable(path: str) -> bool:
    try:
        fd, probe_path = tempfile.mkstemp(prefix="neraium_probe_", dir=path)
        os.close(fd)
        probe = Path(probe_path)
        probe.unlink(missing_ok=True)
        return True
    except OSError:
        return False


def resolve_db_path(configured_db_path: str) -> tuple[str, bool]:
    """Return a writable SQLite path and whether persistence is available."""
    configured = str(configured_db_path or "").strip() or "neraium.db"
    if persistence_available(configured):
        return configured, True

    fallback = "/tmp/neraium.db"
    if persistence_available(fallback):
        logger.warning(
            "Configured NERAIUM_DB_PATH=%s is not writable; falling back to %s.",
            configured,
            fallback,
        )
        return fallback, True

    logger.error(
        "Configured NERAIUM_DB_PATH=%s and fallback=%s are not writable; using in-memory SQLite store.",
        configured,
        fallback,
    )
    return ":memory:", False
