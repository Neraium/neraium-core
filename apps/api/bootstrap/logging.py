from __future__ import annotations

import json
import logging
import os
import traceback


class _JsonFormatter(logging.Formatter):
    """Emit one JSON object per log line for structured log aggregation."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = traceback.format_exception(*record.exc_info)
        extra_skip = {
            "args", "created", "exc_info", "exc_text", "filename", "funcName",
            "levelname", "levelno", "lineno", "message", "module", "msecs",
            "msg", "name", "pathname", "process", "processName", "relativeCreated",
            "stack_info", "taskName", "thread", "threadName",
        }
        for key, val in record.__dict__.items():
            if key not in extra_skip and not key.startswith("_"):
                payload[key] = val
        return json.dumps(payload, default=str)


def configure_logging() -> None:
    raw = str(os.getenv("NERAIUM_LOG_LEVEL", "INFO")).strip().upper()
    level = getattr(logging, raw, logging.INFO)

    runtime_mode = str(
        os.getenv("NERAIUM_RUNTIME_MODE")
        or os.getenv("NERAIUM_RUNTIME_ENV")
        or os.getenv("NERAIUM_ENV")
        or os.getenv("APP_ENV")
        or ""
    ).strip().lower()
    use_json = runtime_mode in {"production", "prod"} or str(os.getenv("NERAIUM_LOG_JSON", "")).strip().lower() in {"1", "true", "yes"}

    handler = logging.StreamHandler()
    if use_json:
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
