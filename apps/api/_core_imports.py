from __future__ import annotations

"""Runtime imports for the API app that work from the `apps/api` root.

This module intentionally does **not** import from `neraium_core.*` so Railway can
boot when only `apps/api` is mounted as `/app`.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable


try:
    from csv_mapping import (
        infer_semantic_mapping,
        parse_csv_sample_for_mapping,
        resolve_mapping,
        row_to_frame_kwargs,
        validate_mapping,
    )
except ImportError:
    # Fallback stubs keep app startup healthy when core modules are not vendored.
    def parse_csv_sample_for_mapping(csv_text: str, max_rows: int = 16) -> tuple[list[str], list[dict[str, str]]]:
        _ = max_rows
        lines = [ln for ln in str(csv_text).splitlines() if ln.strip()]
        if not lines:
            return [], []
        headers = [h.strip() for h in lines[0].split(",") if h.strip()]
        return headers, []

    def infer_semantic_mapping(headers: list[str], *, sample_rows: list[dict[str, str]] | None = None) -> tuple[Any, list[str], dict[str, Any]]:
        _ = sample_rows
        if not headers:
            return None, ["CSV has no header columns."], {}
        return None, ["Semantic mapping module not available in apps/api runtime image."], {"headers": headers}

    def resolve_mapping(mapping: Any, headers: list[str]) -> Any:
        _ = headers
        return mapping

    def validate_mapping(mapping: Any, headers: list[str]) -> list[str]:
        _ = mapping
        _ = headers
        return []

    def row_to_frame_kwargs(row: dict[str, Any], mapping: Any) -> dict[str, Any]:
        _ = mapping
        return {
            "timestamp": row.get("timestamp"),
            "site_id": row.get("site_id"),
            "asset_id": row.get("asset_id"),
            "sensor_values": row.get("sensor_values") if isinstance(row.get("sensor_values"), dict) else {},
        }


try:
    from logging_utils import log_structured, summarize_exception_for_logs
except ImportError:
    import logging

    def log_structured(logger: logging.Logger, *, event: str, fields: dict[str, Any], level: int = logging.INFO) -> None:
        logger.log(level, "event=%s fields=%s", event, fields)

    def summarize_exception_for_logs(exc: Exception) -> dict[str, Any]:
        return {"error_type": type(exc).__name__, "error_message": str(exc)}


try:
    from store import ResultStore
except ImportError:
    class ResultStore:
        def __init__(self, db_path: str = "neraium.db"):
            self.db_path = db_path


@dataclass
class _InMemoryRun:
    run_id: str
    customer_id: str
    name: str
    status: str = "active"


try:
    from service import StructuralMonitoringService
except ImportError:
    class StructuralMonitoringService:
        """Boot-safe fallback service when core runtime modules are unavailable."""

        def __init__(self, store: ResultStore | None = None, **_: Any) -> None:
            self.store = store or ResultStore()
            self._runs: dict[tuple[str, str], _InMemoryRun] = {}

        @staticmethod
        def _resolve_customer(customer_id: str | None) -> str:
            text = str(customer_id or "").strip()
            return text or "default-customer"

        def get_latest_result(self, **_: Any) -> dict[str, Any] | None:
            return None

        def list_results(self, **_: Any) -> list[dict[str, Any]]:
            return []

        def reset(self) -> None:
            return

        def create_run(self, *, name: str, config: dict[str, Any], activate: bool = True, customer_id: str | None = None) -> dict[str, Any]:
            _ = config
            customer = self._resolve_customer(customer_id)
            run_id = f"run_{len(self._runs) + 1}"
            status = "active" if activate else "draft"
            run = _InMemoryRun(run_id=run_id, customer_id=customer, name=name, status=status)
            if activate:
                for key, existing in list(self._runs.items()):
                    if key[0] == customer:
                        existing.status = "inactive"
            self._runs[(customer, run_id)] = run
            return {
                "run_id": run.run_id,
                "customer_id": run.customer_id,
                "name": run.name,
                "status": run.status,
                "config": {},
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }

        def list_runs(self, *, customer_id: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
            customer = self._resolve_customer(customer_id)
            runs = [r for (cid, _), r in self._runs.items() if cid == customer][:limit]
            return [
                {
                    "run_id": r.run_id,
                    "customer_id": r.customer_id,
                    "name": r.name,
                    "status": r.status,
                    "config": {},
                }
                for r in runs
            ]

        def get_active_run(self, *, customer_id: str | None = None) -> dict[str, Any] | None:
            customer = self._resolve_customer(customer_id)
            for (cid, _), r in self._runs.items():
                if cid == customer and r.status == "active":
                    return {
                        "run_id": r.run_id,
                        "customer_id": r.customer_id,
                        "name": r.name,
                        "status": r.status,
                        "config": {},
                    }
            return None

        def get_run(self, run_id: str, *, customer_id: str | None = None) -> dict[str, Any] | None:
            customer = self._resolve_customer(customer_id)
            r = self._runs.get((customer, run_id))
            if r is None:
                return None
            return {
                "run_id": r.run_id,
                "customer_id": r.customer_id,
                "name": r.name,
                "status": r.status,
                "config": {},
            }

        def __getattr__(self, name: str) -> Callable[..., Any]:
            if name.startswith("__"):
                raise AttributeError(name)

            def _missing(*_: Any, **__: Any) -> Any:
                raise RuntimeError(
                    f"Core runtime operation '{name}' is unavailable in this Railway image. "
                    "Deploy with app-local core modules under apps/api to enable full functionality."
                )

            return _missing


__all__ = [
    "ResultStore",
    "StructuralMonitoringService",
    "log_structured",
    "infer_semantic_mapping",
    "parse_csv_sample_for_mapping",
    "resolve_mapping",
    "row_to_frame_kwargs",
    "validate_mapping",
    "summarize_exception_for_logs",
]
