from __future__ import annotations

try:
    from neraium_core.csv_mapping import (
        infer_semantic_mapping,
        parse_csv_sample_for_mapping,
        resolve_mapping,
        row_to_frame_kwargs,
        validate_mapping,
    )
    from neraium_core.logging_utils import log_structured, summarize_exception_for_logs
    from neraium_core.service import StructuralMonitoringService
    from neraium_core.store import ResultStore
except ModuleNotFoundError as exc:  # pragma: no cover - boot-time diagnostic
    raise RuntimeError(
        "Unable to import neraium_core runtime modules. "
        "Install backend dependencies from apps/api/requirements.txt so pip also "
        "installs the local neraium-core package (from ../..)."
    ) from exc


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
