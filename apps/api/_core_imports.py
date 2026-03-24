from __future__ import annotations

import sys
from pathlib import Path


_IMPORT_ERROR_MESSAGE = (
    "Unable to import neraium_core runtime modules. Ensure the repository root "
    "(containing the 'neraium_core' package) is available in the deploy image, "
    "or install neraium-core as a package during build."
)


def _ensure_repo_root_on_sys_path() -> None:
    """Add the nearest ancestor containing `neraium_core/` to sys.path."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "neraium_core").is_dir():
            parent_str = str(parent)
            if parent_str not in sys.path:
                sys.path.insert(0, parent_str)
            return


def _import_core_runtime_modules() -> tuple:
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
        return (
            infer_semantic_mapping,
            parse_csv_sample_for_mapping,
            resolve_mapping,
            row_to_frame_kwargs,
            validate_mapping,
            log_structured,
            summarize_exception_for_logs,
            StructuralMonitoringService,
            ResultStore,
        )
    except ModuleNotFoundError:
        _ensure_repo_root_on_sys_path()
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
            return (
                infer_semantic_mapping,
                parse_csv_sample_for_mapping,
                resolve_mapping,
                row_to_frame_kwargs,
                validate_mapping,
                log_structured,
                summarize_exception_for_logs,
                StructuralMonitoringService,
                ResultStore,
            )
        except ModuleNotFoundError as exc:
            raise RuntimeError(_IMPORT_ERROR_MESSAGE) from exc


(
    infer_semantic_mapping,
    parse_csv_sample_for_mapping,
    resolve_mapping,
    row_to_frame_kwargs,
    validate_mapping,
    log_structured,
    summarize_exception_for_logs,
    StructuralMonitoringService,
    ResultStore,
) = _import_core_runtime_modules()


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
