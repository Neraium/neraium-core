from __future__ import annotations

import importlib
import sys
from pathlib import Path


def _import_or_raise(module_name: str):
    return importlib.import_module(module_name)


def _load_neraium_core_modules() -> tuple[object, object, object, object]:
    try:
        csv_mapping = _import_or_raise("neraium_core.csv_mapping")
        logging_utils = _import_or_raise("neraium_core.logging_utils")
        service = _import_or_raise("neraium_core.service")
        store = _import_or_raise("neraium_core.store")
        return csv_mapping, logging_utils, service, store
    except ModuleNotFoundError:
        repo_root = Path(__file__).resolve().parents[2]
        root_text = str(repo_root)
        if root_text not in sys.path:
            sys.path.insert(0, root_text)
        try:
            csv_mapping = _import_or_raise("neraium_core.csv_mapping")
            logging_utils = _import_or_raise("neraium_core.logging_utils")
            service = _import_or_raise("neraium_core.service")
            store = _import_or_raise("neraium_core.store")
            return csv_mapping, logging_utils, service, store
        except ModuleNotFoundError as exc:  # pragma: no cover - boot-time diagnostic
            raise RuntimeError(
                "Unable to import neraium_core runtime modules. "
                "Install backend dependencies from apps/api/requirements.txt so pip also "
                "installs the local neraium-core package (from ../..)."
            ) from exc


csv_mapping, logging_utils, service, store = _load_neraium_core_modules()

infer_semantic_mapping = csv_mapping.infer_semantic_mapping
parse_csv_sample_for_mapping = csv_mapping.parse_csv_sample_for_mapping
resolve_mapping = csv_mapping.resolve_mapping
row_to_frame_kwargs = csv_mapping.row_to_frame_kwargs
validate_mapping = csv_mapping.validate_mapping

log_structured = logging_utils.log_structured
summarize_exception_for_logs = logging_utils.summarize_exception_for_logs

StructuralMonitoringService = service.StructuralMonitoringService
ResultStore = store.ResultStore


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
