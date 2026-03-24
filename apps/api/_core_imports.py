from __future__ import annotations

import sys
from pathlib import Path


# Ensure core modules are importable when apps/api is the runtime root (e.g. Railway).
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = next((parent for parent in (_HERE, *_HERE.parents) if (parent / "neraium_core").is_dir()), _HERE)
_CORE_DIR = _REPO_ROOT / "neraium_core"

for path in (_REPO_ROOT, _CORE_DIR):
    path_str = str(path)
    if path.is_dir() and path_str not in sys.path:
        sys.path.insert(0, path_str)

from csv_mapping import (
    infer_semantic_mapping,
    parse_csv_sample_for_mapping,
    resolve_mapping,
    row_to_frame_kwargs,
    validate_mapping,
)
from logging_utils import log_structured, summarize_exception_for_logs
from service import StructuralMonitoringService
from store import ResultStore

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
