from __future__ import annotations

import sys
from pathlib import Path


# Ensure the local repo root is importable for monorepo development while
# remaining safe when this file is deployed at a shallow path (e.g. /app).
_HERE = Path(__file__).resolve().parent
_repo_root_candidate = _HERE.parent.parent if len(_HERE.parents) >= 2 else _HERE
_REPO_ROOT = _repo_root_candidate if (_repo_root_candidate / "neraium_core").is_dir() else _HERE
_repo_root_str = str(_REPO_ROOT)
if _repo_root_str not in sys.path:
    sys.path.insert(0, _repo_root_str)

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
