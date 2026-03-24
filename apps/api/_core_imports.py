from __future__ import annotations

import sys
from pathlib import Path


# Railway runs `uvicorn main:app` from `apps/api`. Ensure repo root is importable
# without requiring PYTHONPATH configuration.
_REPO_ROOT = Path(__file__).resolve().parents[2]
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
