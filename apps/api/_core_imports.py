from __future__ import annotations

import os
import sys
from pathlib import Path


_HERE = Path(__file__).resolve().parent


def _resolve_core_package_parent() -> Path | None:
    """Find a directory that contains the `neraium_core` package.

    Railway runs the API with `apps/api` as cwd, so we cannot assume repo root is
    already on PYTHONPATH. We support an explicit env override and a safe upward
    search from this file location.
    """

    configured = str(os.getenv("NERAIUM_CORE_PARENT", "")).strip()
    if configured:
        candidate = Path(configured).expanduser().resolve()
        if (candidate / "neraium_core" / "__init__.py").is_file():
            return candidate

    for parent in (_HERE, *_HERE.parents):
        if (parent / "neraium_core" / "__init__.py").is_file():
            return parent

    return None


_core_parent = _resolve_core_package_parent()
if _core_parent is not None:
    core_parent_str = str(_core_parent)
    if core_parent_str not in sys.path:
        sys.path.insert(0, core_parent_str)


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
        "Ensure the deployment includes the repository root or set "
        "NERAIUM_CORE_PARENT to a directory containing neraium_core/."
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
