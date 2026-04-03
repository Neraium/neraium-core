"""
Compatibility wrapper for FD004 real-data evaluation.

The implementation lives under `examples/fd004/fd004_real.py`.
"""

from __future__ import annotations

from examples.fd004.fd004_real import (  # type: ignore
    Fd004Row,
    fd004_row_to_sii_record,
    load_fd004_dataset,
    run_fd004_real_evaluation,
)

__all__ = [
    "Fd004Row",
    "fd004_row_to_sii_record",
    "load_fd004_dataset",
    "run_fd004_real_evaluation",
]

