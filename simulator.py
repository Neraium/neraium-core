#!/usr/bin/env python3
"""Compatibility wrapper for the moved telemetry simulator script."""

from __future__ import annotations

import runpy
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_TARGET = _ROOT / "experiments" / "scripts" / "simulator.py"

if __name__ == "__main__":
    runpy.run_path(str(_TARGET), run_name="__main__")
