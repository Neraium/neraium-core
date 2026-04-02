#!/usr/bin/env python3
"""Compatibility wrapper for the moved multinode summary helper."""

from __future__ import annotations

import runpy
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_TARGET = _ROOT / "experiments" / "scripts" / "show_summary.py"

if __name__ == "__main__":
    runpy.run_path(str(_TARGET), run_name="__main__")
