#!/usr/bin/env python3
"""Neraium Markets entrypoint: delegates to Day 12 CLI (batch, realtime, health, rebuild)."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from neraium.cli import dispatch_main  # noqa: E402


def main() -> int:
    return dispatch_main()


if __name__ == "__main__":
    raise SystemExit(main())
