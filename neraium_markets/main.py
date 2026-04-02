#!/usr/bin/env python3
"""Neraium Markets Day 1: load, validate, align, print."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root (this directory) is importable
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from neraium.alignment import align_close_series  # noqa: E402
from neraium.data_loader import load_all_assets  # noqa: E402
from neraium.validation import validate_all  # noqa: E402


def main() -> int:
    data = load_all_assets()
    errors = validate_all(data)
    if errors:
        print("Validation failed:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    merged = align_close_series(data)
    print("Merged shape:", merged.shape)
    print(merged.head(10).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
