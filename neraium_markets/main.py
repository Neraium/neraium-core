from __future__ import annotations

import sys

from neraium.alignment import align_close_series
from neraium.data_loader import load_all_assets
from neraium.validation import validate_all


def main() -> int:
    data = load_all_assets()
    errors = validate_all(data)

    if errors:
        print("Validation failed:")
        for err in errors:
            print(f"- {err}")
        return 1

    merged = align_close_series(data)
    print(f"Merged shape: {merged.shape}")
    print(merged.head(10).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
