"""Generate synthetic sample_data CSVs (run from repo: python tools/generate_sample_data.py)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import config  # noqa: E402


def main() -> None:
    out_dir = _ROOT / config.DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    dates = pd.bdate_range("2024-01-02", periods=35, freq="B")

    for seed, asset in enumerate(config.ASSETS):
        r = np.random.default_rng(42 + seed)
        close = 50.0 + seed * 3.0 + np.cumsum(r.normal(0, 0.4, size=len(dates)))
        open_ = np.roll(close, 1)
        open_[0] = close[0]
        high = np.maximum(open_, close) + r.uniform(0.05, 0.8, len(dates))
        low = np.minimum(open_, close) - r.uniform(0.05, 0.8, len(dates))
        volume = r.integers(100_000, 9_000_000, size=len(dates))
        df = pd.DataFrame(
            {
                "timestamp": dates.strftime("%Y-%m-%d"),
                "open": np.round(open_, 4),
                "high": np.round(high, 4),
                "low": np.round(low, 4),
                "close": np.round(close, 4),
                "volume": volume,
            }
        )
        path = out_dir / f"{asset.lower()}.csv"
        df.to_csv(path, index=False)
        print("Wrote", path)


if __name__ == "__main__":
    main()
