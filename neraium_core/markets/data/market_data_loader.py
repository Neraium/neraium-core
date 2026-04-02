"""Load market telemetry from local CSV files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class MarketDataLoader:
    data_dir: Path

    def load(self, timeframe: str) -> pd.DataFrame:
        frame_dir = self.data_dir / timeframe
        if not frame_dir.exists():
            raise FileNotFoundError(f"Missing timeframe folder: {frame_dir}")

        pieces: list[pd.DataFrame] = []
        for csv_path in sorted(frame_dir.glob("*.csv")):
            asset = csv_path.stem.upper()
            df = pd.read_csv(csv_path)
            if "timestamp" not in df.columns or "close" not in df.columns:
                raise ValueError(f"{csv_path} must contain timestamp and close columns")
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.sort_values("timestamp")
            pieces.append(df[["timestamp", "close"]].rename(columns={"close": asset}).set_index("timestamp"))

        if not pieces:
            raise ValueError(f"No CSV files found in {frame_dir}")

        return pd.concat(pieces, axis=1).sort_index()
