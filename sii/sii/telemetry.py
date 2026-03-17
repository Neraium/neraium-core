from __future__ import annotations

import pandas as pd

from sii.models import TelemetryPoint


def points_to_frame(points: list[TelemetryPoint]) -> pd.DataFrame:
    rows = [{"timestamp": p.timestamp, **p.signals} for p in points]
    if not rows:
        return pd.DataFrame(columns=["timestamp"]).set_index("timestamp")
    df = pd.DataFrame(rows).sort_values("timestamp")
    df = df.set_index("timestamp")
    return df.astype(float)


def append_frame(base: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    if base.empty:
        merged = incoming
    else:
        merged = pd.concat([base, incoming], axis=0)
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    return merged.ffill().bfill()
